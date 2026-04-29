"""
Content-addressed cache for Hailo-derived vision features (Phase H).

Per the YouTube visual-intelligence plan, every feature dict produced by
the hailo-vision pipeline is keyed by:

    (sha256(asset_bytes), model_version, pipeline_version)

Same media + same models + same pipeline rev = read from cache.
Bumping any one of those three components produces a fresh row; older
rows persist for archival lookups but are never returned for a key
that no longer matches.

Storage:
  - SQLite at ``<cache_dir>/cache.db``, table ``vision_facts``.
  - Parquet sidecar at ``<cache_dir>/embeddings.parquet`` for CLIP
    vectors. Embeddings are kept out of SQLite to keep row size small;
    pyarrow is required only when an embedding is actually written or
    read.

Eviction: none. Vision-fact rows are ~10 KB JSON each, embeddings are
~2 KB Parquet each; 10k assets ≈ 100 MB.

Thread/process safety: WAL mode is enabled so multi-reader is safe;
writes serialize via SQLite's BEGIN IMMEDIATE. The Parquet sidecar
is rewritten atomically via os.replace.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_CACHE_DIR = Path.home() / "openclaw-output" / "hailo-vision-cache"
DEFAULT_PIPELINE_FALLBACK = "v0.1.0"

EMBEDDING_KEY = "hailo_clip_embedding"
_HAS_EMBEDDING_FLAG = "_has_embedding"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vision_facts (
    asset_path       TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    payload          TEXT NOT NULL,
    computed_at      TEXT NOT NULL,
    PRIMARY KEY (content_hash, model_version, pipeline_version)
);
CREATE INDEX IF NOT EXISTS idx_vf_asset ON vision_facts(asset_path);
CREATE INDEX IF NOT EXISTS idx_vf_computed ON vision_facts(computed_at);
"""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of ``path``'s bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint_hef_dir(
    models_dir: str | Path,
    hefs: Iterable[str] | None = None,
) -> str:
    """Hash a directory of HEFs into a stable model_version string.

    The fingerprint is SHA-256 over ``b'<basename>:<sha256_of_file>\\n'`` for
    each HEF, in lexicographic basename order. Missing files are folded in
    as the literal token ``b'<missing><basename>'`` so a partial set still
    produces a well-defined version.

    Args:
        models_dir: directory holding ``.hef`` files.
        hefs: explicit basenames (in caller-defined order). If None, all
            ``*.hef`` in ``models_dir`` are used, sorted lexicographically.
    """
    md = hashlib.sha256()
    md_path = Path(models_dir)
    if hefs is None:
        hef_paths = sorted(p for p in md_path.glob("*.hef") if p.is_file())
    else:
        hef_paths = [md_path / h for h in hefs]
    for p in hef_paths:
        if not p.exists():
            md.update(b"<missing>")
            md.update(p.name.encode())
            md.update(b"\n")
            continue
        md.update(p.name.encode())
        md.update(b":")
        md.update(sha256_file(p).encode())
        md.update(b"\n")
    return md.hexdigest()


def fingerprint_pipeline(
    repo_dir: str | Path | None = None,
    fallback: str = DEFAULT_PIPELINE_FALLBACK,
) -> str:
    """Return a short git rev for ``repo_dir``; fall back to ``fallback``.

    Truncates to 12 hex chars (matches ``git rev-parse --short=12``). On
    any error (no git, not a repo, timeout) returns ``fallback`` so cache
    keys remain stable for un-versioned deployments.
    """
    if repo_dir is None:
        return fallback
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        rev = out.stdout.strip()
        return rev[:12] if rev else fallback
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return fallback


@dataclass(frozen=True)
class CacheKey:
    content_hash: str
    model_version: str
    pipeline_version: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.content_hash, self.model_version, self.pipeline_version)


class VisionCache:
    """Content-addressed cache for Hailo vision features."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        model_version: str,
        pipeline_version: str,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self.parquet_path = self.cache_dir / "embeddings.parquet"
        self.model_version = model_version
        self.pipeline_version = pipeline_version
        self._init_db()

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            yield c
        finally:
            c.close()

    def _init_db(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)
            c.commit()

    def _key_for(self, image_path: Path) -> CacheKey:
        return CacheKey(
            content_hash=sha256_file(image_path),
            model_version=self.model_version,
            pipeline_version=self.pipeline_version,
        )

    def get(self, image_path: str | Path) -> dict[str, Any] | None:
        """Return cached features for ``image_path`` or None on miss.

        A miss is any of: file does not exist, no row matches the
        ``(content_hash, model_version, pipeline_version)`` triple.
        """
        p = Path(image_path)
        if not p.exists() or not p.is_file():
            return None
        key = self._key_for(p)
        with self._conn() as c:
            row = c.execute(
                "SELECT payload FROM vision_facts "
                "WHERE content_hash=? AND model_version=? AND pipeline_version=?",
                key.as_tuple(),
            ).fetchone()
        if row is None:
            return None
        payload: dict[str, Any] = json.loads(row[0])
        if payload.pop(_HAS_EMBEDDING_FLAG, False):
            payload[EMBEDDING_KEY] = self._read_embedding(key.content_hash)
        return payload

    def put(self, image_path: str | Path, features: dict[str, Any]) -> None:
        """Persist ``features`` for ``image_path``. Overwrites any existing row.

        If the dict carries a non-empty list under ``hailo_clip_embedding``,
        the vector is moved to the Parquet sidecar and replaced with None
        in the JSON payload (a flag preserves the round-trip on read).
        The caller's dict is not mutated.
        """
        p = Path(image_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"cannot cache features for missing file: {p}")
        key = self._key_for(p)

        compact = dict(features)
        emb = compact.get(EMBEDDING_KEY)
        if isinstance(emb, list) and emb:
            compact[EMBEDDING_KEY] = None
            compact[_HAS_EMBEDDING_FLAG] = True
            self._write_embedding(key.content_hash, emb)
        else:
            compact[_HAS_EMBEDDING_FLAG] = False

        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO vision_facts "
                "(asset_path, content_hash, model_version, pipeline_version, payload, computed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(p),
                    key.content_hash,
                    key.model_version,
                    key.pipeline_version,
                    json.dumps(compact),
                    ts,
                ),
            )
            c.commit()

    def get_or_compute(
        self,
        image_path: str | Path,
        compute_fn: Callable[[Path], dict[str, Any]],
    ) -> dict[str, Any]:
        """Return cached features or call ``compute_fn(path)``, persist, return.

        Only persists when the computed dict carries ``image_ok=True`` —
        failed extractions are intentionally not cached so a transient
        device error doesn't poison subsequent runs.
        """
        p = Path(image_path)
        cached = self.get(p)
        if cached is not None:
            return cached
        out = compute_fn(p)
        if out.get("image_ok"):
            self.put(p, out)
        return out

    def stats(self) -> dict[str, int]:
        """Return a small dict of row counts for monitoring."""
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM vision_facts").fetchone()[0]
            current = c.execute(
                "SELECT COUNT(*) FROM vision_facts WHERE model_version=? AND pipeline_version=?",
                (self.model_version, self.pipeline_version),
            ).fetchone()[0]
        return {"rows_total": int(total), "rows_current_version": int(current)}

    @staticmethod
    def _ensure_pyarrow():
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "Caching CLIP embeddings requires pyarrow. "
                "Install with: pip install pyarrow"
            ) from e
        return pa, pc, pq

    def _write_embedding(self, content_hash: str, vector: list[float]) -> None:
        pa, pc, pq = self._ensure_pyarrow()
        new_row = pa.table({"content_hash": [content_hash], "embedding": [vector]})
        if self.parquet_path.exists():
            existing = pq.read_table(self.parquet_path)
            mask = pc.not_equal(existing.column("content_hash"), pa.scalar(content_hash))
            existing = existing.filter(mask)
            combined = pa.concat_tables([existing, new_row])
        else:
            combined = new_row
        tmp = self.parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(combined, tmp)
        os.replace(tmp, self.parquet_path)

    def _read_embedding(self, content_hash: str) -> list[float] | None:
        pa, pc, pq = self._ensure_pyarrow()
        if not self.parquet_path.exists():
            return None
        t = pq.read_table(self.parquet_path)
        col_hash = t.column("content_hash").to_pylist()
        try:
            idx = col_hash.index(content_hash)
        except ValueError:
            return None
        return list(t.column("embedding")[idx].as_py())
