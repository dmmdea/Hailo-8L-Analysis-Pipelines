"""Convenience factory for acquiring a Hailo-8L vision backend from openclaw_shared.

Consumers typically want:

    from openclaw_shared.backends.hailo import maybe_hailo_backend
    from openclaw_shared.features.thumbnail import extract_thumbnail_features

    backend = maybe_hailo_backend()              # None on machines without Hailo
    features = extract_thumbnail_features(p, backend=backend)

`maybe_hailo_backend()` returns a live, initialized `HailoRuntime` when:
  - `HAILO_VISION_ENABLED=1` is set in the environment (or in ~/.openclaw/secrets.env)
  - `hailo_platform` is importable (the hailo venv is active, OR the venv's
    site-packages is on sys.path)
  - The `hailo_runtime` module can be located (by default we add the
    hailo-vision MCP directory to sys.path)
  - The device successfully configures all 4 HEFs

On any failure it returns None and prints a single-line hint to stderr.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HAILO_VISION_MCP_DIR = Path("/home/dmmdea/openclaw-mcp-servers/hailo-vision")
SECRETS_ENV = Path.home() / ".openclaw" / "secrets.env"


def _load_secrets_env_into_os_environ() -> None:
    """Lightweight dotenv-less read of ~/.openclaw/secrets.env.

    Avoids adding `python-dotenv` to every consumer's dep graph. Silently
    skips if the file doesn't exist.
    """
    if not SECRETS_ENV.exists():
        return
    for line in SECRETS_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def maybe_hailo_backend():
    """Return an initialized HailoRuntime, or None if the Hailo lane is unavailable.

    Callers pass the return value to `extract_thumbnail_features(path, backend=...)`.
    """
    _load_secrets_env_into_os_environ()
    if os.environ.get("HAILO_VISION_ENABLED") != "1":
        return None

    if str(HAILO_VISION_MCP_DIR) not in sys.path:
        sys.path.insert(0, str(HAILO_VISION_MCP_DIR))

    try:
        from hailo_runtime import HailoRuntime  # type: ignore[import-not-found]
    except Exception as e:
        print(f"[hailo-backend] hailo_runtime import failed: {e}", file=sys.stderr)
        return None

    try:
        runtime = HailoRuntime()
        runtime.ensure_initialized()
        return runtime
    except Exception as e:
        print(f"[hailo-backend] device init failed: {e}", file=sys.stderr)
        return None
