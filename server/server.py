"""
Hailo-vision MCP server — exposes the Hailo-8L accelerator as 9 MCP tools.

Tools:
  - hailo_face_detect(image_path)        SCrFD-2.5G boxes + 5 landmarks, ~311 FPS on 8L
  - hailo_face_embed(image_path)         ArcFace 512-d identity per face (landmark-aligned)
  - hailo_ocr(image_path)                PaddleOCR v5 det + rec, ~4.59 FPS det bottleneck
  - hailo_embed(image_path)              TinyCLIP ViT-61M image encoder, 512-d vector
  - hailo_object_detect(image_path)      YOLOv8s, 80 COCO classes, on-chip NMS
  - hailo_person_embed(image_path)       OSNet 512-d re-id per detected person
  - hailo_depth(image_path)              Depth-Anything-V2 224-px relative depth map
  - hailo_enhance_low_light(image_path)  Zero-DCE brightening at original resolution
  - hailo_status()                       device + runtime state, safe with driver broken

The server starts even when HAILO_VISION_ENABLED != '1' — tools return structured
error dicts explaining the driver-fix-pending state. Flip the env var once the
kernel VDMA patch is in to make the tools live without restarting the MCP host.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

SECRETS_ENV = Path.home() / ".hailo-vision" / "secrets.env"
if SECRETS_ENV.exists():
    load_dotenv(SECRETS_ENV)

from hailo_runtime import (  # noqa: E402 — import after dotenv so HAILO_VISION_ENABLED is set
    HEFMissing,
    HailoDeviceError,
    HailoRuntime,
    HailoRuntimeDisabled,
    InvalidInput,
)


mcp = FastMCP("hailo-vision")
_runtime = HailoRuntime()


def _err(message: str, **extra: Any) -> dict[str, Any]:
    return {"error": True, "message": message, **extra}


def _guarded(fn):
    try:
        return fn()
    except HailoRuntimeDisabled as e:
        return _err(str(e), kind="hailo_disabled")
    except HEFMissing as e:
        return _err(str(e), kind="hefs_missing")
    except HailoDeviceError as e:
        return _err(str(e), kind="hailo_device_error")
    except NotImplementedError as e:
        return _err(str(e), kind="not_implemented")
    except FileNotFoundError as e:
        return _err(f"image not found: {e}", kind="image_missing")
    except InvalidInput as e:
        # ONLY the deliberate input-contract refusals (no landmarks, empty crop,
        # degenerate alignment, unwritable output). A bare ValueError from numpy
        # is an integration bug and must surface as one, not as "bad input".
        return _err(str(e), kind="invalid_input")


def _face_dict(f) -> dict[str, Any]:
    d = {"x": f.x, "y": f.y, "w": f.w, "h": f.h, "score": round(f.score, 4)}
    if f.kps:
        d["kps"] = [[round(x, 1), round(y, 1)] for x, y in f.kps]
    return d


@mcp.tool()
def hailo_face_detect(image_path: str) -> dict[str, Any]:
    """Detect faces in a JPEG/PNG via SCrFD-2.5G on Hailo-8L.

    Returns {"faces": [{x,y,w,h,score,kps}, ...], "count": N} on success,
    or a structured {"error": True, ...} dict while disabled or on device error.
    kps = the 5 landmarks (left eye, right eye, nose, mouth-L, mouth-R) in image
    pixels — pass a face straight into hailo_face_embed for identity.
    """
    def _run():
        faces = _runtime.face_detect(Path(image_path))
        return {"faces": [_face_dict(f) for f in faces], "count": len(faces)}
    return _guarded(_run)


@mcp.tool()
def hailo_face_embed(image_path: str, max_faces: int = 16) -> dict[str, Any]:
    """Identity vectors for every face in an image (SCrFD detect → ArcFace).

    For each face returns the box plus a 512-d L2-normalised ArcFace embedding.
    Cosine similarity between two embeddings is the identity score: the same
    person scores ≈0.5+ across shots, different people ≈0.3-. Use it to cluster
    who appears where across a whole project without a single cloud call.
    Faces are processed strongest-score first, up to max_faces.
    """
    def _run():
        p = Path(image_path)
        faces = sorted(_runtime.face_detect(p), key=lambda f: f.score, reverse=True)[:max_faces]
        out = []
        for f in faces:
            entry = _face_dict(f)
            if f.kps:
                entry["embedding"] = _runtime.face_embed(p, f)
            else:
                entry["embedding"] = None
                entry["note"] = "no landmarks from the detector; cannot align"
            out.append(entry)
        return {"faces": out, "count": len(out)}
    return _guarded(_run)


@mcp.tool()
def hailo_object_detect(image_path: str, score_threshold: float = 0.3) -> dict[str, Any]:
    """Detect the 80 COCO object classes (person, car, dog, laptop, ...) via
    YOLOv8s with on-chip NMS on Hailo-8L.

    Returns {"objects": [{label, class_id, x, y, w, h, score}, ...], "count": N},
    sorted by score. Boxes are in the image's own pixel space.
    """
    def _run():
        boxes = _runtime.object_detect(Path(image_path), score_threshold=score_threshold)
        return {
            "objects": [
                {"label": b.label, "class_id": b.class_id, "x": b.x, "y": b.y,
                 "w": b.w, "h": b.h, "score": round(b.score, 4)}
                for b in boxes
            ],
            "count": len(boxes),
        }
    return _guarded(_run)


@mcp.tool()
def hailo_depth(image_path: str, out_path: str | None = None) -> dict[str, Any]:
    """Preview-grade relative depth map via Depth-Anything-V2 (ViT-S) on Hailo-8L.

    Writes a 224×224 8-bit PNG (bright = near) to out_path, or next to the
    input as <name>.depth.png. Returns {"depth_path", "min", "max", "mean"}.
    Good for shot analysis and parallax previews; not a production depth pass.
    """
    def _run():
        import cv2
        import numpy as np
        p = Path(image_path)
        d = _runtime.depth(p)
        lo, hi = float(d.min()), float(d.max())
        norm = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
        dest = Path(out_path) if out_path else p.with_suffix(".depth.png")
        # cv2.imwrite reports failure by returning False, never by raising.
        if not cv2.imwrite(str(dest), (norm * 255).astype(np.uint8)):
            raise InvalidInput(f"could not write {dest} — is the destination directory writable?")
        return {"depth_path": str(dest), "min": lo, "max": hi, "mean": float(d.mean())}
    return _guarded(_run)


@mcp.tool()
def hailo_person_embed(image_path: str) -> dict[str, Any]:
    """Re-identification vectors for every PERSON in an image (YOLOv8s → OSNet).

    Unlike face identity this works with no visible face — it keys on clothing
    and body shape — so it tracks the same person across shots and angles.
    Returns per person: the box plus a 512-d L2-normalised embedding; cosine
    similarity between two embeddings is the match score.
    """
    def _run():
        p = Path(image_path)
        people = [b for b in _runtime.object_detect(p) if b.label == "person"]
        out = []
        for b in people:
            out.append({
                "x": b.x, "y": b.y, "w": b.w, "h": b.h, "score": round(b.score, 4),
                "embedding": _runtime.person_embed(p, b),
            })
        return {"people": out, "count": len(out)}
    return _guarded(_run)


@mcp.tool()
def hailo_enhance_low_light(image_path: str, out_path: str | None = None) -> dict[str, Any]:
    """Brighten an under-exposed frame via Zero-DCE on Hailo-8L (preview-grade).

    Writes the enhanced image at the original resolution to out_path, or next
    to the input as <name>.enhanced.png. Returns {"enhanced_path"}.
    """
    def _run():
        import cv2
        p = Path(image_path)
        img = _runtime.enhance_low_light(p)
        dest = Path(out_path) if out_path else p.with_suffix(".enhanced.png")
        if not cv2.imwrite(str(dest), img):
            raise InvalidInput(f"could not write {dest} — is the destination directory writable?")
        return {"enhanced_path": str(dest), "width": int(img.shape[1]), "height": int(img.shape[0])}
    return _guarded(_run)


@mcp.tool()
def hailo_ocr(image_path: str) -> dict[str, Any]:
    """Extract text from a thumbnail/frame via PaddleOCR v5 (det + rec) on Hailo-8L.

    Returns {"text": str, "char_count": int, "boxes": [[x,y,w,h], ...]}.
    """
    def _run():
        r = _runtime.ocr(Path(image_path))
        return {
            "text": r.text,
            "char_count": r.char_count,
            "boxes": [list(b) for b in r.boxes],
        }
    return _guarded(_run)


@mcp.tool()
def hailo_embed(image_path: str) -> dict[str, Any]:
    """Compute a 512-d TinyCLIP image embedding on Hailo-8L (for similarity / clustering)."""
    def _run():
        v = _runtime.embed(Path(image_path))
        return {"embedding": v, "dim": len(v)}
    return _guarded(_run)


@mcp.tool()
def hailo_status() -> dict[str, Any]:
    """Report runtime + device state. Always safe — does not touch the crashing VDMA path.

    Note: loaded_networks reflects THIS moment. The runtime initialises lazily on
    the first real tool call, so a fresh process reports [] here until one runs;
    that is by design (status must never trigger device I/O), not a fault.
    """
    return _runtime.status()


if __name__ == "__main__":
    mcp.run()
