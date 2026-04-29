"""
Hailo-vision MCP server — exposes the Hailo-8L accelerator as 4 MCP tools.

Tools:
  - hailo_face_detect(image_path)   SCrFD-2.5G @ ~311 FPS on 8L
  - hailo_ocr(image_path)           PaddleOCR v5 det + rec, ~4.59 FPS det bottleneck
  - hailo_embed(image_path)         TinyCLIP image encoder, 512-d vector
  - hailo_status()                  device + runtime state, safe with driver broken

The server starts even when HAILO_VISION_ENABLED != '1' — tools return structured
error dicts explaining the driver-fix-pending state. Flip the env var once the
kernel VDMA patch is in to make the tools live without restarting the MCP host.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

SECRETS_ENV = Path.home() / ".openclaw" / "secrets.env"
if SECRETS_ENV.exists():
    load_dotenv(SECRETS_ENV)

from hailo_runtime import (  # noqa: E402 — import after dotenv so HAILO_VISION_ENABLED is set
    HEFMissing,
    HailoDeviceError,
    HailoRuntime,
    HailoRuntimeDisabled,
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


@mcp.tool()
def hailo_face_detect(image_path: str) -> dict[str, Any]:
    """Detect faces in a JPEG/PNG via SCrFD-2.5G on Hailo-8L.

    Returns {"faces": [{x,y,w,h,score}, ...], "count": N} on success,
    or a structured {"error": True, ...} dict while disabled or on device error.
    """
    def _run():
        faces = _runtime.face_detect(Path(image_path))
        return {
            "faces": [
                {"x": f.x, "y": f.y, "w": f.w, "h": f.h, "score": round(f.score, 4)}
                for f in faces
            ],
            "count": len(faces),
        }
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
    """Report runtime + device state. Always safe — does not touch the crashing VDMA path."""
    return _runtime.status()


if __name__ == "__main__":
    mcp.run()
