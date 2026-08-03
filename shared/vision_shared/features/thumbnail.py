"""
Thumbnail image feature extractor — OpenCV-based, CPU-only, no ML deps beyond OpenCV.

Produces a flat dict per image with the packaging-signal features the playbook calls
out (§5 Week 2): compositional, color, contrast, and face-presence features. Defers
DeepFace emotion + CLIP embeddings to a later pass when their dep weight is worth it.

All features are deterministic. Invalid/missing images return a dict with safe zeros
and `image_ok=False` so matched-pair analyses don't crash on a single bad download.

Features emitted per thumbnail:

Geometry:
  width, height, aspect_ratio            pixels + w/h
  megapixels                              w × h / 1e6

Color (HSV):
  mean_hue, mean_sat, mean_value          0-179 (H) / 0-255 (S,V)
  std_hue, std_sat, std_value             spread within each channel
  saturation_high_ratio                   fraction of pixels with S > 128
  brightness_mean                         alias for mean_value (0-255)
  dark_pixel_ratio                        fraction of pixels with V < 64
  bright_pixel_ratio                      fraction of pixels with V > 192
  warm_color_ratio                        fraction of pixels with Hue in [0-30] ∪ [150-179] (reds/yellows/magenta)
  cool_color_ratio                        fraction of pixels with Hue in [90-150] (blues/greens)
  unique_color_ratio                      distinct RGB quantized colors / total pixels (palette richness)

Contrast & detail:
  contrast_std                            std of grayscale intensities (0-127)
  edge_density                            Canny edge pixel count / total pixels
  edge_density_top_third                  edge density restricted to top 1/3 (headline area)
  edge_density_bottom_third               edge density restricted to bottom 1/3 (text overlay area)

Face detection (OpenCV Haar frontal + profile):
  face_count                              integer
  has_face                                bool
  largest_face_area_ratio                 largest face area / image area

Text-overlay proxy:
  high_edge_rows_ratio                    fraction of rows with edge density >2× image mean
                                           (high values suggest text banners)
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class HailoBackend(Protocol):
    """Duck-typed interface that the optional Hailo-8L accelerator implements.

    Any object satisfying these methods can be passed to extract_thumbnail_features
    as backend=. On ANY failure (device down, driver bug, HEF missing, image unreadable),
    the method should raise — the caller merges a null-valued Hailo feature block in
    that case, and existing OpenCV-only features are unaffected. This keeps the
    packaging-analysis pipeline standalone-first: it runs fine with no Hailo on the host.
    """

    def face_detect(self, image_path: str | Path) -> list[Any]: ...
    def ocr(self, image_path: str | Path) -> Any: ...
    def embed(self, image_path: str | Path) -> list[float]: ...


_HAILO_NULL_FEATURES: dict[str, Any] = {
    "hailo_ok": False,
    "hailo_face_count": 0,
    "hailo_has_face": False,
    "hailo_largest_face_area_ratio": 0.0,
    "hailo_face_score_max": 0.0,
    "hailo_ocr_text": "",
    "hailo_ocr_text_corrected": "",
    "hailo_ocr_char_count": 0,
    "hailo_ocr_block_count": 0,
    "hailo_ocr_has_year": False,
    "hailo_ocr_has_digit": False,
    "hailo_ocr_brands": [],
    "hailo_ocr_brand_sources": {},
    "hailo_ocr_models": [],
    "hailo_ocr_years": [],
    "hailo_ocr_years_ambiguous": [],
    "hailo_ocr_brand_count": 0,
    "hailo_ocr_model_count": 0,
    "hailo_ocr_keyword_superlative": [],
    "hailo_ocr_keyword_freshness": [],
    "hailo_ocr_keyword_powertrain": [],
    "hailo_ocr_keyword_category": [],
    "hailo_ocr_keyword_review_format": [],
    "hailo_clip_embedding": None,
    "hailo_vehicle_count": 0,
    "hailo_has_vehicle": False,
    "hailo_largest_vehicle_area_ratio": 0.0,
    "hailo_vehicle_score_max": 0.0,
}


_HAAR_FRONT_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
_HAAR_PROFILE_PATH = cv2.data.haarcascades + "haarcascade_profileface.xml"  # type: ignore[attr-defined]


@functools.lru_cache(maxsize=1)
def _load_cascades() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    """Lazy-load OpenCV's Haar cascades once per process."""
    front = cv2.CascadeClassifier(_HAAR_FRONT_PATH)
    profile = cv2.CascadeClassifier(_HAAR_PROFILE_PATH)
    return front, profile


def _detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) face rectangles — frontal + profile deduplicated."""
    front, profile = _load_cascades()
    faces_front = front.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    faces_profile = profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    rects: list[tuple[int, int, int, int]] = list(faces_front) + list(faces_profile)
    # Deduplicate overlapping rects (profile+frontal on same face)
    deduped: list[tuple[int, int, int, int]] = []
    for r in rects:
        x, y, w, h = r
        overlaps = False
        for dx, dy, dw, dh in deduped:
            # IoU test — >0.3 considered same face
            inter_x1 = max(x, dx)
            inter_y1 = max(y, dy)
            inter_x2 = min(x + w, dx + dw)
            inter_y2 = min(y + h, dy + dh)
            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                continue
            inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union = w * h + dw * dh - inter
            if union > 0 and inter / union > 0.3:
                overlaps = True
                break
        if not overlaps:
            deduped.append((int(x), int(y), int(w), int(h)))
    return deduped


def extract_thumbnail_features(
    image_path: str | Path,
    backend: HailoBackend | None = None,
) -> dict[str, Any]:
    """Extract features from one thumbnail. If `backend` is passed, merge Hailo features.

    The existing 22 OpenCV features are always produced. Hailo fields (prefixed `hailo_`)
    default to nulls unless a working backend is supplied and each per-tool call succeeds.
    Per-tool failures are isolated: OCR can fail while face-detect succeeds, etc.
    """
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return _empty_features(ok=False, reason=f"missing: {p}")

    img_bgr = cv2.imread(str(p))
    if img_bgr is None or img_bgr.size == 0:
        return _empty_features(ok=False, reason=f"unreadable: {p}")

    h, w = img_bgr.shape[:2]
    total_px = h * w
    if total_px == 0:
        return _empty_features(ok=False, reason="zero-size image")

    # Convert to needed color spaces once
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    H, S, V = img_hsv[..., 0], img_hsv[..., 1], img_hsv[..., 2]

    # Palette richness: quantize to 5-bit per channel (32^3 = 32k palette buckets max)
    quantized = (img_bgr.astype(np.int32) >> 3)  # 8-bit → 5-bit
    flat = quantized.reshape(-1, 3)
    # Pack into single int for uniqueness count
    packed = (flat[:, 0] << 10) | (flat[:, 1] << 5) | flat[:, 2]
    unique_colors = int(np.unique(packed).size)

    # Edges
    edges = cv2.Canny(img_gray, 80, 180)
    edge_density = float(np.count_nonzero(edges)) / total_px

    top_third = edges[: h // 3, :]
    bottom_third = edges[-(h // 3):, :]
    edge_top = float(np.count_nonzero(top_third)) / max(1, top_third.size)
    edge_bot = float(np.count_nonzero(bottom_third)) / max(1, bottom_third.size)

    # Text-overlay proxy: rows with edge density > 2× image mean
    row_edge_densities = np.count_nonzero(edges, axis=1) / max(1, w)
    mean_row_edge = float(row_edge_densities.mean()) if row_edge_densities.size else 0.0
    high_edge_rows = int(np.count_nonzero(row_edge_densities > (2 * mean_row_edge))) if mean_row_edge > 0 else 0
    high_edge_rows_ratio = high_edge_rows / max(1, h)

    # Color distribution
    warm_mask = (H < 30) | (H > 150)  # reds + yellows + magenta (OpenCV hue is 0-179)
    cool_mask = (H >= 90) & (H <= 150)  # blues + greens
    sat_high_mask = S > 128
    dark_mask = V < 64
    bright_mask = V > 192

    # Faces
    faces = _detect_faces(img_gray)
    face_count = len(faces)
    largest_face_area = max((fw * fh for _, _, fw, fh in faces), default=0)

    out: dict[str, Any] = {
        "image_ok": True,
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(w / h, 3) if h else 0.0,
        "megapixels": round(total_px / 1_000_000, 3),
        "mean_hue": float(np.mean(H)),
        "mean_sat": float(np.mean(S)),
        "mean_value": float(np.mean(V)),
        "std_hue": float(np.std(H)),
        "std_sat": float(np.std(S)),
        "std_value": float(np.std(V)),
        "saturation_high_ratio": float(np.mean(sat_high_mask)),
        "brightness_mean": float(np.mean(V)),
        "dark_pixel_ratio": float(np.mean(dark_mask)),
        "bright_pixel_ratio": float(np.mean(bright_mask)),
        "warm_color_ratio": float(np.mean(warm_mask)),
        "cool_color_ratio": float(np.mean(cool_mask)),
        "unique_color_ratio": round(unique_colors / total_px, 6),
        "contrast_std": float(np.std(img_gray)),
        "edge_density": round(edge_density, 5),
        "edge_density_top_third": round(edge_top, 5),
        "edge_density_bottom_third": round(edge_bot, 5),
        "face_count": face_count,
        "has_face": face_count > 0,
        "largest_face_area_ratio": round(largest_face_area / total_px, 5),
        "high_edge_rows_ratio": round(high_edge_rows_ratio, 4),
        **_HAILO_NULL_FEATURES,
    }
    if backend is not None:
        _merge_hailo_features(out, backend, p, total_px)
    return out


def _merge_hailo_features(
    out: dict[str, Any],
    backend: HailoBackend,
    image_path: Path,
    total_px: int,
) -> None:
    """Overlay Hailo-derived features onto `out`. Per-tool isolated: one failure ≠ all fail.

    HAILO_OCR_MODE is validated UP-FRONT (outside the per-feature try/except
    blocks) so a misconfigured mode fails loud with ValueError. Device errors
    inside any one feature stay isolated. Default mode is "production" — the
    FINAL_REPORT-recommended Phase 3 multi-scale config; callers that want
    cheap scans set HAILO_OCR_MODE=fast explicitly.
    """
    import os as _os
    import re

    _VALID_OCR_MODES = ("fast", "production", "research")
    ocr_mode = _os.environ.get("HAILO_OCR_MODE", "production")
    if ocr_mode not in _VALID_OCR_MODES:
        raise ValueError(
            f"HAILO_OCR_MODE must be one of {_VALID_OCR_MODES}, got {ocr_mode!r}"
        )

    any_ok = False

    try:
        faces = backend.face_detect(image_path)
        areas = [getattr(f, "w", 0) * getattr(f, "h", 0) for f in faces]
        scores = [getattr(f, "score", 0.0) for f in faces]
        out["hailo_face_count"] = len(faces)
        out["hailo_has_face"] = len(faces) > 0
        out["hailo_largest_face_area_ratio"] = (
            round(max(areas, default=0) / total_px, 5) if total_px else 0.0
        )
        out["hailo_face_score_max"] = round(max(scores, default=0.0), 4)
        any_ok = True
    except Exception:
        pass

    # Vehicle detection (yolov5m_vehicles HEF). Optional — only runs if the
    # backend exposes a vehicle_detect method AND the HEF is loaded.
    try:
        if hasattr(backend, "vehicle_detect"):
            vehicles = backend.vehicle_detect(image_path)
            v_areas = [getattr(v, "w", 0) * getattr(v, "h", 0) for v in vehicles]
            v_scores = [getattr(v, "score", 0.0) for v in vehicles]
            out["hailo_vehicle_count"] = len(vehicles)
            out["hailo_has_vehicle"] = len(vehicles) > 0
            out["hailo_largest_vehicle_area_ratio"] = (
                round(max(v_areas, default=0) / total_px, 5) if total_px else 0.0
            )
            out["hailo_vehicle_score_max"] = round(max(v_scores, default=0.0), 4)
            any_ok = True
    except Exception:
        pass

    try:
        # ocr_mode was validated up-front; backends that don't accept the
        # `mode` kwarg fall back to the default call (older backend stubs).
        try:
            r = backend.ocr(image_path, mode=ocr_mode)
        except TypeError:
            r = backend.ocr(image_path)
        text = getattr(r, "text", "")
        boxes = getattr(r, "boxes", [])

        from vision_shared.features.ocr_correction import correct_text
        from vision_shared.features.ocr_entities import extract_entities

        corrected = correct_text(text)

        out["hailo_ocr_text"] = text
        out["hailo_ocr_text_corrected"] = corrected
        out["hailo_ocr_char_count"] = len(text)
        out["hailo_ocr_block_count"] = len(boxes)
        out["hailo_ocr_has_digit"] = bool(re.search(r"\d", text))

        entities = extract_entities(corrected)
        out["hailo_ocr_brands"] = entities["brands"]
        out["hailo_ocr_brand_sources"] = entities.get("brand_sources", {})
        out["hailo_ocr_models"] = entities["models"]
        out["hailo_ocr_years"] = entities["years"]
        out["hailo_ocr_years_ambiguous"] = entities.get("years_ambiguous", [])
        out["hailo_ocr_brand_count"] = entities["brand_count"]
        out["hailo_ocr_model_count"] = entities["model_count"]
        out["hailo_ocr_has_year"] = bool(entities["years"]) or bool(entities.get("years_ambiguous"))
        out["hailo_ocr_keyword_superlative"] = entities["keywords_superlative"]
        out["hailo_ocr_keyword_freshness"] = entities["keywords_freshness"]
        out["hailo_ocr_keyword_powertrain"] = entities["keywords_powertrain"]
        out["hailo_ocr_keyword_category"] = entities["keywords_category"]
        out["hailo_ocr_keyword_review_format"] = entities["keywords_review_format"]
        any_ok = True
    except Exception:
        pass

    try:
        out["hailo_clip_embedding"] = list(backend.embed(image_path))
        any_ok = True
    except Exception:
        pass

    out["hailo_ok"] = any_ok


def _empty_features(ok: bool = False, reason: str = "") -> dict[str, Any]:
    """Return a safe zero-filled features dict so matched-pair downstream doesn't crash."""
    zero: dict[str, Any] = {
        "image_ok": ok,
        "reason": reason,
        "width": 0, "height": 0, "aspect_ratio": 0.0, "megapixels": 0.0,
        "mean_hue": 0.0, "mean_sat": 0.0, "mean_value": 0.0,
        "std_hue": 0.0, "std_sat": 0.0, "std_value": 0.0,
        "saturation_high_ratio": 0.0, "brightness_mean": 0.0,
        "dark_pixel_ratio": 0.0, "bright_pixel_ratio": 0.0,
        "warm_color_ratio": 0.0, "cool_color_ratio": 0.0,
        "unique_color_ratio": 0.0, "contrast_std": 0.0,
        "edge_density": 0.0, "edge_density_top_third": 0.0, "edge_density_bottom_third": 0.0,
        "face_count": 0, "has_face": False, "largest_face_area_ratio": 0.0,
        "high_edge_rows_ratio": 0.0,
        **_HAILO_NULL_FEATURES,
    }
    return zero
