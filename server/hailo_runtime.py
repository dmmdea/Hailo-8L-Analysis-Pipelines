"""
Core HailoRT wrapper for the hailo-vision MCP and the vision_shared
thumbnail/video feature extractors.

One HailoRuntime per process: persistent VDevice + ROUND_ROBIN scheduler + 3
configured network groups (face detection, OCR detect+recognize, CLIP embed).

This module is importable even when the Hailo device/driver is unavailable.
Actual HailoRT calls are guarded by HAILO_VISION_ENABLED=1 so the scaffold can
be registered as an MCP today while the kernel driver fix is pending. Once the
driver is patched, flipping the env var turns the runtime on without code churn.

Env vars:
  HAILO_VISION_ENABLED   "1" to make the runtime actually touch /dev/hailo0
                         (default "0" — every tool returns a structured error dict)
  HAILO_MODELS_DIR       directory containing the 4 HEFs. Default /mnt/ai/hailo/models
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODELS_DIR = Path("/mnt/ai/hailo/models")

HEF_FACE_DETECT = "scrfd_2.5g.hef"
HEF_OCR_DETECT = "paddle_ocr_v5_mobile_detection.hef"
HEF_OCR_RECOGNIZE = "paddle_ocr_v5_mobile_recognition.hef"
# TinyCLIP ViT-61M: same 224×224 UINT8 → 512-d contract as the 40M it replaces,
# better (67.8 vs 65.7 HW top-1) AND faster (44 vs 35 FPS b=1) per the model-zoo
# HAILO8L table. Embeddings are NOT comparable across the two — re-index on swap.
HEF_CLIP_EMBED = "tinyclip_vit_61m_32_text_29m_laion400m_image_encoder.hef"
HEF_SUPER_RESOLVE = "real_esrgan_x2.hef"  # 512×512 UINT8 → 1024×1024 UINT8
HEF_VEHICLE_DETECT = "yolov5m_vehicles.hef"  # 1920×1080 UINT8 → Hailo-NMS list of vehicle bboxes
# Editor-workstation additions (all HAILO8L builds from model zoo v2.19.0).
HEF_FACE_EMBED = "arcface_mobilefacenet.hef"  # 112×112 aligned face → 512-d identity vector
HEF_OBJECT_DETECT = "yolov8s.hef"  # 640×640 → on-chip NMS, 80 COCO classes
HEF_DEPTH = "depth_anything_v2_vits.hef"  # 224×224 → 224×224×1 relative depth
HEF_PERSON_EMBED = "osnet_x1_0.hef"  # 256×128 person crop → 512-d re-id vector
HEF_LOW_LIGHT = "zero_dce.hef"  # 400×600 → 400×600 enhanced

ALL_HEFS = (HEF_FACE_DETECT, HEF_OCR_DETECT, HEF_OCR_RECOGNIZE, HEF_CLIP_EMBED)
OPTIONAL_HEFS = (  # runtime gracefully handles missing
    HEF_SUPER_RESOLVE,
    HEF_VEHICLE_DETECT,
    HEF_FACE_EMBED,
    HEF_OBJECT_DETECT,
    HEF_DEPTH,
    HEF_PERSON_EMBED,
    HEF_LOW_LIGHT,
)

# ArcFace reference landmarks (left eye, right eye, nose, mouth-L, mouth-R) for a
# 112×112 crop — the canonical insightface template. SCrFD's 5 keypoints are
# emitted in this same order, so a similarity transform maps one onto the other.
ARCFACE_TEMPLATE_112 = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)

# COCO-80 class names in YOLO order, for yolov8s' on-chip NMS output.
COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

OCR_CHARSET_PATH = Path("/mnt/ai/hailo/models/charsets/ppocrv5_dict.txt")

# OCR modes — ordered slowest-to-fastest along the production axis. Every
# caller that flows through HAILO_OCR_MODE or HailoRuntime.ocr(mode=...) must
# pass exactly one of these. The previous "quality" mode entangled four
# independently-tunable knobs (SR / multi-scale / TTA / beam) and the
# closed 9-phase OCR plan proved only multi-scale was monotonically positive
# on the deployed rec HEF. Splitting decouples those knobs and aligns the
# default with the FINAL_REPORT recommended config.
VALID_OCR_MODES = ("fast", "production", "research")


class HailoRuntimeDisabled(RuntimeError):
    """Raised when a HailoRuntime method is called while HAILO_VISION_ENABLED != '1'."""


class HailoDeviceError(RuntimeError):
    """HailoRT / driver rejected an operation. Message carries the original text."""


class HEFMissing(FileNotFoundError):
    """A required HEF file is not present in HAILO_MODELS_DIR."""


class InvalidInput(ValueError):
    """The CALLER's input cannot be processed (no landmarks, empty crop, unwritable
    output path). Deliberately a distinct subclass: the MCP boundary catches only
    this, so a bare ValueError from numpy (a wrong reshape = an integration bug)
    is never relabelled as the caller's fault."""


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    # 5 landmarks (left eye, right eye, nose, mouth-L, mouth-R) in original-image
    # pixels. SCrFD always emits them; they are what ArcFace alignment needs.
    kps: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class ObjectBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str
    class_id: int


@dataclass(frozen=True)
class VehicleBox:
    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass(frozen=True)
class OcrResult:
    text: str
    boxes: list[tuple[int, int, int, int]]  # (x, y, w, h) per text region

    @property
    def char_count(self) -> int:
        return len(self.text)


def _models_dir() -> Path:
    return Path(os.environ.get("HAILO_MODELS_DIR", str(DEFAULT_MODELS_DIR)))


def _enabled() -> bool:
    return os.environ.get("HAILO_VISION_ENABLED", "0") == "1"


def _verify_hefs_present() -> list[str]:
    """Return list of HEF basenames missing from HAILO_MODELS_DIR. Empty list = all good."""
    d = _models_dir()
    return [h for h in ALL_HEFS if not (d / h).exists()]


def _preprocess_for_rec(
    crop_bgr: Any,
    apply_clahe: bool = True,
    width_scale: float = 1.0,
) -> Any:
    """PaddleOCR v5 rec: 48×320 input, aspect-preserving resize with right-pad, RGB uint8.

    Args:
        crop_bgr: cropped image BGR uint8.
        apply_clahe: if True, apply CLAHE contrast normalization (default). Set
            False to keep the crop in its original contrast — gives the Phase 4
            TTA pipeline a second sampling view.
        width_scale: scale factor applied to the aspect-preserved width before
            the final resize/pad. 1.0 = default. 0.75 and 1.25 are used by the
            Phase 4 TTA pipeline to get different aspect-ratio renderings of
            the same crop (the rec HEF's input size is fixed so we trade more
            padding for a different relative character width).
    """
    import cv2
    import numpy as np

    if apply_clahe:
        # CLAHE contrast boost in LAB space — preserves color, enhances luminance
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        crop_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    crop_h, crop_w = crop_bgr.shape[:2]
    target_h, target_w = 48, 320
    ratio = crop_w / max(1, crop_h)
    new_w_base = max(1, min(target_w, int(round(target_h * ratio))))
    new_w = max(1, min(target_w, int(round(new_w_base * width_scale))))
    resized = cv2.resize(crop_bgr, (new_w, target_h))
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[:, :new_w] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0).astype(np.uint8)


# Phase 4 TTA config: pairs of (width_scale, clahe) applied per crop.
# The initial spec was 3 widths × 2 contrasts = 6 augmentations, but the
# width-scaled variants produced enough divergent outputs on this rec HEF
# that the geometric-mean got dragged into gibberish ("ELECTRCoS ... A CA").
# Dropped to 2 augs (CLAHE on + off at native width) — preserves the
# softmax-broadening benefit without drowning in low-quality views.
_TTA_AUGMENTATIONS = (
    (1.0, True),
    (1.0, False),
)


def _merge_boxes_nms(boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Greedy NMS on (x, y, w, h) boxes — no scores, so we sort by area
    (largest first) and drop smaller boxes that overlap a kept one above the
    IoU threshold. Suitable for merging two-pass detection outputs where each
    pass found the same text block from a different resolution.
    """
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: -(b[2] * b[3]))
    kept: list[tuple[int, int, int, int]] = []
    for b in sorted_boxes:
        bx, by, bw, bh = b
        bx2, by2 = bx + bw, by + bh
        suppressed = False
        for k in kept:
            kx, ky, kw_, kh = k
            kx2, ky2 = kx + kw_, ky + kh
            ix1, iy1 = max(bx, kx), max(by, ky)
            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter == 0:
                continue
            union = bw * bh + kw_ * kh - inter
            if inter / max(1, union) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(b)
    return kept


def _unclip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int, ratio: float = 0.12) -> tuple[int, int, int, int]:
    """Grow a tight detection box by `ratio` on each side — catches character
    ascenders/descenders the text-region contour may have clipped. Clamps to image.
    """
    dx = max(2, int(round(w * ratio)))
    dy = max(2, int(round(h * ratio)))
    x2 = max(0, x - dx)
    y2 = max(0, y - dy)
    w2 = min(img_w - x2, w + 2 * dx)
    h2 = min(img_h - y2, h + 2 * dy)
    return x2, y2, w2, h2


class HailoRuntime:
    """Singleton-style HailoRT facade. Instantiate once per process, reuse for every call.

    Until HAILO_VISION_ENABLED=1, every method raises HailoRuntimeDisabled without
    touching the device — safe to import and instantiate on a machine whose driver
    has the kernel-6.17 VDMA oops pending.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._vdevice: Any = None
        self._networks: dict[str, Any] = {}
        self._ocr_charset: list[str] | None = None
        self._beam_decoder: Any = None
        self._hotwords: list[str] | None = None

    # --- lifecycle ---------------------------------------------------------

    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        if not _enabled():
            raise HailoRuntimeDisabled(
                "HAILO_VISION_ENABLED != '1' — runtime is scaffolded but not live. "
                "This guard prevents hitting the kernel VDMA bug on unpatched drivers."
            )
        missing = _verify_hefs_present()
        if missing:
            raise HEFMissing(
                f"Missing HEFs under {_models_dir()}: {missing}. "
                f"Run the hailo-vision setup to populate the model dir."
            )
        self._open_vdevice()
        self._configure_networks()
        self._initialized = True

    def _open_vdevice(self) -> None:
        import hailo_platform as hpf  # local import: only touched when enabled

        params = hpf.VDevice.create_params()
        params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = hpf.VDevice(params=params)

    def _configure_networks(self) -> None:
        import hailo_platform as hpf

        d = _models_dir()
        for basename in ALL_HEFS:
            hef = hpf.HEF(str(d / basename))
            cfg = hpf.ConfigureParams.create_from_hef(
                hef=hef, interface=hpf.HailoStreamInterface.PCIe
            )
            ng = self._vdevice.configure(hef, cfg)[0]
            self._networks[basename] = (hef, ng)
        # Optional HEFs: configure only if the file is on disk. Lets the runtime
        # boot on systems that haven't downloaded the quality-mode extras.
        for basename in OPTIONAL_HEFS:
            path = d / basename
            if not path.exists():
                continue
            hef = hpf.HEF(str(path))
            cfg = hpf.ConfigureParams.create_from_hef(
                hef=hef, interface=hpf.HailoStreamInterface.PCIe
            )
            ng = self._vdevice.configure(hef, cfg)[0]
            self._networks[basename] = (hef, ng)

    def close(self) -> None:
        if self._vdevice is not None:
            try:
                self._vdevice.release()
            except Exception:
                pass
        self._vdevice = None
        self._networks.clear()
        self._initialized = False

    # --- tools -------------------------------------------------------------

    def face_detect(
        self,
        image_path: str | Path,
        score_threshold: float = 0.5,
        nms_iou: float = 0.4,
    ) -> list[FaceBox]:
        """Detect faces via SCrFD-2.5G on Hailo-8L.

        Pipeline: letterbox → UINT8 → Hailo inference → multi-scale decode → NMS →
        rescale to original image coordinates. Boxes returned in the original image's
        pixel space.
        """
        self.ensure_initialized()
        import cv2
        import hailo_platform as hpf
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        orig_h, orig_w = img.shape[:2]
        target = 640
        scale = target / max(orig_h, orig_w)
        new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
        resized = cv2.resize(img, (new_w, new_h))
        padded = np.zeros((target, target, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

        hef, ng = self._networks[HEF_FACE_DETECT]
        in_info = hef.get_input_vstream_infos()[0]
        in_params = hpf.InputVStreamParams.make_from_network_group(
            ng, quantized=True, format_type=hpf.FormatType.UINT8
        )
        out_params = hpf.OutputVStreamParams.make_from_network_group(
            ng, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        with hpf.InferVStreams(ng, in_params, out_params) as pipe:
            out = pipe.infer({in_info.name: tensor})

        # Decode 3 strides × 2 anchors per cell. Output name suffixes per Hailo HEF:
        #   cls  -> conv42/49/55  shape (H, W, 2)
        #   bbox -> conv43/50/56  shape (H, W, 8)  (2 anchors × 4 offsets)
        #   kps  -> conv44/51/57  shape (H, W, 20) (2 anchors × 5 landmarks × (dx, dy))
        strides = [
            ("scrfd_2_5g/conv42", "scrfd_2_5g/conv43", "scrfd_2_5g/conv44", 8, 80),
            ("scrfd_2_5g/conv49", "scrfd_2_5g/conv50", "scrfd_2_5g/conv51", 16, 40),
            ("scrfd_2_5g/conv55", "scrfd_2_5g/conv56", "scrfd_2_5g/conv57", 32, 20),
        ]
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []
        for cls_name, bbox_name, kps_name, stride, fs in strides:
            cls = np.asarray(out[cls_name]).reshape(fs, fs, 2)
            bbox = np.asarray(out[bbox_name]).reshape(fs, fs, 2, 4)
            ys, xs, anchors = np.meshgrid(
                np.arange(fs), np.arange(fs), np.arange(2), indexing="ij"
            )
            anchor_x = (xs + 0.5) * stride
            anchor_y = (ys + 0.5) * stride
            left = bbox[..., 0] * stride
            top = bbox[..., 1] * stride
            right = bbox[..., 2] * stride
            bottom = bbox[..., 3] * stride
            x1 = anchor_x - left
            y1 = anchor_y - top
            x2 = anchor_x + right
            y2 = anchor_y + bottom
            boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
            # Landmarks share the box convention: per-anchor offsets in stride
            # units from the anchor center, so (anchor + offset*stride) is the
            # 640-letterbox pixel position. The head is optional-by-name so a
            # HEF compiled without it still detects (kps stay empty).
            if kps_name in out:
                raw = np.asarray(out[kps_name])
                expected = fs * fs * 2 * 5 * 2
                if raw.size != expected:
                    # A wrong-shaped head is a HEF/runtime contract break, not a
                    # bad image — surface it as a device error, never as input.
                    raise HailoDeviceError(
                        f"{HEF_FACE_DETECT} {kps_name} output has {raw.size} values, "
                        f"expected {expected} (stride {stride})"
                    )
                kraw = raw.reshape(fs, fs, 2, 5, 2)
                kx = anchor_x[..., None] + kraw[..., 0] * stride
                ky = anchor_y[..., None] + kraw[..., 1] * stride
                kps = np.stack([kx, ky], axis=-1).reshape(-1, 5, 2)
            else:
                kps = np.full((boxes.shape[0], 5, 2), np.nan, dtype=np.float32)
            # Hailo-compiled SCrFD emits post-sigmoid probabilities in [0,1]; don't reapply sigmoid.
            scores = cls.reshape(-1)
            keep = scores > score_threshold
            if keep.any():
                all_boxes.append(boxes[keep])
                all_scores.append(scores[keep])
                all_kps.append(kps[keep])

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        kps_all = np.concatenate(all_kps, axis=0)

        # OpenCV NMS expects (x, y, w, h)
        rects = np.stack([
            boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        ], axis=-1).tolist()
        idxs = cv2.dnn.NMSBoxes(rects, scores.tolist(), score_threshold, nms_iou)
        if len(idxs) == 0:
            return []
        idxs = np.asarray(idxs).flatten()

        # Rescale from 640x640 letterbox back to original image
        inv = 1.0 / scale
        results: list[FaceBox] = []
        for i in idxs:
            x1, y1, x2, y2 = boxes[i]
            x1 = max(0, int(round(x1 * inv)))
            y1 = max(0, int(round(y1 * inv)))
            x2 = min(orig_w, int(round(x2 * inv)))
            y2 = min(orig_h, int(round(y2 * inv)))
            if x2 <= x1 or y2 <= y1:
                continue
            k = kps_all[i]
            # Landmarks are usable only if present (not the NaN sentinel) AND
            # actually spread out. A low-confidence anchor can emit all-zero
            # offsets, which decode to 5 coincident points at the anchor centre
            # — "valid" numbers that would align to a black crop downstream.
            spread = float((k.max(axis=0) - k.min(axis=0)).max()) if not np.isnan(k).any() else 0.0
            kps = () if spread < 1.0 else tuple(
                (float(px * inv), float(py * inv)) for px, py in k
            )
            results.append(
                FaceBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1, score=float(scores[i]), kps=kps)
            )
        return results

    def ocr(
        self,
        image_path: str | Path,
        score_threshold: float = 0.5,
        mode: str = "fast",
    ) -> OcrResult:
        """Full OCR — PaddleOCR v5 detect + recognize on Hailo-8L.

        Two-stage: detection HEF emits a binary text-region heatmap, each contour is
        cropped, aspect-preserving-padded to 48×320, and fed through the recognition HEF.
        Recognition output is CTC-decoded (greedy argmax + consecutive-dup collapse +
        blank drop) against the 18,384-char PaddleOCR-v5 charset (18,383 file chars + " ").

        Args:
            image_path: path to image.
            score_threshold: detection heatmap threshold (default 0.5).
            mode (one of `VALID_OCR_MODES`):
                "fast" — single-scale detection on the original, greedy CTC.
                    ~115 s / 90 thumb. Default. Use for high-volume scans.
                "production" — Real-ESRGAN x2 super-resolve, multi-scale
                    detection (original ∪ SR-downsampled boxes via NMS),
                    SR-sourced rec crops, greedy CTC. No TTA, no beam, no LM.
                    Matches the FINAL_REPORT-recommended Phase 3 config:
                    year F1 = 0.930, brand F1 = 0.575, model F1 = 0.654.
                    ~7-10× slower than fast.
                "research" — production stack PLUS per-crop TTA + beam-search
                    CTC with optional KenLM rescoring (env-gated via
                    HAILO_BEAM_LM_PATH / HAILO_BEAM_UNIGRAMS_PATH /
                    HAILO_BEAM_ALPHA / HAILO_BEAM_BETA). Kept for offline
                    ablation against future rec HEFs; the FINAL_REPORT showed
                    these regressed year F1 to 0.850 on the current rec HEF.

        Mode validation runs BEFORE device init so callers get a clean
        ValueError on bad mode strings even when HAILO_VISION_ENABLED=0.

        Returns:
          OcrResult(text=" ".join(per-box decoded strings), boxes=[...])
          Boxes are always reported in *original* image coordinates regardless of mode.
        """
        if mode not in VALID_OCR_MODES:
            raise ValueError(f"ocr mode must be one of {VALID_OCR_MODES}, got {mode!r}")
        self.ensure_initialized()
        import cv2
        import hailo_platform as hpf
        import numpy as np

        # Decouple the four knobs the old `quality` mode entangled. Production
        # keeps the only Phase 3 win (multi-scale + SR-sourced crops); research
        # additionally engages the experimental TTA + beam stack.
        use_sr = mode in ("production", "research")
        use_multi_scale = mode in ("production", "research")
        use_tta = mode == "research"
        use_beam = mode == "research"

        raw = cv2.imread(str(image_path))
        if raw is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        orig_h, orig_w = raw.shape[:2]

        # Production/research: detection runs on the ORIGINAL image (sharper
        # text boundaries → cleaner contours) but recognition crops from the
        # SR'd image (2× pixels per character → cleaner rec input). Box
        # coordinates live in original-image space throughout; we upscale once
        # at crop time.
        if use_sr:
            sr_img = self.super_resolve(raw)
            sr_scale = 2
        else:
            sr_img = None
            sr_scale = 1

        min_area = 20
        min_rec_side = 10

        # Detection pass(es). Fast mode: single pass on the original.
        # Production/research: union of (original 544×960) and (SR-downsampled
        # 544×960) boxes — SR downsample gives detection a different frequency
        # band of the same text, catching blocks the direct downsample missed
        # (e.g. small overlay labels on stylized banners).
        boxes = self._detect_boxes(raw, score_threshold, min_area)
        if use_multi_scale and sr_img is not None:
            boxes_sr = self._detect_boxes(sr_img, score_threshold, min_area)
            # boxes_sr are in SR coordinates (2×). Halve them so all boxes live
            # in original-image space before NMS merge.
            boxes_sr = [(x // 2, y // 2, max(1, w // 2), max(1, h // 2)) for (x, y, w, h) in boxes_sr]
            boxes = _merge_boxes_nms(boxes + boxes_sr, iou_threshold=0.5)

        if not boxes:
            return OcrResult(text="", boxes=[])

        # Recognition stage — one rec call per detected box
        boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))  # top-left reading order
        texts: list[str] = []
        hef_rec, ng_rec = self._networks[HEF_OCR_RECOGNIZE]
        in_info_rec = hef_rec.get_input_vstream_infos()[0]
        out_info_rec = hef_rec.get_output_vstream_infos()[0]
        in_params_rec = hpf.InputVStreamParams.make_from_network_group(
            ng_rec, quantized=True, format_type=hpf.FormatType.UINT8
        )
        out_params_rec = hpf.OutputVStreamParams.make_from_network_group(
            ng_rec, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

        # Pick crop source: SR image (production/research) scales coordinates
        # 2×; fast mode reads from the original.
        crop_src = sr_img if sr_img is not None else raw

        top_beams_per_box: list[list[tuple[str, float]]] = []

        with hpf.InferVStreams(ng_rec, in_params_rec, out_params_rec) as rec_pipe:
            for (x, y, w, h) in boxes_sorted:
                if w < min_rec_side or h < min_rec_side:
                    continue
                cx, cy, cw, ch = x * sr_scale, y * sr_scale, w * sr_scale, h * sr_scale
                crop = crop_src[cy:cy + ch, cx:cx + cw]
                if crop.size == 0:
                    continue
                if use_tta:
                    log_softmax_sum = None
                    n_valid = 0
                    for width_scale, clahe in _TTA_AUGMENTATIONS:
                        rec_tensor = _preprocess_for_rec(crop, apply_clahe=clahe, width_scale=width_scale)
                        rec_out = rec_pipe.infer({in_info_rec.name: rec_tensor})
                        aug_logits = np.asarray(rec_out[out_info_rec.name]).reshape(40, 18385).astype(np.float32)
                        aug_logits = aug_logits - aug_logits.max(axis=-1, keepdims=True)
                        aug_probs = np.exp(aug_logits)
                        aug_probs /= aug_probs.sum(axis=-1, keepdims=True)
                        aug_log_probs = np.log(aug_probs + 1e-9)
                        log_softmax_sum = aug_log_probs if log_softmax_sum is None else (log_softmax_sum + aug_log_probs)
                        n_valid += 1
                    if n_valid == 0 or log_softmax_sum is None:
                        continue
                    mean_log_probs = log_softmax_sum / n_valid
                    # Back to a valid probability distribution via softmax over
                    # the log-mean (renormalizes so each timestep sums to 1).
                    mean_log_probs = mean_log_probs - mean_log_probs.max(axis=-1, keepdims=True)
                    probs = np.exp(mean_log_probs)
                    probs /= probs.sum(axis=-1, keepdims=True)
                    # pyctcdecode expects probabilities (not logits). Pass the
                    # aggregated distribution directly — _ctc_decode_beam will
                    # re-softmax but since inputs are already probabilities the
                    # normalization is idempotent to numerical precision.
                    logits = np.log(probs + 1e-9)
                else:
                    rec_tensor = _preprocess_for_rec(crop)
                    rec_out = rec_pipe.infer({in_info_rec.name: rec_tensor})
                    logits = np.asarray(rec_out[out_info_rec.name]).reshape(40, 18385)
                if use_beam:
                    decoded, beams = self._ctc_decode_beam(logits)
                    if beams:
                        top_beams_per_box.append(beams)
                else:
                    decoded = self._ctc_decode(logits)
                if decoded:
                    texts.append(decoded)

        # Attach per-box top-K beams to the result via a side-channel dict on
        # OcrResult (when in beam mode) so downstream Phase 8 can consume them.
        result = OcrResult(text=" ".join(texts), boxes=boxes_sorted)
        if use_beam and top_beams_per_box:
            # Stash on the instance as the most-recent-call side-channel since
            # OcrResult is frozen; consumers that want N-best can read it.
            self._last_top_beams = top_beams_per_box
        return result

    def _get_ocr_charset(self) -> list[str]:
        if self._ocr_charset is None:
            if not OCR_CHARSET_PATH.exists():
                raise HEFMissing(f"PaddleOCR charset missing at {OCR_CHARSET_PATH}")
            with OCR_CHARSET_PATH.open("r", encoding="utf-8") as f:
                chars = [line.rstrip("\n").rstrip("\r") for line in f]
            chars.append(" ")  # PaddleOCR v5 uses use_space_char=True
            self._ocr_charset = chars
        return self._ocr_charset

    def _ctc_decode(self, logits: Any) -> str:
        """Greedy CTC decode: argmax per timestep, collapse consecutive duplicates,
        remove blank (class 0). Class i (i>=1) maps to charset[i-1]."""
        import numpy as np

        charset = self._get_ocr_charset()
        preds = np.argmax(logits, axis=-1)  # shape (40,)
        result_chars: list[str] = []
        prev = -1
        for idx in preds:
            idx = int(idx)
            if idx != prev and idx != 0 and 1 <= idx <= len(charset):
                result_chars.append(charset[idx - 1])
            prev = idx
        return "".join(result_chars)

    def _get_hotwords(self) -> list[str]:
        """CAR_BRANDS + CAR_MODELS from the shared vocab, ASCII-folded for the
        beam decoder's hotword matcher (which does literal substring compare)."""
        if self._hotwords is not None:
            return self._hotwords
        try:
            import sys
            sys.path.insert(0, "DATA_ROOT (see README)")
            from vision_shared.features.title import CAR_BRANDS, CAR_MODELS  # noqa: E402
        except Exception:
            self._hotwords = []
            return self._hotwords
        seen: set[str] = set()
        words: list[str] = []
        for item in tuple(CAR_BRANDS) + tuple(CAR_MODELS):
            # The rec charset is mostly-ASCII Latin; upper-case hotwords match
            # what the rec model tends to emit on thumbnail banners.
            s = item.upper().strip()
            if s and s not in seen:
                seen.add(s)
                words.append(s)
        self._hotwords = words
        return self._hotwords

    def _get_beam_decoder(self) -> Any:
        """Lazy-build the pyctcdecode decoder with the PaddleOCR charset.

        labels[0] is blank ("" per pyctcdecode convention), labels[1:] mirror the
        charset — so class index alignment matches the rec HEF's output layout
        (blank at 0, charset starts at 1).

        The ppocrv5 charset contains ~10 flag emojis that are 2-codepoint
        sequences (regional indicators). pyctcdecode auto-detects BPE when any
        token has length >1, so we rewrite those to single-char Unicode PUA
        placeholders (U+E000+). They're exceedingly unlikely to appear in
        automotive thumbnail text; mapping them to PUA preserves index
        alignment while keeping pyctcdecode in char-level mode.

        KenLM rescoring (Phase 6): if env var HAILO_BEAM_LM_PATH points to a
        .klm file, the decoder is built with that LM and the tuning knobs
        HAILO_BEAM_ALPHA (LM weight) and HAILO_BEAM_BETA (per-word bonus)
        control rescoring. alpha=0 effectively disables the LM, so grid
        search / A-B toggles happen via env vars without code changes.
        """
        if self._beam_decoder is not None:
            return self._beam_decoder
        from pyctcdecode import build_ctcdecoder  # lazy import
        charset = self._get_ocr_charset()
        labels = [""]
        pua_idx = 0
        for tok in charset:
            # Replace anything that would flip pyctcdecode into BPE mode: tokens
            # longer than a single character (flag emojis) or tokens that start
            # with the BPE boundary marker '▁' / the HuggingFace '##' prefix.
            if len(tok) != 1 or tok.startswith("▁") or tok.startswith("##"):
                labels.append(chr(0xE000 + pua_idx))
                pua_idx += 1
            else:
                labels.append(tok)
        lm_path = os.environ.get("HAILO_BEAM_LM_PATH", "").strip()
        unigrams_path = os.environ.get("HAILO_BEAM_UNIGRAMS_PATH", "").strip()
        alpha = float(os.environ.get("HAILO_BEAM_ALPHA", "0.5"))
        beta = float(os.environ.get("HAILO_BEAM_BETA", "1.0"))
        unigrams: list[str] | None = None
        if unigrams_path and Path(unigrams_path).exists():
            with open(unigrams_path, encoding="utf-8") as f:
                unigrams = [line.strip() for line in f if line.strip()]
        if lm_path and Path(lm_path).exists():
            self._beam_decoder = build_ctcdecoder(
                labels, kenlm_model_path=lm_path, unigrams=unigrams, alpha=alpha, beta=beta,
            )
        else:
            self._beam_decoder = build_ctcdecoder(labels, unigrams=unigrams)
        return self._beam_decoder

    def _ctc_decode_beam(
        self,
        logits: Any,
        beam_width: int = 100,
        hotword_weight: float = 10.0,
        top_k: int = 5,
    ) -> tuple[str, list[tuple[str, float]]]:
        """Beam-search CTC decode with CAR_BRANDS + CAR_MODELS hotword boost.

        Softmaxes the (T, V) logits first so pyctcdecode sees probabilities.
        Returns (top1_text, top_k_beams_with_scores).
        """
        import numpy as np

        decoder = self._get_beam_decoder()
        hotwords = self._get_hotwords()

        # softmax over the vocab axis; numerically stable
        x = np.asarray(logits, dtype=np.float32)
        x = x - x.max(axis=-1, keepdims=True)
        np.exp(x, out=x)
        x /= x.sum(axis=-1, keepdims=True)

        beams = decoder.decode_beams(
            x,
            beam_width=beam_width,
            hotwords=hotwords if hotwords else None,
            hotword_weight=hotword_weight if hotwords else 0.0,
        )
        if not beams:
            return "", []
        # beams: list of (text, last_state, text_frames, logit_score, lm_score)
        top = beams[:top_k]
        top_scored = [(b[0], float(b[3]) + float(b[4])) for b in top]
        return top[0][0], top_scored

    def _detect_boxes(
        self,
        image_bgr: Any,
        score_threshold: float,
        min_area: int,
    ) -> list[tuple[int, int, int, int]]:
        """Run the PaddleOCR detection HEF on `image_bgr` and return bounding
        boxes in the input image's coordinate space.

        Input image is always resized to the HEF's fixed 960×544 regardless of
        source dimensions — caller is responsible for choosing which image to
        pass (original vs SR).
        """
        import cv2
        import hailo_platform as hpf
        import numpy as np

        work_h, work_w = image_bgr.shape[:2]
        h_t, w_t = 544, 960
        scale_x = w_t / work_w
        scale_y = h_t / work_h
        resized = cv2.resize(image_bgr, (w_t, h_t))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

        hef_det, ng_det = self._networks[HEF_OCR_DETECT]
        in_info_det = hef_det.get_input_vstream_infos()[0]
        out_info_det = hef_det.get_output_vstream_infos()[0]
        in_params = hpf.InputVStreamParams.make_from_network_group(
            ng_det, quantized=True, format_type=hpf.FormatType.UINT8
        )
        out_params = hpf.OutputVStreamParams.make_from_network_group(
            ng_det, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        with hpf.InferVStreams(ng_det, in_params, out_params) as pipe:
            out = pipe.infer({in_info_det.name: tensor})

        heatmap = np.asarray(out[out_info_det.name]).reshape(h_t, w_t)
        binary = (heatmap > score_threshold).astype(np.uint8) * 255
        binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2)), iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple[int, int, int, int]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < min_area:
                continue
            x_orig = int(round(x / scale_x))
            y_orig = int(round(y / scale_y))
            w_orig = max(1, int(round(w / scale_x)))
            h_orig = max(1, int(round(h / scale_y)))
            x_orig, y_orig, w_orig, h_orig = _unclip_box(x_orig, y_orig, w_orig, h_orig, work_w, work_h, ratio=0.12)
            boxes.append((x_orig, y_orig, w_orig, h_orig))
        return boxes

    def vehicle_detect(self, image_path: str | Path, score_threshold: float = 0.3) -> list[VehicleBox]:
        """YOLOv5m vehicle detector via Hailo-8L.

        The HEF has a fixed 1920×1080 UINT8 input and an on-device NMS
        post-process that emits a list of (batch, num_detections, 5)
        float32 arrays where each row is (y1, x1, y2, x2, score) in
        normalized [0, 1] coordinates. Single class ("vehicle") — the
        per-class NMS score_thresh=0.2 and IoU=0.6 are baked into the
        HEF; the `score_threshold` arg here is an additional filter on
        the returned detections.

        Boxes returned in the ORIGINAL input image's pixel space.
        """
        self.ensure_initialized()
        if HEF_VEHICLE_DETECT not in self._networks:
            raise HEFMissing(
                f"{HEF_VEHICLE_DETECT} not loaded — drop it in {_models_dir()} "
                "and re-instantiate HailoRuntime"
            )
        import cv2
        import hailo_platform as hpf
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        orig_h, orig_w = img.shape[:2]
        resized = cv2.resize(img, (1920, 1080))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

        hef, ng = self._networks[HEF_VEHICLE_DETECT]
        in_info = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        in_p = hpf.InputVStreamParams.make_from_network_group(ng, quantized=True, format_type=hpf.FormatType.UINT8)
        out_p = hpf.OutputVStreamParams.make_from_network_group(ng, quantized=False, format_type=hpf.FormatType.FLOAT32)
        with hpf.InferVStreams(ng, in_p, out_p) as pipe:
            raw = pipe.infer({in_info.name: tensor})

        # Output is a list of per-batch arrays shape (1, N_detections, 5).
        # 5 values = (y1, x1, y2, x2, score) in normalized coords.
        result = raw[out_info.name]
        if isinstance(result, list):
            result = result[0]
        arr = np.asarray(result)
        if arr.size == 0:
            return []
        # Squeeze batch dim if present
        if arr.ndim == 3:
            arr = arr[0]

        boxes: list[VehicleBox] = []
        for row in arr:
            y1, x1, y2, x2, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
            if score < score_threshold:
                continue
            px1 = max(0, int(round(x1 * orig_w)))
            py1 = max(0, int(round(y1 * orig_h)))
            px2 = min(orig_w, int(round(x2 * orig_w)))
            py2 = min(orig_h, int(round(y2 * orig_h)))
            if px2 <= px1 or py2 <= py1:
                continue
            boxes.append(VehicleBox(x=px1, y=py1, w=px2 - px1, h=py2 - py1, score=score))
        return boxes

    def super_resolve(self, image_bgr: Any) -> Any:
        """Real-ESRGAN x2 super-resolution via Hailo-8L.

        HEF I/O is fixed at 512×512 UINT8 → 1024×1024 UINT8. For arbitrary input
        sizes we tile with 128-pixel overlap (input coords) and triangular-window
        blend the 2× output tiles to suppress seam artifacts at tile boundaries.

        Args:
            image_bgr: H×W×3 uint8 BGR image (any dimensions).

        Returns:
            (2H)×(2W)×3 uint8 BGR image.

        Raises:
            HEFMissing: if real_esrgan_x2.hef isn't on disk / wasn't configured.
        """
        self.ensure_initialized()
        if HEF_SUPER_RESOLVE not in self._networks:
            raise HEFMissing(
                f"{HEF_SUPER_RESOLVE} not loaded — place it in {_models_dir()} "
                "and re-instantiate HailoRuntime"
            )
        import cv2
        import hailo_platform as hpf
        import numpy as np

        tile_in, tile_out, stride = 512, 1024, 384  # 128-pixel input overlap
        h, w = image_bgr.shape[:2]

        def _positions(length: int, tile: int, step: int) -> list[int]:
            if length <= tile:
                return [0]
            positions: list[int] = []
            pos = 0
            while pos + tile <= length:
                positions.append(pos)
                pos += step
            if positions[-1] + tile < length:
                positions.append(length - tile)
            return sorted(set(positions))

        # If the image is smaller than a tile in either dimension, reflect-pad.
        pad_h = max(0, tile_in - h)
        pad_w = max(0, tile_in - w)
        if pad_h or pad_w:
            image_bgr = cv2.copyMakeBorder(
                image_bgr, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101
            )
        H, W = image_bgr.shape[:2]
        ys = _positions(H, tile_in, stride)
        xs = _positions(W, tile_in, stride)

        # Triangular blending window (tapered on all 4 edges).
        overlap_out_px = (tile_in - stride) * 2  # 256 px in output space
        w1d = np.ones(tile_out, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, overlap_out_px, endpoint=False, dtype=np.float32)
        w1d[:overlap_out_px] = ramp
        w1d[-overlap_out_px:] = ramp[::-1]
        win = np.outer(w1d, w1d).astype(np.float32)[..., None]

        out_H, out_W = H * 2, W * 2
        sr = np.zeros((out_H, out_W, 3), dtype=np.float32)
        wsum = np.zeros((out_H, out_W, 1), dtype=np.float32)

        hef, ng = self._networks[HEF_SUPER_RESOLVE]
        in_info = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        in_params = hpf.InputVStreamParams.make_from_network_group(
            ng, quantized=True, format_type=hpf.FormatType.UINT8
        )
        out_params = hpf.OutputVStreamParams.make_from_network_group(
            ng, quantized=True, format_type=hpf.FormatType.UINT8
        )
        with hpf.InferVStreams(ng, in_params, out_params) as pipe:
            for y in ys:
                for x in xs:
                    tile_rgb = cv2.cvtColor(
                        image_bgr[y:y + tile_in, x:x + tile_in], cv2.COLOR_BGR2RGB
                    )
                    tensor = np.expand_dims(tile_rgb, axis=0).astype(np.uint8)
                    result = pipe.infer({in_info.name: tensor})
                    sr_tile = np.asarray(result[out_info.name]).reshape(tile_out, tile_out, 3).astype(np.float32)
                    oy, ox = y * 2, x * 2
                    sr[oy:oy + tile_out, ox:ox + tile_out] += sr_tile * win
                    wsum[oy:oy + tile_out, ox:ox + tile_out] += win

        sr = sr / np.maximum(wsum, 1e-6)
        sr_bgr = cv2.cvtColor(np.clip(sr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # Strip the reflection padding in output space
        return sr_bgr[: h * 2, : w * 2]

    def embed(self, image_path: str | Path) -> list[float]:
        """TinyCLIP image encoder on Hailo-8L. Returns a flat float vector (typically 512-d)."""
        self.ensure_initialized()
        import cv2
        import hailo_platform as hpf
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor = np.expand_dims(img_224, axis=0).astype(np.uint8)

        hef, ng = self._networks[HEF_CLIP_EMBED]
        in_info = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        in_params = hpf.InputVStreamParams.make_from_network_group(
            ng, quantized=True, format_type=hpf.FormatType.UINT8
        )
        out_params = hpf.OutputVStreamParams.make_from_network_group(
            ng, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        with hpf.InferVStreams(ng, in_params, out_params) as pipe:
            output = pipe.infer({in_info.name: tensor})
        vector = np.asarray(output[out_info.name]).flatten().astype(float)
        return vector.tolist()

    # --- editor-workstation tools -----------------------------------------

    def _require(self, basename: str) -> tuple[Any, Any]:
        """The (hef, network_group) for an optional HEF, or HEFMissing naming it."""
        self.ensure_initialized()
        if basename not in self._networks:
            raise HEFMissing(
                f"{basename} not loaded — drop it in {_models_dir()} "
                "and re-instantiate HailoRuntime"
            )
        return self._networks[basename]

    def _infer_single(self, basename: str, tensor: Any, quantized_out: bool = False) -> dict[str, Any]:
        """Run one UINT8 NHWC tensor through a HEF; return {output_name: array}.

        quantized_out=False dequantizes every output to FLOAT32 (what every
        vector/regression head here wants). Pass True only for a model whose
        output is itself an image (zero_dce) so it comes back as UINT8 pixels.
        """
        import hailo_platform as hpf

        hef, ng = self._require(basename)
        in_info = hef.get_input_vstream_infos()[0]
        in_p = hpf.InputVStreamParams.make_from_network_group(
            ng, quantized=True, format_type=hpf.FormatType.UINT8
        )
        out_p = hpf.OutputVStreamParams.make_from_network_group(
            ng,
            quantized=quantized_out,
            format_type=hpf.FormatType.UINT8 if quantized_out else hpf.FormatType.FLOAT32,
        )
        with hpf.InferVStreams(ng, in_p, out_p) as pipe:
            out = pipe.infer({in_info.name: tensor})
        # Every caller takes "the" output. If a swapped-in HEF has several heads,
        # dict order would silently pick one of them — refuse instead.
        if len(out) != 1:
            raise HailoDeviceError(
                f"{basename} has {len(out)} output vstreams; _infer_single expects exactly 1"
            )
        return out

    def face_embed(self, image_path: str | Path, face: FaceBox) -> list[float]:
        """ArcFace identity vector (512-d, L2-normalised) for ONE detected face.

        ArcFace is only meaningful on an ALIGNED 112×112 crop: the 5 SCrFD
        landmarks are mapped onto the canonical insightface template with a
        similarity transform (rotation + uniform scale + translation), exactly
        as the reference pipeline does. Cosine similarity between two of these
        vectors is the identity score; same person ≈ >0.5, different ≈ <0.3 in
        the usual 112-crop regime. Without landmarks we refuse rather than embed
        an unaligned box — that silently produces vectors that cluster by pose
        instead of identity.
        """
        import cv2
        import numpy as np

        if len(face.kps) != 5:
            raise InvalidInput(
                "face_embed needs the 5 SCrFD landmarks (FaceBox.kps); this box has none"
            )
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")

        src = np.asarray(face.kps, dtype=np.float32)
        dst = np.asarray(ARCFACE_TEMPLATE_112, dtype=np.float32)
        # estimateAffinePartial2D = similarity (4 DOF), the standard ArcFace warp.
        m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if m is None:
            raise InvalidInput("could not estimate the face alignment transform")
        # estimateAffinePartial2D returns a finite-looking but singular matrix on
        # coincident/collinear landmarks instead of None. That warps to a black
        # 112×112 crop which ArcFace would embed into a plausible garbage vector.
        # A similarity transform's uniform scale is sqrt(a²+b²) of its first column.
        if not np.all(np.isfinite(m)):
            raise InvalidInput("face alignment transform is not finite (bad landmarks)")
        scale2 = float(m[0, 0] ** 2 + m[1, 0] ** 2)
        if scale2 < 1e-6:
            raise InvalidInput("face alignment transform is degenerate (landmarks coincident/collinear)")
        crop = cv2.warpAffine(img, m, (112, 112), borderValue=(0, 0, 0))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

        out = self._infer_single(HEF_FACE_EMBED, tensor)
        vec = np.asarray(next(iter(out.values()))).flatten().astype(np.float64)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise HailoDeviceError("ArcFace returned a zero vector")
        return (vec / norm).tolist()

    def object_detect(self, image_path: str | Path, score_threshold: float = 0.3) -> list[ObjectBox]:
        """YOLOv8s, 80 COCO classes, via Hailo's on-chip NMS.

        The HEF's output is HAILO NMS BY CLASS: one array PER CLASS (80 of
        them), each (n_i, 5) rows of (y1, x1, y2, x2, score) in normalised
        [0, 1] coordinates. That is a different shape from the single-class
        vehicle HEF — never reuse that decoder here. Boxes are returned in the
        original image's pixel space.
        """
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        orig_h, orig_w = img.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(img, (640, 640)), cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

        out = self._infer_single(HEF_OBJECT_DETECT, tensor)
        result = next(iter(out.values()))
        # Batch of 1 → the per-class list is result[0].
        per_class = result[0] if isinstance(result, (list, tuple)) and len(result) == 1 and isinstance(result[0], (list, tuple)) else result
        # The contract is a FIXED 80-slot list where position == COCO class id.
        # A sparse/shorter list would make enumerate() silently relabel every
        # detection (a car reported as "person"); a non-list means a different
        # HEF variant was dropped in. Both are device/HEF errors, not input.
        if not isinstance(per_class, (list, tuple)):
            raise HailoDeviceError(
                f"{HEF_OBJECT_DETECT} NMS output is {type(per_class).__name__}, expected a per-class list"
            )
        if len(per_class) != len(COCO80):
            raise HailoDeviceError(
                f"{HEF_OBJECT_DETECT} NMS output has {len(per_class)} classes, expected {len(COCO80)} "
                "— is this a non-NMS or non-COCO build?"
            )

        boxes: list[ObjectBox] = []
        for class_id, arr in enumerate(per_class):
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size == 0:
                continue
            # Validate the (n, 5) row shape explicitly — reshape(-1, 5) would
            # silently reinterpret a (n, 4) array whenever n*4 divides by 5.
            if arr.ndim == 1 and arr.size == 5:
                arr = arr.reshape(1, 5)
            elif not (arr.ndim == 2 and arr.shape[1] == 5):
                raise HailoDeviceError(
                    f"{HEF_OBJECT_DETECT} class {class_id} rows have shape {arr.shape}, expected (n, 5)"
                )
            if not np.all(np.isfinite(arr)):
                raise HailoDeviceError(f"{HEF_OBJECT_DETECT} class {class_id} emitted non-finite values")
            for y1, x1, y2, x2, score in arr:
                if float(score) < score_threshold:
                    continue
                px1 = max(0, int(round(float(x1) * orig_w)))
                py1 = max(0, int(round(float(y1) * orig_h)))
                px2 = min(orig_w, int(round(float(x2) * orig_w)))
                py2 = min(orig_h, int(round(float(y2) * orig_h)))
                if px2 <= px1 or py2 <= py1:
                    continue
                label = COCO80[class_id] if class_id < len(COCO80) else f"class_{class_id}"
                boxes.append(ObjectBox(
                    x=px1, y=py1, w=px2 - px1, h=py2 - py1,
                    score=float(score), label=label, class_id=class_id,
                ))
        boxes.sort(key=lambda b: b.score, reverse=True)
        return boxes

    def depth(self, image_path: str | Path) -> Any:
        """Depth-Anything-V2 ViT-S relative depth map, float32 (224, 224).

        Higher = closer (inverse-depth convention, as the model emits it). The
        224 working resolution is a preview-grade map: good for shot analysis
        and parallax previews, not for a production depth pass.
        """
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)
        out = self._infer_single(HEF_DEPTH, tensor)
        raw = np.asarray(next(iter(out.values())))
        if raw.size != 224 * 224:
            raise HailoDeviceError(f"{HEF_DEPTH} output has {raw.size} values, expected {224 * 224}")
        return raw.reshape(224, 224).astype(np.float32)

    def person_embed(self, image_path: str | Path, box: ObjectBox | None = None) -> list[float]:
        """OSNet re-id vector (512-d, L2-normalised) for a person crop.

        Re-identification works WITHOUT a visible face — it keys on clothing
        and body shape — so it is the tool for tracking the same person across
        shots where ArcFace has nothing to see. Pass the person's ObjectBox
        (from object_detect) to crop; with box=None the whole image is used.
        """
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        if box is not None:
            img = img[box.y:box.y + box.h, box.x:box.x + box.w]
            if img.size == 0:
                raise InvalidInput("person box is empty after cropping")
            # A 1-2 px box survives the clamp, upsamples to a flat 128×256 image
            # and embeds into a real-looking vector with no identity signal in it.
            if min(img.shape[:2]) < 20:
                raise InvalidInput(
                    f"person crop too small for re-id ({img.shape[1]}×{img.shape[0]} px, need ≥20)"
                )
        rgb = cv2.cvtColor(cv2.resize(img, (128, 256)), cv2.COLOR_BGR2RGB)  # (w, h)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)
        out = self._infer_single(HEF_PERSON_EMBED, tensor)
        vec = np.asarray(next(iter(out.values()))).flatten().astype(np.float64)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise HailoDeviceError("OSNet returned a zero vector")
        return (vec / norm).tolist()

    def enhance_low_light(self, image_path: str | Path) -> Any:
        """Zero-DCE low-light enhancement; returns a BGR uint8 image at the
        ORIGINAL resolution (the NPU works at 600×400, result is resized back).
        A preview-grade look-brightening pass — not a replacement for a grade.
        """
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        orig_h, orig_w = img.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(img, (600, 400)), cv2.COLOR_BGR2RGB)  # (w, h)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)
        out = self._infer_single(HEF_LOW_LIGHT, tensor, quantized_out=True)
        raw = np.asarray(next(iter(out.values())))
        if raw.size != 400 * 600 * 3:
            raise HailoDeviceError(f"{HEF_LOW_LIGHT} output has {raw.size} values, expected {400 * 600 * 3}")
        enhanced = raw.reshape(400, 600, 3).astype(np.uint8)
        bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    def status(self) -> dict[str, Any]:
        """Return device state without touching the VDMA path. Safe when disabled."""
        info: dict[str, Any] = {
            "enabled": _enabled(),
            "initialized": self._initialized,
            "models_dir": str(_models_dir()),
            "hefs_missing": _verify_hefs_present(),
            "loaded_networks": sorted(self._networks.keys()),
        }
        if _enabled():
            try:
                import subprocess
                r = subprocess.run(
                    ["hailortcli", "fw-control", "identify"],
                    capture_output=True, text=True, timeout=10,
                )
                info["fw_control_identify_rc"] = r.returncode
                info["fw_control_identify_tail"] = r.stdout.splitlines()[-5:] if r.stdout else []
            except Exception as e:
                info["fw_control_error"] = str(e)
        return info
