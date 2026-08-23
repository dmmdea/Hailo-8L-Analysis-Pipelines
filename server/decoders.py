"""Host-side decoders for RAW-HEAD Hailo HEFs (pose / instance segmentation).

Ported from hailo_model_zoo/core/postprocessing (MIT, hailo-ai/hailo_model_zoo
master @ 2026-08-23): `_yolov8_decoding`, the pose and seg `non_max_suppression`
variants, `process_mask`/`crop_mask` — the exact math the zoo evaluates its
compiled v2.19.0 HEFs with, so the decode matches what the accuracy tables
measured. Two deliberate adaptations, both recorded here rather than silent:

- The zoo's `cnms` (cython) is replaced by a plain-numpy greedy NMS with the
  same class-shift trick — no compiled extension on the deployment box.
- Score tensors are auto-ranged: the compiled HEFs emit post-sigmoid scores
  (the zoo thresholds them raw), but if a build ever emits logits (values
  outside [0, 1]) we apply sigmoid once and note it in the result, instead of
  silently thresholding logits — that failure mode reads as "no detections".

Everything here is pure numpy on host; the NPU only ran the backbone+heads.
"""
from __future__ import annotations

from typing import Any

import numpy as np

REG_MAX = 15  # yolov8 DFL regression length (16 bins) — the zoo's fixed value
STRIDES = (8, 16, 32)  # fine → coarse; tensors are matched to strides by H×W


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _greedy_nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Indices kept by greedy IoU suppression, highest score first (cnms port)."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=int)


def _dfl_boxes(box_distribute: np.ndarray, stride: int, image_dims: tuple[int, int]) -> np.ndarray:
    """One scale of the zoo's `_yolov8_decoding`: DFL distribution → xywh boxes
    in input-pixel space. box_distribute: (1, H, W, 4*(REG_MAX+1))."""
    shape = [int(x / stride) for x in image_dims]
    grid_x = np.arange(shape[1]) + 0.5
    grid_y = np.arange(shape[0]) + 0.5
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    ct_row = grid_y.flatten() * stride
    ct_col = grid_x.flatten() * stride
    center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

    reg_range = np.arange(REG_MAX + 1)
    box_distribute = np.reshape(
        box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, REG_MAX + 1)
    )
    box_distance = _softmax(box_distribute)
    box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
    box_distance = np.sum(box_distance, axis=-1) * stride
    box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
    decode_box = np.expand_dims(center, axis=0) + box_distance

    xmin, ymin = decode_box[:, :, 0], decode_box[:, :, 1]
    xmax, ymax = decode_box[:, :, 2], decode_box[:, :, 3]
    return np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])


def _group_scales(outputs: dict[str, np.ndarray], image_dims: tuple[int, int]) -> dict[int, dict[str, np.ndarray]]:
    """Group raw output tensors by stride, tagging each tensor's role by its
    trailing channel count. Order-independent: HailoRT vstream naming/order is
    a compilation detail, but (H, W, C) identifies every yolov8 head tensor
    unambiguously: C==64 box-DFL, C==51 keypoints, C==32 at H<proto mask
    coefficients, anything else the class-score tensor."""
    scales: dict[int, dict[str, np.ndarray]] = {}
    for name, t in outputs.items():
        if t.ndim != 4:
            raise ValueError(f"unexpected output rank for {name}: {t.shape}")
        h = t.shape[1]
        stride = image_dims[0] // h if h else 0
        if stride not in STRIDES:
            continue  # the 160×160 proto tensor (stride 4) is handled by the caller
        role_by_c = {64: "box", 51: "kpt", 32: "coeff"}
        role = role_by_c.get(t.shape[3], "score")
        scales.setdefault(stride, {})[role] = t
    missing = [s for s in STRIDES if s not in scales]
    if missing:
        raise ValueError(f"missing yolov8 head scales for strides {missing}; got {[(n, o.shape) for n, o in outputs.items()]}")
    return scales


def _scores_maybe_sigmoid(scores: np.ndarray) -> tuple[np.ndarray, bool]:
    if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
        return _sigmoid(scores), True
    return scores, False


def yolov8_pose_decode(
    outputs: dict[str, np.ndarray],
    image_dims: tuple[int, int] = (640, 640),
    score_thres: float = 0.3,
    iou_thres: float = 0.45,
    max_det: int = 100,
) -> dict[str, Any]:
    """Zoo yolov8_pose_estimation_postprocess, single image, 1 class (person).

    Returns dict: boxes (n,4 xyxy input-px), scores (n,), keypoints (n,17,2
    input-px), joint_scores (n,17) [0..1], plus scores_sigmoid_applied.
    """
    scales = _group_scales(outputs, image_dims)
    boxes_all, scores_all, kpts_all = [], [], []
    applied = False
    for stride in STRIDES:
        sc = scales[stride]
        if "kpt" not in sc:
            raise ValueError(f"stride {stride} lacks the 51-channel keypoint tensor — is this a pose HEF?")
        boxes_all.append(_dfl_boxes(sc["box"].astype(np.float32), stride, image_dims))
        s = np.reshape(sc["score"].astype(np.float32), (1, -1, sc["score"].shape[3]))
        s, a = _scores_maybe_sigmoid(s)
        applied = applied or a
        scores_all.append(s)
        k = np.reshape(sc["kpt"].astype(np.float32), (1, -1, 17, 3))
        # zoo kpt decode: xy*2, then stride*(xy-0.5)+center
        shape = [int(x / stride) for x in image_dims]
        gx, gy = np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)
        center = np.stack((gx.flatten() * stride, gy.flatten() * stride), axis=1)
        k[..., :2] *= 2
        k[..., :2] = stride * (k[..., :2] - 0.5) + np.expand_dims(center, axis=1)
        kpts_all.append(k)

    boxes = np.concatenate(boxes_all, axis=1)[0]          # (N, 4) xywh
    scores = np.concatenate(scores_all, axis=1)[0]        # (N, nc)
    kpts = np.concatenate(kpts_all, axis=1)[0]            # (N, 17, 3)

    conf = scores.max(axis=1)
    cand = conf > score_thres
    if not np.any(cand):
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)),
                "keypoints": np.zeros((0, 17, 2)), "joint_scores": np.zeros((0, 17)),
                "scores_sigmoid_applied": applied}
    boxes_xyxy = _xywh2xyxy(boxes[cand])
    conf = conf[cand]
    kpts = kpts[cand]
    order = conf.argsort()[::-1]
    boxes_xyxy, conf, kpts = boxes_xyxy[order], conf[order], kpts[order]
    keep = _greedy_nms(boxes_xyxy, conf, iou_thres)[:max_det]
    return {
        "boxes": boxes_xyxy[keep],
        "scores": conf[keep],
        "keypoints": kpts[keep][..., :2],
        "joint_scores": _sigmoid(kpts[keep][..., 2]),
        "scores_sigmoid_applied": applied,
    }


def yolov8_seg_decode(
    outputs: dict[str, np.ndarray],
    num_classes: int,
    image_dims: tuple[int, int] = (640, 640),
    score_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 50,
) -> dict[str, Any]:
    """Zoo yolov8_seg_postprocess, single image. Covers yolov8s_seg (80 classes)
    AND fast_sam_s (1 class — FastSAM-s is the same architecture).

    Returns dict: boxes (n,4 xyxy input-px), scores (n,), classes (n,),
    masks (n, H, W) bool in input space, scores_sigmoid_applied.
    """
    import cv2

    proto = None
    heads = {}
    for name, t in outputs.items():
        if t.ndim == 4 and t.shape[1] not in (image_dims[0] // s for s in STRIDES):
            proto = t  # the (1, 160, 160, 32) prototype tensor
        else:
            heads[name] = t
    if proto is None:
        raise ValueError(f"no prototype tensor found among outputs {[(n, o.shape) for n, o in outputs.items()]}")
    scales = _group_scales(heads, image_dims)

    boxes_all, scores_all, coeffs_all = [], [], []
    applied = False
    for stride in STRIDES:
        sc = scales[stride]
        if "coeff" not in sc:
            raise ValueError(f"stride {stride} lacks the 32-channel mask-coefficient tensor — is this a seg HEF?")
        boxes_all.append(_dfl_boxes(sc["box"].astype(np.float32), stride, image_dims))
        s = np.reshape(sc["score"].astype(np.float32), (1, -1, sc["score"].shape[3]))
        s, a = _scores_maybe_sigmoid(s)
        applied = applied or a
        scores_all.append(s)
        coeffs_all.append(np.reshape(sc["coeff"].astype(np.float32), (1, -1, 32)))

    boxes = np.concatenate(boxes_all, axis=1)[0]
    scores = np.concatenate(scores_all, axis=1)[0][:, :num_classes]
    coeffs = np.concatenate(coeffs_all, axis=1)[0]

    conf = scores.max(axis=1)
    cls = scores.argmax(axis=1)
    cand = conf > score_thres
    if not np.any(cand):
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "classes": np.zeros((0,), dtype=int),
                "masks": np.zeros((0, *image_dims), dtype=bool), "scores_sigmoid_applied": applied}
    boxes_xyxy = _xywh2xyxy(boxes[cand])
    conf, cls, coeffs = conf[cand], cls[cand], coeffs[cand]
    order = conf.argsort()[::-1]
    boxes_xyxy, conf, cls, coeffs = boxes_xyxy[order], conf[order], cls[order], coeffs[order]
    # per-class NMS via the zoo's class-shift trick
    shift = cls.astype(np.float32)[:, None] * 7680.0
    keep = _greedy_nms(boxes_xyxy + np.concatenate([shift] * 4, axis=1), conf, iou_thres)[:max_det]
    boxes_xyxy, conf, cls, coeffs = boxes_xyxy[keep], conf[keep], cls[keep], coeffs[keep]

    # zoo process_mask: sigmoid(coeff @ proto) → upsample to input dims → crop to box
    ph, pw, pc = proto.shape[1:]
    masks = _sigmoid(coeffs @ proto[0].reshape((-1, pc)).T).reshape((-1, ph, pw))
    masks = cv2.resize(np.transpose(masks, (1, 2, 0)), image_dims, interpolation=cv2.INTER_LINEAR)
    if masks.ndim == 2:
        masks = masks[..., np.newaxis]
    masks = np.transpose(masks, (2, 0, 1))
    ib = np.ceil(boxes_xyxy).astype(int)
    ib = np.where(ib > 0, ib, 0)
    for k in range(masks.shape[0]):
        x1, y1, x2, y2 = ib[k, 0], ib[k, 1], ib[k, 2], ib[k, 3]
        masks[k, :y1, :] = 0
        masks[k, y2:, :] = 0
        masks[k, :, :x1] = 0
        masks[k, :, x2:] = 0
    return {"boxes": boxes_xyxy, "scores": conf, "classes": cls,
            "masks": masks > 0.5, "scores_sigmoid_applied": applied}
