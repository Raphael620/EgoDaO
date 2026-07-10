"""Hand tracking — MediaPipe ONNX backend.

Two-stage pipeline:
  1. HandDetector (ONNX) → boxes + keypoints
  2. HandLandmarkDetector (ONNX) → 21×3 landmarks

Matches the public API of hand_tracker.py:
  process(bgr) → [(label, landmarks_21x3_pixel_coords), ...]
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ── model paths ────────────────────────────────────────────────────

_MODEL_DIR = Path(__file__).resolve().parent
_ONNX_DIR = _MODEL_DIR.parent / "test" / "onnx_models" / "mediapipe_hand-onnx-float"
_DETECTOR_PATH = str(_ONNX_DIR / "hand_detector.onnx")
_LANDMARK_PATH = str(_ONNX_DIR / "hand_landmark_detector.onnx")
_ANCHORS_PATH = str(_MODEL_DIR.parent / "test" / "MediaPipePyTorch" / "anchors_palm.npy")

# ── constants ──────────────────────────────────────────────────────

PD_INPUT = 256
LM_INPUT = 256
NUM_ANCHORS = 2944
NUM_COORDS = 18
SCORE_CLIP = 100
MIN_DET_SCORE = 0.5
NMS_IOU = 0.3
WRIST_KP = 0
MIDDLE_KP = 2
D_SCALE = 2.6
D_Y = -0.5
ROT_OFFSET = np.pi / 2


class HandTrackerONNX:
    """MediaPipe hand tracking via ONNX runtime."""

    def __init__(self, **kwargs):
        if not os.path.exists(_DETECTOR_PATH):
            raise FileNotFoundError(f"ONNX detector not found: {_DETECTOR_PATH}")
        if not os.path.exists(_LANDMARK_PATH):
            raise FileNotFoundError(f"ONNX landmark model not found: {_LANDMARK_PATH}")

        self._det = ort.InferenceSession(_DETECTOR_PATH)
        self._lm = ort.InferenceSession(_LANDMARK_PATH)
        self._anchors = np.load(_ANCHORS_PATH)

    # ── public API ─────────────────────────────────────────────────

    def process(self, bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Run hand detection on a BGR frame.

        Returns list of (label, landmarks_21x3_pixel_coords).
        """
        if bgr is None or bgr.size == 0:
            return []

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = self._detect_hands(rgb)
        if not dets:
            return []

        results = []
        for det in dets:
            lm, is_right = self._detect_landmarks(rgb, det)
            if lm is not None:
                label = "Right" if is_right else "Left"
                results.append((label, lm))

        return results

    # ── internal: hand detection ────────────────────────────────────

    def _detect_hands(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Decode → NMS → denormalize."""
        img1, (pad_y, pad_x) = self._resize_pad(rgb)

        blob = img1.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]
        box_coords, box_scores = self._det.run(None, {"image": blob})
        raw_boxes = box_coords[0]
        raw_scores = box_scores[0, :, 0]

        scores = 1.0 / (1.0 + np.exp(-np.clip(
            raw_scores.astype(np.float64), -SCORE_CLIP, SCORE_CLIP)))
        decoded = self._decode_boxes(raw_boxes, self._anchors)

        detections = self._weighted_nms(decoded, scores, MIN_DET_SCORE, NMS_IOU)
        if not detections:
            return []

        # denormalize to original image coords
        scale = (rgb.shape[0] / 256.0) if rgb.shape[0] >= rgb.shape[1] \
            else (rgb.shape[1] / 256.0)

        for det in detections:
            det[0] = det[0] * 256.0 * scale - pad_y
            det[1] = det[1] * 256.0 * scale - pad_x
            det[2] = det[2] * 256.0 * scale - pad_y
            det[3] = det[3] * 256.0 * scale - pad_x
            for k in range(7):
                det[4 + k * 2]     = det[4 + k * 2] * 256.0 * scale - pad_y
                det[4 + k * 2 + 1] = det[4 + k * 2 + 1] * 256.0 * scale - pad_x

        return detections

    # ── internal: landmark regression ───────────────────────────────

    def _detect_landmarks(self, rgb: np.ndarray,
                          det: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """detection2roi → affine warp → LM → inverse affine."""
        xc = (det[1] + det[3]) / 2.0
        yc = (det[0] + det[2]) / 2.0
        roi_s = (det[3] - det[1]) * D_SCALE
        yc += D_Y * (det[3] - det[1])

        x0, y0 = det[4 + WRIST_KP * 2 + 1], det[4 + WRIST_KP * 2]
        x1, y1 = det[4 + MIDDLE_KP * 2 + 1], det[4 + MIDDLE_KP * 2]
        theta = np.arctan2(y0 - y1, x0 - x1) - ROT_OFFSET

        # affine from original frame → 256×256 upright hand
        pts = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=np.float32)
        pts = pts * roi_s / 2.0
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        src = (R @ pts + np.array([[xc], [yc]])).T
        dst = np.array([[0, 0], [0, LM_INPUT - 1],
                        [LM_INPUT - 1, 0]], dtype=np.float32)
        affine = cv2.getAffineTransform(src[:3].astype(np.float32), dst)
        crop = cv2.warpAffine(rgb, affine, (LM_INPUT, LM_INPUT))

        blob = crop.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]
        _scores, lr, kpts_raw = self._lm.run(None, {"image": blob})

        is_right = bool(lr[0] > 0.5)
        affine_inv = cv2.invertAffineTransform(affine)
        kp = kpts_raw[0]

        landmarks = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            px = kp[i, 0].item() * LM_INPUT
            py = kp[i, 1].item() * LM_INPUT
            ox, oy = affine_inv @ np.array([px, py, 1.0])
            landmarks[i, 0] = ox
            landmarks[i, 1] = oy
            landmarks[i, 2] = kp[i, 2].item()

        return landmarks, is_right

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _resize_pad(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = img.shape[:2]
        if h >= w:
            nw = PD_INPUT * w // h
            nh = PD_INPUT
            ph, pw = 0, PD_INPUT - nw
        else:
            nh = PD_INPUT * h // w
            nw = PD_INPUT
            ph, pw = PD_INPUT - nh, 0

        ph1, ph2 = ph // 2, ph // 2 + ph % 2
        pw1, pw2 = pw // 2, pw // 2 + pw % 2
        resized = cv2.resize(img, (nw, nh))
        padded = np.pad(resized, ((ph1, ph2), (pw1, pw2), (0, 0)))
        scale = w / nw if h >= w else h / nh
        pad_xy = (int(ph1 * scale), int(pw1 * scale))
        return padded, pad_xy

    @staticmethod
    def _decode_boxes(raw: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        n = anchors.shape[0]
        decoded = np.zeros((n, NUM_COORDS + 1), dtype=np.float32)
        for i in range(n):
            a, r = anchors[i], raw[i]
            cx = r[0] / PD_INPUT * a[2] + a[0]
            cy = r[1] / PD_INPUT * a[3] + a[1]
            w = r[2] / PD_INPUT * a[2]
            h = r[3] / PD_INPUT * a[3]
            decoded[i, 0] = cy - h / 2
            decoded[i, 1] = cx - w / 2
            decoded[i, 2] = cy + h / 2
            decoded[i, 3] = cx + w / 2
            for k in range(7):
                kx = r[4 + k * 2] / PD_INPUT * a[2] + a[0]
                ky = r[4 + k * 2 + 1] / PD_INPUT * a[3] + a[1]
                decoded[i, 4 + k * 2] = ky
                decoded[i, 4 + k * 2 + 1] = kx
        return decoded

    @staticmethod
    def _weighted_nms(decoded: np.ndarray, scores: np.ndarray,
                      min_score: float, iou_thresh: float) -> list[np.ndarray]:
        mask = scores >= min_score
        if not mask.any():
            return []

        dets = decoded[mask].copy()
        for i in range(len(dets)):
            dets[i, NUM_COORDS] = scores[mask][i]

        order = np.argsort(scores[mask])[::-1]
        keep = []

        while len(order) > 0:
            keep.append(dets[order[0]])
            if len(order) == 1:
                break

            b1 = dets[order[0], :4]
            mask_keep = np.ones(len(order) - 1, dtype=bool)
            for j, oi in enumerate(order[1:]):
                b2 = dets[oi, :4]
                xi1 = max(b1[1], b2[1])
                yi1 = max(b1[0], b2[0])
                xi2 = min(b1[3], b2[3])
                yi2 = min(b1[2], b2[2])
                inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
                area1 = max(0.0, b1[3] - b1[1]) * max(0.0, b1[2] - b1[0])
                area2 = max(0.0, b2[3] - b2[1]) * max(0.0, b2[2] - b2[0])
                iou = inter / max(area1 + area2 - inter, 1e-8)
                if iou >= iou_thresh:
                    mask_keep[j] = False
            order = order[1:][mask_keep]

        return keep
