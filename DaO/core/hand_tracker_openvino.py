"""Hand tracking — MediaPipe pipeline powered by OpenVINO on Intel GPU.

Uses the official MediaPipe palm_detection + hand_landmark models via
OpenVINO inference, with anchor-based decoding, NMS, rect transformation,
and affine warp matching the MediaPipe CPU pipeline.

Falls back to CPU if GPU is unavailable.

Matches the public API of hand_tracker.py:
  process(bgr) → (pixel_hands, humanego_hands)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

# ── import MediaPipe utils from the reference project ──
_REF_DIR = os.path.join(os.path.sep, "D:", os.path.sep, "WorkSpace", "ei",
                         "EgoDaO", "openvino_hand_tracker")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

from mediapipe_utils import (SSDAnchorOptions, generate_anchors, decode_bboxes,
                              non_max_suppression, detections_to_rect,
                              rect_transformation, warp_rect_img)

# ── model paths ──────────────────────────────────────────────────

_PD_XML = os.path.join(_REF_DIR, "models", "palm_detection_FP32.xml")
_LM_XML = os.path.join(_REF_DIR, "models", "hand_landmark_FP32.xml")

_LM_INPUT = 224
_PD_INPUT = 128

# ── tracker ──────────────────────────────────────────────────────

class HandTrackerOpenVINO:
    def __init__(self, **kwargs):
        if not os.path.isfile(_PD_XML):
            raise FileNotFoundError(f"PD model not found: {_PD_XML}")
        if not os.path.isfile(_LM_XML):
            raise FileNotFoundError(f"LM model not found: {_LM_XML}")

        core = ov.Core()
        device = "GPU" if "GPU" in core.available_devices else "CPU"

        self._pd_model = core.compile_model(core.read_model(_PD_XML), device)
        self._pd_infer = self._pd_model.create_infer_request()

        self._lm_model = core.compile_model(core.read_model(_LM_XML), device)
        self._lm_infer = self._lm_model.create_infer_request()

        # Generate anchors once (MediaPipe SSD)
        anchor_options = SSDAnchorOptions(
            num_layers=4, min_scale=0.1484375, max_scale=0.75,
            input_size_height=_PD_INPUT, input_size_width=_PD_INPUT,
            anchor_offset_x=0.5, anchor_offset_y=0.5,
            strides=[8, 16, 16, 16], aspect_ratios=[1.0],
            reduce_boxes_in_lowest_layer=False,
            interpolated_scale_aspect_ratio=1.0, fixed_anchor_size=True)
        self._anchors = generate_anchors(anchor_options)

        self._K = None  # camera intrinsics not used for OV pipeline

    def process(self, bgr, apply_filter=True, compute_metric_3d=False):
        if bgr is None or bgr.size == 0:
            return [], []

        h_img, w_img = bgr.shape[:2]
        frame_size = max(h_img, w_img)

        # ── pad to square ──
        pad_h = (frame_size - h_img) // 2
        pad_w = (frame_size - w_img) // 2
        square = cv2.copyMakeBorder(bgr, pad_h, pad_h, pad_w, pad_w,
                                     cv2.BORDER_CONSTANT)

        # ── palm detection (128×128) ──
        pd_input = cv2.resize(square, (_PD_INPUT, _PD_INPUT),
                               interpolation=cv2.INTER_AREA)
        pd_blob = np.transpose(pd_input, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        self._pd_infer.set_input_tensor(ov.Tensor(pd_blob))
        self._pd_infer.infer()
        scores = np.squeeze(self._pd_infer.get_tensor("classificators").data)
        bboxes = self._pd_infer.get_tensor("regressors").data[0]
        regions = decode_bboxes(0.5, scores, bboxes, self._anchors)
        regions = non_max_suppression(regions, 0.3)
        if not regions:
            return [], []

        detections_to_rect(regions)
        rect_transformation(regions, frame_size, frame_size)

        # ── landmark regression (224×224 per hand) ──
        pixel_hands = []
        for r in regions:
            warp = warp_rect_img(r.rect_points, square, _LM_INPUT, _LM_INPUT)
            lm_blob = np.transpose(warp, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

            self._lm_infer.set_input_tensor(ov.Tensor(lm_blob))
            self._lm_infer.infer()
            lm_score = float(self._lm_infer.get_tensor("Identity_1").data[0, 0])
            handedness = float(self._lm_infer.get_tensor("Identity_2").data[0, 0])
            lm_raw = np.squeeze(
                self._lm_infer.get_tensor("Identity_dense/BiasAdd/Add").data)

            if lm_score < 0.5:
                continue

            # Normalised landmark coords (divide by LM_INPUT)
            lm_norm = lm_raw.reshape(21, 3) / _LM_INPUT

            # Map from warp space → square image via inverse affine
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array(r.rect_points[1:], dtype=np.float32)
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(lm_norm[:, :2], axis=0)
            lm_px = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.float32)

            # Map from square → original image
            lm_px[:, 0] -= pad_w
            lm_px[:, 1] -= pad_h

            lms = np.zeros((21, 3), dtype=np.float32)
            lms[:, :2] = lm_px
            lms[:, 2] = lm_norm[:, 2]

            label = "Right" if handedness > 0.5 else "Left"
            pixel_hands.append((label, lms))

        return pixel_hands, []

    def close(self):
        pass


def create_openvino_tracker(**kwargs):
    try:
        return HandTrackerOpenVINO(**kwargs)
    except Exception as e:
        import sys
        sys.stderr.write(f"OpenVINO tracker init fail: {e}\n")
        return None
