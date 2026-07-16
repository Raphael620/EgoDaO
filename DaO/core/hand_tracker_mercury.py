"""Hand tracking — Monado Mercury ONNX backend (single camera, 3D depth).

Two-stage pipeline:
  Stage 1: grayscale_detection_160x160.onnx → 2 hand ROIs
  Stage 2: grayscale_keypoint_jan18.onnx → 21×22×22 heatmaps + 21×22 depth

3D depth is recovered from the detection bounding-box size via the pin-hole
camera model: Z ≈ focal * physical_hand_size / size_px

Temporal coherence: the previous frame's keypoint output (in model-space
[0,128] coordinates) is fed back as `lastKeypoints`, matching Monado's
pose-predicted-input path (hg_model.cpp line 737-750).
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
import onnxruntime as ort

_MONADO_MODELS = "D:/WorkSpace/ei/monado-v25.1.0/utilities/hand-tracking-models"
_DETECTOR_PATH = os.path.join(_MONADO_MODELS, "grayscale_detection_160x160.onnx")
_KEYPOINT_PATH = os.path.join(_MONADO_MODELS, "grayscale_keypoint_jan18.onnx")

_DET_INPUT = 160
_KP_INPUT = 128
_HM = 22
_NJ = 21
_MIN_DET_CONF = 0.4
_HAND_EXIST = 0.97

# Physical hand size: wrist-to-middle-mcp ≈ 8.5 cm (Monado STANDARD_HAND_SIZE)
_HAND_SIZE_M = 0.085


class MercuryHandTracker:
    """Monado Mercury hand tracker with 3D depth estimation."""

    def __init__(self, K=None, **kw):
        if not os.path.isfile(_DETECTOR_PATH):
            raise FileNotFoundError(_DETECTOR_PATH)
        if not os.path.isfile(_KEYPOINT_PATH):
            raise FileNotFoundError(_KEYPOINT_PATH)

        sopts = ort.SessionOptions()
        sopts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sopts.intra_op_num_threads = 1
        sopts.inter_op_num_threads = 1

        self._det = ort.InferenceSession(_DETECTOR_PATH, sess_options=sopts,
                                         providers=['CPUExecutionProvider'])
        self._kp = ort.InferenceSession(_KEYPOINT_PATH, sess_options=sopts,
                                        providers=['CPUExecutionProvider'])
        self._K = np.asarray(K, dtype=np.float64) if K is not None else None

        # Temporal state per hand (0=left, 1=right)
        self._prev_kpts: list[np.ndarray | None] = [None, None]  # (21,2) in model-space
        self._prev_roi: list[tuple | None] = [None, None]
        self._hand_active: list[bool] = [False, False]

    # ── public API ─────────────────────────────────────────────────

    def process(self, bgr, apply_filter=True, compute_metric_3d=False):
        if bgr is None or bgr.size == 0:
            return [], []

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        h, w = gray.shape

        dets = self._detect_hands(gray, h, w)
        pixel_hands, he_hands = [], []

        for hand_idx, cx, cy, size_px, conf in dets:
            kps, depths, depth_m = self._estimate_keypoints(
                gray, cx, cy, size_px, h, w, hand_idx)
            if kps is None:
                continue

            # Build (21, 3) in camera frame: x, y (pixels), z (meters)
            kps_cam = np.zeros((21, 3), dtype=np.float32)
            kps_cam[:, :2] = kps

            # Depth: model depth heatmap gives relative depth (unitless).
            # We scale by the absolute depth from detection size.
            # Positive Z = forward from camera.
            z_abs = max(depth_m, 0.05)  # clamp min depth
            kps_cam[:, 2] = z_abs + depths * z_abs * 0.5  # relative + absolute

            label = "Left" if hand_idx == 0 else "Right"
            pixel_hands.append((label, kps_cam))

            # HumanEgo output (metric 3D)
            if compute_metric_3d and self._K is not None:
                he_hands.append((label, self._build_he_dict(kps_cam, hand_idx)))

        return pixel_hands, he_hands

    def close(self):
        pass

    # ── detection ───────────────────────────────────────────────────

    def _detect_hands(self, gray, h_img, w_img):
        sc = min(_DET_INPUT / w_img, _DET_INPUT / h_img)
        nw, nh = int(w_img * sc), int(h_img * sc)
        pad = np.zeros((_DET_INPUT, _DET_INPUT), dtype=np.uint8)
        res = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dx = (_DET_INPUT - nw) // 2
        dy = (_DET_INPUT - nh) // 2
        pad[dy:dy + nh, dx:dx + nw] = res

        inp = _monado_norm(pad)[np.newaxis, np.newaxis, :, :].astype(np.float32)
        he = self._det.run(None, {"inputImg": inp})
        hand_exists, cx_r, cy_r, sz_r = he[0][0], he[1][0], he[2][0], he[3][0]

        dets = []
        for hi in range(2):
            cf = float(hand_exists[hi])
            if cf < _MIN_DET_CONF:
                continue
            cdx = _map(float(cx_r[hi]), -1, 1, 0, _DET_INPUT)
            cdy = _map(float(cy_r[hi]), -1, 1, 0, _DET_INPUT)
            sz = float(sz_r[hi]) * _DET_INPUT * 2.0
            ocx = float((cdx - dx) / sc)
            ocy = float((cdy - dy) / sc)
            osz = float(sz / sc)
            if osz < 20 or osz > 800:
                continue
            dets.append((hi, ocx, ocy, osz, cf))

        # Keep left-hand (idx 0) on the left side of image
        if len(dets) == 2 and dets[0][1] > dets[1][1]:
            dets.reverse()

        return dets

    # ── keypoints ───────────────────────────────────────────────────

    def _estimate_keypoints(self, gray, cx, cy, size_px, h_img, w_img, hand_idx):
        """Return (keypoints_21x2_pixel, depths_21_relative, depth_absolute_m)."""
        expand = 1.7
        half = max(size_px * expand * 0.5, 10.0)

        # Affine warp: detection ROI → 128×128
        src = np.array([
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx - half, cy + half],
        ], dtype=np.float32)
        dst = np.array([
            [0, 0],
            [_KP_INPUT - 1.0, 0],
            [0, _KP_INPUT - 1.0],
        ], dtype=np.float32)

        aff = cv2.getAffineTransform(src, dst)
        warped = cv2.warpAffine(gray, aff, (_KP_INPUT, _KP_INPUT),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Prepare temporal input
        use_last = 0.0
        last_kp = np.zeros((1, 42), dtype=np.float32)
        if (self._hand_active[hand_idx] and self._prev_kpts[hand_idx] is not None
                and self._prev_roi[hand_idx] is not None):
            px, py, psz = self._prev_roi[hand_idx]
            # Only use lastKeypoints if ROI is close (hand hasn't moved far)
            roi_dist = math.hypot(cx - px, cy - py)
            if roi_dist < psz * 1.5:
                use_last = 1.0
                # Map previous keypoints through the NEW affine to get model-space coords
                pk = self._prev_kpts[hand_idx]  # (21,2) in original pixel space
                ones = np.ones((21, 1), dtype=np.float32)
                pt = np.hstack([pk, ones]) @ aff.T  # (21, 2) in 128x128 space
                pt[:, 0] = np.clip(pt[:, 0], 0, 127.0)
                pt[:, 1] = np.clip(pt[:, 1], 0, 127.0)
                last_kp[0, ::2] = pt[:, 0]
                last_kp[0, 1::2] = pt[:, 1]

        inp = _monado_norm(warped)[np.newaxis, np.newaxis, :, :].astype(np.float32)
        out = self._kp.run(None, {
            "inputImg": inp,
            "lastKeypoints": last_kp,
            "useLastKeypoints": np.array([use_last], dtype=np.float32),
        })

        hm_xy = out[0][0]    # (21, 22, 22)
        hm_dp = out[1][0]    # (21, 22)
        extras = out[2][0]   # (8,)

        he = 1.0 / (1.0 + math.exp(-float(extras[0])))
        if he < _HAND_EXIST:
            self._hand_active[hand_idx] = False
            return None

        self._hand_active[hand_idx] = True
        self._prev_roi[hand_idx] = (cx, cy, size_px)

        # Inverse affine to map from model-space back to pixel space
        inv_aff = cv2.invertAffineTransform(aff)
        kps = np.zeros((_NJ, 2), dtype=np.float32)
        dps = np.zeros(_NJ, dtype=np.float32)

        for j in range(_NJ):
            hm = hm_xy[j]
            ai = int(np.argmax(hm))
            ry, rx = ai // _HM, ai % _HM
            # Sub-pixel refinement
            k = 10
            kx = min(k, rx, _HM - 1 - rx)
            ky = min(k, ry, _HM - 1 - ry)
            x0, y0 = max(0, rx - kx), max(0, ry - ky)
            x1, y1 = min(_HM, rx + kx + 1), min(_HM, ry + ky + 1)
            patch = hm[y0:y1, x0:x1]
            s = patch.sum()
            if s > 1e-6:
                gy, gx = np.mgrid[y0:y1, x0:x1]
                rrx = (patch * gx).sum() / s
                rry = (patch * gy).sum() / s
            else:
                rrx, rry = float(rx) + 0.5, float(ry) + 0.5

            # Heatmap [0,22] → model-space pixel [0,128]
            kx_n = rrx / _HM * _KP_INPUT
            ky_n = rry / _HM * _KP_INPUT

            # Inverse affine → original image
            o = inv_aff @ np.array([kx_n, ky_n, 1.0])
            kps[j, 0] = float(np.clip(o[0], 0, w_img - 1))
            kps[j, 1] = float(np.clip(o[1], 0, h_img - 1))

            # Depth (model output, unitless relative scale)
            dd = np.clip(hm_dp[j], 0, None)
            ds = dd.sum()
            if ds > 1e-6:
                avg = (dd * np.arange(_HM)).sum() / ds
                dps[j] = float(((avg + 0.5) / _HM - 0.5) * 2.0 * 1.5)
            else:
                dps[j] = 0.0

        self._prev_kpts[hand_idx] = kps.copy()

        # Absolute depth from detection size
        if self._K is not None:
            focal = (self._K[0, 0] + self._K[1, 1]) / 2.0
        else:
            # Approximate for OAK 1280px ≈ 96° HFOV
            focal = w_img * 0.445
        depth_m = focal * _HAND_SIZE_M / max(size_px, 1.0)

        return kps, dps, float(depth_m)

    # ── HumanEgo builder ────────────────────────────────────────────

    def _build_he_dict(self, kps_cam, hand_idx):
        """Build HumanEgo-compatible dict from camera-frame keypoints."""
        wrist = kps_cam[0]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = wrist
        return {
            "d2c": None, "c2w": None, "confidence": 0.8, "grasp_state": 0,
            "wrist_pose": pose.tolist(), "palm_pose": pose.tolist(),
            "kpts_3d": kps_cam.tolist(), "kpts_2d": kps_cam[:, :2].tolist(),
            "joint_angles": {},
            "wrist_pose_raw_world": pose.tolist(),
            "wrist_pose_opt_world": pose.tolist(),
            "wrist_lin_vel_raw_world": [0, 0, 0],
            "wrist_ang_vel_raw_world": [0, 0, 0],
            "wrist_lin_vel_opt_world": [0, 0, 0],
            "wrist_ang_vel_opt_world": [0, 0, 0],
            "index_translation_raw_world": kps_cam[8].tolist()[:3],
            "index_translation_opt_world": kps_cam[8].tolist()[:3],
            "thumb_translation_raw_world": kps_cam[4].tolist()[:3],
            "thumb_translation_opt_world": kps_cam[4].tolist()[:3],
            "midpoint_pose_raw_world": pose.tolist(),
            "midpoint_pose_opt_world": pose.tolist(),
            "midpoint_translation_raw_world": wrist.tolist()[:3],
            "midpoint_orientation_raw_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "midpoint_translation_opt_world": wrist.tolist()[:3],
            "midpoint_orientation_opt_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "midpoint_lin_vel_raw_world": [0, 0, 0],
            "midpoint_ang_vel_raw_world": [0, 0, 0],
            "midpoint_lin_vel_opt_world": [0, 0, 0],
            "midpoint_ang_vel_opt_world": [0, 0, 0],
            "distance_midpoint2wrist_raw_world": 0.0,
            "distance_midpoint2wrist_opt_world": 0.0,
        }


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _monado_norm(img):
    f32 = img.astype(np.float32) / 255.0
    m, s = f32.mean(), f32.std()
    if s < 1e-6:
        return np.full_like(f32, 0.5, dtype=np.float32)
    return ((f32 - m) / s * 0.25 + 0.5).astype(np.float32)


def _map(x, in_lo, in_hi, out_lo, out_hi):
    return (x - in_lo) / (in_hi - in_lo) * (out_hi - out_lo) + out_lo


def create_mercury_tracker(**kw):
    try:
        return MercuryHandTracker(**kw)
    except Exception as e:
        import sys; sys.stderr.write(f"Mercury init fail: {e}\n"); return None
