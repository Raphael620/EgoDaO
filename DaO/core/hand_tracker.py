"""Hand tracking — MediaPipe 0.10+ Tasks API (host-side).

Provides online temporal smoothing (outlier rejection + median filter + EMA),
absolute 3D metric recovery in camera frame, and Aria MPS keypoint ordering.
"""
from __future__ import annotations

import os
import threading
import urllib.request
import numpy as np
import cv2

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Average adult hand: wrist to middle MCP ≈ 0.085m
_HAND_SIZE_WRIST_TO_MIDDLE_MCP_M = 0.085

_MEDIAPIPE_TRACKER_CACHE = None
_MEDIAPIPE_TRACKER_LOCK = threading.Lock()

# ── MediaPipe 21-point → Aria MPS 21-point index mapping ──────────
# MediaPipe: 0=Wrist, 1=ThumbCMC, 2=ThumbMCP, 3=ThumbIP, 4=ThumbTip,
#            5=IndexMCP, 6=IndexPIP, 7=IndexDIP, 8=IndexTip,
#            9=MiddleMCP, 10=MiddlePIP, 11=MiddleDIP, 12=MiddleTip,
#            13=RingMCP, 14=RingPIP, 15=RingDIP, 16=RingTip,
#            17=PinkyMCP, 18=PinkyPIP, 19=PinkyDIP, 20=PinkyTip
# Aria:  0=ThumbTip, 1=IndexTip, 2=MiddleTip, 3=RingTip, 4=PinkyTip,
#        5=Wrist, 6=ThumbMCP, 7=ThumbIP, 8=IndexMCP, 9=IndexPIP, 10=IndexDIP,
#        11=MiddleMCP, 12=MiddlePIP, 13=MiddleDIP,
#        14=RingMCP, 15=RingPIP, 16=RingDIP,
#        17=PinkyMCP, 18=PinkyPIP, 19=PinkyDIP,
#        20=PalmCenter (computed)
_MP_TO_ARIA = [
    4,   # Aria 0  = ThumbTip     ← MP 4
    8,   # Aria 1  = IndexTip     ← MP 8
    12,  # Aria 2  = MiddleTip    ← MP 12
    16,  # Aria 3  = RingTip      ← MP 16
    20,  # Aria 4  = PinkyTip     ← MP 20
    0,   # Aria 5  = Wrist        ← MP 0
    2,   # Aria 6  = ThumbMCP     ← MP 2
    3,   # Aria 7  = ThumbIP      ← MP 3
    5,   # Aria 8  = IndexMCP     ← MP 5
    6,   # Aria 9  = IndexPIP     ← MP 6
    7,   # Aria 10 = IndexDIP     ← MP 7
    9,   # Aria 11 = MiddleMCP    ← MP 9
    10,  # Aria 12 = MiddlePIP    ← MP 10
    11,  # Aria 13 = MiddleDIP    ← MP 11
    13,  # Aria 14 = RingMCP      ← MP 13
    14,  # Aria 15 = RingPIP      ← MP 14
    15,  # Aria 16 = RingDIP      ← MP 15
    17,  # Aria 17 = PinkyMCP     ← MP 17
    18,  # Aria 18 = PinkyPIP     ← MP 18
    19,  # Aria 19 = PinkyDIP     ← MP 19
    -1,  # Aria 20 = PalmCenter   ← computed (mean of Wrist, IndexMCP, MiddleMCP)
]


def _remap_mp_to_aria(kpts_mp_21: np.ndarray) -> np.ndarray:
    """Remap (21, 3) MediaPipe keypoints to Aria MPS ordering."""
    kpts_aria = np.zeros((21, 3), dtype=kpts_mp_21.dtype)
    for aria_idx in range(20):
        mp_idx = _MP_TO_ARIA[aria_idx]
        kpts_aria[aria_idx] = kpts_mp_21[mp_idx]
    # Aria 20 = PalmCenter ≈ mean(Wrist=MP0, IndexMCP=MP5, MiddleMCP=MP9)
    kpts_aria[20] = (kpts_mp_21[0] + kpts_mp_21[5] + kpts_mp_21[9]) / 3.0
    return kpts_aria


def _remap_2d_mp_to_aria(kpts_mp_21: np.ndarray) -> np.ndarray:
    """Remap (21, 2) MediaPipe 2D keypoints to Aria MPS ordering."""
    kpts3d = np.column_stack([kpts_mp_21, np.zeros(21, dtype=kpts_mp_21.dtype)])
    return _remap_mp_to_aria(kpts3d)[:, :2]


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


# ── Minimal temporal filter ──────────────────────────────────────

class HandLandmarkFilter:
    """Temporal filter for hand landmarks with label stability.

    - Outlier rejection: wrist jumps > ``wrist_speed_limit_px`` discarded.
    - Adaptive smoothing: more at rest, less during fast movement.
    - Handedness stabilisation: when MediaPipe flips left/right labels,
      corrects based on wrist proximity to the previous track.
    """

    def __init__(self, wrist_speed_limit_px=200, smoothing=0.5,
                 miss_reset_frames=5):
        self._wrist_speed_limit = float(wrist_speed_limit_px)
        self._smoothing = float(smoothing)
        self._miss_reset = miss_reset_frames

        self._prev: dict[str, np.ndarray] = {}      # true_label -> (21,3)
        self._misses: dict[str, int] = {}
        self._label_map: dict[str, str] = {}         # raw label -> true_label

    def apply(self, raw_hands: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
        # ── resolve handedness ──
        mapped: list[tuple[str, np.ndarray]] = []
        if len(raw_hands) == 2 and len(self._prev) == 2:
            # Two hands tracked — check for label swap
            l0, k0 = raw_hands[0]
            l1, k1 = raw_hands[1]
            prev_keys = list(self._prev.keys())
            d00 = np.linalg.norm(k0[0] - self._prev[prev_keys[0]][0])
            d11 = np.linalg.norm(k1[0] - self._prev[prev_keys[1]][0])
            d01 = np.linalg.norm(k0[0] - self._prev[prev_keys[1]][0])
            d10 = np.linalg.norm(k1[0] - self._prev[prev_keys[0]][0])
            if d01 + d10 < d00 + d11:
                # Swap back
                self._label_map[l0] = prev_keys[1]
                self._label_map[l1] = prev_keys[0]
                mapped.append((prev_keys[1], k0))
                mapped.append((prev_keys[0], k1))
            else:
                self._label_map[l0] = prev_keys[0]
                self._label_map[l1] = prev_keys[1]
                mapped.append((prev_keys[0], k0))
                mapped.append((prev_keys[1], k1))
        else:
            for label, lms in raw_hands:
                true = self._label_map.get(label, label)
                self._label_map[label] = true
                mapped.append((true, lms))

        # ── filter ──
        filtered: list[tuple[str, np.ndarray]] = []
        active_labels: set[str] = set()

        for true_label, lms in mapped:
            active_labels.add(true_label)
            lms64 = lms.astype(np.float64)

            prev = self._prev.get(true_label)
            if prev is not None:
                jump = float(np.linalg.norm(lms64[0] - prev[0]))
                if jump > self._wrist_speed_limit:
                    self._misses[true_label] = self._misses.get(true_label, 0) + 1
                    if self._misses.get(true_label, 0) >= self._miss_reset:
                        self._reset_label(true_label)
                    filtered.append((true_label, prev.astype(np.float32)))
                    continue

                speed = jump
                if speed < 5.0:
                    alpha = self._smoothing
                elif speed > 60.0:
                    alpha = 1.0
                else:
                    t = (speed - 5.0) / 55.0
                    alpha = self._smoothing + (1.0 - self._smoothing) * t

                smoothed = (1.0 - alpha) * prev + alpha * lms64
            else:
                smoothed = lms64

            self._misses[true_label] = 0
            self._prev[true_label] = smoothed
            filtered.append((true_label, smoothed.astype(np.float32)))

        for label in list(self._prev.keys()):
            if label not in active_labels:
                self._misses[label] = self._misses.get(label, 0) + 1
                if self._misses.get(label, 0) >= self._miss_reset:
                    self._reset_label(label)

        return filtered

    def _reset_label(self, label: str) -> None:
        self._prev.pop(label, None)
        self._misses.pop(label, None)
        # Remove any mapping that points to this label
        for k, v in list(self._label_map.items()):
            if v == label:
                del self._label_map[k]

    def reset(self) -> None:
        self._prev.clear()
        self._misses.clear()
        self._label_map.clear()


# ── MediaPipe hand tracker ───────────────────────────────────────

class MediaPipeHandTracker:
    """MediaPipe HandLandmarker with optional temporal smoothing and
    3D metric recovery in camera frame.

    Uses ``RunningMode.VIDEO`` so ``hand_world_landmarks`` are available
    for absolute depth estimation.
    """

    def __init__(self, max_num_hands=2, min_detection_confidence=0.3,
                 K=None, enable_filter=True):
        _ensure_model()
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as mp_base
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._K: np.ndarray | None = np.asarray(K, dtype=np.float64) if K is not None else None
        self._frame_timestamp_ms = 0
        self._filter = HandLandmarkFilter() if enable_filter else None

    def set_intrinsics(self, K: np.ndarray) -> None:
        self._K = np.asarray(K, dtype=np.float64)

    @property
    def has_intrinsics(self) -> bool:
        return self._K is not None

    def process(self, bgr: np.ndarray, apply_filter: bool = True,
                compute_metric_3d: bool = False):
        """Run hand detection on a BGR frame.

        Args:
            bgr: BGR or grayscale image.
            apply_filter: Apply temporal smoothing (outlier rejection +
                median + EMA).  Only valid when the filter is enabled.
            compute_metric_3d: Recover absolute 3D in camera frame (meters)
                and remap to Aria MPS ordering. Requires ``set_intrinsics()``.

        Returns:
            ``(pixel_hands, humanego_hands)`` tuple.

            *pixel_hands*: ``[(label, (21,3) pixel-coord array), ...]`` for UI.
            *humanego_hands*: ``[(label, dict), ...]`` with metric 3D data
            (empty when ``compute_metric_3d`` is False or K is not set).
        """
        h, w = bgr.shape[:2]
        if len(bgr.shape) == 2 or bgr.shape[2] == 1:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        self._frame_timestamp_ms += 33  # ~30 FPS
        result = self._detector.detect_for_video(mp_img, self._frame_timestamp_ms)

        raw_hands: list[tuple[str, np.ndarray, np.ndarray | None]] = []
        if result.hand_landmarks and result.handedness:
            world_landmarks = result.hand_world_landmarks or [None] * len(result.hand_landmarks)
            for lms, world_lms, hns in zip(result.hand_landmarks, world_landmarks,
                                            result.handedness):
                label = hns[0].category_name
                arr = np.zeros((21, 3), dtype=np.float32)
                for i, lm in enumerate(lms):
                    arr[i] = [lm.x * w, lm.y * h, lm.z * w]
                wl_arr = None
                if world_lms is not None:
                    wl_arr = np.array([[lm.x, lm.y, lm.z] for lm in world_lms],
                                      dtype=np.float32)
                raw_hands.append((label, arr, wl_arr))

        # ── Apply temporal filter (pixel space) ──
        pixel_hands: list[tuple[str, np.ndarray]]
        if self._filter is not None and apply_filter:
            # Feed raw (label, array) pairs into the filter
            raw_for_filter = [(l, a) for l, a, _ in raw_hands]
            pixel_hands = self._filter.apply(raw_for_filter)
        else:
            pixel_hands = [(l, a) for l, a, _ in raw_hands]

        # ── HumanEgo-compatible metric 3D output ──
        humanego_hands: list[tuple[str, dict]] = []
        if compute_metric_3d and self._K is not None:
            for label, arr, wl_arr in raw_hands:
                if wl_arr is None or wl_arr.shape != (21, 3):
                    continue
                kpts_cam_mp = self._recover_absolute_3d(arr[:, :2], wl_arr, h, w)
                if kpts_cam_mp is None:
                    continue

                # Remap to Aria ordering
                kpts_cam_aria = _remap_mp_to_aria(kpts_cam_mp)
                kpts_2d_aria = _remap_2d_mp_to_aria(arr[:, :2])

                is_right = label.lower().startswith("r")

                # Build wrist pose
                wrist_pose = _build_wrist_pose(kpts_cam_aria)
                wrist_pos_cam = kpts_cam_aria[5]  # Aria 5 = Wrist

                # Grasp state
                grasp_state = _compute_grasp_state(kpts_cam_aria)

                he_dict = {
                    "d2c": None,
                    "c2w": None,
                    "confidence": 0.8,
                    "grasp_state": grasp_state,
                    "wrist_pose": wrist_pose.tolist() if wrist_pose is not None else np.eye(4).tolist(),
                    "palm_pose": wrist_pose.tolist() if wrist_pose is not None else np.eye(4).tolist(),
                    "kpts_3d": kpts_cam_aria.tolist(),
                    "kpts_2d": kpts_2d_aria.tolist(),
                    "joint_angles": {},
                    "wrist_pose_raw_world": wrist_pose.tolist() if wrist_pose is not None else np.eye(4).tolist(),
                    "wrist_pose_opt_world": wrist_pose.tolist() if wrist_pose is not None else np.eye(4).tolist(),
                    "wrist_lin_vel_raw_world": [0, 0, 0],
                    "wrist_ang_vel_raw_world": [0, 0, 0],
                    "wrist_lin_vel_opt_world": [0, 0, 0],
                    "wrist_ang_vel_opt_world": [0, 0, 0],
                    "index_translation_raw_world": kpts_cam_aria[1].tolist()[:3],   # Aria 1=IndexTip
                    "index_translation_opt_world": kpts_cam_aria[1].tolist()[:3],
                    "thumb_translation_raw_world": kpts_cam_aria[0].tolist()[:3],   # Aria 0=ThumbTip
                    "thumb_translation_opt_world": kpts_cam_aria[0].tolist()[:3],
                    "midpoint_pose_raw_world": wrist_pose.tolist() if wrist_pose is not None else np.eye(4).tolist(),
                    "midpoint_pose_opt_world": wrist_pose.tolist() if wrist_pose is not None else np.eye(4).tolist(),
                    "midpoint_translation_raw_world": wrist_pos_cam.tolist()[:3],
                    "midpoint_orientation_raw_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "midpoint_translation_opt_world": wrist_pos_cam.tolist()[:3],
                    "midpoint_orientation_opt_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "midpoint_lin_vel_raw_world": [0, 0, 0],
                    "midpoint_ang_vel_raw_world": [0, 0, 0],
                    "midpoint_lin_vel_opt_world": [0, 0, 0],
                    "midpoint_ang_vel_opt_world": [0, 0, 0],
                    "distance_midpoint2wrist_raw_world": 0.0,
                    "distance_midpoint2wrist_opt_world": 0.0,
                    "is_right": is_right,
                }
                humanego_hands.append((label, he_dict))

        return pixel_hands, humanego_hands

    # ── 3D metric recovery ──────────────────────────────────────

    def _recover_absolute_3d(
        self,
        kpts_2d_mp: np.ndarray,       # (21, 2) pixel coords
        kpts_world_mp: np.ndarray,    # (21, 3) hand-centered meters
        h_img: int,
        w_img: int,
    ) -> np.ndarray | None:
        """Recover absolute 3D keypoints in camera frame (meters).

        Strategy:
        1. Measure wrist→middle_MCP physical distance from world_landmarks.
        2. Measure same distance in 2D pixels.
        3. Estimate wrist depth: z = focal * physical_dist / pixel_dist.
        4. Back-project wrist to camera frame.
        5. Add relative offsets from world_landmarks.

        Returns (21, 3) in metres or None if invalid.
        """
        K = self._K
        if K is None:
            return None
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        focal = (fx + fy) / 2.0

        wrist_2d = kpts_2d_mp[0]
        middle_mcp_2d = kpts_2d_mp[9]
        wrist_world = kpts_world_mp[0]
        middle_mcp_world = kpts_world_mp[9]

        physical_dist = float(np.linalg.norm(middle_mcp_world - wrist_world))
        if physical_dist < 0.01:
            physical_dist = _HAND_SIZE_WRIST_TO_MIDDLE_MCP_M

        pixel_dist = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
        if pixel_dist < 5.0:
            return None

        z_wrist = focal * physical_dist / pixel_dist
        if z_wrist < 0.05 or z_wrist > 3.0:
            return None

        x_wrist = (wrist_2d[0] - cx) * z_wrist / fx
        y_wrist = (wrist_2d[1] - cy) * z_wrist / fy
        wrist_cam = np.array([x_wrist, y_wrist, z_wrist], dtype=np.float32)

        offsets = kpts_world_mp - kpts_world_mp[0:1]
        kpts_cam = wrist_cam[np.newaxis, :] + offsets

        kpts_cam[:, 2] = np.clip(kpts_cam[:, 2], 0.01, None)
        return kpts_cam.astype(np.float32)

    def close(self):
        # MediaPipe 0.10.35 synchronously flushes Clearcut telemetry from
        # detector.close().  On an offline machine that blocks for ~42 s.
        # The factory keeps one process-wide detector for reconnects, so reset
        # only the temporal state and let process teardown release the graph.
        if self._filter is not None:
            self._filter.reset()


# ── Wrist pose builder ────────────────────────────────────────────

def _build_wrist_pose(kpts_cam_aria: np.ndarray) -> np.ndarray | None:
    """Build wrist SE(3) pose from Aria-ordered keypoints in camera frame.

    Y = wrist→palm direction, Z = palm normal, X = cross(Y, Z).
    Uses: Aria 5=Wrist, 20=PalmCenter, 8=IndexMCP, 11=MiddleMCP.
    """
    wrist_pos = kpts_cam_aria[5]
    palm_center = kpts_cam_aria[20]
    index_mcp = kpts_cam_aria[8]
    middle_mcp = kpts_cam_aria[11]

    v_wrist_palm = palm_center - wrist_pos
    norm = np.linalg.norm(v_wrist_palm)
    if norm < 1e-6:
        return None
    y_axis = v_wrist_palm / norm

    v_lateral = index_mcp - middle_mcp
    x_axis = np.cross(y_axis, v_lateral)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return None
    x_axis /= x_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= (np.linalg.norm(z_axis) + 1e-6)
    y_axis = np.cross(z_axis, x_axis)  # re-orthonormalise

    wrist_pose = np.eye(4, dtype=np.float64)
    wrist_pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    wrist_pose[:3, 3] = wrist_pos
    return wrist_pose


def _compute_grasp_state(kpts_cam_aria: np.ndarray) -> int:
    """Binary grasp: 1=closed, 0=open.

    Based on thumb-tip (Aria 0) to index-tip (Aria 1) distance,
    normalised by palm size (wrist Aria 5 → middle_MCP Aria 11).
    """
    thumb_tip = kpts_cam_aria[0]
    index_tip = kpts_cam_aria[1]
    wrist = kpts_cam_aria[5]
    mid_mcp = kpts_cam_aria[11]

    distance = float(np.linalg.norm(thumb_tip - index_tip))
    palm_size = float(np.linalg.norm(mid_mcp - wrist))
    if palm_size > 0.01:
        ratio = distance / palm_size
        return 1 if ratio < 1.0 else 0
    return 0


# ── Factory ──────────────────────────────────────────────────────

def create_hand_tracker(backend="mediapipe", **kwargs):
    if backend == "mediapipe":
        global _MEDIAPIPE_TRACKER_CACHE
        try:
            with _MEDIAPIPE_TRACKER_LOCK:
                if _MEDIAPIPE_TRACKER_CACHE is None:
                    _MEDIAPIPE_TRACKER_CACHE = MediaPipeHandTracker(**kwargs)
                else:
                    K = kwargs.get("K")
                    if K is not None:
                        _MEDIAPIPE_TRACKER_CACHE.set_intrinsics(K)
                    if _MEDIAPIPE_TRACKER_CACHE._filter is not None:
                        _MEDIAPIPE_TRACKER_CACHE._filter.reset()
                return _MEDIAPIPE_TRACKER_CACHE
        except ImportError:
            print("MediaPipe not installed — hand tracking disabled")
            return None
    if backend == "mercury":
        try:
            from DaO.core.hand_tracker_mercury import create_mercury_tracker
            return create_mercury_tracker(**kwargs)
        except Exception as e:
            print(f"Mercury hand tracker init failed: {e}")
            return None
    if backend == "openvino":
        try:
            from DaO.core.hand_tracker_openvino import create_openvino_tracker
            return create_openvino_tracker(**kwargs)
        except Exception as e:
            print(f"OpenVINO hand tracker init failed: {e}")
            return None
    raise ValueError(f"Unknown hand tracker backend: {backend}")
