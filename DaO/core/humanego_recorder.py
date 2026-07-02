"""HumanEgo-compatible data recorder.

Writes per-frame data in the format expected by the HumanEgo preprocessing
pipeline (Preprocess.py / DatasetGen.py).  The directory layout mirrors the
output of ``aria_mps single`` + AriaCam/AriaHands/AriaSlam generators:

    Data/HumanEgo/mps_{session}_vrs/preprocess/
        all_data/
            00000/
                rgb.png
                aria_cam_rgb.json
                aria_slam.json
                aria_hands.json
                training_data.json
            00001/
                ...
        vis/                         -- optionally populated by downstream

Recording is lock-free:
  - write_*()  methods only append to in-memory lists  (fast, called from UI thread)
  - stop()     hands a copy of the data to a background QThread that does
               all disk I/O and JSON serialisation
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from DaO.config import AppConfig


# ── background writer thread (pure Python threading, not QThread) ──

def _flush_worker(session_dir: Path, payload: dict[str, Any],
                  camera_config, K_default: np.ndarray, fov_default: float):
    """Runs in a plain ``threading.Thread`` — zero Qt involvement."""
    total = payload["frame_idx"]
    if total == 0:
        return

    slam_frames = _compute_slam_frames(payload["vio_frames"], payload["timestamps_us"])
    K = K_default
    fov = fov_default
    cfg_cam = camera_config
    w, h = cfg_cam.resolution

    for i in range(total):
        frame_dir = session_dir / f"{i:05d}"
        os.makedirs(frame_dir, exist_ok=True)

        if i < len(payload["rgb_frames"]) and payload["rgb_frames"][i] is not None:
            cv2.imwrite(str(frame_dir / "rgb.png"), payload["rgb_frames"][i])

        _write_cam_json(frame_dir, i, K, fov, payload["vio_frames"],
                        payload["timestamps_us"], w, h, cfg_cam.fps)
        _write_slam_json(frame_dir, i, slam_frames)
        _write_hands_json(frame_dir, i, payload["hand_data_left"],
                          payload["hand_data_right"], payload["timestamps_us"])
        _write_training_data_json(frame_dir, i, K, fov, payload["vio_frames"],
                                  payload["timestamps_us"], w, h, cfg_cam.fps)


# ── public API ────────────────────────────────────────────────────

class HumanEgoRecorder:
    """HumanEgo data collector backed by plain ``threading.Thread``.

    ``write_*`` methods only append to lists — O(1), no I/O.
    ``stop()`` hands data to a daemon thread and returns immediately.
    """

    _DEFAULT_FOV_DEG = 72.0

    def __init__(self, config: AppConfig | None = None):
        self._cfg = config or AppConfig()
        self._session_dir: Path | None = None
        self._frame_idx = 0
        self._active = False
        self._vio_frames: list[np.ndarray] = []
        self._timestamps_us: list[int] = []
        self._hand_data_left: list[list] = []
        self._hand_data_right: list[list] = []
        self._rgb_frames: list[np.ndarray] = []
        self._K: np.ndarray | None = None
        self._latest_vio: np.ndarray | None = None
        self._latest_hands: list | None = None

    @property
    def is_recording(self) -> bool:
        return self._active

    def start(self) -> Path:
        if self._active:
            return self._session_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = (
            self._cfg.recording.data_root
            / self._cfg.recording.humanego_subdir
            / f"mps_{ts}_vrs"
            / "preprocess"
            / "all_data"
        )
        os.makedirs(self._session_dir, exist_ok=True)
        self._frame_idx = 0
        self._active = True
        self._vio_frames = []
        self._timestamps_us = []
        self._hand_data_left = []
        self._hand_data_right = []
        self._rgb_frames = []
        self._latest_vio = None
        self._latest_hands = None
        return self._session_dir

    def stop(self):
        if not self._active:
            return
        self._active = False
        total = self._frame_idx
        if total == 0:
            return

        K = self._K if self._K is not None else self._default_intrinsics()
        payload = {
            "frame_idx": total,
            "vio_frames": self._vio_frames,
            "timestamps_us": self._timestamps_us,
            "hand_data_left": self._hand_data_left,
            "hand_data_right": self._hand_data_right,
            "rgb_frames": self._rgb_frames,
        }
        self._vio_frames = []
        self._timestamps_us = []
        self._hand_data_left = []
        self._hand_data_right = []
        self._rgb_frames = []
        self._latest_vio = None
        self._latest_hands = None

        t = threading.Thread(
            target=_flush_worker,
            args=(self._session_dir, payload, self._cfg.camera,
                  K, self._DEFAULT_FOV_DEG),
            daemon=True,
        )
        t.start()

    def set_camera_intrinsics(self, K: np.ndarray):
        self._K = np.asarray(K, dtype=np.float64)

    # ── data ingestion (called from UI signal handlers) ────────────

    def write_frame_rgb(self, bgr: np.ndarray, timestamp_us: int = 0):
        if not self._active or bgr is None or bgr.size == 0:
            return
        self._rgb_frames.append(bgr.copy())
        self._timestamps_us.append(timestamp_us)
        # Align VIO and hands data with this RGB frame
        if self._latest_vio is not None:
            self._vio_frames.append(self._latest_vio.copy())
        else:
            self._vio_frames.append(np.eye(4, dtype=np.float64))
        if self._latest_hands is not None:
            hands_l = [h for h in self._latest_hands if h[0].lower().startswith("l")]
            hands_r = [h for h in self._latest_hands if h[0].lower().startswith("r")]
            self._hand_data_left.append(hands_l if hands_l else [])
            self._hand_data_right.append(hands_r if hands_r else [])
        else:
            self._hand_data_left.append([])
            self._hand_data_right.append([])
        self._frame_idx += 1

    def write_vio(self, transform: np.ndarray):
        if self._active:
            self._latest_vio = np.asarray(transform, dtype=np.float64)

    def write_hands(self, role: str, hands: list):
        if not self._active:
            return
        self._latest_hands = hands

    # ── helpers ────────────────────────────────────────────────────

    def _default_intrinsics(self) -> np.ndarray:
        w, h = self._cfg.camera.resolution
        fx = w / (2.0 * np.tan(np.radians(self._DEFAULT_FOV_DEG / 2.0)))
        return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)


# ── per-frame JSON writers (module-level, called from worker thread) ──

def _write_cam_json(frame_dir, idx, K, fov, vio_frames, timestamps_us,
                    w, h, fps):
    c2w = vio_frames[idx].tolist() if idx < len(vio_frames) else np.eye(4).tolist()
    ts_ns = int(timestamps_us[idx] * 1000) if idx < len(timestamps_us) else 0
    data = {
        "idx": idx, "ts": ts_ns, "fov": fov, "h": h, "w": w,
        "k": K.tolist(), "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "c2w": c2w, "c2d": np.eye(4).tolist(), "d2w": np.eye(4).tolist(),
        "rgb_path": f"preprocess/all_data/{idx:05d}/rgb.png",
        "fps": fps,
    }
    with open(frame_dir / "aria_cam_rgb.json", "w") as f:
        json.dump(data, f, indent=2)


def _write_slam_json(frame_dir, idx, slam_frames):
    if idx < len(slam_frames):
        data = slam_frames[idx]
    else:
        data = {"idx": idx, "ts": 0, "t_world": [0,0,0], "rpy_deg": [0,0,0],
                "delta_t_world": [0,0,0], "delta_rpy_deg": [0,0,0],
                "linear_speed_mps": 0.0, "angular_speed_rps": 0.0,
                "yaw_unwrapped_deg": 0.0}
    with open(frame_dir / "aria_slam.json", "w") as f:
        json.dump(data, f, indent=2)


def _write_hands_json(frame_dir, idx, hand_data_left, hand_data_right, timestamps_us):
    def _pack(entries):
        if not entries:
            return None
        for label, lms in entries:
            lms_np = np.asarray(lms, dtype=np.float64)
            if lms_np.shape[0] < 21:
                continue
            wrist = lms_np[0].copy()
            pose = np.eye(4); pose[:3, 3] = wrist
            return {
                "d2c": None, "c2w": None, "confidence": 0.8, "grasp_state": 0,
                "wrist_pose": pose.tolist(), "palm_pose": pose.tolist(),
                "kpts_3d": lms_np.tolist(), "kpts_2d": lms_np[:, :2].tolist(),
                "joint_angles": {},
                "wrist_pose_raw_world": pose.tolist(),
                "wrist_pose_opt_world": pose.tolist(),
                "wrist_lin_vel_raw_world": [0, 0, 0],
                "wrist_ang_vel_raw_world": [0, 0, 0],
                "wrist_lin_vel_opt_world": [0, 0, 0],
                "wrist_ang_vel_opt_world": [0, 0, 0],
                "index_translation_raw_world": lms_np[8].tolist()[:3],
                "index_translation_opt_world": lms_np[8].tolist()[:3],
                "thumb_translation_raw_world": lms_np[4].tolist()[:3],
                "thumb_translation_opt_world": lms_np[4].tolist()[:3],
                "midpoint_pose_raw_world": pose.tolist(),
                "midpoint_pose_opt_world": pose.tolist(),
                "midpoint_translation_raw_world": wrist.tolist()[:3],
                "midpoint_orientation_raw_world": [[1,0,0],[0,1,0],[0,0,1]],
                "midpoint_translation_opt_world": wrist.tolist()[:3],
                "midpoint_orientation_opt_world": [[1,0,0],[0,1,0],[0,0,1]],
                "midpoint_lin_vel_raw_world": [0,0,0],
                "midpoint_ang_vel_raw_world": [0,0,0],
                "midpoint_lin_vel_opt_world": [0,0,0],
                "midpoint_ang_vel_opt_world": [0,0,0],
                "distance_midpoint2wrist_raw_world": 0.0,
                "distance_midpoint2wrist_opt_world": 0.0,
            }
        return None

    n_ts = len(timestamps_us)
    data = {
        "idx": idx,
        "ts": int(timestamps_us[idx] * 1000) if idx < n_ts and timestamps_us[idx] > 0 else 0,
        "hand_l": _pack(hand_data_left[idx] if idx < len(hand_data_left) else []),
        "hand_r": _pack(hand_data_right[idx] if idx < len(hand_data_right) else []),
    }
    with open(frame_dir / "aria_hands.json", "w") as f:
        json.dump(data, f, indent=2)


def _write_training_data_json(frame_dir, idx, K, fov, vio_frames, timestamps_us,
                              w, h, fps):
    c2w = vio_frames[idx].tolist() if idx < len(vio_frames) else np.eye(4).tolist()
    ts_ns = int(timestamps_us[idx] * 1000) if idx < len(timestamps_us) else 0
    data = {
        "metadata": {
            "idx": idx, "ts": ts_ns, "w": w, "h": h, "fps": fps,
            "k": K.tolist(), "c2w": c2w, "anchor_key": "obj1",
            "is_finished": 0.0,
            "world_transforms": {
                "cam0": np.eye(4).tolist(),
                "virtual_static_anchor": np.eye(4).tolist(),
            },
        },
        "obs": {
            "rgb_path": f"preprocess/all_data/{idx:05d}/rgb.png",
            "mask_arm_path": "", "mask_obj_path": "",
            "rgb_WArmObjKpts_path": "", "rgb_WoArm_path": "",
            "rgb_WoArm_WArmObjKpts_path": "",
        },
        "entities": {"hands": {}, "objects": {}},
    }
    with open(frame_dir / "training_data.json", "w") as f:
        json.dump(data, f, indent=2)


# ── SLAM helpers ──────────────────────────────────────────────────

def _compute_slam_frames(transforms, timestamps_us):
    n = len(transforms)
    if n == 0:
        return []
    t_world = np.array([m[:3, 3] for m in transforms], dtype=np.float64)
    rpy_deg = np.array([_rotmat_to_rpy_zyx_deg(m[:3, :3]) for m in transforms], dtype=np.float64)
    delta_t = t_world - t_world[0]
    delta_rpy = rpy_deg - rpy_deg[0]
    yaws = np.unwrap(np.radians(rpy_deg[:, 2]))
    n_ts = len(timestamps_us)
    frames = []
    for i in range(n):
        ts_ns = int(timestamps_us[i] * 1000) if i < n_ts else 0
        if i > 0 and i < n_ts and (i - 1) < n_ts:
            dt_us = max(timestamps_us[i] - timestamps_us[i - 1], 1)
            dt = dt_us / 1e6
        else:
            dt = 1.0 / 30.0  # fallback: assume 30 FPS
        v = float(np.linalg.norm(t_world[i] - t_world[i - 1]) / max(dt, 1e-6)) if i > 0 else 0.0
        w = float(abs(yaws[i] - yaws[i - 1]) / max(dt, 1e-6)) if i > 0 else 0.0
        frames.append({
            "idx": i, "ts": ts_ns,
            "t_world": t_world[i].tolist(),
            "rpy_deg": rpy_deg[i].tolist(),
            "delta_t_world": delta_t[i].tolist(),
            "delta_rpy_deg": delta_rpy[i].tolist(),
            "linear_speed_mps": v, "angular_speed_rps": w,
            "yaw_unwrapped_deg": float(np.degrees(yaws[i])),
        })
    return frames


def _rotmat_to_rpy_zyx_deg(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])
