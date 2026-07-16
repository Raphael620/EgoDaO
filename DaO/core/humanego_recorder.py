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
        vis/

During recording only lightweight metadata (VIO transforms, timestamps,
hand data) is accumulated in memory.  RGB frames are written *once* at
stop() time to avoid the ~3 MB/frame heap pressure of live PNG encoding.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from DaO.core.data_formats import (default_intrinsics, compute_slam_frames,
                                    rotmat_to_rpy_zyx_deg, pack_hand, build_training_data)
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from DaO.config import AppConfig

_STOP_SENTINEL = object()


# ── background I/O worker ────────────────────────────────────────

def _io_worker(q):
    while True:
        item = q.get()
        if item is _STOP_SENTINEL:
            break
        try:
            _handle_io_item(item)
        except Exception as e:
            import sys, traceback
            sys.stderr.write(f"[HE-I/O] {e}\n")
            traceback.print_exc(file=sys.stderr)


def _handle_io_item(item: dict):
    action = item["action"]
    if action == "write_png":
        os.makedirs(os.path.dirname(item["path"]), exist_ok=True)
        cv2.imwrite(item["path"], item["frame"])
    elif action == "write_json":
        os.makedirs(os.path.dirname(item["path"]), exist_ok=True)
        with open(item["path"], "w", encoding="utf-8") as f:
            json.dump(item["data"], f, indent=2, ensure_ascii=False)
    elif action == "mkdir":
        os.makedirs(item["path"], exist_ok=True)


# ── public API ────────────────────────────────────────────────────

class HumanEgoRecorder:
    """HumanEgo data collector.

    Recording-path operation:

    1. ``write_*()`` —  store lightweight metadata only  (no frame copies).
    2. ``stop()``     —  spawn a daemon thread that:
       a. Reads RGB frames from the concurrently-recorded mp4, or
          —if unavailable— accepts a fallback list of numpy frames.
       b. Writes all per-frame PNG + JSON sidecar files.
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
        self._hand_data_humanego: list[list[dict]] = []
        # Backup: if mp4 source is not available, store frames here.
        # Bounded to _MAX_BACKUP_FRAMES to prevent OOM.
        self._frame_backup: list[np.ndarray] = []
        self._K: np.ndarray | None = None
        self._fov = self._DEFAULT_FOV_DEG
        self._w, self._h = 1280, 800
        self._fps = 30
        self._latest_vio: np.ndarray | None = None
        self._latest_hands: list | None = None
        self._latest_hands_he: list[dict] | None = None
        self._io_queue = None
        self._io_thread = None
        self._mp4_source: str | None = None

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
        self._hand_data_humanego = []
        self._frame_backup = []
        self._w, self._h = self._cfg.camera.resolution
        self._fps = self._cfg.camera.fps
        self._latest_vio = None
        self._latest_hands = None
        self._latest_hands_he = None
        self._mp4_source = None
        return self._session_dir

    def stop(self):
        if not self._active:
            return
        self._active = False
        total = self._frame_idx

        if total == 0:
            self._vio_frames = []
            self._timestamps_us = []
            self._hand_data_left = []
            self._hand_data_right = []
            self._hand_data_humanego = []
            self._frame_backup = []
            self._latest_vio = None
            self._latest_hands = None
            self._latest_hands_he = None
            return

        # Gather data that the worker needs (no copies — worker runs
        # after all write_*() calls have stopped)
        K = self._K if self._K is not None else self.default_intrinsics()
        payload = {
            "session_dir": self._session_dir,
            "total": total,
            "K": K,
            "fov": self._fov,
            "w": self._w, "h": self._h,
            "fps": self._fps,
            "vio_frames": self._vio_frames,
            "timestamps_us": self._timestamps_us,
            "hand_data_left": self._hand_data_left,
            "hand_data_right": self._hand_data_right,
            "hand_data_humanego": self._hand_data_humanego,
            "mp4_source": self._mp4_source,
            "frame_backup": self._frame_backup,
        }

        self._vio_frames = []
        self._timestamps_us = []
        self._hand_data_left = []
        self._hand_data_right = []
        self._hand_data_humanego = []
        self._frame_backup = []
        self._latest_vio = None
        self._latest_hands = None
        self._latest_hands_he = None

        t = threading.Thread(target=_flush_worker,
                             args=(payload,), daemon=True)
        t.start()

    def set_camera_intrinsics(self, K: np.ndarray):
        self._K = np.asarray(K, dtype=np.float64)

    def set_mp4_source(self, center_mp4_path: str):
        """Tell the recorder where the center-camera mp4 is for frame extraction."""
        self._mp4_source = center_mp4_path

    # ── data ingestion (called from UI signal handlers) ────────────

    def write_frame_rgb(self, bgr: np.ndarray, timestamp_us: int = 0):
        if not self._active:
            return
        if bgr is not None and bgr.size == 0:
            return

        # Store up to 90 backup frames (3 seconds) as safety net if mp4
        # source is unavailable at stop() time.  Oldest frames are dropped.
        # This is ~270 MB max, FAR less than the previous unbounded growth.
        if self._mp4_source is None:
            self._frame_backup.append(bgr.copy())
            if len(self._frame_backup) > 90:
                self._frame_backup.pop(0)  # only keep last 3 seconds

        self._timestamps_us.append(timestamp_us)
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

        if self._latest_hands_he is not None:
            self._hand_data_humanego.append(list(self._latest_hands_he))
        else:
            self._hand_data_humanego.append([])

        self._frame_idx += 1

    def write_vio(self, transform: np.ndarray):
        if self._active:
            self._latest_vio = np.asarray(transform, dtype=np.float64)

    def write_hands(self, role: str, hands: list):
        if not self._active:
            return
        self._latest_hands = hands

    def write_hands_humanego(self, role: str, humanego_hands: list[dict]):
        if not self._active:
            return
        self._latest_hands_he = humanego_hands

    def default_intrinsics(self) -> np.ndarray:
        w, h = self._w, self._h
        fx = w / (2.0 * np.tan(np.radians(self._fov / 2.0)))
        return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)


# ── flush worker (background thread spawned at stop()) ────────────

def _flush_worker(payload: dict):
    import queue  # local import: runs in background thread
    session_dir = payload["session_dir"]
    total = payload["total"]
    K = payload["K"]
    fov = payload["fov"]
    w, h, fps = payload["w"], payload["h"], payload["fps"]
    vio_frames = payload["vio_frames"]
    timestamps_us = payload["timestamps_us"]
    hand_data_left = payload["hand_data_left"]
    hand_data_right = payload["hand_data_right"]
    hand_data_humanego = payload["hand_data_humanego"]
    mp4_src = payload.get("mp4_source")
    frame_backup = payload.get("frame_backup", [])

    slam_frames = compute_slam_frames(vio_frames, timestamps_us)

    # Open mp4 for frame extraction if available
    cap = None
    if mp4_src and os.path.isfile(mp4_src):
        try:
            cap = cv2.VideoCapture(mp4_src)
            if cap.isOpened():
                mp4_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if mp4_total == 0:
                    cap.release()
                    cap = None
            else:
                cap.release()
                cap = None
        except Exception:
            cap = None

    # I/O queue pipeline for parallelism (PNG encoding is slow)
    q = queue.Queue(maxsize=200)
    worker = threading.Thread(target=_io_worker, args=(q,), daemon=True)
    worker.start()

    for idx in range(total):
        frame_dir = session_dir / f"{idx:05d}"
        os.makedirs(frame_dir, exist_ok=True)

        # Extract frame
        frame = None
        if cap is not None and cap.isOpened():
            ret, frm = cap.read()
            if ret:
                frame = frm
        if frame is None and idx < len(frame_backup):
            frame = frame_backup[idx]
        if frame is not None:
            q.put({
                "action": "write_png",
                "path": str(frame_dir / "rgb.png"),
                "frame": frame,
            })
        # else: frame not available (edge case — mp4 missing, no backup)

        # aria_cam_rgb.json
        c2w = vio_frames[idx].tolist() if idx < len(vio_frames) else np.eye(4).tolist()
        ts_ns = int(timestamps_us[idx] * 1000) if idx < len(timestamps_us) else 0
        q.put({
            "action": "write_json",
            "path": str(frame_dir / "aria_cam_rgb.json"),
            "data": {
                "idx": idx, "ts": ts_ns, "fov": fov, "h": h, "w": w,
                "k": K.tolist(), "d": [0.0, 0.0, 0.0, 0.0, 0.0],
                "c2w": c2w, "c2d": np.eye(4).tolist(), "d2w": np.eye(4).tolist(),
                "rgb_path": f"preprocess/all_data/{idx:05d}/rgb.png",
                "fps": fps,
            },
        })

        # aria_slam.json
        slam_data = (slam_frames[idx] if idx < len(slam_frames)
                     else {"idx": idx, "ts": 0, "t_world": [0,0,0],
                           "rpy_deg": [0,0,0], "delta_t_world": [0,0,0],
                           "delta_rpy_deg": [0,0,0],
                           "linear_speed_mps": 0.0, "angular_speed_rps": 0.0,
                           "yaw_unwrapped_deg": 0.0})
        q.put({"action": "write_json",
               "path": str(frame_dir / "aria_slam.json"),
               "data": slam_data})

        # aria_hands.json
        he_left = [h for h in (hand_data_humanego[idx] if idx < len(hand_data_humanego) else [])
                   if h[0].lower().startswith("l")]
        he_right = [h for h in (hand_data_humanego[idx] if idx < len(hand_data_humanego) else [])
                    if h[0].lower().startswith("r")]
        fbl = hand_data_left[idx] if idx < len(hand_data_left) else []
        fbr = hand_data_right[idx] if idx < len(hand_data_right) else []
        q.put({
            "action": "write_json",
            "path": str(frame_dir / "aria_hands.json"),
            "data": {
                "idx": idx,
                "ts": int(timestamps_us[idx] * 1000) if idx < len(timestamps_us) and timestamps_us[idx] > 0 else 0,
                "hand_l": pack_hand(he_left, fbl),
                "hand_r": pack_hand(he_right, fbr),
            },
        })

        # training_data.json
        q.put({
            "action": "write_json",
            "path": str(frame_dir / "training_data.json"),
            "data": build_training_data(idx, ts_ns, w, h, fps, K, c2w),
        })

    if cap is not None:
        cap.release()

    q.put(_STOP_SENTINEL)
    worker.join(timeout=300)
