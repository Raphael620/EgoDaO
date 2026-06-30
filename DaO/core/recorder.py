from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from DaO.config import AppConfig


class DataRecorder:
    """Raw data recorder: mp4 video + imu.csv + hands.json + vio.json.

    VideoWriters are created lazily on first frame so the recorded
    resolution matches the actual camera output.
    """

    def __init__(self, config: AppConfig | None = None):
        self._cfg = config or AppConfig()
        self._session_dir: Path | None = None
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._writer_paths: dict[str, str] = {}
        self._imu_file: Path | None = None
        self._imu_writer = None
        self._hands_frames: list[dict] = []
        self._vio_frames: list[dict] = []
        self._active = False
        self._frame_counters: dict[str, int] = {}

    @property
    def is_recording(self) -> bool:
        return self._active

    def start(self) -> Path:
        if self._active:
            return self._session_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = (
            self._cfg.recording.data_root / self._cfg.recording.raw_subdir / ts
        )
        os.makedirs(self._session_dir, exist_ok=True)

        self._writers = {}
        self._writer_paths = {}
        for role in ("left", "center", "right"):
            self._writer_paths[role] = str(self._session_dir / f"{role}_cam.mp4")

        self._imu_file = self._session_dir / "imu.csv"
        self._imu_writer = open(self._imu_file, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._imu_writer)
        self._csv.writerow(["timestamp_us", "acc_x_g", "acc_y_g", "acc_z_g",
                            "gyr_x_dps", "gyr_y_dps", "gyr_z_dps"])

        self._hands_frames = []
        self._vio_frames = []
        self._frame_counters = {"left": 0, "center": 0, "right": 0}
        self._active = True
        return self._session_dir

    def stop(self):
        if not self._active:
            return
        for w in self._writers.values():
            w.release()
        self._writers.clear()
        if self._imu_writer is not None:
            self._imu_writer.close()
            self._imu_writer = None

        with open(self._session_dir / "hands.json", "w", encoding="utf-8") as f:
            json.dump(self._hands_frames, f, indent=2, ensure_ascii=False)
        with open(self._session_dir / "vio.json", "w", encoding="utf-8") as f:
            json.dump(self._vio_frames, f, indent=2, ensure_ascii=False)
        self._active = False

    def write_frame(self, role: str, bgr: np.ndarray):
        if bgr is None or bgr.size == 0:
            return
        # Normalise mono → 3-channel (left/right cameras are grayscale)
        if len(bgr.shape) == 2 or bgr.shape[2] == 1:
            frame = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        else:
            frame = bgr
        # Lazy VideoWriter initialisation
        if role not in self._writers:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self._cfg.recording.video_codec)
            writer = cv2.VideoWriter(
                self._writer_paths[role], fourcc, self._cfg.camera.fps, (w, h))
            if writer.isOpened():
                self._writers[role] = writer
            else:
                return
        w = self._writers.get(role)
        if w is not None:
            w.write(frame)
            self._frame_counters[role] = self._frame_counters.get(role, 0) + 1

    def write_imu(self, readings: list[dict]):
        if self._imu_writer is None:
            return
        for r in readings:
            acc = r.get("acc_g", [0, 0, 0])
            gyr = r.get("gyro_dps", [0, 0, 0])
            self._csv.writerow([
                r.get("t_us", 0),
                f"{acc[0]:.4f}", f"{acc[1]:.4f}", f"{acc[2]:.4f}",
                f"{gyr[0]:.4f}", f"{gyr[1]:.4f}", f"{gyr[2]:.4f}",
            ])

    def write_hands(self, hands_data: dict):
        if self._active:
            self._hands_frames.append({
                "frame_idx": self._frame_counters.get("left", 0),
                **hands_data,
            })

    def write_vio(self, transform: np.ndarray):
        if self._active:
            self._vio_frames.append({
                "frame_idx": self._frame_counters.get("left", 0),
                "transform": transform.tolist(),
            })
