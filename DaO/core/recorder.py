from __future__ import annotations

import csv
import json
import os
import subprocess
import abc
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from DaO.config import AppConfig


# ── video writer interfaces ───────────────────────────────────────

class _BaseVideoWriter(abc.ABC):
    """Abstract video sink."""

    @abc.abstractmethod
    def write(self, frame: np.ndarray) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class _OpenCVWriter(_BaseVideoWriter):
    """Standard cv2.VideoWriter (software encoding)."""

    def __init__(self, path: str, codec: str, fps: int, w: int, h: int):
        for c in (codec, "mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*c)
            self._w = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if self._w.isOpened():
                break
            self._w.release()
        if not self._w.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed for {path}")

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def close(self) -> None:
        self._w.release()


class _FFmpegHWWriter(_BaseVideoWriter):
    """FFmpeg h264_mf (MediaFoundation) hardware encoder via subprocess pipe."""

    _FFMPEG_PATH = "D:/Develop/bin/ffmpeg-n8.1-latest-win64-gpl-8.1/bin/ffmpeg.exe"

    def __init__(self, path: str, fps: int, w: int, h: int):
        self._path = path
        self._h, self._w = h, w
        self._proc = None
        self._stream = None

        cmd = [
            self._FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{w}x{h}",
            "-framerate", str(fps),
            "-i", "pipe:0",
            "-c:v", "h264_mf",
            "-b:v", "8M",
            "-pix_fmt", "yuv420p",
            path,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._stream = self._proc.stdin
        except Exception as e:
            raise RuntimeError(f"FFmpeg failed to start: {e}")

    def write(self, frame: np.ndarray) -> None:
        if self._stream is None:
            return
        # Ensure BGR 3-channel
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        try:
            self._stream.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            pass  # FFmpeg may have exited

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None


def _create_writer(path: str, backend: str, codec: str,
                   fps: int, w: int, h: int) -> _BaseVideoWriter | None:
    """Factory: create a video writer, falling back to OpenCV if HW fails."""
    if backend == "ffmpeg_hw":
        try:
            return _FFmpegHWWriter(path, fps, w, h)
        except Exception:
            pass  # Fall through to OpenCV
    # Default: OpenCV
    try:
        return _OpenCVWriter(path, codec, fps, w, h)
    except Exception:
        return None


# ── DataRecorder ───────────────────────────────────────────────────

class DataRecorder:
    """Raw data recorder: mp4 video + imu.csv + hands.jsonl + vio.jsonl.

    Video encoding uses the backend specified in ``RecordingConfig.video_backend``:
      - ``"opencv"``:  cv2.VideoWriter (software)
      - ``"ffmpeg_hw"``: FFmpeg h264_mf (Windows MediaFoundation HW)
    Falls back to OpenCV if the FFmpeg backend fails.
    """

    def __init__(self, config: AppConfig | None = None):
        self._cfg = config or AppConfig()
        self._session_dir: Path | None = None
        self._writers: dict[str, _BaseVideoWriter] = {}
        self._imu_file: Path | None = None
        self._imu_writer = None
        self._hands_file = None
        self._vio_file = None
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
        self._hands_file = None
        self._vio_file = None

        self._imu_file = self._session_dir / "imu.csv"
        self._imu_writer = open(self._imu_file, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._imu_writer)
        self._csv.writerow(["timestamp_us", "acc_x_g", "acc_y_g", "acc_z_g",
                            "gyr_x_dps", "gyr_y_dps", "gyr_z_dps"])

        self._frame_counters = {"left": 0, "center": 0, "right": 0}
        self._active = True
        return self._session_dir

    def stop(self):
        if not self._active:
            return
        for w in self._writers.values():
            w.close()
        self._writers.clear()
        if self._imu_writer is not None:
            self._imu_writer.close()
            self._imu_writer = None
        if self._hands_file is not None:
            self._hands_file.close()
            self._hands_file = None
        if self._vio_file is not None:
            self._vio_file.close()
            self._vio_file = None
        self._active = False

    def write_frame(self, role: str, bgr: np.ndarray):
        if bgr is None or bgr.size == 0:
            return
        # Lazy VideoWriter initialisation
        if role not in self._writers:
            h, w = bgr.shape[:2]
            path = str(self._session_dir / f"{role}_cam.mp4")
            wtr = _create_writer(path, self._cfg.recording.video_backend,
                                 self._cfg.recording.video_codec,
                                 self._cfg.camera.fps, w, h)
            if wtr is not None:
                self._writers[role] = wtr
            else:
                return
        self._writers[role].write(bgr)
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
        if not self._active:
            return
        entry = json.dumps({
            "frame_idx": self._frame_counters.get("center", 0),
            **hands_data,
        }, ensure_ascii=False) + "\n"
        if self._hands_file is None:
            self._hands_file = open(
                str(self._session_dir / "hands.jsonl"), "w", encoding="utf-8")
        self._hands_file.write(entry)

    def write_vio(self, transform: np.ndarray):
        if not self._active:
            return
        entry = json.dumps({
            "frame_idx": self._frame_counters.get("left", 0),
            "transform": transform.tolist(),
        }, ensure_ascii=False) + "\n"
        if self._vio_file is None:
            self._vio_file = open(
                str(self._session_dir / "vio.jsonl"), "w", encoding="utf-8")
        self._vio_file.write(entry)
