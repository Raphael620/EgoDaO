import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "Ego Daq-O"
APP_VERSION = "0.3.0"


@dataclass
class CameraConfig:
    resolution: tuple[int, int] = (1280, 800)
    fps: int = 30


@dataclass
class ImuConfig:
    accel_rate_hz: int = 200
    gyro_rate_hz: int = 200
    batch_threshold: int = 1
    max_batch_reports: int = 20


@dataclass
class RecordingConfig:
    data_root: Path = field(default_factory=lambda: Path(os.getcwd()) / "Data")
    raw_subdir: str = "Raw"
    humanego_subdir: str = "HumanEgo"
    video_codec: str = "mp4v"
    video_backend: str = "ffmpeg_hw"    # "ffmpeg_hw" (Intel QSV) or "opencv" (cv2.VideoWriter)
    hw_encoder: str = "h264_qsv"        # encoder: h264_qsv, h264_mf, h264_d3d12va, etc.
    enable_raw: bool = True          # record raw mp4 + csv/jsonl
    enable_humanego: bool = False    # record HumanEgo-format per-frame data
    disk_keep_days: int = 3          # auto-clean raw folders older than N days (0 = disabled)


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    enable_vio: bool = True
    enable_hand_tracking: bool = True
    hand_tracker_backend: str = "mediapipe"  # "mediapipe" (CPU), "openvino" (Intel GPU), "mercury" (ONNX experimental)
    vio_camera_resolution: tuple[int, int] = (640, 400)
