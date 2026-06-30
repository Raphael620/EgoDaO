import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CameraConfig:
    resolution: tuple[int, int] = (1280, 800)
    fps: int = 30


@dataclass
class ImuConfig:
    accel_rate_hz: int = 480
    gyro_rate_hz: int = 400
    batch_threshold: int = 5
    max_batch_reports: int = 20


@dataclass
class RecordingConfig:
    data_root: Path = field(default_factory=lambda: Path(os.getcwd()) / "Data")
    raw_subdir: str = "Raw"
    humanego_subdir: str = "HumanEgo"
    video_codec: str = "mp4v"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    enable_vio: bool = True
    enable_hand_tracking: bool = True
    vio_camera_resolution: tuple[int, int] = (640, 400)
