"""Ego Daq-O V0.2.1 — Ego 数据采集与实时处理系统."""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def _load_config() -> dict:
    """Load optional config JSON from the project root or command line."""
    # Priority: 1) --config <path> on command line  2) ./config.json
    config_path = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            break
    if config_path is None:
        default = Path(_root) / "config.json"
        if default.is_file():
            config_path = str(default)

    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def _apply_config(app_config, cfg: dict):
    """Merge JSON config keys into AppConfig."""
    rec = cfg.get("recording", {})
    if "data_root" in rec:
        app_config.recording.data_root = Path(rec["data_root"])
    if "raw_subdir" in rec:
        app_config.recording.raw_subdir = rec["raw_subdir"]
    if "humanego_subdir" in rec:
        app_config.recording.humanego_subdir = rec["humanego_subdir"]
    if "video_codec" in rec:
        app_config.recording.video_codec = rec["video_codec"]
    if "video_backend" in rec:
        app_config.recording.video_backend = rec["video_backend"]

    cam = cfg.get("camera", {})
    if "resolution" in cam:
        app_config.camera.resolution = tuple(cam["resolution"])
    if "fps" in cam:
        app_config.camera.fps = cam["fps"]

    imu = cfg.get("imu", {})
    if "accel_rate_hz" in imu:
        app_config.imu.accel_rate_hz = imu["accel_rate_hz"]
    if "gyro_rate_hz" in imu:
        app_config.imu.gyro_rate_hz = imu["gyro_rate_hz"]

    app = cfg.get("app", {})
    if "enable_vio" in app:
        app_config.enable_vio = app["enable_vio"]
    if "enable_hand_tracking" in app:
        app_config.enable_hand_tracking = app["enable_hand_tracking"]


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("Ego Daq-O")
    app.setApplicationVersion("0.2.1")

    # Load config
    cfg = _load_config()
    from DaO.config import AppConfig
    app_config = AppConfig()
    if cfg:
        _apply_config(app_config, cfg)

    from DaO.ui.main_window import MainWindow
    window = MainWindow(app_config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
