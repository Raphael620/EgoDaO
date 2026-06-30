"""Main window for Ego Daq-O."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSplitter, QStatusBar, QToolBar, QWidget,
)

from DaO.config import AppConfig
from DaO.core.capture_worker import CaptureWorker
from DaO.core.humanego_recorder import HumanEgoRecorder
from DaO.core.recorder import DataRecorder
from DaO.ui.camera_view import CameraViewWidget
from DaO.ui.imu_3d_widget import Imu3DWidget
from DaO.ui.imu_panel import ImuPanel
from DaO.ui.vio_3d_widget import Vio3DWidget

STYLESHEET = """
QMainWindow { background-color: #1a1a1a; }
QWidget { color: #ddd; }
QToolBar { background: #222; border: none; spacing: 8px; padding: 4px 8px; }
QStatusBar { background: #222; color: #aaa; font-size: 11px; border-top: 1px solid #333; }
QPushButton { background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px; padding: 6px 18px; font-size: 13px; font-weight: bold; }
QPushButton:hover { background: #444; }
QPushButton:pressed { background: #555; }
QPushButton:disabled { color: #666; background: #252525; }
QPushButton#btnRecord { color: #ff6644; }
QPushButton#btnRecord:disabled { color: #553322; }
QSplitter::handle { background: #2a2a2a; width: 2px; }
QLabel#cameraTitle { background: #252525; color: #999; font-size: 12px; font-weight: bold; padding: 2px; }
QFrame#cameraPane { border: 1px solid #333; background: #111; }
"""


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig | None = None):
        super().__init__()
        self._cfg = config or AppConfig()
        self._capture: CaptureWorker | None = None
        self._recorder: DataRecorder | None = None
        self._he_recorder: HumanEgoRecorder | None = None
        self._frame_count = 0
        self._fps = 0.0
        self._vio_pt_count = 0
        self._hand_counts = {"left": 0, "right": 0}
        self._setup_ui()
        self._setup_status_timer()

    # ── UI construction ────────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle("Ego Daq-O V0.2.1 — Ego 数据采集与实时处理系统")
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(STYLESHEET)

        # icon
        icon_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icon.png")
        if os.path.exists(icon_file):
            self.setWindowIcon(QIcon(icon_file))

        # toolbar
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self._btn_capture = QPushButton("采集")
        self._btn_capture.setToolTip("连接硬件")
        self._btn_capture.clicked.connect(self._toggle_capture)
        toolbar.addWidget(self._btn_capture)
        toolbar.addSeparator()
        self._btn_record = QPushButton("录制")
        self._btn_record.setObjectName("btnRecord")
        self._btn_record.setToolTip("开始/停止录制")
        self._btn_record.clicked.connect(self._toggle_recording)
        self._btn_record.setEnabled(False)
        toolbar.addWidget(self._btn_record)
        toolbar.addSeparator()
        self._label_device = QLabel("设备未连接")
        self._label_device.setStyleSheet("color: #888; padding: 0 8px;")
        toolbar.addWidget(self._label_device)

        # camera views
        self._camera_view = CameraViewWidget()

        # bottom panel
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(4, 4, 4, 4)
        bottom_layout.setSpacing(4)
        self._imu_panel = ImuPanel()
        self._imu_3d = Imu3DWidget()
        self._vio_3d = Vio3DWidget()
        bottom_layout.addWidget(self._imu_panel, 2)
        bottom_layout.addWidget(self._imu_3d, 1)
        bottom_layout.addWidget(self._vio_3d, 3)
        bottom.setMaximumHeight(220)

        # splitter
        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Vertical)
        splitter.addWidget(self._camera_view)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 200])
        self.setCentralWidget(splitter)

        # status bar
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self._sb_status = QLabel("就绪")
        self._sb_fps = QLabel("FPS: --")
        self._sb_frames = QLabel("Frames: 0")
        self._sb_rpy = QLabel("RPY: --")
        self._sb_hand = QLabel("Hand: --")
        self._sb_vio = QLabel("VIO: --")
        for lbl in [self._sb_status, self._sb_fps, self._sb_frames, self._sb_rpy, self._sb_hand, self._sb_vio]:
            lbl.setStyleSheet("padding: 0 10px;")
            status_bar.addPermanentWidget(lbl)

    def _setup_status_timer(self):
        self._status_timer = QTimer(self)
        self._status_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start(100)

    # ── capture ────────────────────────────────────────────────────

    def _toggle_capture(self):
        if self._capture and self._capture.is_running:
            self._stop_capture()
        else:
            self._start_capture()

    def _start_capture(self):
        self._capture = CaptureWorker(self._cfg)
        self._capture.frame_ready.connect(self._on_frame)
        self._capture.hands_ready.connect(self._on_hands)
        self._capture.imu_ready.connect(self._on_imu)
        self._capture.vio_ready.connect(self._on_vio)
        self._capture.pipeline_stats.connect(self._on_stats)
        self._capture.pipeline_error.connect(self._on_error)
        self._capture.pipeline_started.connect(self._on_pipeline_started)
        self._capture.pipeline_stopped.connect(self._on_pipeline_stopped)
        self._capture.start()
        self._frame_count = 0
        self._vio_pt_count = 0
        self._hand_counts = {"left": 0, "right": 0}
        self._btn_capture.setText("停止采集")
        self._sb_status.setText("连接中...")
        self._label_device.setText("正在连接设备...")

    def _stop_capture(self):
        if self._recorder and self._recorder.is_recording:
            self._stop_recording()
        if self._capture:
            self._capture.stop()
            self._capture.join(3.0)
            self._capture = None
        self._btn_capture.setText("采集")
        self._btn_record.setEnabled(False)
        self._btn_record.setText("录制")
        self._sb_status.setText("已停止")
        self._label_device.setText("设备已断开")

    # ── recording ──────────────────────────────────────────────────

    def _toggle_recording(self):
        if self._recorder and self._recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self._recorder = DataRecorder(self._cfg)
        d = self._recorder.start()
        self._he_recorder = HumanEgoRecorder(self._cfg)
        self._he_recorder.start()
        self._btn_record.setText("停止录制")
        self._sb_status.setText(f"录制中 — {d.name}")

    def _stop_recording(self):
        if self._recorder:
            self._recorder.stop()
            self._recorder = None
        if self._he_recorder:
            self._he_recorder.stop()
            self._he_recorder = None
        self._btn_record.setText("录制")
        self._sb_status.setText("录制已保存")

    # ── signal handlers ────────────────────────────────────────────

    @Slot(str, np.ndarray)
    def _on_frame(self, role: str, bgr: np.ndarray):
        self._camera_view.set_frame(role, bgr)
        self._frame_count += 1
        if self._recorder and self._recorder.is_recording:
            self._recorder.write_frame(role, bgr)
        if self._he_recorder and self._he_recorder.is_recording and role == "center":
            self._he_recorder.write_frame_rgb(bgr)

    @Slot(str, list)
    def _on_hands(self, role: str, hands: list):
        self._camera_view.set_hands(role, hands)
        self._hand_counts[role] = len(hands)
        if self._recorder and self._recorder.is_recording:
            hands_dict = {}
            for label, lms in hands:
                hands_dict[label] = lms.tolist() if hasattr(lms, "tolist") else lms
            self._recorder.write_hands({role: hands_dict})
        if self._he_recorder and self._he_recorder.is_recording:
            self._he_recorder.write_hands(role, hands)

    @Slot(list)
    def _on_imu(self, readings: list[dict]):
        self._imu_panel.update_imu(readings)
        if self._recorder and self._recorder.is_recording:
            self._recorder.write_imu(readings)

    @Slot(np.ndarray)
    def _on_vio(self, transform: np.ndarray):
        self._vio_3d.update_pose(transform)
        self._vio_pt_count += 1
        if self._recorder and self._recorder.is_recording:
            self._recorder.write_vio(transform)
        if self._he_recorder and self._he_recorder.is_recording:
            self._he_recorder.write_vio(transform)

    @Slot(dict)
    def _on_stats(self, stats: dict):
        self._fps = stats.get("fps", 0)

    @Slot()
    def _on_pipeline_started(self):
        self._btn_record.setEnabled(True)
        self._sb_status.setText("采集中")
        self._label_device.setText("设备已连接 · 运行中")
        self._btn_capture.setText("停止采集")

    @Slot()
    def _on_pipeline_stopped(self):
        self._sb_status.setText("已停止")
        self._label_device.setText("设备已断开")

    @Slot(str)
    def _on_error(self, msg: str):
        self._sb_status.setText(f"错误: {msg[:60]}")
        self._stop_capture()
        QMessageBox.critical(self, "采集错误", msg)

    def _update_status_bar(self):
        self._sb_fps.setText(f"FPS: {self._fps:.1f}" if self._fps > 0 else "FPS: --")
        self._sb_frames.setText(f"Frames: {self._frame_count}")
        roll, pitch, yaw = self._imu_panel.attitude_rpy
        rd, pd, yd = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
        self._sb_rpy.setText(f"RPY: {rd:+.0f} {pd:+.0f} {yd:+.0f}")
        self._imu_3d.set_rotation(self._imu_panel.attitude_matrix, (rd, pd, yd))
        lh, rh = self._hand_counts.get("left", 0), self._hand_counts.get("right", 0)
        self._sb_hand.setText(f"Hand: L={lh} R={rh}")
        self._sb_vio.setText(f"VIO: {self._vio_pt_count} pts")

    def closeEvent(self, event):
        self._stop_capture()
        self._status_timer.stop()
        event.accept()
