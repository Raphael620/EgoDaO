"""IMU data display panel — bar gauges for accelerometer and gyroscope."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

ACCEL_RANGE = 4.0   # +/- G
GYRO_RANGE = 500.0  # +/- dps

# fmt: off
_AXIS_COLORS = {
    "X": QColor("#ff5555"), "Y": QColor("#55ff55"), "Z": QColor("#5588ff"),
}
# fmt: on


class ImuBarWidget(QWidget):
    """Horizontal bar gauge for a single IMU axis."""

    def __init__(self, axis: str, unit: str, min_val: float, max_val: float, parent=None):
        super().__init__(parent)
        self._axis = axis
        self._unit = unit
        self._color = _AXIS_COLORS.get(axis, QColor("#888"))
        self._min = min_val
        self._max = max_val
        self._value = 0.0
        self.setMinimumHeight(22)
        self.setMaximumHeight(22)

    def set_value(self, v: float):
        self._value = float(v) if abs(v) < 999 else 0.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        bar_left, bar_right = 40, 40
        bar_w = w - bar_left - bar_right

        # Background
        p.fillRect(0, 0, w, h, QColor(25, 25, 30))

        # Filled bar
        ratio = max(0.0, min(1.0, (self._value - self._min) / (self._max - self._min)))
        bw = int(bar_w * ratio)
        if bw > 0:
            c = QColor(self._color)
            c.setAlpha(170)
            p.fillRect(bar_left, 2, bw, h - 4, c)

        # Center line
        cx = bar_left + int(bar_w * (0.5 - self._min / (self._max - self._min)))
        p.setPen(QPen(QColor(60, 60, 70), 1))
        p.drawLine(cx, 0, cx, h)

        # Axis label (left)
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Consolas", 9))
        p.drawText(2, 0, bar_left - 4, h, int(Qt.AlignVCenter) | int(Qt.AlignLeft),
                   f"{self._axis}:")

        # Value label (right)
        val_s = f"{self._value:+.2f}" if abs(self._value) < 10 else f"{self._value:+.1f}"
        p.drawText(w - bar_right, 0, bar_right - 4, h, int(Qt.AlignVCenter) | int(Qt.AlignRight), val_s)
        p.end()


class ImuPanel(QWidget):
    """Panel showing accelerometer and gyroscope bar gauges with integrated attitude estimation.

    Complementary filter is updated per-IMU-sample.
    """

    _BIAS_SAMPLES_TARGET = 600  # ~2 s at 400 Hz

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)

        # Attitude estimation state
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._rot_matrix = np.eye(3, dtype=np.float64)
        self._gyro_bias = np.zeros(3, dtype=np.float64)
        self._bias_samples = 0
        self._bias_locked = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Accelerometer group
        accel_group = QGroupBox("Accelerometer (G)")
        accel_layout = QVBoxLayout(accel_group)
        accel_layout.setContentsMargins(4, 4, 4, 4)
        accel_layout.setSpacing(2)
        self._accel_bars: list[ImuBarWidget] = []
        for ax in ("X", "Y", "Z"):
            bar = ImuBarWidget(ax, "G", -ACCEL_RANGE, ACCEL_RANGE)
            self._accel_bars.append(bar)
            accel_layout.addWidget(bar)
        layout.addWidget(accel_group)

        # Gyroscope group
        gyro_group = QGroupBox("Gyroscope (dps)")
        gyro_layout = QVBoxLayout(gyro_group)
        gyro_layout.setContentsMargins(4, 4, 4, 4)
        gyro_layout.setSpacing(2)
        self._gyro_bars: list[ImuBarWidget] = []
        for ax in ("X", "Y", "Z"):
            bar = ImuBarWidget(ax, "dps", -GYRO_RANGE, GYRO_RANGE)
            self._gyro_bars.append(bar)
            gyro_layout.addWidget(bar)
        layout.addWidget(gyro_group)

        # Attitude readout
        self._attitude_label = QLabel("RPY: --")
        self._attitude_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._attitude_label)
        layout.addStretch()

    # ── public properties ─────────────────────────────────────────

    @property
    def attitude_rpy(self) -> tuple[float, float, float]:
        return self._roll, self._pitch, self._yaw

    @property
    def attitude_matrix(self) -> np.ndarray:
        return self._rot_matrix.copy()

    # ── data update ───────────────────────────────────────────────

    def update_imu(self, readings: list[dict]):
        """Ingest a batch of decoded IMU readings, update attitude and bar gauges."""
        if not readings:
            return

        # Estimate dt from the batch
        n = len(readings)
        if n >= 2:
            dt = (readings[-1]["t_us"] - readings[0]["t_us"]) / (n - 1) / 1_000_000
        else:
            dt = 1.0 / 400.0
        dt = max(dt, 1e-6)

        for r in readings:
            acc = r.get("acc_g", [0, 0, 0])
            gyr_dps = r.get("gyro_dps", [0, 0, 0])

            # Gyro bias calibration (first ~2 s)
            if not self._bias_locked:
                self._gyro_bias[0] += math.radians(gyr_dps[0])
                self._gyro_bias[1] += math.radians(gyr_dps[1])
                self._gyro_bias[2] += math.radians(gyr_dps[2])
                self._bias_samples += 1
                if self._bias_samples >= self._BIAS_SAMPLES_TARGET:
                    self._gyro_bias /= self._bias_samples
                    self._bias_locked = True
                continue

            # Complementary filter
            gx = math.radians(gyr_dps[0]) - self._gyro_bias[0]
            gy = math.radians(gyr_dps[1]) - self._gyro_bias[1]
            gz = math.radians(gyr_dps[2]) - self._gyro_bias[2]

            sp = math.sin(self._pitch)
            cp = math.cos(self._pitch)
            sr = math.sin(self._roll)
            cr = math.cos(self._roll)

            # Gyro integration
            self._roll += dt * (gx + gy * sr * sp / cp + gz * cr * sp / cp)
            self._pitch += dt * (gy * cr - gz * sr)
            self._yaw += dt * (gy * sr / cp + gz * cr / cp)

            # Accel correction (low-pass)
            ax_norm = acc[0] / 1000.0
            ay_norm = acc[1] / 1000.0
            az_norm = acc[2] / 1000.0
            accel_roll = math.atan2(ay_norm, az_norm)
            accel_pitch = math.atan2(-ax_norm, math.sqrt(ay_norm * ay_norm + az_norm * az_norm))
            alpha = 0.02
            self._roll = (1 - alpha) * self._roll + alpha * accel_roll
            self._pitch = (1 - alpha) * self._pitch + alpha * accel_pitch

        # Update rotation matrix from latest Euler angles
        cr, cp, cy = math.cos(self._roll), math.cos(self._pitch), math.cos(self._yaw)
        sr, sp, sy = math.sin(self._roll), math.sin(self._pitch), math.sin(self._yaw)
        self._rot_matrix = np.array([
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp,     sr * cp,                cr * cp],
        ], dtype=np.float64)

        # Update bar gauges with latest reading
        last = readings[-1]
        acc_last = last.get("acc_g", [0, 0, 0])
        gyr_last = last.get("gyro_dps", [0, 0, 0])

        if self._bias_locked:
            gyr_corrected = [
                gyr_last[0] - math.degrees(self._gyro_bias[0]),
                gyr_last[1] - math.degrees(self._gyro_bias[1]),
                gyr_last[2] - math.degrees(self._gyro_bias[2]),
            ]
        else:
            gyr_corrected = gyr_last

        for bar, val in zip(self._accel_bars, acc_last):
            bar.set_value(val)
        for bar, val in zip(self._gyro_bars, gyr_corrected):
            bar.set_value(val)

        self._attitude_label.setText(
            f"RPY: R={math.degrees(self._roll):+.0f}  "
            f"P={math.degrees(self._pitch):+.0f}  "
            f"Y={math.degrees(self._yaw):+.0f}"
        )
