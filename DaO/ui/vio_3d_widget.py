"""VIO 3D trajectory widget — real-time position trail via QPainter."""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class Vio3DWidget(QWidget):
    """Renders VIO trajectory as an orthographic 3D line-strip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 160)
        self._trail_x = deque(maxlen=2000)
        self._trail_y = deque(maxlen=2000)
        self._trail_z = deque(maxlen=2000)
        self._latest_pos = np.zeros(3)
        self._view_az = 45.0
        self._view_el = 25.0
        self._auto_scale = 2.0  # dynamic scale, starts at 2m view
        self.setStyleSheet("background-color: #0e0e14;")

    def update_pose(self, transform: np.ndarray):
        pos = transform[:3, 3].copy()
        self._trail_x.append(float(pos[0]))
        self._trail_y.append(float(pos[1]))
        self._trail_z.append(float(pos[2]))
        self._latest_pos = pos

        # Dynamic auto-scaling
        if len(self._trail_x) > 50:
            all_x = np.array(self._trail_x)
            all_y = np.array(self._trail_y)
            all_z = np.array(self._trail_z)
            span = max(
                np.ptp(all_x[-200:]),
                np.ptp(all_y[-200:]),
                np.ptp(all_z[-200:]),
                0.02  # minimum 2cm
            )
            self._auto_scale = span * 2.5

        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        p.fillRect(0, 0, w, h, QColor(14, 14, 20))

        az = math.radians(self._view_az)
        el = math.radians(self._view_el)
        Rv = np.array([
            [math.cos(az), 0, math.sin(az)],
            [math.sin(el) * math.sin(az), math.cos(el), -math.sin(el) * math.cos(az)],
        ], dtype=np.float64)

        half_span = max(self._auto_scale, 0.05)
        view_size = min(w, h) * 0.85
        scale_factor = view_size / (half_span * 2)
        origin = np.array([cx, cy])

        def project(x, y, z):
            p2 = Rv @ np.array([x, y, z])
            return origin + np.array([p2[0], -p2[1]]) * scale_factor

        # Grid at adaptive interval
        grid_step = 10 ** math.floor(math.log10(half_span + 1e-6))
        if half_span / grid_step < 2:
            grid_step /= 2
        ng = int(half_span / grid_step) + 2
        p.setPen(QPen(QColor(30, 30, 40), 1, Qt.DotLine))
        for i in range(-ng, ng + 1):
            v = i * grid_step
            p1 = project(v, 0, -half_span)
            p2 = project(v, 0, half_span)
            p.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            p1 = project(-half_span, 0, v)
            p2 = project(half_span, 0, v)
            p.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        # Trail
        if len(self._trail_x) > 1:
            pen = QPen(QColor(60, 200, 230, 220), 2.5, Qt.SolidLine, Qt.RoundCap)
            p.setPen(pen)
            pts = []
            for i in range(len(self._trail_x)):
                pt = project(self._trail_x[i], self._trail_y[i], self._trail_z[i])
                pts.append((int(pt[0]), int(pt[1])))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])

            last_pt = pts[-1]
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 230, 50))
            p.drawEllipse(last_pt[0] - 5, last_pt[1] - 5, 10, 10)

        # Position text
        p.setPen(QColor(150, 150, 170))
        p.setFont(QFont("Consolas", 9))
        p3 = self._latest_pos
        p.drawText(8, h - 8,
                   f"({p3[0]:.3f}, {p3[1]:.3f}, {p3[2]:.3f}) m  |  {len(self._trail_x)} pts  |  span {half_span:.2f}m")

        p.end()
