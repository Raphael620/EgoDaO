"""IMU 3D pose widget — renders a rotating coordinate frame via QPainter (no OpenGL dependency)."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class Imu3DWidget(QWidget):
    """Shows IMU orientation as a projected 3D coordinate frame with RPY labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 160)
        self._rot = np.eye(3, dtype=np.float64)
        self._euler = (0.0, 0.0, 0.0)
        self.setStyleSheet("background-color: #101016;")

    def set_rotation(self, rot_matrix: np.ndarray, euler_deg: tuple[float, float, float] = (0, 0, 0)):
        self._rot = np.asarray(rot_matrix, dtype=np.float64)
        self._euler = euler_deg
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        p.fillRect(0, 0, w, h, QColor(16, 16, 22))

        # Simple orthographic projection: drop Z, rotate by fixed viewpoint angles
        view_az = math.radians(20)
        view_el = math.radians(30)

        Rv = np.array([
            [math.cos(view_az), 0, math.sin(view_az)],
            [math.sin(view_el) * math.sin(view_az), math.cos(view_el), -math.sin(view_el) * math.cos(view_az)],
        ], dtype=np.float64)

        scale = min(w, h) * 0.28
        origin_2d = np.array([cx, cy])

        def project(vec3):
            p2 = Rv @ self._rot @ np.asarray(vec3, dtype=np.float64)
            return origin_2d + np.array([p2[0], -p2[1]]) * scale

        # Axes
        axis_colors = [
            (QColor(230, 60, 60), "X"),
            (QColor(60, 230, 60), "Y"),
            (QColor(60, 100, 240), "Z"),
        ]
        directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        o = project((0, 0, 0))

        for (color, label), v in zip(axis_colors, directions):
            tip = project(v)
            pen = QPen(color, 2.5, Qt.SolidLine, Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(int(o[0]), int(o[1]), int(tip[0]), int(tip[1]))
            # Arrowhead
            head_size = 5
            dir_vec = tip - o
            dlen = np.linalg.norm(dir_vec)
            if dlen > 0:
                ndir = dir_vec / dlen
                perp = np.array([-ndir[1], ndir[0]])
                ah = tip - ndir * head_size * 2
                p1 = ah + perp * head_size
                p2 = ah - perp * head_size
                p.setBrush(color)
                p.setPen(Qt.NoPen)
                poly = QPolygonF()
                from PySide6.QtCore import QPointF
                poly << QPointF(tip[0], tip[1]) << QPointF(p1[0], p1[1]) << QPointF(p2[0], p2[1])
                p.drawPolygon(poly)

            # Label near tip
            p.setPen(color.lighter(140))
            p.setFont(QFont("Consolas", 10, QFont.Bold))
            lt = tip + (tip - o) * 0.15 / max(dlen, 1e-6) * scale
            p.drawText(int(lt[0]) - 6, int(lt[1]) + 4, label)

        # Center dot
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(200, 200, 200))
        p.drawEllipse(int(o[0]) - 3, int(o[1]) - 3, 6, 6)

        # RPY text
        roll, pitch, yaw = self._euler
        p.setPen(QColor(150, 150, 170))
        p.setFont(QFont("Consolas", 9))
        p.drawText(8, h - 8, f"R{roll:+.0f}  P{pitch:+.0f}  Y{yaw:+.0f}")

        p.end()
