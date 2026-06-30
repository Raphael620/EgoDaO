"""Three-camera view widget — side-by-side display with hand skeleton overlay on left/right."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

CAMERA_LABELS = {"left": "Left", "center": "Center", "right": "Right"}
CAMERA_ORDER = ("left", "center", "right")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Pre-allocate colors to avoid QColor construction every frame
_COLOR_LEFT = QColor(100, 200, 255)
_COLOR_RIGHT = QColor(255, 150, 80)
_COLOR_LEFT_BRIGHT = QColor(220, 220, 255)
_COLOR_KEYPOINT_LEFT = QColor(100, 200, 255)
_COLOR_KEYPOINT_RIGHT = QColor(255, 150, 80)

class CameraPane(QFrame):
    """A single camera feed pane with hand skeleton overlay.

    Rendering is deferred: set_frame/set_hands mark the pane dirty, and
    _do_render is called at most once per 33 ms (30 FPS) via a coalescing timer.
    """

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self._role = role
        self.setObjectName("cameraPane")
        self.setFrameStyle(int(QFrame.Shape.StyledPanel))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._label_title = QLabel(CAMERA_LABELS.get(role, role))
        self._label_title.setAlignment(Qt.AlignCenter)
        self._label_title.setObjectName("cameraTitle")
        self._label_title.setFixedHeight(22)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(320, 200)

        layout.addWidget(self._label_title)
        layout.addWidget(self._image_label, 1)

        self._bgr: np.ndarray | None = None
        self._hands: list[tuple[str, np.ndarray]] = []

        # Coalescing timer — throttles repaints to 30 FPS max
        self._dirty = False
        self._render_timer = QTimer(self)
        self._render_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._do_render)

    def set_frame(self, bgr: np.ndarray | None):
        self._bgr = bgr
        self._dirty = True
        self._render_timer.start(33)  # throttle at ~30 fps

    def set_hands(self, hands: list[tuple[str, np.ndarray]]):
        """Set hand landmarks. Expects pixel-coordinate landmarks (21, 2) or (21, 3).

        Calling with an empty list immediately clears the skeleton overlay.
        Otherwise the draw happens on the next frame render.
        """
        if not hands and self._hands:
            self._hands = []
            self._dirty = True
            self._render_timer.start(0)  # Immediate render (next event-loop tick)
            return
        self._hands = hands

    def _do_render(self):
        self._dirty = False
        if self._bgr is None:
            return

        h, w = self._bgr.shape[:2]
        lbl_w = self._image_label.width()
        lbl_h = self._image_label.height()
        if lbl_w < 2 or lbl_h < 2:
            return

        scale = min(lbl_w / w, lbl_h / h)
        dw, dh = int(w * scale), int(h * scale)
        if dw < 1 or dh < 1:
            return

        # Normalise to 3-channel BGR (left/right may be mono), then → RGB → QPixmap
        disp = self._bgr
        if len(disp.shape) == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        elif disp.shape[2] == 1:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            dw, dh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        # Draw hand skeletons
        if self._hands:
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen()
            pen.setWidth(2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)

            for label, lms in self._hands:
                lms_np = np.asarray(lms, dtype=np.float32)
                if lms_np.shape[0] < 21:
                    continue
                sx = lms_np[:, 0] * scale
                sy = lms_np[:, 1] * scale

                side = "left" if label.lower().startswith("l") else "right"
                color = _COLOR_LEFT if side == "left" else _COLOR_RIGHT

                # Connections
                pen.setColor(color)
                painter.setPen(pen)
                for a, b in HAND_CONNECTIONS:
                    if a < len(sx) and b < len(sx):
                        painter.drawLine(int(sx[a]), int(sy[a]), int(sx[b]), int(sy[b]))

                # Keypoints
                painter.setPen(Qt.PenStyle.NoPen)
                kp_color = _COLOR_KEYPOINT_LEFT if side == "left" else _COLOR_KEYPOINT_RIGHT
                painter.setBrush(kp_color)
                for j in range(21):
                    r = 4 if j == 0 else 3
                    painter.drawEllipse(int(sx[j]) - r, int(sy[j]) - r, r * 2, r * 2)

                # Label
                painter.setPen(_COLOR_LEFT_BRIGHT)
                painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                painter.drawText(int(sx[0]) + 6, int(sy[0]) - 6, label)

            painter.end()

        self._image_label.setPixmap(pix)

    def paintEvent(self, event):
        super().paintEvent(event)
        # During resize, re-render if we already have a frame
        if self._bgr is not None and self._image_label.pixmap() is None:
            self._do_render()


class CameraViewWidget(QWidget):
    """Horizontal layout of three CameraPane widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._panes: dict[str, CameraPane] = {}
        for role in CAMERA_ORDER:
            pane = CameraPane(role)
            self._panes[role] = pane
            layout.addWidget(pane, 1)

    def set_frame(self, role: str, bgr: np.ndarray):
        pane = self._panes.get(role)
        if pane is not None:
            pane.set_frame(bgr)

    def set_hands(self, role: str, hands: list[tuple[str, np.ndarray]]):
        pane = self._panes.get(role)
        if pane is not None:
            pane.set_hands(hands)
