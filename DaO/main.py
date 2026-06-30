"""Ego Daq-O V0.2.1 — Ego 数据采集与实时处理系统."""
from __future__ import annotations

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("Ego Daq-O")
    app.setApplicationVersion("0.2.1")

    from DaO.ui.main_window import MainWindow
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
