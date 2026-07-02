import sys, time, traceback, threading
import depthai as dai
import numpy as np
from PySide6.QtCore import QObject, Signal
from DaO.config import AppConfig
from DaO.core.pipeline import create_pipeline


class CaptureWorker(QObject):
    """Reads camera/IMU/VIO from OAK device in a ``threading.Thread``.

    All ``Signal.emit()`` calls are cross-thread — Qt auto-dispatches
    them to the main-thread event loop.
    """
    frame_ready = Signal(str, np.ndarray, int)
    hands_ready = Signal(str, list)
    imu_ready = Signal(list)
    vio_ready = Signal(np.ndarray)
    pipeline_stats = Signal(dict)
    pipeline_error = Signal(str)
    pipeline_started = Signal()
    pipeline_stopped = Signal()

    _HAND_INTERVAL = 1

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self._cfg = config or AppConfig()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._device = None
        self._hand_tracker = None
        self._hand_skip_counter = {"center": 0}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return (self._thread is not None
                and self._thread.is_alive()
                and not self._stop_event.is_set())

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    # ── main loop ──────────────────────────────────────────────────

    def _run(self):
        self._stop_event.clear()
        try:
            infos = dai.Device.getAllConnectedDevices()
            if not infos:
                self.pipeline_error.emit("No OAK device.")
                return
            self._device = dai.Device(infos[0].getDeviceId())
        except Exception as e:
            self.pipeline_error.emit(f"Device: {e}")
            return

        try:
            pip, cam_q, imu_q, vio_q = create_pipeline(self._device, self._cfg)
        except Exception as e:
            self.pipeline_error.emit(f"Pipeline: {e}")
            self._device.close()
            return

        if self._cfg.enable_hand_tracking:
            try:
                from DaO.core.hand_tracker import create_hand_tracker
                self._hand_tracker = create_hand_tracker("mediapipe")
            except Exception as e:
                sys.stderr.write(f"HT init failed: {e}\n")
                self._hand_tracker = None

        pip.start()
        self.pipeline_started.emit()

        fc, fps_fc = 0, 0
        fps_t0 = time.perf_counter()

        try:
            while pip.isRunning() and not self._stop_event.is_set():
                any_data = False

                for role, q in cam_q.items():
                    try:
                        pkt = q.tryGet()
                    except Exception:
                        pkt = None
                    if pkt is None:
                        continue
                    bgr = pkt.getCvFrame()
                    if bgr is None or bgr.size == 0:
                        continue

                    # Extract timestamp in microseconds
                    ts_us = 0
                    try:
                        ts = pkt.getTimestamp()
                        if ts is not None:
                            ts_us = int(ts.total_seconds() * 1e6)
                    except Exception:
                        pass

                    self.frame_ready.emit(role, bgr, ts_us)
                    any_data = True

                    if self._hand_tracker is not None and role == "center":
                        cnt = self._hand_skip_counter["center"] + 1
                        if cnt >= self._HAND_INTERVAL:
                            self._hand_skip_counter["center"] = 0
                            try:
                                self.hands_ready.emit(role, self._hand_tracker.process(bgr))
                            except Exception as ex:
                                sys.stderr.write(f"HT {role}: {ex}\n")
                        else:
                            self._hand_skip_counter["center"] = cnt

                if imu_q is not None:
                    try:
                        imu_pkt = imu_q.tryGet()
                    except Exception:
                        imu_pkt = None
                    if imu_pkt is not None:
                        readings = _decode_imu(imu_pkt.packets)
                        if readings:
                            self.imu_ready.emit(readings)
                            any_data = True

                if vio_q is not None:
                    try:
                        vio_pkt = vio_q.tryGet()
                    except Exception:
                        vio_pkt = None
                    if vio_pkt is not None:
                        m = _transform(vio_pkt)
                        if m is not None:
                            self.vio_ready.emit(m)
                            any_data = True

                if any_data:
                    fc += 1
                    fps_fc += 1

                elapsed = time.perf_counter() - fps_t0
                if elapsed >= 1.0:
                    self.pipeline_stats.emit({"fps": fps_fc / elapsed, "frames": fc})
                    fps_fc, fps_t0 = 0, time.perf_counter()

                if not any_data:
                    time.sleep(0.005)

        except Exception as e:
            self.pipeline_error.emit(f"Error: {e}\n{traceback.format_exc()}")
        finally:
            if pip.isRunning():
                pip.stop()
            if self._device:
                self._device.close()
            self.pipeline_stopped.emit()


def _decode_imu(packets):
    result = []
    for p in packets:
        a = p.acceleroMeter
        g = p.gyroscope
        result.append({
            "acc_g": [a.x / 9.80665, a.y / 9.80665, a.z / 9.80665],
            "gyro_dps": [np.degrees(g.x), np.degrees(g.y), np.degrees(g.z)],
            "t_us": int(a.getTimestamp().total_seconds() * 1e6),
        })
    return result


def _transform(tf):
    try:
        t, q = tf.getTranslation(), tf.getQuaternion()
    except Exception:
        return None
    qw, qx, qy, qz = q.qw, q.qx, q.qy, q.qz
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R
    m[:3, 3] = [t.x, t.y, t.z]
    return m
