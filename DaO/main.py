"""Ego Daq-O V0.2.3 — Ego 数据采集与实时处理系统.

Usage:
  python -m DaO.main                     # GUI mode (default)
  python -m DaO.main --no-gui            # Headless recording mode
  python -m DaO.main --config my.json    # Use custom config file
"""
from __future__ import annotations

import json
import sys
import os
import time
import threading
from pathlib import Path

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


# ── config loading ───────────────────────────────────────────────

def _load_config() -> dict:
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
    if config_path is None or not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_config(cfg_obj, cfg: dict):
    rec = cfg.get("recording", {})
    for key in ("data_root", "raw_subdir", "humanego_subdir",
                "video_codec", "video_backend", "hw_encoder"):
        if key in rec:
            setattr(cfg_obj.recording, key,
                    Path(rec[key]) if key == "data_root" else rec[key])
    for key in ("enable_raw", "enable_humanego", "disk_keep_days"):
        if key in rec:
            setattr(cfg_obj.recording, key, rec[key])

    cam = cfg.get("camera", {})
    for key in ("resolution", "fps"):
        if key in cam:
            setattr(cfg_obj.camera, key,
                    tuple(cam[key]) if key == "resolution" else cam[key])

    imu = cfg.get("imu", {})
    for key in ("accel_rate_hz", "gyro_rate_hz", "batch_threshold",
                 "max_batch_reports"):
        if key in imu:
            setattr(cfg_obj.imu, key, imu[key])

    app = cfg.get("app", {})
    for key in ("enable_vio", "enable_hand_tracking", "hand_tracker_backend"):
        if key in app:
            setattr(cfg_obj, key, app[key])


def _has_arg(name: str) -> bool:
    return name in sys.argv[1:]


# ── disk cleanup ─────────────────────────────────────────────────

def _cleanup_old_data(config):
    """Keep only the most recent N distinct-date folders under Raw/.
    A 'date' is extracted from the folder name prefix 'YYYYMMDD'."""
    keep = config.recording.disk_keep_days
    if keep <= 0:
        return
    raw_root = config.recording.data_root / config.recording.raw_subdir
    if not raw_root.is_dir():
        return

    import re, shutil
    # Collect folders with date prefix
    dated = []
    for entry in raw_root.iterdir():
        if not entry.is_dir():
            continue
        m = re.match(r"^(\d{8})", entry.name)
        if not m:
            continue
        dated.append((m.group(1), entry))

    # Group by date string, keep entries sorted newest first
    dates = sorted(set(d for d, _ in dated), reverse=True)
    if len(dates) <= keep:
        return

    old_dates = set(dates[keep:])
    for date_str, entry in dated:
        if date_str in old_dates:
            try:
                shutil.rmtree(entry, ignore_errors=True)
                _logger.info(f"Cleanup: removed {entry.name}")
            except Exception as e:
                _logger.warning(f"Cleanup: failed to remove {entry.name}: {e}")

_logger = None


# ── headless recording entry ─────────────────────────────────────

def _run_headless(config):
    """No-GUI recording loop.  Ctrl+C to stop; saves data on exit."""
    # Use a lightweight event queue (NOT QApplication) to avoid any GUI.
    # CaptureWorker's Qt signals are bridged via direct callbacks in a
    # background thread.
    import queue
    import traceback

    from DaO.core.recorder import DataRecorder
    from DaO.core.humanego_recorder import HumanEgoRecorder

    # ── create camera pipeline directly (no Qt signals) ──
    raw_rec = [None]
    he_rec = [None]
    frame_count = [0]
    q = queue.Queue(maxsize=500)

    # Camera thread: polls OAK devices, calls callbacks directly
    def _camera_loop():
        try:
            import depthai as dai
            from DaO.core.pipeline import create_pipeline
            from DaO.core.hand_tracker import create_hand_tracker
            from DaO.core.capture_worker import (_decode_imu, _transform,
                                                  _get_center_camera_intrinsics)

            infos = dai.Device.getAllConnectedDevices()
            if not infos:
                q.put(("error", "No OAK device"))
                return
            device = dai.Device(infos[0].getDeviceId())
            pip, cam_q, imu_q, vio_q = create_pipeline(device, config)

            hand_tracker = None
            if config.enable_hand_tracking:
                try:
                    K = _get_center_camera_intrinsics(device)
                    hand_tracker = create_hand_tracker(config.hand_tracker_backend, K=K)
                except Exception:
                    pass

            pip.start()
            q.put(("started", None))

            fc = 0
            fps_fc = 0
            fps_t0 = time.time()

            while pip.isRunning() and not stop_event.is_set():
                any_data = False
                for role, cq in cam_q.items():
                    try:
                        pkt = cq.tryGet()
                    except Exception:
                        continue
                    if pkt is None:
                        continue
                    bgr = pkt.getCvFrame()
                    if bgr is None or bgr.size == 0:
                        continue

                    q.put(("frame", (role, bgr)))
                    any_data = True
                    fc += 1
                    if role == "center":
                        fps_fc += 1

                    if hand_tracker is not None and role == "center":
                        try:
                            pix, heh = hand_tracker.process(bgr, apply_filter=True, compute_metric_3d=True)
                            if pix:
                                q.put(("hands", (role, pix)))
                            if heh:
                                q.put(("hands_he", (role, heh)))
                        except Exception:
                            pass

                if imu_q is not None:
                    try:
                        pkt = imu_q.tryGet()
                        if pkt is not None:
                            readings = _decode_imu(pkt.packets)
                            if readings:
                                q.put(("imu", readings))
                                any_data = True
                    except Exception:
                        pass

                if vio_q is not None:
                    try:
                        pkt = vio_q.tryGet()
                        if pkt is not None:
                            m = _transform(pkt)
                            if m is not None:
                                q.put(("vio", m))
                                any_data = True
                    except Exception:
                        pass

                elapsed = time.time() - fps_t0
                if elapsed >= 1.0:
                    q.put(("stats", {"fps": fps_fc / elapsed, "frames": fc}))
                    fps_fc = 0
                    fps_t0 = time.time()

                if not any_data:
                    time.sleep(0.005)

            pip.stop()
            device.close()
            if hand_tracker:
                hand_tracker.close()
            q.put(("stopped", None))
        except Exception as e:
            q.put(("error", str(e) + "\n" + traceback.format_exc()))
    stop_event = threading.Event()

    # ── record toggle ──
    def _toggle_record_on():
        if raw_rec[0] is not None:
            return
        if config.recording.enable_raw:
            r = DataRecorder(config)
            r.start()
            raw_rec[0] = r
            _logger.info("Headless: raw recording started (session=%s)", r._session_dir)
        if config.recording.enable_humanego:
            r2 = HumanEgoRecorder(config)
            r2.start()
            he_rec[0] = r2
            _logger.info("Headless: HumanEgo recording started")

    def _toggle_record_off():
        try:
            if he_rec[0]:
                center_mp4 = None
                if raw_rec[0] and hasattr(raw_rec[0], '_session_dir') and raw_rec[0]._session_dir:
                    center_mp4 = str(raw_rec[0]._session_dir / "center_cam.mp4")
                if center_mp4:
                    try:
                        he_rec[0].set_mp4_source(center_mp4)
                    except Exception:
                        pass
                he_rec[0].stop()
                he_rec[0] = None
                _logger.info("Headless: HumanEgo recording saved")
        except Exception as e:
            _logger.error("Headless: HE stop failed: %s", e)
        try:
            if raw_rec[0]:
                raw_rec[0].stop()
                raw_rec[0] = None
                _logger.info("Headless: raw recording saved")
        except Exception as e:
            _logger.error("Headless: raw stop failed: %s", e)

    # ── socket command listener ──
    def _cmd_listener():
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(2.0)
        try:
            sock.bind(("127.0.0.1", 9876))
            sock.listen(1)
        except OSError:
            _logger.warning("Headless: socket 9876 in use")
            sock.close()
            return
        _logger.info("Headless: cmd socket on 127.0.0.1:9876")
        while not stop_event.is_set():
            try:
                conn, _ = sock.accept()
                data = conn.recv(1024).decode().strip().lower()
                if data == "start":
                    q.put(("toggle_rec", "on"))
                elif data == "stop":
                    q.put(("toggle_rec", "off"))
                elif data == "quit":
                    stop_event.set()
                conn.sendall(b"ok\n")
                conn.close()
            except socket.timeout:
                continue
            except OSError:
                break
        sock.close()

    cam_thread = threading.Thread(target=_camera_loop, daemon=True)
    cam_thread.start()

    cmd_thread = threading.Thread(target=_cmd_listener, daemon=True)
    cmd_thread.start()

    # Do NOT auto-start — wait for Ctrl+Q hotkey

    # ── graceful shutdown on SIGTERM / Ctrl+C / taskkill ──
    import signal as _signal
    def _graceful(signum, frame):
        if not stop_event.is_set():
            stop_event.set()
    for sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(sig, _graceful)
        except (ValueError, OSError):
            pass  # not available in sub-interpreter or on Windows

    # ── global hotkey (Ctrl+Q) ──
    try:
        import keyboard
        def _hotkey_cb():
            if raw_rec[0] is not None:
                _toggle_record_off()
            else:
                _toggle_record_on()
        keyboard.add_hotkey("ctrl+q", _hotkey_cb, suppress=False)
        _logger.info("Headless: Ctrl+Q hotkey registered")
    except Exception as e:
        _logger.warning("Headless: hotkey registration failed: %s", e)

    # ── main event loop ──
    _logger.info("Headless: running (Ctrl+Q to toggle recording, Ctrl+C in terminal to exit)")
    try:
        while not stop_event.is_set():
            try:
                msg = q.get(timeout=0.5)
            except queue.Empty:
                continue

            mtype, mdata = msg
            if mtype == "error":
                _logger.error("Headless: %s", mdata)
                stop_event.set()
            elif mtype == "stopped":
                _logger.info("Headless: pipeline stopped")
                stop_event.set()
            elif mtype == "frame":
                role, bgr = mdata
                frame_count[0] += 1
                r = raw_rec[0]
                if r is not None and r.is_recording:
                    r.write_frame(role, bgr)
                h = he_rec[0]
                if h is not None and h.is_recording and role == "center":
                    h.write_frame_rgb(bgr, frame_count[0])
            elif mtype == "hands":
                role, hands = mdata
                r = raw_rec[0]
                if r is not None and r.is_recording:
                    d = {}
                    for lbl, lms in hands:
                        d[lbl] = lms.tolist() if hasattr(lms, "tolist") else lms
                    r.write_hands({role: d})
                h = he_rec[0]
                if h is not None and h.is_recording:
                    h.write_hands(role, hands)
            elif mtype == "hands_he":
                role, he_hands = mdata
                h = he_rec[0]
                if h is not None and h.is_recording:
                    h.write_hands_humanego(role, he_hands)
            elif mtype == "imu":
                r = raw_rec[0]
                if r is not None and r.is_recording:
                    r.write_imu(mdata)
            elif mtype == "vio":
                r = raw_rec[0]
                if r is not None and r.is_recording:
                    r.write_vio(mdata)
                h = he_rec[0]
                if h is not None and h.is_recording:
                    h.write_vio(mdata)
            elif mtype == "stats":
                if frame_count[0] % 150 == 0:
                    _logger.debug("FPS=%.1f", mdata.get("fps", 0))
            elif mtype == "toggle_rec":
                if mdata == "on":
                    _toggle_record_on()
                else:
                    _toggle_record_off()
    except KeyboardInterrupt:
        _logger.info("Headless: interrupted")
    finally:
        stop_event.set()
        _toggle_record_off()
        cam_thread.join(timeout=5.0)
        time.sleep(1.0)  # let OS flush disk buffers
        _logger.info("Headless: done")


# ── main ─────────────────────────────────────────────────────────

def main():
    global _logger
    from DaO.core.logger import setup_logging
    _logger = setup_logging()

    cfg = _load_config()
    from DaO.config import AppConfig
    app_config = AppConfig()
    if cfg:
        _apply_config(app_config, cfg)

    _logger.info(f"EgoDaO starting (data_root={app_config.recording.data_root})")

    # Auto-clean old data
    _cleanup_old_data(app_config)

    if _has_arg("--no-gui"):
        _run_headless(app_config)
        return

    # ── GUI mode ──
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("Ego Daq-O")
    app.setApplicationVersion("0.2.1")

    from DaO.ui.main_window import MainWindow
    window = MainWindow(app_config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
