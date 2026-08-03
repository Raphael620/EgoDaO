"""Raw → HumanEgo converter (importable module).

Handles the conversion from a raw recording session directory
(containing center_cam.mp4 + hands.jsonl + vio.jsonl)
to the HumanEgo per-frame directory layout.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from DaO.core.data_formats import (default_intrinsics, compute_slam_frames,
                                    pack_hand, build_training_data)

_FOV_DEG = 72.0


def _read_metadata(raw_dir):
    """Read sparse Raw metadata keyed by center-camera frame index."""
    vio_events = {}
    p = raw_dir / "vio.jsonl"
    if p.is_file():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                idx = max(0, int(entry.get("frame_idx", 0)))
                vio_events[idx] = (
                    np.array(entry["transform"], dtype=np.float64).reshape(4, 4),
                    int(entry.get("timestamp_us", 0)),
                )

    hand_events = {}
    hf = raw_dir / "hands.jsonl"
    if hf.is_file():
        with open(hf, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                idx = max(0, int(entry.get("frame_idx", 0)))
                center = entry.get("center", {})
                left = _find_hand(center, "l")
                right = _find_hand(center, "r")
                hand_events[idx] = (
                    [("Left", left)] if left is not None else [],
                    [("Right", right)] if right is not None else [],
                )

    frame_timestamps = {}
    tf = raw_dir / "frame_timestamps.jsonl"
    if tf.is_file():
        with open(tf, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                frame_timestamps[max(0, int(entry.get("frame_idx", 0)))] = int(
                    entry.get("timestamp_us", 0))

    return vio_events, hand_events, frame_timestamps


def _find_hand(d, prefix):
    for k, v in d.items():
        if k.lower().startswith(prefix):
            return v
    return None


def _expand_metadata(total: int, fps: int, vio_events: dict,
                     hand_events: dict, frame_timestamps: dict):
    """Align sparse VIO/hand samples to every center-camera video frame."""
    interval_us = max(1, round(1_000_000 / max(fps, 1)))
    first_known = next(((i, frame_timestamps[i])
                        for i in sorted(frame_timestamps)
                        if frame_timestamps[i] > 0), (0, 0))
    base_ts = first_known[1] - first_known[0] * interval_us
    timestamps = []
    for idx in range(total):
        ts = frame_timestamps.get(idx, 0)
        if ts <= 0:
            ts = base_ts + idx * interval_us
        timestamps.append(ts)

    vios, hands_l, hands_r = [], [], []
    latest_vio = np.eye(4, dtype=np.float64)
    latest_left, latest_right = [], []
    for idx in range(total):
        if idx in vio_events:
            latest_vio = vio_events[idx][0]
        if idx in hand_events:
            latest_left, latest_right = hand_events[idx]
        vios.append(np.asarray(latest_vio, dtype=np.float64).copy())
        hands_l.append(latest_left)
        hands_r.append(latest_right)
    return vios, timestamps, hands_l, hands_r


def convert_session(raw_dir: Path, data_root: Path, he_subdir: str = "HumanEgo"):
    mp4 = raw_dir / "center_cam.mp4"
    if not mp4.is_file():
        raise FileNotFoundError(f"center_cam.mp4 not found in {raw_dir}")

    ts = raw_dir.name
    out = data_root / he_subdir / f"mps_{ts}_vrs" / "preprocess" / "all_data"
    os.makedirs(out, exist_ok=True)

    import logging
    log = logging.getLogger("egodao")

    vio_events, hand_events, frame_timestamps = _read_metadata(raw_dir)
    log.info("Converter: %d VIO, %d hand samples", len(vio_events), len(hand_events))

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open {mp4}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    K = default_intrinsics(w, h, _FOV_DEG)
    vios, stamps, hands_l, hands_r = _expand_metadata(
        total, fps, vio_events, hand_events, frame_timestamps)
    slam = compute_slam_frames(vios, stamps)
    log.info("Converter: %d frames, %dx%d, %d fps", total, w, h, fps)

    idx = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fd = out / f"{idx:05d}"
            os.makedirs(fd, exist_ok=True)
            cv2.imwrite(str(fd / "rgb.png"), frame)

            c2w = vios[idx].tolist() if idx < len(vios) else np.eye(4).tolist()
            ts_ns = int(stamps[idx] * 1000) if idx < len(stamps) else 0

            with open(fd / "aria_cam_rgb.json", "w", encoding="utf-8") as f:
                json.dump({
                    "idx": idx, "ts": ts_ns, "fov": _FOV_DEG, "h": h, "w": w,
                    "k": K.tolist(), "d": [0, 0, 0, 0, 0],
                    "c2w": c2w, "c2d": np.eye(4).tolist(), "d2w": np.eye(4).tolist(),
                    "rgb_path": f"preprocess/all_data/{idx:05d}/rgb.png", "fps": fps,
                }, f, indent=2)

            sl = slam[idx] if idx < len(slam) else {}
            with open(fd / "aria_slam.json", "w", encoding="utf-8") as f:
                json.dump(sl, f, indent=2)

            with open(fd / "aria_hands.json", "w", encoding="utf-8") as f:
                json.dump({
                    "idx": idx, "ts": ts_ns,
                    "hand_l": pack_hand(hands_l[idx] if idx < len(hands_l) else []),
                    "hand_r": pack_hand(hands_r[idx] if idx < len(hands_r) else []),
                }, f, indent=2)

            with open(fd / "training_data.json", "w", encoding="utf-8") as f:
                json.dump(build_training_data(idx, ts_ns, w, h, fps, K, c2w), f, indent=2)

            idx += 1
            if idx % 30 == 0:
                elapsed = time.time() - t0
                log.debug("Converter: %d/%d (%.0f fps)", idx, total, idx / max(elapsed, 0.1))
    finally:
        cap.release()
    elapsed = time.time() - t0
    log.info("Converter: done %d frames in %.1fs", idx, elapsed)
