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


def _read_hands_vio(raw_dir):
    vios, stamps = [], []
    for fname, parser in [
        ("vio.jsonl", lambda e: (np.array(e["transform"], dtype=np.float64).reshape(4, 4),
                                  e.get("frame_idx", 0))),
    ]:
        p = raw_dir / fname
        if not p.is_file():
            continue
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                m, ts = parser(e)
                vios.append(m)
                stamps.append(ts)

    hands_l, hands_r = [], []
    hf = raw_dir / "hands.jsonl"
    if hf.is_file():
        with open(hf) as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                c = e.get("center", {})
                hands_l.append(_find_hand(c, "l") or [])
                hands_r.append(_find_hand(c, "r") or [])

    return vios, stamps, hands_l, hands_r


def _find_hand(d, prefix):
    for k, v in d.items():
        if k.lower().startswith(prefix):
            return v
    return None


def convert_session(raw_dir: Path, data_root: Path, he_subdir: str = "HumanEgo"):
    mp4 = raw_dir / "center_cam.mp4"
    if not mp4.is_file():
        raise FileNotFoundError(f"center_cam.mp4 not found in {raw_dir}")

    ts = raw_dir.name
    out = data_root / he_subdir / f"mps_{ts}_vrs" / "preprocess" / "all_data"
    os.makedirs(out, exist_ok=True)

    import logging
    log = logging.getLogger("egodao")

    vios, stamps, hands_l, hands_r = _read_hands_vio(raw_dir)
    slam = compute_slam_frames(vios, stamps)
    log.info("Converter: %d VIO, %d hand frames", len(vios), len(hands_l))

    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    K = default_intrinsics(w, h, _FOV_DEG)
    log.info("Converter: %d frames, %dx%d, %d fps", total, w, h, fps)

    idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fd = out / f"{idx:05d}"
        os.makedirs(fd, exist_ok=True)
        cv2.imwrite(str(fd / "rgb.png"), frame)

        nv = len(vios)
        c2w = vios[idx].tolist() if idx < nv else np.eye(4).tolist()
        ts_ns = int(stamps[idx] * 1000) if idx < len(stamps) else 0

        with open(fd / "aria_cam_rgb.json", "w") as f:
            json.dump({
                "idx": idx, "ts": ts_ns, "fov": _FOV_DEG, "h": h, "w": w,
                "k": K.tolist(), "d": [0, 0, 0, 0, 0],
                "c2w": c2w, "c2d": np.eye(4).tolist(), "d2w": np.eye(4).tolist(),
                "rgb_path": f"preprocess/all_data/{idx:05d}/rgb.png", "fps": fps,
            }, f, indent=2)

        sl = slam[idx] if idx < len(slam) else {}
        with open(fd / "aria_slam.json", "w") as f:
            json.dump(sl, f, indent=2)

        with open(fd / "aria_hands.json", "w") as f:
            json.dump({
                "idx": idx, "ts": ts_ns,
                "hand_l": pack_hand(hands_l[idx] if idx < len(hands_l) else []),
                "hand_r": pack_hand(hands_r[idx] if idx < len(hands_r) else []),
            }, f, indent=2)

        with open(fd / "training_data.json", "w") as f:
            json.dump(build_training_data(idx, ts_ns, w, h, fps, K, c2w), f, indent=2)

        idx += 1
        if idx % 30 == 0:
            elapsed = time.time() - t0
            log.debug("Converter: %d/%d (%.0f fps)", idx, total, idx / max(elapsed, 0.1))

    cap.release()
    elapsed = time.time() - t0
    log.info("Converter: done %d frames in %.1fs", idx, elapsed)
