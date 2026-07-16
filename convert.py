"""Raw → HumanEgo data converter.

Reads a Raw recording session (center_cam.mp4 + imu.csv + hands.jsonl +
vio.jsonl) and produces HumanEgo-compatible per-frame data.

Usage:
  python convert.py                           # Convert latest raw session
  python convert.py Data/Raw/20260711_120000  # Convert specific session
  python convert.py --data-root D:/Recordings # Use custom data root
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Default data root (matches config)
_DATA_ROOT = Path(os.getcwd()) / "Data"
_RAW_SUBDIR = "Raw"
_HE_SUBDIR = "HumanEgo"
_FOV_DEG = 72.0


def _find_latest_raw(data_root: Path) -> Path | None:
    raw_dir = data_root / _RAW_SUBDIR
    if not raw_dir.is_dir():
        return None
    dirs = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return dirs[0] if dirs else None


def _default_intrinsics(w: int, h: int, fov: float) -> np.ndarray:
    fx = w / (2.0 * np.tan(np.radians(fov / 2.0)))
    return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)


def _read_imuvio(raw_dir: Path) -> tuple[list, list, list, list]:
    """Read imu.csv, vio.jsonl, hands.jsonl.  Returns (vios, timestamps, hands_left, hands_right)."""
    vios = []
    timestamps = []

    # Read VIO
    vio_file = raw_dir / "vio.jsonl"
    if vio_file.is_file():
        with open(vio_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                vios.append(np.array(entry["transform"], dtype=np.float64).reshape(4, 4))
                timestamps.append(entry.get("frame_idx", 0))

    # Read hands
    hands_left = []
    hands_right = []
    hands_file = raw_dir / "hands.jsonl"
    if hands_file.is_file():
        with open(hands_file, "r") as f:
            for _line in f:
                line = _line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # hands dict is {role: {label: [[21,3], ...]}}
                center_data = entry.get("center", {})
                left_raw = center_data.get("Left", None)
                right_raw = center_data.get("Right", None)
                # Also check for lower-case variants
                if left_raw is None:
                    for k in center_data:
                        if k.lower().startswith("l"):
                            left_raw = center_data[k]
                            break
                if right_raw is None:
                    for k in center_data:
                        if k.lower().startswith("r") and k != list(center_data.keys())[0] if left_raw else True:
                            right_raw = center_data[k]
                            break
                hands_left.append(left_raw if left_raw else [])
                hands_right.append(right_raw if right_raw else [])

    return vios, timestamps, hands_left, hands_right


def _compute_slam_frames(transforms, timestamps):
    """Compute SLAM frames from VIO transforms (matching humanego_recorder.py)."""
    n = len(transforms)
    if n == 0:
        return []
    t_world = np.array([m[:3, 3] for m in transforms], dtype=np.float64)
    rpy_deg = np.array([[_rotmat_to_rpy_zyx_deg(m[:3, :3])] for m in transforms],
                       dtype=np.float64)
    delta_t = t_world - t_world[0]
    delta_rpy = rpy_deg - rpy_deg[0]
    yaws = np.unwrap(np.radians(rpy_deg[:, 2]))
    n_ts = len(timestamps)
    frames = []
    for i in range(n):
        ts_ns = int(timestamps[i] * 1000) if i < n_ts else 0
        if i > 0 and i < n_ts and (i - 1) < n_ts:
            dt_us = max(timestamps[i] - timestamps[i - 1], 1)
            dt = dt_us / 1e6
        else:
            dt = 1.0 / 30.0
        v = float(np.linalg.norm(t_world[i] - t_world[i - 1]) / max(dt, 1e-6)) if i > 0 else 0.0
        w = float(abs(yaws[i] - yaws[i - 1]) / max(dt, 1e-6)) if i > 0 else 0.0
        frames.append({
            "idx": i, "ts": ts_ns,
            "t_world": t_world[i].tolist(),
            "rpy_deg": rpy_deg[i].tolist(),
            "delta_t_world": delta_t[i].tolist(),
            "delta_rpy_deg": delta_rpy[i].tolist(),
            "linear_speed_mps": v, "angular_speed_rps": w,
            "yaw_unwrapped_deg": float(np.degrees(yaws[i])),
        })
    return frames


def _rotmat_to_rpy_zyx_deg(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def _pack_hand(entries):
    """Build hand dict from raw data entries."""
    if not entries:
        return None
    lms = np.asarray(entries, dtype=np.float64)
    if lms.shape[0] < 21:
        return None
    wrist = lms[0].copy()
    pose = np.eye(4); pose[:3, 3] = wrist
    return {
        "d2c": None, "c2w": None, "confidence": 0.8, "grasp_state": 0,
        "wrist_pose": pose.tolist(), "palm_pose": pose.tolist(),
        "kpts_3d": lms.tolist(), "kpts_2d": lms[:, :2].tolist(),
        "joint_angles": {},
        "wrist_pose_raw_world": pose.tolist(),
        "wrist_pose_opt_world": pose.tolist(),
        "wrist_lin_vel_raw_world": [0, 0, 0],
        "wrist_ang_vel_raw_world": [0, 0, 0],
        "wrist_lin_vel_opt_world": [0, 0, 0],
        "wrist_ang_vel_opt_world": [0, 0, 0],
        "index_translation_raw_world": lms[8].tolist()[:3],
        "index_translation_opt_world": lms[8].tolist()[:3],
        "thumb_translation_raw_world": lms[4].tolist()[:3],
        "thumb_translation_opt_world": lms[4].tolist()[:3],
        "midpoint_pose_raw_world": pose.tolist(),
        "midpoint_pose_opt_world": pose.tolist(),
        "midpoint_translation_raw_world": wrist.tolist()[:3],
        "midpoint_orientation_raw_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "midpoint_translation_opt_world": wrist.tolist()[:3],
        "midpoint_orientation_opt_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "midpoint_lin_vel_raw_world": [0, 0, 0],
        "midpoint_ang_vel_raw_world": [0, 0, 0],
        "midpoint_lin_vel_opt_world": [0, 0, 0],
        "midpoint_ang_vel_opt_world": [0, 0, 0],
        "distance_midpoint2wrist_raw_world": 0.0,
        "distance_midpoint2wrist_opt_world": 0.0,
    }


def _build_training_data(idx, ts_ns, w, h, fps, K, c2w):
    return {
        "metadata": {
            "idx": idx, "ts": ts_ns, "w": w, "h": h, "fps": fps,
            "k": K.tolist(), "c2w": c2w, "anchor_key": "obj1",
            "is_finished": 0.0,
            "world_transforms": {
                "cam0": np.eye(4).tolist(),
                "virtual_static_anchor": np.eye(4).tolist(),
            },
        },
        "obs": {
            "rgb_path": f"preprocess/all_data/{idx:05d}/rgb.png",
            "mask_arm_path": "", "mask_obj_path": "",
            "rgb_WArmObjKpts_path": "", "rgb_WoArm_path": "",
            "rgb_WoArm_WArmObjKpts_path": "",
        },
        "entities": {"hands": {}, "objects": {}},
    }


def convert(raw_dir: Path, data_root: Path | None = None):
    if data_root is None:
        data_root = _DATA_ROOT
    mp4_path = raw_dir / "center_cam.mp4"
    if not mp4_path.is_file():
        print(f"ERROR: center_cam.mp4 not found in {raw_dir}")
        return

    # Create output dir
    ts = raw_dir.name
    out_dir = data_root / _HE_SUBDIR / f"mps_{ts}_vrs" / "preprocess" / "all_data"
    os.makedirs(out_dir, exist_ok=True)

    # Read metadata
    vios, timestamps, hands_left, hands_right = _read_imuvio(raw_dir)
    slam_frames = _compute_slam_frames(vios, timestamps)

    # Read video
    cap = cv2.VideoCapture(str(mp4_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    K = _default_intrinsics(w, h, _FOV_DEG)
    print(f"Video: {total_frames} frames, {w}x{h}, {fps} fps")

    idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dir = out_dir / f"{idx:05d}"
        os.makedirs(frame_dir, exist_ok=True)

        cv2.imwrite(str(frame_dir / "rgb.png"), frame)

        n_v = len(vios)
        c2w = vios[idx].tolist() if idx < n_v else np.eye(4).tolist()
        ts_ns = int(timestamps[idx] * 1000) if idx < len(timestamps) else 0

        # aria_cam_rgb.json
        json.dump({
            "idx": idx, "ts": ts_ns, "fov": _FOV_DEG, "h": h, "w": w,
            "k": K.tolist(), "d": [0, 0, 0, 0, 0],
            "c2w": c2w, "c2d": np.eye(4).tolist(), "d2w": np.eye(4).tolist(),
            "rgb_path": f"preprocess/all_data/{idx:05d}/rgb.png", "fps": fps,
        }, open(frame_dir / "aria_cam_rgb.json", "w"), indent=2)

        # aria_slam.json
        slam = slam_frames[idx] if idx < len(slam_frames) else {
            "idx": idx, "ts": 0,
            "t_world": [0, 0, 0], "rpy_deg": [0, 0, 0],
            "delta_t_world": [0, 0, 0], "delta_rpy_deg": [0, 0, 0],
            "linear_speed_mps": 0, "angular_speed_rps": 0,
            "yaw_unwrapped_deg": 0,
        }
        json.dump(slam, open(frame_dir / "aria_slam.json", "w"), indent=2)

        # aria_hands.json
        hand_l = _pack_hand(hands_left[idx] if idx < len(hands_left) else [])
        hand_r = _pack_hand(hands_right[idx] if idx < len(hands_right) else [])
        json.dump({
            "idx": idx, "ts": ts_ns, "hand_l": hand_l, "hand_r": hand_r,
        }, open(frame_dir / "aria_hands.json", "w"), indent=2)

        # training_data.json
        json.dump(_build_training_data(idx, ts_ns, w, h, fps, K, c2w),
                  open(frame_dir / "training_data.json", "w"), indent=2)

        idx += 1
        if idx % 30 == 0:
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 0.1)
            print(f"\r  Frame {idx}/{total_frames} ({rate:.0f} fps)...", end="", flush=True)

    cap.release()
    elapsed = time.time() - t0
    print(f"\rDone: {idx} frames in {elapsed:.1f}s ({idx / max(elapsed, 0.1):.0f} fps)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert Raw data to HumanEgo format")
    parser.add_argument("session", nargs="?", help="Path to Raw session directory")
    parser.add_argument("--data-root", default=None, help="Data root directory")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else _DATA_ROOT
    raw_dir = Path(args.session) if args.session else _find_latest_raw(data_root)

    if raw_dir is None:
        print("No Raw sessions found. Specify a path as argument.")
        sys.exit(1)

    print(f"Converting: {raw_dir}")
    convert(raw_dir, data_root)


if __name__ == "__main__":
    main()
