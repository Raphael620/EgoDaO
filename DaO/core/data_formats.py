"""Shared HumanEgo data format builders.

Eliminates duplication between humanego_recorder.py, converter.py, and convert.py.
"""
import numpy as np

_DEFAULT_FOV_DEG = 72.0


def default_intrinsics(w: int, h: int, fov_deg: float = _DEFAULT_FOV_DEG) -> np.ndarray:
    fx = w / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)


def rotmat_to_rpy_zyx_deg(R: np.ndarray):
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


def compute_slam_frames(transforms: list[np.ndarray], timestamps_us: list[int]):
    n = len(transforms)
    if n == 0:
        return []
    t_world = np.array([m[:3, 3] for m in transforms], dtype=np.float64)
    rpy_deg = np.array([rotmat_to_rpy_zyx_deg(m[:3, :3]) for m in transforms], dtype=np.float64)
    delta_t = t_world - t_world[0]
    delta_rpy = rpy_deg - rpy_deg[0]
    yaws = np.unwrap(np.radians(rpy_deg[:, 2]))
    n_ts = len(timestamps_us)
    frames = []
    for i in range(n):
        ts_ns = int(timestamps_us[i] * 1000) if i < n_ts else 0
        if i > 0 and i < n_ts and (i - 1) < n_ts:
            dt_us = max(timestamps_us[i] - timestamps_us[i - 1], 1)
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


def pack_hand(entries, he_entries=None):
    """Build HumanEgo-format hand dict.

    If he_entries (pre-formatted tracker output) is provided and non-empty,
    returns the first entry directly.  Otherwise falls back to raw landmarks.
    """
    if he_entries:
        for _label, he_dict in he_entries:
            return he_dict
    if not entries:
        return None
    for _label, lms in entries:
        lms_np = np.asarray(lms, dtype=np.float64)
        if lms_np.shape[0] < 21:
            continue
        wrist = lms_np[0].copy()
        pose = np.eye(4)
        pose[:3, 3] = wrist
        return {
            "d2c": None, "c2w": None, "confidence": 0.8, "grasp_state": 0,
            "wrist_pose": pose.tolist(), "palm_pose": pose.tolist(),
            "kpts_3d": lms_np.tolist(), "kpts_2d": lms_np[:, :2].tolist(),
            "joint_angles": {},
            "wrist_pose_raw_world": pose.tolist(),
            "wrist_pose_opt_world": pose.tolist(),
            "wrist_lin_vel_raw_world": [0, 0, 0],
            "wrist_ang_vel_raw_world": [0, 0, 0],
            "wrist_lin_vel_opt_world": [0, 0, 0],
            "wrist_ang_vel_opt_world": [0, 0, 0],
            "index_translation_raw_world": lms_np[8].tolist()[:3],
            "index_translation_opt_world": lms_np[8].tolist()[:3],
            "thumb_translation_raw_world": lms_np[4].tolist()[:3],
            "thumb_translation_opt_world": lms_np[4].tolist()[:3],
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
    return None


def build_training_data(idx: int, ts_ns: int, w: int, h: int,
                        fps: int, K: np.ndarray, c2w: list) -> dict:
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
