"""Hand tracking — RKNN backend for Rockchip NPU (RK3588).

Simple detect+landmark every frame. Stable but runs both models per frame (~31ms).
"""
from __future__ import annotations

import os
import numpy as np
import cv2

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                          "rknn-mediapipe", "models")


def _get_model_path(filename: str) -> str:
    local = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(local):
        return local
    return os.path.abspath(os.path.join(_MODEL_DIR, filename))


def _generate_anchors_192():
    anchors = []
    for stride, n_per_cell in [(8, 2), (16, 6)]:
        grid = 192 // stride
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                for _ in range(n_per_cell):
                    anchors.append([cx, cy, 1.0, 1.0])
    return np.array(anchors, dtype=np.float32)


_ANCHORS = _generate_anchors_192()


class RKNNPalmDetector:
    def __init__(self, model_path: str):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime()
        self.input_size = 192
        self.min_score = 0.55
        self.nms_thresh = 0.3

    def detect(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        padded, scale, (pl, pt) = self._letterbox(bgr, (self.input_size, self.input_size))
        inp = np.expand_dims(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), axis=0)

        outputs = self.rknn.inference(inputs=[inp])
        out0 = np.asarray(outputs[0])
        out1 = np.asarray(outputs[1])

        if out0.shape[-1] == 18:
            reg_raw, cls_raw = out0.reshape(-1, 18), out1.reshape(-1)
        else:
            reg_raw, cls_raw = out1.reshape(-1, 18), out0.reshape(-1)

        scores = 1.0 / (1.0 + np.exp(-np.clip(cls_raw.astype(np.float64), -100, 100)))
        mask = scores > self.min_score
        if not np.any(mask):
            return []

        reg, scr, anc = reg_raw[mask], scores[mask], _ANCHORS[mask]
        cx = reg[:, 0] / self.input_size + anc[:, 0]
        cy = reg[:, 1] / self.input_size + anc[:, 1]
        bw = reg[:, 2] / self.input_size
        bh = reg[:, 3] / self.input_size

        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        cv_boxes = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
        idx = cv2.dnn.NMSBoxes(cv_boxes.tolist(), scr.tolist(), self.min_score, self.nms_thresh)

        dets = []
        if len(idx) > 0:
            best = int(np.array(idx).flatten()[0])
            bx = boxes[best]
            dets.append({
                'xmin': np.clip((bx[0] * self.input_size - pl) / scale, 0, w),
                'ymin': np.clip((bx[1] * self.input_size - pt) / scale, 0, h),
                'xmax': np.clip((bx[2] * self.input_size - pl) / scale, 0, w),
                'ymax': np.clip((bx[3] * self.input_size - pt) / scale, 0, h),
                'score': float(scr[best]),
            })
        return dets

    @staticmethod
    def _letterbox(img, target):
        h, w = img.shape[:2]
        scale = min(target[0] / h, target[1] / w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh))
        top, left = (target[0] - nh) // 2, (target[1] - nw) // 2
        return cv2.copyMakeBorder(resized, top, target[0] - nh - top, left, target[1] - nw - left,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0)), scale, (left, top)

    def close(self):
        try:
            self.rknn.release()
        except Exception:
            pass


class RKNNLandmarkDetector:
    def __init__(self, model_path: str):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime()
        self.input_size = 224

    def predict(self, bgr: np.ndarray, palm_detections: list):
        all_landmarks = []
        all_handedness = []
        for det in palm_detections:
            roi, M = self._extract_roi(bgr, det)
            inp = np.expand_dims(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), axis=0)

            outputs = self.rknn.inference(inputs=[inp])
            raw = None
            for o in outputs:
                arr = np.asarray(o)
                if arr.shape[-1] == 63 and arr.size >= 63:
                    raw = arr.reshape(-1, 3)[:21]
                    break
            if raw is None:
                continue

            is_left = False
            if len(outputs) >= 3:
                is_left = float(np.squeeze(np.asarray(outputs[2]))) > 0.5

            Minv = cv2.invertAffineTransform(M)
            kpts_px = np.zeros((21, 3), dtype=np.float32)
            kpts_px[:, 0] = Minv[0, 0] * raw[:, 0] + Minv[0, 1] * raw[:, 1] + Minv[0, 2]
            kpts_px[:, 1] = Minv[1, 0] * raw[:, 0] + Minv[1, 1] * raw[:, 1] + Minv[1, 2]
            kpts_px[:, 2] = raw[:, 2]

            all_handedness.append(not is_left)
            all_landmarks.append(kpts_px)
        return all_landmarks, all_handedness

    def _extract_roi(self, img, det):
        cx = (det['xmin'] + det['xmax']) / 2.0
        cy = (det['ymin'] + det['ymax']) / 2.0
        side = max(det['xmax'] - det['xmin'], det['ymax'] - det['ymin']) * 2.6
        half = side / 2.0
        src = np.float32([[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half]])
        dst = np.float32([[0, 0], [self.input_size, 0], [0, self.input_size]])
        M = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(img, M, (self.input_size, self.input_size)), M

    def close(self):
        try:
            self.rknn.release()
        except Exception:
            pass


class RKNNHandTracker:
    def __init__(self, K=None, **kwargs):
        self.palm = RKNNPalmDetector(_get_model_path("hand_detector.rknn"))
        self.landmark = RKNNLandmarkDetector(_get_model_path("hand_landmarks_detector.rknn"))
        self.K = K

    def process(self, bgr, apply_filter=True, compute_metric_3d=False):
        detections = self.palm.detect(bgr)
        pixel_hands = []
        humanego_hands = []

        if detections:
            lm_list, hd_list = self.landmark.predict(bgr, detections)
            for lm, is_right in zip(lm_list, hd_list):
                label = "Right" if is_right else "Left"
                pixel_hands.append((label, lm.astype(np.float32)))
                if compute_metric_3d and self.K is not None:
                    humanego_hands.append((label, _build_humanego(lm, is_right)))

        return pixel_hands, humanego_hands

    def close(self):
        self.palm.close()
        self.landmark.close()


def _build_humanego(kpts_px, is_right):
    mp2aria = [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, -1]
    aria2d, aria3d = [], []
    for idx in mp2aria:
        px = (kpts_px[0] + kpts_px[5] + kpts_px[9]) / 3.0 if idx == -1 else kpts_px[idx]
        aria2d.append([float(px[0]), float(px[1])])
        aria3d.append([float(px[0]), float(px[1]), float(px[2])])
    wp = [float(kpts_px[0, 0]), float(kpts_px[0, 1]), float(kpts_px[0, 2])]
    pose = [[1, 0, 0, wp[0]], [0, 1, 0, wp[1]], [0, 0, 1, wp[2]], [0, 0, 0, 1]]
    return {
        "d2c": None, "c2w": None, "confidence": 0.8, "grasp_state": 0,
        "wrist_pose": pose, "palm_pose": pose,
        "kpts_3d": aria3d, "kpts_2d": aria2d, "joint_angles": {},
        "wrist_pose_raw_world": pose, "wrist_pose_opt_world": pose,
        "wrist_lin_vel_raw_world": [0, 0, 0], "wrist_ang_vel_raw_world": [0, 0, 0],
        "wrist_lin_vel_opt_world": [0, 0, 0], "wrist_ang_vel_opt_world": [0, 0, 0],
        "index_translation_raw_world": aria3d[1], "index_translation_opt_world": aria3d[1],
        "thumb_translation_raw_world": aria3d[0], "thumb_translation_opt_world": aria3d[0],
        "midpoint_pose_raw_world": pose, "midpoint_pose_opt_world": pose,
        "midpoint_translation_raw_world": wp,
        "midpoint_orientation_raw_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "midpoint_translation_opt_world": wp,
        "midpoint_orientation_opt_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "midpoint_lin_vel_raw_world": [0, 0, 0], "midpoint_ang_vel_raw_world": [0, 0, 0],
        "midpoint_lin_vel_opt_world": [0, 0, 0], "midpoint_ang_vel_opt_world": [0, 0, 0],
        "distance_midpoint2wrist_raw_world": 0.0, "distance_midpoint2wrist_opt_world": 0.0,
        "is_right": is_right,
    }


def create_rknn_tracker(**kwargs):
    return RKNNHandTracker(**kwargs)
