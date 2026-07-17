"""Hand tracking — TFLite backend (CPU) for RK3588.

Falls back to TFLite CPU inference when RKNN models are not available for RK3588.
Uses the same palm detection + landmark regression two-stage pipeline.
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


def _generate_anchors():
    """Generate 2016 anchors matching MediaPipe palm detection v0.10 lite."""
    strides = [8, 16, 16, 16]
    input_size = 192
    min_scale = 0.1484375
    max_scale = 0.75
    aspect_ratios = [1.0]
    interpolated_scale_aspect_ratio = 1.0
    anchor_offset_x = 0.5
    anchor_offset_y = 0.5
    reduce_boxes_in_lowest_layer = False
    fixed_anchor_size = True

    def calc_scale(min_s, max_s, idx, total):
        if total == 1:
            return (max_s + min_s) * 0.5
        return min_s + (max_s - min_s) * idx / (total - 1.0)

    anchors = []
    layer_id = 0
    while layer_id < len(strides):
        last_same = layer_id
        reps = []
        while (last_same < len(strides) and
               strides[last_same] == strides[layer_id]):
            scale = calc_scale(min_scale, max_scale, last_same, len(strides))
            if last_same == 0 and reduce_boxes_in_lowest_layer:
                for ar, s in zip([1.0, 2.0, 0.5], [0.1, scale, scale]):
                    reps.append((s, ar))
            else:
                for ar in aspect_ratios:
                    reps.append((scale, ar))
                if interpolated_scale_aspect_ratio > 0.0:
                    nxt = 1.0 if last_same == len(strides) - 1 else calc_scale(
                        min_scale, max_scale, last_same + 1, len(strides))
                    reps.append((np.sqrt(scale * nxt), interpolated_scale_aspect_ratio))
            last_same += 1

        stride = strides[layer_id]
        fh = int(np.ceil(input_size / stride))
        fw = int(np.ceil(input_size / stride))

        for y in range(fh):
            for x in range(fw):
                for s, ar in reps:
                    xc = (x + anchor_offset_x) / fw
                    yc = (y + anchor_offset_y) / fh
                    w = s * np.sqrt(ar) if not fixed_anchor_size else 1.0
                    h = s / np.sqrt(ar) if not fixed_anchor_size else 1.0
                    anchors.append({'x_center': xc, 'y_center': yc,
                                    'width': w, 'height': h})
        layer_id = last_same
    return anchors


_ANCHORS = _generate_anchors()  # pre-compute once


class TFLitePalmDetector:
    """TFLite palm detector (192x192 input, 2016 anchors)."""

    def __init__(self, model_path: str):
        import tflite_runtime.interpreter as tflite
        self.interp = tflite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.in_detail = self.interp.get_input_details()[0]
        self.out_detail = self.interp.get_output_details()
        # Infer num_anchors from model output shape
        self.num_anchors = self.out_detail[0]['shape'][1]
        self.num_coords = 18
        self.score_clip = 100.0
        self.x_scale = 192.0
        self.y_scale = 192.0
        self.w_scale = 192.0
        self.h_scale = 192.0
        self.min_score = 0.5  # lower threshold for fisheye cameras
        self.nms_thresh = 0.3
        self.anchors = _ANCHORS[:self.num_anchors]

    def detect(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        padded, scale, (pl, pt) = self._resize_pad(bgr, (192, 192))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.expand_dims(rgb, axis=0)

        self.interp.set_tensor(self.in_detail['index'], inp)
        self.interp.invoke()
        out_reg = self.interp.get_tensor(self.out_detail[0]['index'])[0]  # (2016, 18)
        out_clf = self.interp.get_tensor(self.out_detail[1]['index'])[0]  # (2016, 1)

        clipped = np.clip(np.asarray(out_clf[:, 0], dtype=np.float64),
                          -self.score_clip, self.score_clip)
        scores = 1.0 / (1.0 + np.exp(-clipped))

        detections = []
        for i in range(self.num_anchors):
            if scores[i] < self.min_score:
                continue
            raw = out_reg[i]
            a = self.anchors[i]
            xc = raw[0] / self.x_scale * a['width'] + a['x_center']
            yc = raw[1] / self.y_scale * a['height'] + a['y_center']
            bw = raw[2] / self.w_scale * a['width']
            bh = raw[3] / self.h_scale * a['height']
            detections.append({
                'xmin': (xc - bw / 2.0) * 192,
                'ymin': (yc - bh / 2.0) * 192,
                'xmax': (xc + bw / 2.0) * 192,
                'ymax': (yc + bh / 2.0) * 192,
                'score': float(scores[i]),
            })

        detections.sort(key=lambda d: d['score'], reverse=True)
        kept = []
        while detections and len(kept) < 2:
            best = detections.pop(0)
            kept.append(best)
            detections = [d for d in detections
                          if self._iou(best, d) < self.nms_thresh]

        for d in kept:
            d['xmin'] = np.clip((d['xmin'] - pl) / scale, 0, w)
            d['ymin'] = np.clip((d['ymin'] - pt) / scale, 0, h)
            d['xmax'] = np.clip((d['xmax'] - pl) / scale, 0, w)
            d['ymax'] = np.clip((d['ymax'] - pt) / scale, 0, h)
        return kept

    @staticmethod
    def _resize_pad(img, target):
        h, w = img.shape[:2]
        scale = min(target[0] / h, target[1] / w)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((target[0], target[1], 3), dtype=np.uint8)
        pt = (target[0] - nh) // 2
        pl = (target[1] - nw) // 2
        padded[pt:pt + nh, pl:pl + nw] = resized
        return padded, scale, (pl, pt)

    @staticmethod
    def _iou(a, b):
        x1 = max(a['xmin'], b['xmin'])
        y1 = max(a['ymin'], b['ymin'])
        x2 = min(a['xmax'], b['xmax'])
        y2 = min(a['ymax'], b['ymax'])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a['xmax'] - a['xmin']) * (a['ymax'] - a['ymin'])
        area_b = (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


class TFLiteLandmarkDetector:
    """TFLite hand landmark detector (224x224 input, outputs 21 xyz keypoints)."""

    def __init__(self, model_path: str):
        import tflite_runtime.interpreter as tflite
        self.interp = tflite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.in_detail = self.interp.get_input_details()[0]
        self.out_detail = self.interp.get_output_details()
        self.input_size = 224

    def predict(self, bgr: np.ndarray, palm_detections: list):
        all_landmarks = []
        all_handedness = []

        for det in palm_detections:
            roi, M = self._extract_roi(bgr, det)
            inp = np.expand_dims(
                roi.astype(np.float32) / 255.0, axis=0)

            self.interp.set_tensor(self.in_detail['index'], inp)
            self.interp.invoke()

            # out[0]=Identity(landmarks 63), out[1]=Identity_1(hand_flag),
            # out[2]=Identity_2(handedness)
            raw = self.interp.get_tensor(self.out_detail[0]['index'])[0]  # (63,)
            landmarks = raw.reshape(21, 3)

            # handedness: output[2] > 0.5 means left hand (C++ convention)
            if len(self.out_detail) >= 3:
                hd_val = self.interp.get_tensor(self.out_detail[2]['index'])
                is_left = float(np.squeeze(hd_val)) > 0.5
            else:
                is_left = landmarks[2, 0] > landmarks[17, 0]
            is_right = not is_left

            # Denormalize from ROI to original image
            Minv = cv2.invertAffineTransform(M)
            kpts_px = np.zeros((21, 3), dtype=np.float32)
            for j in range(21):
                kpts_px[j, 0] = (Minv[0, 0] * landmarks[j, 0] +
                                 Minv[0, 1] * landmarks[j, 1] + Minv[0, 2])
                kpts_px[j, 1] = (Minv[1, 0] * landmarks[j, 0] +
                                 Minv[1, 1] * landmarks[j, 1] + Minv[1, 2])
                kpts_px[j, 2] = landmarks[j, 2]

            all_handedness.append(is_right)
            all_landmarks.append(kpts_px)

        return all_landmarks, all_handedness

    def _extract_roi(self, img, det):
        cx = (det['xmin'] + det['xmax']) / 2.0
        cy = (det['ymin'] + det['ymax']) / 2.0
        box_size = max(det['xmax'] - det['xmin'], det['ymax'] - det['ymin'])
        roi_size = box_size * 2.6
        rx0 = cx - roi_size / 2.0
        ry0 = cy - roi_size / 2.0

        src = np.array([[rx0, ry0],
                        [rx0 + roi_size, ry0],
                        [rx0, ry0 + roi_size]], dtype=np.float32)
        dst = np.array([[0, 0], [self.input_size, 0],
                        [0, self.input_size]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        roi = cv2.warpAffine(img, M, (self.input_size, self.input_size))
        return roi, M


class TFLiteHandTracker:
    """TFLite-based hand tracker (CPU fallback for RK3588).

    Implements the full EgoDaO hand tracker interface.
    """

    def __init__(self, K=None, **kwargs):
        self.palm = TFLitePalmDetector(_get_model_path("hand_detector.tflite"))
        self.landmark = TFLiteLandmarkDetector(
            _get_model_path("hand_landmarks_detector.tflite"))
        self.K = K

    def process(self, bgr, apply_filter=True, compute_metric_3d=False):
        detections = self.palm.detect(bgr)
        pixel_hands = []
        humanego_hands = []

        if not detections:
            return pixel_hands, humanego_hands

        landmarks_list, handedness = self.landmark.predict(bgr, detections)

        for lm, is_right in zip(landmarks_list, handedness):
            label = "Right" if is_right else "Left"
            pixel_hands.append((label, lm.astype(np.float32)))

            if compute_metric_3d and self.K is not None:
                he_dict = self._build_humanego_dict(lm, is_right)
                humanego_hands.append((label, he_dict))

        return pixel_hands, humanego_hands

    @staticmethod
    def _build_humanego_dict(kpts_px, is_right):
        mp_to_aria = [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7,
                      9, 10, 11, 13, 14, 15, 17, 18, 19, -1]
        aria_2d = []
        aria_3d = []
        for idx in mp_to_aria:
            if idx == -1:
                px = (kpts_px[0] + kpts_px[5] + kpts_px[9]) / 3.0
            else:
                px = kpts_px[idx]
            aria_2d.append([float(px[0]), float(px[1])])
            aria_3d.append([float(px[0]), float(px[1]), float(px[2])])

        wrist_pos = [float(kpts_px[0, 0]), float(kpts_px[0, 1]),
                     float(kpts_px[0, 2])]
        pose = [[1, 0, 0, wrist_pos[0]], [0, 1, 0, wrist_pos[1]],
                [0, 0, 1, wrist_pos[2]], [0, 0, 0, 1]]

        return {
            "d2c": None, "c2w": None,
            "confidence": 0.8, "grasp_state": 0,
            "wrist_pose": pose, "palm_pose": pose,
            "kpts_3d": aria_3d, "kpts_2d": aria_2d,
            "joint_angles": {},
            "wrist_pose_raw_world": pose,
            "wrist_pose_opt_world": pose,
            "wrist_lin_vel_raw_world": [0, 0, 0],
            "wrist_ang_vel_raw_world": [0, 0, 0],
            "wrist_lin_vel_opt_world": [0, 0, 0],
            "wrist_ang_vel_opt_world": [0, 0, 0],
            "index_translation_raw_world": aria_3d[1],
            "index_translation_opt_world": aria_3d[1],
            "thumb_translation_raw_world": aria_3d[0],
            "thumb_translation_opt_world": aria_3d[0],
            "midpoint_pose_raw_world": pose,
            "midpoint_pose_opt_world": pose,
            "midpoint_translation_raw_world": wrist_pos,
            "midpoint_orientation_raw_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "midpoint_translation_opt_world": wrist_pos,
            "midpoint_orientation_opt_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "midpoint_lin_vel_raw_world": [0, 0, 0],
            "midpoint_ang_vel_raw_world": [0, 0, 0],
            "midpoint_lin_vel_opt_world": [0, 0, 0],
            "midpoint_ang_vel_opt_world": [0, 0, 0],
            "distance_midpoint2wrist_raw_world": 0.0,
            "distance_midpoint2wrist_opt_world": 0.0,
            "is_right": is_right,
        }

    def close(self):
        pass  # TFLite interpreters are garbage-collected


def create_tflite_tracker(**kwargs):
    """Factory function for EgoDaO's create_hand_tracker protocol."""
    return TFLiteHandTracker(**kwargs)


# ── quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    tracker = create_tflite_tracker(K=None)
    img_path = sys.argv[1] if len(sys.argv) > 1 else "../../sample.png"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read {img_path}")
        sys.exit(1)
    pixel_hands, he_hands = tracker.process(img)
    for label, kpts in pixel_hands:
        print(f"{label}: {kpts.shape}")
        print(f"  wrist=({kpts[0,0]:.0f},{kpts[0,1]:.0f})")
        print(f"  thumb=({kpts[4,0]:.0f},{kpts[4,1]:.0f})")
        print(f"  index=({kpts[8,0]:.0f},{kpts[8,1]:.0f})")
        print(f"  pinky=({kpts[20,0]:.0f},{kpts[20,1]:.0f})")
    tracker.close()
