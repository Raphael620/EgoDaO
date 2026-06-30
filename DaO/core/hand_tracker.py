"""Hand tracking — MediaPipe 0.10+ Tasks API (host-side)."""
from __future__ import annotations

import os, urllib.request
import numpy as np
import cv2

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


class MediaPipeHandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.3):
        _ensure_model()
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as mp_base
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def process(self, bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        h, w = bgr.shape[:2]
        if len(bgr.shape) == 2 or bgr.shape[2] == 1:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = bgr[..., ::-1]
        rgb = np.ascontiguousarray(rgb)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)
        hands = []
        if result.hand_landmarks and result.handedness:
            for lms, hns in zip(result.hand_landmarks, result.handedness):
                label = hns[0].category_name
                arr = np.zeros((21, 3), dtype=np.float32)
                for i, lm in enumerate(lms):
                    arr[i] = [lm.x * w, lm.y * h, lm.z * w]
                hands.append((label, arr))
        return hands

    def close(self):
        self._detector.close()


def create_hand_tracker(backend="mediapipe", **kwargs):
    if backend == "mediapipe":
        try:
            return MediaPipeHandTracker(**kwargs)
        except ImportError:
            print("MediaPipe not installed — hand tracking disabled")
            return None
    raise ValueError(f"Unknown hand tracker backend: {backend}")
