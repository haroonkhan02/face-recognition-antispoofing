"""Face detection with OpenCV's YuNet: boxes, 5 landmarks and a confidence per face."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR, YUNET


@dataclass
class Face:
    box: np.ndarray  # x, y, w, h (pixels)
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose tip, right / left mouth corner
    score: float

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])


class FaceDetector:
    """Thin wrapper around ``cv2.FaceDetectorYN`` (a 75k-parameter CNN, ~1 ms per frame)."""

    def __init__(self, model: Path = MODELS_DIR / YUNET[1], score_threshold: float = 0.7,
                 nms_threshold: float = 0.3):
        if not Path(model).exists():
            raise FileNotFoundError(f"{model} missing; run `faceauth-download models`.")
        self.net = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score_threshold,
                                             nms_threshold, 5000)

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        self.net.setInputSize((w, h))
        _, raw = self.net.detect(bgr)
        if raw is None:
            return []
        faces = [Face(r[:4].copy(), r[4:14].reshape(5, 2).copy(), float(r[14])) for r in raw]
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def largest(self, bgr: np.ndarray) -> Face | None:
        faces = self.detect(bgr)
        return faces[0] if faces else None
