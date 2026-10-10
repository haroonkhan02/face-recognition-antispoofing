"""Face embeddings: map an aligned 112x112 face to a unit-length feature vector.

Two pretrained recognisers are compared:
- ArcFace MobileFaceNet (InsightFace ``w600k_mbf``): 512-d, trained on WebFace600K with the
  additive angular margin (ArcFace) loss. ONNX, run with onnxruntime.
- SFace (OpenCV Zoo): 128-d, MobileFaceNet-style network trained with a sigmoid-constrained
  hypersphere loss. Run with OpenCV's ``FaceRecognizerSF``.
Both output vectors that are L2-normalised here, so cosine similarity = dot product.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ARCFACE_FILE, MODELS_DIR, SFACE


def l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of (…, d) arrays (inputs need not be normalised)."""
    return (l2_normalize(a) * l2_normalize(b)).sum(-1)


class ArcFaceEmbedder:
    name = "arcface"
    dim = 512

    def __init__(self, model: Path = MODELS_DIR / ARCFACE_FILE):
        import onnxruntime as ort

        if not Path(model).exists():
            raise FileNotFoundError(f"{model} missing; run `faceauth-download models`.")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # the ONNX graph declares batch 1 but runs batched fine
        self.session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        """Aligned 112x112 BGR crops -> (N, 512) unit vectors."""
        blob = cv2.dnn.blobFromImages(faces_bgr, 1 / 127.5, (112, 112), (127.5,) * 3,
                                      swapRB=True)
        return l2_normalize(self.session.run(None, {self.input: blob})[0])


class SFaceEmbedder:
    name = "sface"
    dim = 128

    def __init__(self, model: Path = MODELS_DIR / SFACE[1]):
        if not Path(model).exists():
            raise FileNotFoundError(f"{model} missing; run `faceauth-download models`.")
        self.net = cv2.FaceRecognizerSF.create(str(model), "")

    def __call__(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        feats = [self.net.feature(np.ascontiguousarray(f)).reshape(-1) for f in faces_bgr]
        return l2_normalize(np.stack(feats).astype(np.float32))


EMBEDDERS = {"arcface": ArcFaceEmbedder, "sface": SFaceEmbedder}


def get_embedder(name: str):
    if name not in EMBEDDERS:
        raise ValueError(f"Unknown embedder {name!r}; choose from {sorted(EMBEDDERS)}")
    return EMBEDDERS[name]()
