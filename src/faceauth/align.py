"""Face alignment: a least-squares similarity transform from 5 landmarks to the ArcFace template.

A similarity transform (rotation + uniform scale + translation, 4 degrees of freedom) is the most
a face crop should be warped: it removes in-plane tilt, size and position, but keeps the face's
shape. Recognition networks are trained on crops aligned exactly this way, so feeding them an
un-aligned crop is a train/test mismatch.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import ARCFACE_TEMPLATE

TEMPLATE = np.asarray(ARCFACE_TEMPLATE, dtype=np.float64)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity matrix M minimising sum ||M [src;1] - dst||^2 (Umeyama, 1991).

    Steps: centre both point sets, take the SVD of their cross-covariance to get the optimal
    rotation (with a reflection guard), the scale is the ratio of 'explained' variance, and the
    translation maps the source centroid onto the destination centroid.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # avoid a mirror image
        d[-1] = -1
    rot = u @ np.diag(d) @ vt
    var_s = (s_c ** 2).sum() / len(src)
    scale = (sig * d).sum() / var_s if var_s > 0 else 1.0
    t = mu_d - scale * rot @ mu_s
    return np.hstack([scale * rot, t[:, None]])


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp ``image`` so the 5 landmarks land on the ArcFace template; returns size x size crop."""
    dst = TEMPLATE * (size / 112.0)
    m = umeyama(np.asarray(landmarks, dtype=np.float64).reshape(5, 2), dst)
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def center_crop(image: np.ndarray, box: np.ndarray, size: int = 112, scale: float = 1.0
                ) -> np.ndarray:
    """Un-aligned baseline: square crop around the detector box, resized (used in the ablation)."""
    x, y, w, h = (float(v) for v in box[:4])
    side = max(w, h) * scale
    cx, cy = x + w / 2, y + h / 2
    m = np.array([[size / side, 0, size / 2 - cx * size / side],
                  [0, size / side, size / 2 - cy * size / side]])
    return cv2.warpAffine(image, m, (size, size), borderMode=cv2.BORDER_REPLICATE)
