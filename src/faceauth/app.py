"""Webcam access-control demo: detect -> anti-spoof -> head-turn challenge -> identify.

A face is only identified after (1) the passive anti-spoofing CNN scores it as live on a run of
frames and (2) the randomised head-turn challenge is passed. Keys: q = quit, r = restart.
Needs a camera, so it's never run in CI; ``--image`` runs one still frame through the same
pipeline (useful to check the install).

Examples:
    faceauth-enroll me selfie1.jpg selfie2.jpg     # only people who consent
    faceauth-webcam
    faceauth-webcam --image photo.jpg
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from .align import align_face
from .config import MODELS_DIR, SpoofConfig
from .detect import Face, FaceDetector
from .embed import get_embedder
from .gallery import DEFAULT_THRESHOLD, Gallery
from .liveness import HeadTurnChallenge
from .spoof_data import square_crop
from .train_spoof import load_spoof_model, pick_device, predict_scores, run_name


class SpoofScorer:
    """Runs the anti-spoofing CNN on a face box from a live frame (same cropping as training)."""

    def __init__(self, scale: float = 1.6, device=None):
        cfg = SpoofConfig(crop_scale=scale)
        path = MODELS_DIR / f"{run_name(cfg)}.pt"
        self.device = device or pick_device()
        self.model, ckpt = load_spoof_model(path, self.device)
        self.threshold = ckpt["threshold"]
        self.scale, self.size = scale, ckpt["config"]["image_size"]

    def __call__(self, bgr: np.ndarray, face: Face) -> float:
        from PIL import Image

        x, y, w, h = face.box
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        crop = square_crop(rgb, [x, y, x + w, y + h], self.scale, self.size)
        with torch.no_grad():
            return float(predict_scores(self.model, crop[None], self.device)[0])


def process_frame(bgr, detector, spoof, embedder, gallery, challenge, history, t, threshold):
    """One step of the pipeline; returns (face, status text, colour)."""
    face = detector.largest(bgr)
    challenge.update(face.landmarks if face is not None else None, t)
    if face is None:
        return None, "no face", (0, 165, 255)
    history.append(spoof(bgr, face))
    attack = np.mean(history) >= spoof.threshold
    if attack:
        return face, f"SPOOF? attack score {np.mean(history):.2f}", (0, 0, 255)
    if challenge.state != "passed":
        return face, challenge.prompt, (0, 200, 255)
    name, sim = gallery.match(embedder([align_face(bgr, face.landmarks)])[0], threshold)
    return face, f"{name or 'unknown'} ({sim:.2f})", (0, 200, 0) if name else (0, 200, 255)


def draw(bgr, face, text, colour):
    if face is not None:
        x, y, w, h = face.box.astype(int)
        cv2.rectangle(bgr, (x, y), (x + w, y + h), colour, 2)
        for px, py in face.landmarks.astype(int):
            cv2.circle(bgr, (px, py), 2, (255, 255, 0), -1)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(bgr, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return bgr


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", type=Path, help="run one still image instead of the camera")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args(argv)
    detector, spoof = FaceDetector(), SpoofScorer()
    embedder, gallery = get_embedder("arcface"), Gallery.load()
    challenge, history = HeadTurnChallenge(), deque(maxlen=8)
    if args.image:
        bgr = cv2.imread(str(args.image))
        challenge.state = "passed"  # a still image can't do the challenge; report the rest
        face, text, colour = process_frame(bgr, detector, spoof, embedder, gallery, challenge,
                                           history, 0.0, args.threshold)
        print(text)
        out = args.image.with_name(args.image.stem + "_faceauth.jpg")
        cv2.imwrite(str(out), draw(bgr, face, text, colour))
        print(f"wrote {out}")
        return
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Camera not available (grant camera permission to your terminal).")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view: "left" = the user's left on screen
        face, text, colour = process_frame(frame, detector, spoof, embedder, gallery, challenge,
                                           history, time.monotonic(), args.threshold)
        cv2.imshow("faceauth", draw(frame, face, text, colour))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            challenge, history = HeadTurnChallenge(), deque(maxlen=8)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
