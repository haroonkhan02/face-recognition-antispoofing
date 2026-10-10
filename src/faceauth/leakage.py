"""Do the CelebA-Spoof validation / test faces belong to people seen in training?

The mirror has no subject IDs, so this uses the project's own face recogniser: every face crop is
embedded with ArcFace, and a validation (or test) face counts as a "seen identity" if its best
cosine match among the training faces clears the 1:N threshold. If validation shares identities
with training and test does not, validation scores overstate how the model does on new people.

Example:
    python -m faceauth.leakage           # -> reports/spoof/identity_overlap.json
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from tqdm import tqdm

from .align import align_face
from .config import REPORTS_DIR
from .detect import FaceDetector
from .embed import get_embedder
from .gallery import IDENTIFY_THRESHOLD
from .spoof_data import load_split


def embed_crops(x_rgb: np.ndarray, detector: FaceDetector, embedder) -> tuple[np.ndarray, int]:
    """ArcFace embeddings of the largest face in each crop (rows without a face are dropped)."""
    crops = []
    for img in tqdm(x_rgb, desc="embed", leave=False):
        bgr = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_RGB2BGR)
        face = detector.largest(bgr)
        if face is not None:
            crops.append(align_face(bgr, face.landmarks))
    emb = np.concatenate([embedder(crops[s:s + 64]) for s in range(0, len(crops), 64)])
    return emb, len(x_rgb) - len(crops)


def main(sample: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    detector, embedder = FaceDetector(score_threshold=0.6), get_embedder("arcface")
    x_tr, _ = load_split("train", 1.6)
    train_emb, miss_tr = embed_crops(x_tr, detector, embedder)
    out = {"train_faces": int(len(train_emb)), "train_no_face": miss_tr,
           "threshold": IDENTIFY_THRESHOLD, "splits": {}}
    for split in ("valid", "test"):
        x, _ = load_split(split, 1.6)
        idx = np.sort(rng.choice(len(x), min(sample, len(x)), replace=False))
        emb, miss = embed_crops(x[idx], detector, embedder)
        best = (emb @ train_emb.T).max(1)
        out["splits"][split] = {
            "sampled": int(len(idx)), "no_face": miss,
            "seen_identity_share": float((best >= IDENTIFY_THRESHOLD).mean()),
            "median_best_similarity": float(np.median(best)),
            "share_above_0.6": float((best >= 0.6).mean())}
        print(split, out["splits"][split])
    (REPORTS_DIR / "spoof").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "spoof" / "identity_overlap.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
