"""Enrol identities and identify faces: a gallery of embeddings matched by cosine similarity.

Only embeddings are stored (data/gallery/gallery.npz), never the photos. Identification is
*open-set*: the best match is accepted only if its similarity clears a threshold, otherwise the
face is "unknown". ``--benchmark-lfw`` measures this on LFW (see ``lfw_open_set``).

Examples (enrol only people who have consented):
    faceauth-enroll alice photos/alice_*.jpg
    faceauth-identify photo.jpg
    faceauth-identify --benchmark-lfw          # open-set identification on LFW -> reports/lfw/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .align import align_face
from .config import GALLERY_DIR, LFW_DIR, REPORTS_DIR
from .detect import FaceDetector
from .embed import get_embedder, l2_normalize

# Cosine thresholds for ArcFace (w600k_mbf), both measured on LFW and rounded up:
# - 1:1 verification ("is this the claimed person?"): FAR = 1e-3 on the 3,000 impostor pairs at
#   0.234 (reports/lfw/results.json) -> 0.25.
# - 1:N identification ("who is this, if anyone?"): an unknown face is compared with *every*
#   enrolled template, so false alarms grow with gallery size (~ 1 - (1 - FAR)^N). With 1,680
#   identities, 0.25 lets 43 % of strangers match someone; FPIR = 1 % needs 0.343
#   (reports/lfw/open_set.json) -> 0.35.
VERIFY_THRESHOLD = 0.25
IDENTIFY_THRESHOLD = 0.35
DEFAULT_THRESHOLD = IDENTIFY_THRESHOLD


class Gallery:
    def __init__(self, names: list[str] | None = None, embeddings: np.ndarray | None = None):
        self.names = list(names or [])
        self.embeddings = embeddings  # (identities, dim) unit vectors, or None when empty

    def add(self, name: str, embeddings: np.ndarray) -> None:
        """One template per identity: the re-normalised mean of its enrolment embeddings."""
        emb = l2_normalize(np.atleast_2d(np.asarray(embeddings, np.float32)))
        template = l2_normalize(emb.mean(0))
        if name in self.names:
            self.embeddings[self.names.index(name)] = template
        elif self.embeddings is None:
            self.names, self.embeddings = [name], template[None]
        else:
            self.names.append(name)
            self.embeddings = np.vstack([self.embeddings, template])

    def match(self, embedding: np.ndarray, threshold: float = DEFAULT_THRESHOLD
              ) -> tuple[str | None, float]:
        """(name or None, best similarity)."""
        if not self.names:
            return None, -1.0
        sims = self.embeddings @ l2_normalize(np.asarray(embedding, np.float32))
        i = int(np.argmax(sims))
        return (self.names[i] if sims[i] >= threshold else None), float(sims[i])

    def save(self, path: Path = GALLERY_DIR / "gallery.npz") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.array(self.names), embeddings=self.embeddings)

    @classmethod
    def load(cls, path: Path = GALLERY_DIR / "gallery.npz") -> Gallery:
        if not path.exists():
            return cls()
        d = np.load(path)
        return cls(d["names"].tolist(), d["embeddings"])


def embed_images(paths: list[Path], embedder, detector: FaceDetector, central: bool = False
                 ) -> tuple[np.ndarray, list[bool]]:
    """Embed the main face of each image; returns (embeddings, found-a-face flags)."""
    from .verify import central_face, fallback_face

    crops, found = [], []
    for p in tqdm(paths, desc="embed", leave=False):
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)
        faces = detector.detect(bgr)
        face = central_face(faces, bgr.shape) if central else (faces[0] if faces else None)
        found.append(face is not None)
        crops.append(align_face(bgr, (face or fallback_face(bgr.shape)).landmarks))
    out = [embedder(crops[s:s + 64]) for s in range(0, len(crops), 64)]
    return np.concatenate(out) if out else np.zeros((0, embedder.dim), np.float32), found


def lfw_open_set(root: Path = LFW_DIR, model: str = "arcface") -> dict:
    """Open-set identification on LFW.

    Gallery = the first photo of each of the 1,680 people with >= 2 photos. Known probes = their
    other photos; unknown probes = the photos of the 4,069 people with a single photo (never
    enrolled). DIR(t) = known probes whose top match is correct *and* >= t; FPIR(t) = unknown
    probes whose top match is >= t (false alarms).
    """
    people = sorted(p for p in (root / "images").iterdir() if p.is_dir())
    files = {p.name: sorted(p.glob("*.jpg")) for p in people}
    multi = [n for n, f in files.items() if len(f) >= 2]
    single = [n for n, f in files.items() if len(f) == 1]
    paths = [files[n][0] for n in multi] + [f for n in multi for f in files[n][1:]] + \
            [files[n][0] for n in single]
    embedder, detector = get_embedder(model), FaceDetector()
    emb, found = embed_images(paths, embedder, detector, central=True)
    g = emb[:len(multi)]
    known = emb[len(multi):len(multi) + sum(len(files[n]) - 1 for n in multi)]
    known_id = np.array([i for i, n in enumerate(multi) for _ in files[n][1:]])
    unknown = emb[len(multi) + len(known):]
    sk, su = known @ g.T, unknown @ g.T
    top_k, best_k = sk.argmax(1), sk.max(1)
    best_u = su.max(1)
    rank1 = float((top_k == known_id).mean())
    out = {"model": model, "gallery_identities": len(multi), "known_probes": int(len(known)),
           "unknown_probes": int(len(unknown)), "detection_misses": int(len(found) - sum(found)),
           "rank1_closed_set": rank1, "dir_at_fpir": {}}
    for fpir in (0.1, 0.01, 0.001):
        t = float(np.quantile(best_u, 1 - fpir))
        out["dir_at_fpir"][str(fpir)] = {
            "threshold": t, "dir": float(((top_k == known_id) & (best_k >= t)).mean())}
    for label, t in (("verify_threshold", VERIFY_THRESHOLD),
                     ("identify_threshold", IDENTIFY_THRESHOLD)):
        out[label] = {"threshold": t, "dir": float(((top_k == known_id) & (best_k >= t)).mean()),
                      "fpir": float((best_u >= t).mean())}
    return out


def enroll_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Enrol a consenting person from 1+ photos.")
    ap.add_argument("name")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--model", default="arcface")
    args = ap.parse_args(argv)
    emb, found = embed_images(args.images, get_embedder(args.model), FaceDetector())
    if not any(found):
        raise SystemExit("No face found in any image.")
    gallery = Gallery.load()
    gallery.add(args.name, emb[np.array(found)])
    gallery.save()
    print(f"enrolled {args.name} from {sum(found)} photo(s); gallery has {len(gallery.names)}")


def identify_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Identify faces against the enrolled gallery.")
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--benchmark-lfw", action="store_true")
    args = ap.parse_args(argv)
    if args.benchmark_lfw:
        res = lfw_open_set()
        (REPORTS_DIR / "lfw").mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "lfw" / "open_set.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    gallery, embedder, detector = Gallery.load(), get_embedder("arcface"), FaceDetector()
    for p in args.images:
        bgr = cv2.imread(str(p))
        for face in detector.detect(bgr):
            name, sim = gallery.match(embedder([align_face(bgr, face.landmarks)])[0],
                                      args.threshold)
            print(f"{p.name}: {name or 'unknown'} (similarity {sim:.3f}) at {face.box.astype(int)}")


if __name__ == "__main__":
    identify_main()
