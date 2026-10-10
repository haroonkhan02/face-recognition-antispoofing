"""LFW face verification: "are these two photos the same person?" on the standard 6,000 pairs.

Protocol (Huang et al., 2007, 'unrestricted with labeled outside data'): 10 folds of 300 genuine
+ 300 impostor pairs. For each fold the decision threshold is chosen on the other 9 folds and
accuracy is measured on the held-out fold; we report mean ± std over the 10 folds. ROC, EER and
TAR@FAR are computed on all 6,000 pair scores.

Runs every (embedder, crop) combination: ArcFace / SFace x 5-point aligned / un-aligned crop.

Example:
    faceauth-lfw                        # all four combinations -> reports/lfw/
    faceauth-lfw --models arcface --modes aligned
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .align import align_face, center_crop
from .config import ARCFACE_TEMPLATE, LFW_DIR, REPORTS_DIR
from .detect import Face, FaceDetector
from .embed import get_embedder
from .metrics import auc, best_accuracy_threshold, eer, far_frr, roc_curve, tar_at_far


def parse_pairs(path: Path) -> list[tuple[str, str, bool, int]]:
    """pairs.txt -> [(image_a, image_b, same_person, fold)] with paths relative to images/."""
    lines = path.read_text().strip().splitlines()
    n_folds, n_per = (int(v) for v in lines[0].split())
    out, i = [], 1
    for fold in range(n_folds):
        for same in (True, False):
            for _ in range(n_per):
                p = lines[i].split()
                i += 1
                if same:
                    a, b = (p[0], int(p[1])), (p[0], int(p[2]))
                else:
                    a, b = (p[0], int(p[1])), (p[2], int(p[3]))
                out.append((f"{a[0]}/{a[0]}_{a[1]:04d}.jpg", f"{b[0]}/{b[0]}_{b[1]:04d}.jpg",
                            same, fold))
    return out


def central_face(faces: list[Face], shape: tuple[int, ...]) -> Face | None:
    """LFW photos are centred on the labelled person; other faces in the frame are bystanders."""
    if not faces:
        return None
    cy, cx = shape[0] / 2, shape[1] / 2

    def dist(f: Face) -> float:
        x, y, w, h = f.box
        return (x + w / 2 - cx) ** 2 + (y + h / 2 - cy) ** 2

    return min(faces, key=dist)


def fallback_face(shape: tuple[int, ...]) -> Face:
    """If detection fails, assume the standard LFW framing (face ~ central 100 px)."""
    h, w = shape[:2]
    box = np.array([w / 2 - 50, h / 2 - 60, 100, 120], np.float32)
    lm = (np.asarray(ARCFACE_TEMPLATE) - 56) * (100 / 112) + [w / 2, h / 2]
    return Face(box, lm.astype(np.float32), 0.0)


def crops_for(images: list[str], root: Path, mode: str, detector: FaceDetector
              ) -> tuple[list[np.ndarray], int]:
    crops, missed = [], 0
    for rel in tqdm(images, desc=f"detect+{mode}", leave=False):
        bgr = cv2.imread(str(root / rel))
        face = central_face(detector.detect(bgr), bgr.shape)
        if face is None:
            missed += 1
            face = fallback_face(bgr.shape)
        crops.append(align_face(bgr, face.landmarks) if mode == "aligned"
                     else center_crop(bgr, face.box))
    return crops, missed


def fold_accuracies(scores: np.ndarray, same: np.ndarray, folds: np.ndarray) -> list[float]:
    accs = []
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        thr = best_accuracy_threshold(scores[train & same], scores[train & ~same])
        pred = scores[test] >= thr
        accs.append(float((pred == same[test]).mean()))
    return accs


def evaluate(scores: np.ndarray, same: np.ndarray, folds: np.ndarray) -> dict:
    g, i = scores[same], scores[~same]
    accs = fold_accuracies(scores, same, folds)
    _, far, tar = roc_curve(g, i)
    e, e_thr = eer(g, i)
    t2, thr2 = tar_at_far(g, i, 1e-2)
    t3, thr3 = tar_at_far(g, i, 1e-3)
    thr = best_accuracy_threshold(g, i)
    far_b, frr_b = far_frr(g, i, thr)
    return {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "fold_accuracies": accs, "auc": auc(far, tar), "eer": e, "eer_threshold": e_thr,
            "tar_at_far_1e-2": t2, "threshold_far_1e-2": thr2,
            "tar_at_far_1e-3": t3, "threshold_far_1e-3": thr3,
            "best_threshold_all": thr, "far_at_best": far_b, "frr_at_best": frr_b,
            "genuine_mean": float(g.mean()), "impostor_mean": float(i.mean())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=LFW_DIR)
    ap.add_argument("--models", nargs="+", default=["arcface", "sface"])
    ap.add_argument("--modes", nargs="+", default=["aligned", "unaligned"])
    ap.add_argument("--limit", type=int, help="evenly spaced subset of ~N pairs (smoke test)")
    args = ap.parse_args(argv)
    pairs = parse_pairs(args.root / "pairs.txt")
    if args.limit:  # every k-th pair keeps all folds and both classes
        pairs = pairs[::max(1, len(pairs) // args.limit)]
    images = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    index = {im: k for k, im in enumerate(images)}
    same = np.array([p[2] for p in pairs])
    folds = np.array([p[3] for p in pairs])
    detector = FaceDetector()
    out_dir = REPORTS_DIR / "lfw"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"pairs": len(pairs), "images": len(images), "runs": {}}
    all_scores = {}
    for mode in args.modes:
        t0 = time.perf_counter()
        crops, missed = crops_for(images, args.root / "images", mode, detector)
        prep_ms = 1000 * (time.perf_counter() - t0) / len(images)
        for name in args.models:
            embedder = get_embedder(name)
            t0 = time.perf_counter()
            emb = np.concatenate([embedder(crops[s:s + 64]) for s in range(0, len(crops), 64)])
            emb_ms = 1000 * (time.perf_counter() - t0) / len(images)
            a = emb[[index[p[0]] for p in pairs]]
            b = emb[[index[p[1]] for p in pairs]]
            scores = (a * b).sum(1)
            key = f"{name}_{mode}"
            all_scores[key] = scores
            results["runs"][key] = {"embedder": name, "crop": mode, "dim": int(emb.shape[1]),
                                    "detection_misses": missed,
                                    "detect_align_ms_per_image": round(prep_ms, 2),
                                    "embed_ms_per_image_cpu": round(emb_ms, 2),
                                    **evaluate(scores, same, folds)}
            r = results["runs"][key]
            print(f"{key:>18}: acc {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}  "
                  f"AUC {r['auc']:.4f}  EER {r['eer']:.4f}  TAR@1e-3 {r['tar_at_far_1e-3']:.4f}")
    np.savez(out_dir / "pair_scores.npz", same=same, folds=folds,
             images_a=np.array([p[0] for p in pairs]), images_b=np.array([p[1] for p in pairs]),
             **all_scores)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    from .plots import plot_lfw

    plot_lfw(all_scores, same, results, out_dir)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
