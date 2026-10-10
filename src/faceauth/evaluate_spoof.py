"""Evaluate anti-spoofing models on the CelebA-Spoof test subset (ISO/IEC 30107-3 metrics).

For every trained crop scale: APCER / BPCER / ACER at the threshold fixed on validation, test
AUC and EER, and BPCER20 / BPCER100 (BPCER when the threshold is set so that APCER = 5 % / 1 %).
Writes reports/spoof/results_test.json, spoof_scores.png, spoof_errors.jpg, training_curves.png.

Example:
    faceauth-evaluate-spoof
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from PIL import Image, ImageDraw

from .config import MODELS_DIR, REPORTS_DIR, SpoofConfig
from .metrics import auc, eer, pad_metrics, roc_curve
from .plots import plot_spoof
from .spoof_data import load_split
from .train_spoof import load_spoof_model, pick_device, predict_scores, run_name


def bpcer_at_apcer(attack: np.ndarray, bonafide: np.ndarray, apcer: float) -> float:
    """BPCER at the threshold where at most ``apcer`` of attacks slip through."""
    thr = np.quantile(attack, apcer)  # attacks scoring below thr are accepted as live
    return float((bonafide >= thr).mean())


def error_gallery(x: np.ndarray, y: np.ndarray, scores: np.ndarray, thr: float, n: int = 8,
                  size: int = 128) -> np.ndarray:
    """Top row: attacks the model let through (lowest scores). Bottom: real faces it rejected."""
    rows = []
    for wrong, title in (((y == 1) & (scores < thr), "attack accepted"),
                         ((y == 0) & (scores >= thr), "real face rejected")):
        idx = np.flatnonzero(wrong)
        idx = idx[np.argsort(scores[idx] if title.startswith("attack") else -scores[idx])][:n]
        tiles = []
        for i in idx:
            im = Image.fromarray(x[i]).resize((size, size))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, size, 12], fill=(0, 0, 0))
            d.text((2, 0), f"{title[:6]} p={scores[i]:.2f}", fill=(255, 255, 255))
            tiles.append(np.asarray(im))
        while len(tiles) < n:
            tiles.append(np.full((size, size, 3), 255, np.uint8))
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.6, 2.4])
    ap.add_argument("--main-scale", type=float, default=1.6, help="model used for the figures")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    dev = pick_device(args.device)
    out_dir = REPORTS_DIR / "spoof"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"models": {}}
    for scale in args.scales:
        cfg = SpoofConfig(crop_scale=scale)
        path = MODELS_DIR / f"{run_name(cfg)}.pt"
        if not path.exists():
            print(f"skip scale {scale}: {path.name} not trained")
            continue
        model, ckpt = load_spoof_model(path, dev)
        x, y = load_split("test", scale, ckpt["config"]["image_size"])
        t0 = time.perf_counter()
        scores = predict_scores(model, x, dev)
        ms = 1000 * (time.perf_counter() - t0) / len(x)
        attack, live = scores[y == 1], scores[y == 0]
        _, fpr, tpr = roc_curve(attack, live)
        test_eer, _ = eer(attack, live)
        results["models"][f"s{scale:g}"] = {
            "crop_scale": scale, "best_epoch": ckpt["epoch"], "val_eer": ckpt["val_eer"],
            **pad_metrics(attack, live, ckpt["threshold"]), "auc": auc(fpr, tpr),
            "test_eer": test_eer, "bpcer20": bpcer_at_apcer(attack, live, 0.05),
            "bpcer100": bpcer_at_apcer(attack, live, 0.01), "ms_per_face_batched": round(ms, 3)}
        results["test_images"] = int(len(y))
        results["test_spoof"] = int(y.sum())
        r = results["models"][f"s{scale:g}"]
        print(f"scale {scale}: APCER {r['apcer']:.4f}  BPCER {r['bpcer']:.4f}  "
              f"ACER {r['acer']:.4f}  AUC {r['auc']:.4f}  EER {r['test_eer']:.4f}")
        if scale == args.main_scale:
            hist_path = out_dir / f"history_{run_name(cfg)}.json"
            hist = json.loads(hist_path.read_text()) if hist_path.exists() else None
            plot_spoof(scores, y, ckpt["threshold"], out_dir, hist)
            Image.fromarray(error_gallery(x, y, scores, ckpt["threshold"])).save(
                out_dir / "spoof_errors.jpg", quality=90)
    (out_dir / "results_test.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
