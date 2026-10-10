"""Matplotlib figures for the reports (headless backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import roc_curve  # noqa: E402

LABELS = {"arcface_aligned": "ArcFace · 5-pt aligned", "arcface_unaligned": "ArcFace · box crop",
          "sface_aligned": "SFace · 5-pt aligned", "sface_unaligned": "SFace · box crop"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_lfw(scores: dict[str, np.ndarray], same: np.ndarray, results: dict, out_dir: Path) -> None:
    """ROC curves (log FAR) for every run, and genuine/impostor score histograms."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for key, s in scores.items():
        _, far, tar = roc_curve(s[same], s[~same])
        r = results["runs"][key]
        ax.plot(np.clip(far, 1e-4, 1), tar,
                label=f"{LABELS.get(key, key)} (acc {r['accuracy_mean']:.3f})")
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 1)
    ax.set_ylim(0.5, 1.005)
    ax.set_xlabel("false accept rate (FAR)")
    ax.set_ylabel("true accept rate (TAR = 1 − FRR)")
    ax.set_title("LFW verification ROC (6,000 pairs)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out_dir / "roc.png")

    keys = [k for k in scores if k.endswith("_aligned")]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 3.8), squeeze=False)
    for ax, key in zip(axes[0], keys, strict=True):
        s, r = scores[key], results["runs"][key]
        bins = np.linspace(min(s.min(), -0.3), 1, 70)
        ax.hist(s[~same], bins, alpha=0.6, color="#d1495b", label="different people")
        ax.hist(s[same], bins, alpha=0.6, color="#2e8b57", label="same person")
        ax.axvline(r["best_threshold_all"], color="k", ls="--", lw=1,
                   label=f"threshold {r['best_threshold_all']:.2f}")
        ax.set_xlabel("cosine similarity")
        ax.set_title(LABELS.get(key, key))
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("pairs")
    _save(fig, out_dir / "score_histograms.png")


def plot_spoof(scores: np.ndarray, labels: np.ndarray, thr: float, out_dir: Path,
               history: dict | None = None) -> None:
    """Score distributions of live vs spoof test faces, plus the DET-style error trade-off."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 60)
    ax1.hist(scores[labels == 0], bins, alpha=0.6, color="#2e8b57", label="live (bona fide)")
    ax1.hist(scores[labels == 1], bins, alpha=0.6, color="#d1495b", label="spoof (attack)")
    ax1.axvline(thr, color="k", ls="--", lw=1, label=f"threshold {thr:.3f} (from val)")
    ax1.set_yscale("log")
    ax1.set_xlabel("attack score (softmax probability of 'spoof')")
    ax1.set_ylabel("test faces (log)")
    ax1.legend(fontsize=8)
    ax1.set_title("Test score distributions")

    thrs = np.linspace(0, 1, 501)
    apcer = [(scores[labels == 1] < t).mean() for t in thrs]
    bpcer = [(scores[labels == 0] >= t).mean() for t in thrs]
    ax2.plot(thrs, apcer, color="#d1495b", label="APCER (attacks let through)")
    ax2.plot(thrs, bpcer, color="#2e8b57", label="BPCER (real users rejected)")
    ax2.axvline(thr, color="k", ls="--", lw=1)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1)
    ax2.set_xlabel("decision threshold")
    ax2.set_ylabel("error rate (log)")
    ax2.set_title("The security / convenience trade-off")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    _save(fig, out_dir / "spoof_scores.png")

    if history:
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
        ep = [r["epoch"] for r in history["epochs"]]
        a.plot(ep, [r["train_loss"] for r in history["epochs"]], "o-")
        a.set_title("training loss")
        b.plot(ep, [r["val_eer"] for r in history["epochs"]], "o-", color="#d1495b")
        b.set_title("validation EER (APCER = BPCER)")
        for ax in (a, b):
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        _save(fig, out_dir / "training_curves.png")
