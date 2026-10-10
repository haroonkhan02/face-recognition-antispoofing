"""Biometric metrics for verification (FAR / FRR / EER / TAR@FAR) and spoof detection (ISO 30107-3).

Conventions: a *score* is higher for "same person" (verification) or "attack" (spoofing).
- FAR  (false accept rate)  = impostor pairs accepted / all impostor pairs
- FRR  (false reject rate)  = genuine pairs rejected / all genuine pairs;  TAR = 1 - FRR
- EER  = the error rate where FAR = FRR
- APCER (attack presentation classification error rate) = attacks accepted as bona fide
- BPCER (bona fide presentation classification error rate) = real faces rejected as attacks
- ACER = (APCER + BPCER) / 2
"""

from __future__ import annotations

import numpy as np


def far_frr(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    """Accept when score >= threshold."""
    far = float((np.asarray(impostor) >= threshold).mean()) if len(impostor) else 0.0
    frr = float((np.asarray(genuine) < threshold).mean()) if len(genuine) else 0.0
    return far, frr


def roc_curve(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, ...]:
    """(thresholds, FAR, TAR) evaluated at every distinct score, thresholds descending."""
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last index of each distinct score
    return s[last], fp[last] / max(len(impostor), 1), tp[last] / max(len(genuine), 1)


def auc(far: np.ndarray, tar: np.ndarray) -> float:
    x = np.r_[0.0, far, 1.0]
    y = np.r_[0.0, tar, 1.0]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(EER, threshold) at the point where FAR and FRR cross."""
    thr, far, tar = roc_curve(genuine, impostor)
    frr = 1 - tar
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, target_far: float) -> tuple[float, float]:
    """Highest TAR with FAR <= target, and its threshold."""
    thr, far, tar = roc_curve(genuine, impostor)
    ok = np.flatnonzero(far <= target_far)
    if not len(ok):
        return 0.0, float("inf")
    i = ok[np.argmax(tar[ok])]
    return float(tar[i]), float(thr[i])


def best_accuracy_threshold(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Threshold maximising (correct accepts + correct rejects) / all pairs.

    Every threshold between the lowest accepted score and the next lower score gives the same
    accuracy on these pairs; the midpoint of that gap generalises best to held-out pairs (taking
    the score itself would reject a held-out genuine pair that lands just below it).
    """
    thr, far, tar = roc_curve(genuine, impostor)
    n_g, n_i = len(genuine), len(impostor)
    acc = (tar * n_g + (1 - far) * n_i) / (n_g + n_i)
    k = int(np.argmax(acc))
    return float((thr[k] + thr[k + 1]) / 2) if k + 1 < len(thr) else float(thr[k])


def pad_metrics(attack_scores: np.ndarray, bonafide_scores: np.ndarray, threshold: float
                ) -> dict[str, float]:
    """Presentation-attack-detection errors; 'attack' is declared when score >= threshold."""
    apcer = float((np.asarray(attack_scores) < threshold).mean())
    bpcer = float((np.asarray(bonafide_scores) >= threshold).mean())
    return {"apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2, "threshold": threshold}


def threshold_for_bpcer(bonafide_scores: np.ndarray, target: float) -> float:
    """Smallest threshold whose BPCER <= target (fixes the inconvenience to real users)."""
    s = np.sort(np.asarray(bonafide_scores))
    k = int(np.ceil((1 - target) * len(s))) - 1
    return float(np.nextafter(s[min(max(k, 0), len(s) - 1)], np.inf))
