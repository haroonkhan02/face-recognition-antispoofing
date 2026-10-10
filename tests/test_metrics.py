import numpy as np
import pytest

from faceauth.embed import cosine, l2_normalize
from faceauth.metrics import (
    auc,
    best_accuracy_threshold,
    eer,
    far_frr,
    pad_metrics,
    roc_curve,
    tar_at_far,
    threshold_for_bpcer,
)

GEN = np.array([0.9, 0.8, 0.7, 0.4])  # genuine (same person) scores
IMP = np.array([0.5, 0.3, 0.2, 0.1])  # impostor scores


def test_far_frr_at_a_threshold():
    assert far_frr(GEN, IMP, 0.45) == (0.25, 0.25)  # one impostor accepted, one genuine rejected


def test_roc_and_auc():
    thr, far, tar = roc_curve(GEN, IMP)
    assert thr[0] == 0.9 and far[-1] == 1 and tar[-1] == 1
    # 15 of the 16 genuine/impostor pairs are ordered correctly -> AUC = 15/16
    assert auc(far, tar) == pytest.approx(15 / 16)
    assert auc(*roc_curve(np.ones(3), np.zeros(3))[1:]) == pytest.approx(1.0)


def test_eer_and_tar_at_far():
    e, _ = eer(GEN, IMP)
    assert e == pytest.approx(0.25)
    tar, thr = tar_at_far(GEN, IMP, 0.0)
    assert tar == pytest.approx(0.75) and thr == pytest.approx(0.7)


def test_best_accuracy_threshold_separates():
    g, i = np.array([0.8, 0.9]), np.array([0.1, 0.2])
    t = best_accuracy_threshold(g, i)
    assert far_frr(g, i, t) == (0.0, 0.0)


def test_pad_metrics_and_bpcer_threshold():
    attack = np.array([0.9, 0.8, 0.3])
    live = np.array([0.1, 0.2, 0.6, 0.05])
    m = pad_metrics(attack, live, 0.5)
    assert m["apcer"] == pytest.approx(1 / 3) and m["bpcer"] == pytest.approx(1 / 4)
    assert m["acer"] == pytest.approx((1 / 3 + 1 / 4) / 2)
    t = threshold_for_bpcer(live, 0.25)
    assert (live >= t).mean() <= 0.25


def test_cosine_and_normalisation():
    a = np.array([3.0, 4.0])
    assert np.linalg.norm(l2_normalize(a)) == pytest.approx(1)
    assert cosine(a, 2 * a) == pytest.approx(1) and cosine(a, -a) == pytest.approx(-1)
