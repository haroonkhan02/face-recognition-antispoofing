import numpy as np
import pytest
import torch

from faceauth.detect import Face
from faceauth.evaluate_spoof import bpcer_at_apcer, error_gallery
from faceauth.gallery import Gallery
from faceauth.liveness import HeadTurnChallenge, yaw_proxy
from faceauth.spoof_data import _process_shard, square_crop
from faceauth.train_spoof import Normalizer, augment, build_model, predict_scores
from faceauth.verify import central_face, fallback_face, fold_accuracies, parse_pairs


def landmarks(yaw: float) -> np.ndarray:
    """Eyes at x=40/60, nose shifted by yaw * inter-eye distance."""
    return np.array([[40, 50], [60, 50], [50 + 20 * yaw, 60], [43, 70], [57, 70]], float)


def test_parse_pairs(tmp_path):
    lines = ["2\t1", "A\t1\t2", "A\t1\tB\t1", "C\t2\t3", "C\t1\tD\t4"]
    (tmp_path / "pairs.txt").write_text("\n".join(lines))
    pairs = parse_pairs(tmp_path / "pairs.txt")
    assert pairs[0] == ("A/A_0001.jpg", "A/A_0002.jpg", True, 0)
    assert pairs[1] == ("A/A_0001.jpg", "B/B_0001.jpg", False, 0)
    assert pairs[3][1] == "D/D_0004.jpg" and pairs[3][3] == 1


def test_fold_accuracies_use_held_out_thresholds():
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    same = np.array([True, False] * 3)
    folds = np.array([0, 0, 1, 1, 2, 2])
    assert fold_accuracies(scores, same, folds) == [1.0, 1.0, 1.0]


def test_central_face_and_fallback():
    off = Face(np.array([0, 0, 40, 40.0]), landmarks(0), 0.9)
    mid = Face(np.array([100, 100, 50, 50.0]), landmarks(0), 0.8)
    assert central_face([off, mid], (250, 250, 3)) is mid
    assert central_face([], (250, 250, 3)) is None
    assert fallback_face((250, 250, 3)).landmarks.shape == (5, 2)


def test_gallery_open_set_matching(tmp_path):
    rng = np.random.default_rng(0)
    alice, bob = rng.normal(size=512), rng.normal(size=512)
    g = Gallery()
    g.add("alice", np.stack([alice + 0.1 * rng.normal(size=512) for _ in range(3)]))
    g.add("bob", bob)
    assert g.match(alice + 0.2 * rng.normal(size=512))[0] == "alice"
    assert g.match(rng.normal(size=512), threshold=0.5)[0] is None  # a stranger is unknown
    g.save(tmp_path / "g.npz")
    assert Gallery.load(tmp_path / "g.npz").names == ["alice", "bob"]
    assert Gallery().match(alice) == (None, -1.0)


def test_yaw_proxy_sign():
    assert yaw_proxy(landmarks(0)) == pytest.approx(0)
    assert yaw_proxy(landmarks(-0.4)) < 0 < yaw_proxy(landmarks(0.4))


def test_head_turn_challenge_passes_in_order_and_times_out():
    c = HeadTurnChallenge(steps=["left", "right"])
    seq = [0.0, -0.4, 0.0, 0.4]  # centre, left, centre, right
    for t, yaw in enumerate(seq):
        c.update(landmarks(yaw), float(t))
    assert c.state == "passed"
    wrong = HeadTurnChallenge(steps=["left", "right"], timeout_s=3)
    for t, yaw in enumerate([0.0, 0.4, 0.4, 0.4, 0.4]):  # turns the wrong way, then times out
        wrong.update(landmarks(yaw), float(t))
    assert wrong.state == "failed"
    assert len(HeadTurnChallenge(seed=1).steps) == 2


def test_square_crop_and_shard_processing(spoof_shard):
    from PIL import Image

    img = Image.fromarray(np.zeros((100, 80, 3), np.uint8))
    assert square_crop(img, [10, 10, 50, 60], 1.6, 64).shape == (64, 64, 3)
    out = _process_shard((str(spoof_shard), (1.0, 2.0), 32))
    assert out["crops"][1.0].shape == (5, 32, 32, 3) and out["skipped"] == 1
    assert out["labels"].tolist() == [0, 1, 0, 1, 0]


def test_spoof_model_forward_and_scores():
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False)
    x = np.random.default_rng(0).integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
    s = predict_scores(model, x, torch.device("cpu"))
    assert s.shape == (3,) and np.all((s >= 0) & (s <= 1))
    xb = Normalizer("cpu")(torch.from_numpy(x))
    assert augment(xb, torch.Generator().manual_seed(0)).shape == (3, 3, 64, 64)


def test_bpcer_at_apcer_and_error_gallery():
    attack = np.linspace(0.5, 1.0, 100)
    live = np.linspace(0.0, 0.6, 100)
    assert bpcer_at_apcer(attack, live, 0.0) == pytest.approx((live >= 0.5).mean())
    x = np.zeros((4, 32, 32, 3), np.uint8)
    g = error_gallery(x, np.array([1, 1, 0, 0]), np.array([0.1, 0.9, 0.8, 0.2]), 0.5, n=3,
                      size=32)
    assert g.shape == (64, 96, 3)
