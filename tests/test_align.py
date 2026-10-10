import cv2
import numpy as np

from faceauth.align import TEMPLATE, align_face, center_crop, umeyama


def similarity(scale, angle_deg, tx, ty):
    a = np.deg2rad(angle_deg)
    r = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.hstack([r, [[tx], [ty]]])


def test_umeyama_recovers_a_known_similarity():
    m = similarity(1.7, 23, 40, -12)
    src = np.random.default_rng(0).uniform(0, 100, (5, 2))
    dst = src @ m[:, :2].T + m[:, 2]
    np.testing.assert_allclose(umeyama(src, dst), m, atol=1e-8)


def test_umeyama_never_mirrors():
    src = TEMPLATE.copy()
    dst = src * [-1, 1]  # a mirror image cannot be reached by rotation + scale
    m = umeyama(src, dst)
    assert np.linalg.det(m[:, :2]) > 0


def test_align_face_puts_landmarks_on_the_template():
    m = similarity(2.3, -15, 60, 35)  # where the template lands in a bigger, tilted photo
    lm = TEMPLATE @ m[:, :2].T + m[:, 2]
    img = np.zeros((400, 400, 3), np.uint8)
    for x, y in lm:
        cv2.circle(img, (round(x), round(y)), 4, (255, 255, 255), -1)
    out = align_face(img, lm)
    assert out.shape == (112, 112, 3)
    for x, y in TEMPLATE:
        assert out[round(y), round(x)].max() > 200  # each dot lands on its template point


def test_center_crop_shape():
    img = np.zeros((200, 300, 3), np.uint8)
    assert center_crop(img, np.array([100, 50, 80, 100])).shape == (112, 112, 3)
