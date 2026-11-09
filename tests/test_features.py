import numpy as np
import pytest

from phcamera.features import Standardization, extract_features, mean_color_spaces, power_polynomial
from tests.conftest import uniform_bgr


def test_mean_color_spaces_uniform_patch_matches_opencv_scaling():
    rgb, hsv, lab = mean_color_spaces(uniform_bgr((255, 0, 0)))
    assert np.allclose(rgb, [255, 0, 0])
    assert hsv[0] == 0 and hsv[1] == 255 and hsv[2] == 255  # pure red: H=0 (0-179 scale)
    assert lab[1] > 128 and lab[2] > 128  # a*, b* positive (offset 128)

    _, _, lab_white = mean_color_spaces(uniform_bgr((255, 255, 255)))
    assert np.allclose(lab_white, [255, 128, 128])  # L*=100 -> 255, neutral a*/b* -> 128


def test_mean_color_spaces_green_hue_is_60_on_opencv_scale():
    _, hsv, _ = mean_color_spaces(uniform_bgr((0, 255, 0)))
    assert hsv[0] == 60


def test_mean_is_taken_over_all_pixels():
    img = uniform_bgr((0, 0, 0), size=10)
    img[:, :5] = (0, 0, 255)  # left half red (BGR)
    rgb, _, _ = mean_color_spaces(img)
    assert np.allclose(rgb, [127.5, 0, 0])


def test_non_uint8_input_is_clipped_and_cast():
    img = np.full((4, 4, 3), 300.0)
    rgb, _, _ = mean_color_spaces(img)
    assert np.allclose(rgb, 255)


@pytest.mark.parametrize("bad", [np.zeros((4, 4), np.uint8), np.zeros((0, 4, 3), np.uint8), np.zeros((4, 4, 4), np.uint8)])
def test_invalid_shapes_raise(bad):
    with pytest.raises(ValueError):
        mean_color_spaces(bad)


def test_power_polynomial_order_is_degree_major():
    out = power_polynomial([2.0, 3.0], 3)
    assert out.tolist() == [2, 3, 4, 9, 8, 27]


def test_extract_features_standardizes_selected_variables():
    std = Standardization(mean=(100.0, 50.0, 150.0), std=(10.0, 5.0, 20.0))
    feats = extract_features(uniform_bgr((120, 60, 30)), std, degree=2)
    assert feats.mean_rgb == (120.0, 60.0, 30.0)
    assert feats.raw[0] == 120.0  # R
    assert feats.raw[1] == feats.mean_hsv[0]  # H
    assert feats.raw[2] == feats.mean_lab[2]  # b*
    assert feats.standardized[0] == pytest.approx((120 - 100) / 10)
    assert feats.polynomial.shape == (6,)
    assert feats.polynomial[0] == pytest.approx(feats.standardized[0])
    assert feats.polynomial[3] == pytest.approx(feats.standardized[0] ** 2)
