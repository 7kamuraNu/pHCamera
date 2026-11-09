import numpy as np
import pytest

from phcamera.local_mae import local_mae, per_level_mae


def test_local_mae_is_max_of_per_level_mae():
    y_true = [1, 1, 2, 2, 3, 3]
    y_pred = [1.1, 0.9, 2.5, 2.5, 3.0, 3.0]
    levels = per_level_mae(y_true, y_pred)
    assert levels == {1.0: pytest.approx(0.1), 2.0: pytest.approx(0.5), 3.0: 0.0}
    assert local_mae(y_true, y_pred) == pytest.approx(0.5)


def test_local_mae_matches_paper_snippet():
    """Reference implementation from Fig. 2 of the paper (loop over np.arange(1.0, 11.1, 0.5))."""
    rng = np.random.default_rng(0)
    levels = np.arange(1.0, 11.1, 0.5)
    y_true = np.repeat(levels, 40)
    y_pred = y_true + rng.normal(scale=0.2, size=y_true.size)

    ph_mae_list = []
    for ph in levels:
        mask = np.isclose(y_true, ph)
        ph_mae_list.append(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    assert local_mae(y_true, y_pred) == pytest.approx(max(ph_mae_list))


def test_local_mae_penalizes_a_single_bad_region_more_than_overall_mae():
    y_true = np.repeat([1.0, 2.0, 3.0, 4.0], 10)
    y_pred = y_true.copy()
    y_pred[:10] += 1.0  # only pH 1 is wrong
    assert np.mean(np.abs(y_true - y_pred)) == pytest.approx(0.25)
    assert local_mae(y_true, y_pred) == pytest.approx(1.0)


def test_input_validation():
    with pytest.raises(ValueError):
        local_mae([1, 2], [1])
    with pytest.raises(ValueError):
        local_mae([], [])
