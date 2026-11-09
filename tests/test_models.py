"""Tests for the NumPy inference code and its equivalence to the scikit-learn pickles."""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from phcamera.models import DEFAULT_CLASSIFIER_PARAMS, DEFAULT_REGRESSION_PARAMS, LinearSVCOvo, PHEstimator, PolynomialRegression
from phcamera.features import Standardization
from tests.conftest import STRIP_COLORS, uniform_bgr

MODELS = Path(__file__).resolve().parent.parent / "models"
sklearn = pytest.importorskip("sklearn", reason="scikit-learn is only needed to compare against the original pickles")


def _load_legacy(path):
    warnings.simplefilter("ignore")
    with open(path, "rb") as f:
        m = pickle.load(f)
    if type(m).__name__ == "SVC":  # attributes renamed after scikit-learn 0.22
        m._probA = m.__dict__.get("_probA", m.__dict__.get("probA_", np.empty(0)))
        m._probB = m.__dict__.get("_probB", m.__dict__.get("probB_", np.empty(0)))
    return m


@pytest.fixture(scope="module")
def legacy_svm():
    return _load_legacy(MODELS / "RGB_SVM_model.sav")


@pytest.fixture(scope="module")
def legacy_reg():
    return _load_legacy(MODELS / "RHb_model.sav")


def test_regression_json_matches_pickle(legacy_reg):
    p = json.loads(DEFAULT_REGRESSION_PARAMS.read_text())
    assert p["variables"] == ["R", "H", "b*"]
    assert np.allclose(p["coef"], legacy_reg.coef_)
    assert p["intercept"] == pytest.approx(float(legacy_reg.intercept_))
    assert p["degree"] * 3 == len(p["coef"])


def test_regression_numpy_equals_sklearn(legacy_reg):
    reg = PolynomialRegression.from_json(DEFAULT_REGRESSION_PARAMS)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, reg.coef.size))
    ours = np.array([reg.predict_features(x) for x in X])
    assert np.allclose(ours, legacy_reg.predict(X))


def test_svm_json_matches_pickle(legacy_svm):
    p = json.loads(DEFAULT_CLASSIFIER_PARAMS.read_text())
    assert p["classes"] == [100.0 * k for k in range(1, 12)]
    assert np.allclose(p["coef"], legacy_svm.coef_)
    assert np.allclose(p["intercept"], legacy_svm.intercept_)


def test_svm_numpy_equals_sklearn_on_random_rgb_and_support_vectors(legacy_svm):
    svm = LinearSVCOvo.from_json(DEFAULT_CLASSIFIER_PARAMS)
    rng = np.random.default_rng(2)
    X = rng.integers(0, 256, size=(50_000, 3)).astype(float)
    assert np.array_equal(svm.predict(X), legacy_svm.predict(X))
    sv = legacy_svm.support_vectors_
    assert np.array_equal(svm.predict(sv), legacy_svm.predict(sv))


def test_regression_and_classifier_agree_on_support_vector_colors(legacy_svm, estimator):
    """On the SVM training colors the regression should land within 1 pH unit of the class."""
    sv = legacy_svm.support_vectors_
    labels = legacy_svm.predict(sv) / 100.0
    for rgb, label in zip(sv, labels):
        pred = estimator.predict(uniform_bgr(np.round(rgb).astype(int)))
        assert pred.ph_classification == label
        assert abs(pred.ph_regression - label) < 1.0, (rgb, label, pred)
        assert pred.source == "regression"


@pytest.mark.parametrize("ph,rgb", sorted(STRIP_COLORS.items()))
def test_typical_strip_colors_are_classified_correctly(estimator, ph, rgb):
    pred = estimator.predict(uniform_bgr(rgb))
    assert pred.ph_classification == ph
    assert abs(pred.ph - ph) < 1.0


def test_classification_overrides_regression_when_far_apart():
    class FakeReg(PolynomialRegression):
        def __init__(self):
            super().__init__(coef=[0, 0, 0], intercept=9.3, degree=1, variables=["R", "H", "b*"],
                             standardization=Standardization((0, 0, 0), (1, 1, 1)))

    class FakeSVM:
        def predict(self, X):
            return np.array([300.0])

    est = PHEstimator(FakeReg(), FakeSVM())
    pred = est.predict(uniform_bgr((10, 10, 10)))
    assert pred.source == "classification"
    assert pred.ph == 3.0
    assert pred.ph_regression == pytest.approx(9.3)


def test_regression_kept_when_difference_below_threshold():
    class FakeReg(PolynomialRegression):
        def __init__(self):
            super().__init__(coef=[0, 0, 0], intercept=3.9, degree=1, variables=["R", "H", "b*"],
                             standardization=Standardization((0, 0, 0), (1, 1, 1)))

    class FakeSVM:
        def predict(self, X):
            return np.array([300.0])

    pred = PHEstimator(FakeReg(), FakeSVM()).predict(uniform_bgr((10, 10, 10)))
    assert pred.source == "regression"
    assert pred.ph == pytest.approx(3.9)
    assert pred.in_range


def test_to_dict_rounds_values(estimator):
    d = estimator.predict(uniform_bgr(STRIP_COLORS[5.0])).to_dict()
    assert set(d) >= {"ph", "source", "ph_regression", "ph_classification", "mean_rgb", "raw_features", "in_range"}
    assert d["ph"] == round(d["ph"], 2)
    assert set(d["raw_features"]) == {"R", "H", "b*"}
