"""Lightweight inference for the bundled pH models.

The models were trained with scikit-learn 0.22 and stored as pickles
(``models/*.sav``).  Pickles are tied to the scikit-learn version that wrote
them (the SVC pickle no longer loads cleanly in current releases), so the
application does not unpickle them at runtime.  Instead the fitted parameters
are exported once to JSON (``scripts/export_models.py``) and evaluated here
with NumPy only.  ``tests/test_models.py`` checks that this implementation
reproduces scikit-learn's predictions exactly.

Two models are combined, as described in the paper:

* ``PolynomialRegression`` - the RHb* polynomial regression that outputs a
  continuous pH value (the "Regression" block of the graphical abstract).
* ``LinearSVCOvo`` - a linear support vector machine on mean RGB that
  classifies the strip into one of the 11 integer pH classes 1-11 (the
  "Classification" block).

If the regression output and the classification result differ by 1.0 or more,
the classification result is trusted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import ColorFeatures, Standardization, extract_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_REGRESSION_PARAMS = MODELS_DIR / "rhb_poly_regression.json"
DEFAULT_CLASSIFIER_PARAMS = MODELS_DIR / "rgb_linear_svm.json"

#: |regression - classification| at or above this value -> trust classification.
CLASSIFICATION_OVERRIDE_THRESHOLD = 1.0


class PolynomialRegression:
    """Linear model on standardized power-polynomial features (R, H, b*)."""

    def __init__(
        self,
        coef,
        intercept: float,
        degree: int,
        variables,
        standardization: Standardization,
        meta: dict[str, Any] | None = None,
    ):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.degree = int(degree)
        self.variables = tuple(variables)
        self.standardization = standardization
        self.meta = meta or {}
        expected = len(self.variables) * self.degree
        if self.coef.shape != (expected,):
            raise ValueError(f"coef has shape {self.coef.shape}, expected ({expected},)")

    @classmethod
    def from_json(cls, path: str | Path) -> "PolynomialRegression":
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        return cls(
            coef=p["coef"],
            intercept=p["intercept"],
            degree=p["degree"],
            variables=p["variables"],
            standardization=Standardization(
                tuple(p["standardization"]["mean"]), tuple(p["standardization"]["std"])
            ),
            meta=p.get("meta", {}),
        )

    def features(self, bgr: np.ndarray) -> ColorFeatures:
        return extract_features(bgr, self.standardization, self.degree)

    def predict_features(self, polynomial) -> float:
        return float(np.dot(np.asarray(polynomial, dtype=float), self.coef) + self.intercept)

    def predict(self, bgr: np.ndarray) -> float:
        return self.predict_features(self.features(bgr).polynomial)


class LinearSVCOvo:
    """Linear-kernel SVC with one-vs-one voting (libsvm / scikit-learn semantics)."""

    def __init__(self, classes, coef, intercept, meta: dict[str, Any] | None = None):
        self.classes = np.asarray(classes, dtype=float)
        self.coef = np.asarray(coef, dtype=float)  # (n_pairs, n_features)
        self.intercept = np.asarray(intercept, dtype=float)  # (n_pairs,)
        self.meta = meta or {}
        n = len(self.classes)
        self.pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if self.coef.shape[0] != len(self.pairs) or self.intercept.shape != (len(self.pairs),):
            raise ValueError("coef / intercept do not match the number of class pairs")

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearSVCOvo":
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        return cls(p["classes"], p["coef"], p["intercept"], p.get("meta", {}))

    def decision_function(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X @ self.coef.T + self.intercept

    def predict(self, X) -> np.ndarray:
        dec = self.decision_function(X)
        votes = np.zeros((dec.shape[0], len(self.classes)), dtype=int)
        for k, (i, j) in enumerate(self.pairs):
            positive = dec[:, k] > 0
            votes[positive, i] += 1
            votes[~positive, j] += 1
        return self.classes[np.argmax(votes, axis=1)]


@dataclass
class PHPrediction:
    ph: float
    """Final pH value shown to the user."""

    source: str
    """``"regression"`` or ``"classification"`` (which model produced ``ph``)."""

    ph_regression: float
    ph_classification: float
    mean_rgb: tuple[float, float, float]
    raw_features: dict[str, float]
    in_range: bool
    """True when the final pH lies within the strip range of the training data (1-11)."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ph"] = round(self.ph, 2)
        d["ph_regression"] = round(self.ph_regression, 2)
        d["ph_classification"] = round(self.ph_classification, 2)
        d["mean_rgb"] = [round(v, 1) for v in self.mean_rgb]
        d["raw_features"] = {k: round(v, 2) for k, v in self.raw_features.items()}
        return d


class PHEstimator:
    """Regression + classification ensemble used by the demo application."""

    def __init__(
        self,
        regression: PolynomialRegression,
        classifier: LinearSVCOvo,
        class_scale: float = 100.0,
        threshold: float = CLASSIFICATION_OVERRIDE_THRESHOLD,
        ph_range: tuple[float, float] = (1.0, 11.0),
    ):
        self.regression = regression
        self.classifier = classifier
        # SVM classes are stored as pH * 100 (100, 200, ..., 1100).
        self.class_scale = class_scale
        self.threshold = threshold
        self.ph_range = ph_range

    def predict(self, bgr: np.ndarray) -> PHPrediction:
        feats = self.regression.features(bgr)
        ph_reg = self.regression.predict_features(feats.polynomial)
        ph_cls = float(self.classifier.predict([feats.mean_rgb])[0]) / self.class_scale
        if abs(ph_reg - ph_cls) >= self.threshold:
            ph, source = ph_cls, "classification"
        else:
            ph, source = ph_reg, "regression"
        lo, hi = self.ph_range
        return PHPrediction(
            ph=ph,
            source=source,
            ph_regression=ph_reg,
            ph_classification=ph_cls,
            mean_rgb=feats.mean_rgb,
            raw_features={"R": feats.raw[0], "H": feats.raw[1], "b*": feats.raw[2]},
            in_range=lo <= ph <= hi,
        )


def load_default_estimator(
    regression_path: str | Path = DEFAULT_REGRESSION_PARAMS,
    classifier_path: str | Path = DEFAULT_CLASSIFIER_PARAMS,
) -> PHEstimator:
    return PHEstimator(
        PolynomialRegression.from_json(regression_path), LinearSVCOvo.from_json(classifier_path)
    )
