"""Export the fitted parameters of the scikit-learn pickles to JSON.

Usage (requires scikit-learn, any recent version)::

    python scripts/export_models.py

Reads ``models/RHb_model.sav`` (polynomial regression on standardized
R, H, b*) and ``models/RGB_SVM_model.sav`` (linear SVC on mean RGB) and writes
``models/rhb_poly_regression.json`` and ``models/rgb_linear_svm.json``.
The application only needs the JSON files, so it does not depend on the
scikit-learn version used for training (0.22.2).

The standardization constants were not stored in the pickle; they are the
training-set statistics hard-coded in the original demo application.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# Training-set mean / std of R (RGB), H (OpenCV HSV, 0-179), b* (OpenCV Lab, 0-255)
STANDARDIZATION = {
    "variables": ["R", "H", "b*"],
    "mean": [106.87626803684113, 53.69784646955657, 154.32425094322426],
    "std": [79.18476934463334, 38.6981975345093, 27.213700253049293],
}


def pickled_sklearn_version(path: Path) -> str:
    """Read the ``_sklearn_version`` string stored in the pickle (dropped on unpickling)."""
    raw = path.read_bytes()
    m = re.search(rb"_sklearn_version.{1,8}?(\d+\.\d+\.\d+(?:\.post\d+)?)", raw, re.S)
    return m.group(1).decode() if m else "unknown"


def load_pickle(path: Path):
    """Unpickle a scikit-learn 0.22 estimator and patch renamed private attributes."""
    warnings.simplefilter("ignore")
    with open(path, "rb") as f:
        model = pickle.load(f)
    model._pickled_sklearn_version = pickled_sklearn_version(path)
    if type(model).__name__ == "SVC":
        d = model.__dict__
        model._probA = d.get("_probA", d.get("probA_", np.empty(0)))
        model._probB = d.get("_probB", d.get("probB_", np.empty(0)))
    return model


def export_regression(reg) -> dict:
    coef = np.asarray(reg.coef_, dtype=float).ravel()
    variables = STANDARDIZATION["variables"]
    if coef.size % len(variables):
        raise SystemExit(f"unexpected coef size {coef.size} for {len(variables)} variables")
    degree = coef.size // len(variables)
    return {
        "model": "polynomial_regression",
        "description": (
            "Power polynomial (no cross terms) of standardized R, H, b*; variables selected by "
            "Local MAE (Nakamura & Nagao, Chem. Lett. 2025, 54, upaf018)."
        ),
        "variables": variables,
        "degree": degree,
        "standardization": {"mean": STANDARDIZATION["mean"], "std": STANDARDIZATION["std"]},
        "feature_order": [f"{v}^{d}" for d in range(1, degree + 1) for v in variables],
        "coef": coef.tolist(),
        "intercept": float(np.ravel(reg.intercept_)[0]),
        "meta": {
            "source_file": "models/RHb_model.sav",
            "sklearn_estimator": type(reg).__name__,
            "sklearn_version_trained": reg._pickled_sklearn_version,
        },
    }


def one_vs_one_linear_coef(svm) -> np.ndarray:
    """Pairwise weight vectors of a linear SVC, computed deterministically.

    scikit-learn's ``SVC.coef_`` is a property that multiplies ``dual_coef_`` by
    the support vectors with BLAS at access time; the summation order (and so
    the last bits of the result) differs between platforms.  Here every weight
    is the correctly rounded sum of the same products (``math.fsum``), so the
    exported JSON is byte-identical on every machine.
    """
    dual = np.asarray(svm.dual_coef_, dtype=float)        # (n_class - 1, n_SV)
    sv = np.asarray(svm.support_vectors_, dtype=float)    # (n_SV, n_features)
    n_class = dual.shape[0] + 1
    locs = np.concatenate([[0], np.cumsum(svm.n_support_)])
    coef = []
    for c1 in range(n_class):
        for c2 in range(c1 + 1, n_class):
            a1, s1 = dual[c2 - 1, locs[c1]:locs[c1 + 1]], sv[locs[c1]:locs[c1 + 1]]
            a2, s2 = dual[c1, locs[c2]:locs[c2 + 1]], sv[locs[c2]:locs[c2 + 1]]
            alphas = np.concatenate([a1, a2])
            vecs = np.concatenate([s1, s2])
            coef.append([math.fsum(float(a) * float(v) for a, v in zip(alphas, vecs[:, f]))
                         for f in range(sv.shape[1])])
    coef = np.array(coef)
    if not np.allclose(coef, svm.coef_, rtol=1e-12, atol=1e-12):
        raise SystemExit("deterministic coef differs from scikit-learn's coef_")
    return coef


def export_svm(svm) -> dict:
    if svm.kernel != "linear":
        raise SystemExit("only a linear-kernel SVC can be exported")
    return {
        "model": "linear_svc_ovo",
        "description": (
            "Linear-kernel SVC on mean RGB (0-255) of the strip; classes are pH x 100. "
            "Trained on 440 images; one-vs-one voting."
        ),
        "features": ["R", "G", "B"],
        "classes": np.asarray(svm.classes_, dtype=float).tolist(),
        "coef": one_vs_one_linear_coef(svm).tolist(),
        "intercept": np.asarray(svm.intercept_, dtype=float).tolist(),
        "meta": {
            "source_file": "models/RGB_SVM_model.sav",
            "sklearn_estimator": "SVC",
            "sklearn_version_trained": svm._pickled_sklearn_version,
            "C": svm.C,
            "n_support": np.asarray(svm.n_support_).tolist(),
            "n_training_samples": int(svm.shape_fit_[0]),
        },
    }


def main() -> int:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn is required to read the .sav files: pip install scikit-learn", file=sys.stderr)
        return 1

    reg = load_pickle(MODELS / "RHb_model.sav")
    svm = load_pickle(MODELS / "RGB_SVM_model.sav")
    reg_json = export_regression(reg)
    svm_json = export_svm(svm)

    # Verify that the exported parameters reproduce scikit-learn's predictions.
    sys.path.insert(0, str(ROOT))
    from phcamera.features import Standardization
    from phcamera.models import LinearSVCOvo, PolynomialRegression

    rng = np.random.default_rng(0)
    X_rgb = rng.integers(0, 256, size=(100_000, 3)).astype(float)
    ours = LinearSVCOvo(svm_json["classes"], svm_json["coef"], svm_json["intercept"]).predict(X_rgb)
    agreement = float(np.mean(ours == svm.predict(X_rgb)))
    print(f"SVC: NumPy one-vs-one agrees with scikit-learn on {agreement:.4%} of 100k random RGB values")
    if agreement < 1.0:
        raise SystemExit("SVC export mismatch")

    X_poly = rng.normal(size=(1000, len(reg_json["coef"])))
    pr = PolynomialRegression(
        reg_json["coef"],
        reg_json["intercept"],
        reg_json["degree"],
        reg_json["variables"],
        Standardization(tuple(STANDARDIZATION["mean"]), tuple(STANDARDIZATION["std"])),
    )
    ours_r = np.array([pr.predict_features(x) for x in X_poly])
    if not np.allclose(ours_r, reg.predict(X_poly)):
        raise SystemExit("regression export mismatch")
    print(f"Regression: degree-{reg_json['degree']} polynomial of {reg_json['variables']} reproduced exactly")

    (MODELS / "rhb_poly_regression.json").write_text(json.dumps(reg_json, indent=2) + "\n", encoding="utf-8")
    (MODELS / "rgb_linear_svm.json").write_text(json.dumps(svm_json, indent=2) + "\n", encoding="utf-8")
    print("wrote models/rhb_poly_regression.json and models/rgb_linear_svm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
