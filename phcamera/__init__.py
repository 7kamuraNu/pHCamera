"""pH Camera: machine learning-based pH quantification from pH test strip images.

Reference implementation of the demo web application described in

    Y. Nakamura, M. Nagao, "Machine learning-based pH quantification from
    test strip images using multiple color spaces", Chem. Lett. 2025, 54,
    upaf018. https://doi.org/10.1093/chemle/upaf018
"""

from .features import ColorFeatures, Standardization, extract_features
from .local_mae import local_mae, per_level_mae
from .models import LinearSVCOvo, PHEstimator, PHPrediction, PolynomialRegression, load_default_estimator

__all__ = [
    "ColorFeatures",
    "Standardization",
    "extract_features",
    "LinearSVCOvo",
    "PHEstimator",
    "PHPrediction",
    "PolynomialRegression",
    "load_default_estimator",
    "local_mae",
    "per_level_mae",
]

__version__ = "1.0.0"
