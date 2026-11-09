"""Color feature extraction from a pH test strip image region.

The paper derives 12 candidate variables per image (R, G, B, H, S, V, X, Y, Z,
L*, a*, b*) from the color of the cropped test strip and selects the
combination R (RGB), H (HSV) and b* (CIE L*a*b*) by the Local MAE criterion
(the "RHb*" model).  This module reproduces that pipeline for a single region
of interest (ROI) using OpenCV's 8-bit color conversions, which is what the
bundled models were trained with:

* OpenCV ``BGR2HSV`` stores hue in the range 0-179 (degrees / 2).
* OpenCV ``BGR2LAB`` stores a* and b* with an offset of 128 (0-255).

Each selected variable is standardized with the training-set mean / standard
deviation and expanded into a pure power polynomial (x, x**2, ..., x**degree
for every variable, no cross terms), matching the original demo application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Standardization:
    """Per-variable mean and standard deviation from the training set."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def apply(self, values: Sequence[float]) -> np.ndarray:
        return (np.asarray(values, dtype=float) - np.asarray(self.mean)) / np.asarray(self.std)


@dataclass(frozen=True)
class ColorFeatures:
    """Result of :func:`extract_features`."""

    mean_rgb: tuple[float, float, float]
    """Mean R, G, B of the ROI (0-255).  Input of the SVM classifier."""

    mean_hsv: tuple[float, float, float]
    """Mean H, S, V of the ROI in OpenCV scaling (H: 0-179, S/V: 0-255)."""

    mean_lab: tuple[float, float, float]
    """Mean L*, a*, b* of the ROI in OpenCV 8-bit scaling (0-255)."""

    raw: tuple[float, float, float]
    """Selected raw variables (R, H, b*)."""

    standardized: tuple[float, float, float]
    """Standardized (R, H, b*)."""

    polynomial: np.ndarray = field(repr=False)
    """Polynomial feature vector fed to the regression model."""


def mean_color_spaces(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean (RGB, HSV, Lab) of an 8-bit BGR image in OpenCV scaling."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {bgr.shape}")
    if bgr.shape[0] == 0 or bgr.shape[1] == 0:
        raise ValueError("empty image region")
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return (
        rgb.reshape(-1, 3).mean(axis=0),
        hsv.reshape(-1, 3).mean(axis=0),
        lab.reshape(-1, 3).mean(axis=0),
    )


def power_polynomial(values: Sequence[float], degree: int) -> np.ndarray:
    """Expand ``values`` into ``[v1, v2, .., v1**2, v2**2, .., v1**degree, ..]``."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    v = np.asarray(values, dtype=float)
    return np.concatenate([v**d for d in range(1, degree + 1)])


def extract_features(bgr: np.ndarray, standardization: Standardization, degree: int) -> ColorFeatures:
    """Compute the RHb* feature set for a BGR region of interest."""
    rgb, hsv, lab = mean_color_spaces(bgr)
    raw = (float(rgb[0]), float(hsv[0]), float(lab[2]))  # R, H, b*
    std = standardization.apply(raw)
    poly = power_polynomial(std, degree)
    return ColorFeatures(
        mean_rgb=tuple(float(x) for x in rgb),
        mean_hsv=tuple(float(x) for x in hsv),
        mean_lab=tuple(float(x) for x in lab),
        raw=raw,
        standardized=tuple(float(x) for x in std),
        polynomial=poly,
    )
