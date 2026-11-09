# Model files

| File | What it is | Used by the app |
|------|------------|-----------------|
| `rhb_poly_regression.json` | Parameters of the RHb\* polynomial regression (coefficients, intercept, standardization constants, feature order). Generated from `RHb_model.sav`. | yes |
| `rgb_linear_svm.json` | Parameters of the linear SVM pH classifier (11 classes = pH 1 … 11, one-vs-one weights). Generated from `RGB_SVM_model.sav`. | yes |
| `RHb_model.sav` | Original scikit-learn 0.22 pickle (`LinearRegression`, 15 coefficients = 3 variables × 5 powers). | no (source of the JSON) |
| `RGB_SVM_model.sav` | Original scikit-learn 0.22 pickle (`SVC`, linear kernel, trained on 440 images). | no (source of the JSON) |
| `archive/model.sav` | Older scikit-learn 0.19 `Ridge` pickle (35 coefficients) from early experiments. Kept for provenance only; its input variables are undocumented. | no |

The application reads only the JSON files, so it does not depend on the
scikit-learn version that produced the pickles (the `SVC` pickle cannot be
used directly with current scikit-learn releases).  Regenerate the JSON with

```bash
pip install scikit-learn
python scripts/export_models.py
```

The script verifies that the exported parameters reproduce the pickles'
predictions before writing anything; `tests/test_models.py` repeats the check.

## Input definition

Mean color of the cropped test-strip region, computed with OpenCV 8-bit
conversions (`BGR2HSV`: H in 0–179; `BGR2LAB`: a\*, b\* offset by 128).

| Variable | Color space | Training mean | Training std |
|----------|-------------|---------------|--------------|
| R  | RGB   | 106.876 | 79.185 |
| H  | HSV   | 53.698  | 38.698 |
| b\* | L\*a\*b\* | 154.324 | 27.214 |

Regression features: `[R, H, b*, R², H², b*², …, R⁵, H⁵, b*⁵]` of the
standardized variables (no cross terms).  Classifier features: raw mean
`[R, G, B]` (0–255).
