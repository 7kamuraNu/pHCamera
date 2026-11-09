import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Mean RGB of the SVM support vectors per class (see tests/test_models.py);
# handy "typical strip colors" for integration tests.
STRIP_COLORS = {
    1.0: (216, 96, 81),
    2.0: (211, 113, 75),
    3.0: (215, 131, 72),
    4.0: (176, 141, 62),
    5.0: (103, 123, 48),
    6.0: (69, 101, 47),
    7.0: (60, 93, 51),
    8.0: (44, 68, 55),
    9.0: (44, 54, 64),
    10.0: (40, 42, 61),
    11.0: (41, 40, 59),
}


def uniform_bgr(rgb, size=32):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = np.array(rgb[::-1], dtype=np.uint8)  # RGB -> BGR
    return img


def png_data_url(rgb, size=96):
    im = Image.new("RGB", (size, size), tuple(int(v) for v in rgb))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="session")
def estimator():
    from phcamera import load_default_estimator

    return load_default_estimator()


@pytest.fixture(scope="session")
def flask_app():
    from app import app

    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(flask_app):
    return flask_app.test_client()
