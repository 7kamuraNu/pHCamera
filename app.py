"""pH Camera - Flask web application.

Serves the single-page camera UI (``templates/index.html``) and a JSON
prediction endpoint.  Run locally with::

    python app.py                 # http://localhost:5000
    python app.py --https         # https://<your-ip>:5000 (self-signed, for phones)

or with a WSGI server::

    gunicorn app:app
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import logging
import os
import re
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from phcamera import __version__, load_default_estimator

log = logging.getLogger("phcamera")

MAX_IMAGE_BYTES = 8 * 1024 * 1024
_DATA_URL_RE = re.compile(r"^data:image/[\w.+-]+;base64,", re.IGNORECASE)


class BadRequest(ValueError):
    """Client error that is reported as HTTP 400 with a JSON body."""


def decode_image(payload: str | bytes) -> np.ndarray:
    """Decode a data URL / base64 string / raw bytes into an 8-bit BGR array."""
    if isinstance(payload, str):
        payload = _DATA_URL_RE.sub("", payload.strip(), count=1)
        try:
            raw = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise BadRequest("image is not valid base64") from exc
    else:
        raw = bytes(payload)
    if not raw:
        raise BadRequest("empty image")
    if len(raw) > MAX_IMAGE_BYTES:
        raise BadRequest("image too large")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            rgb = np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise BadRequest("could not decode image") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def crop_roi(bgr: np.ndarray, roi: dict[str, Any] | None) -> np.ndarray:
    """Crop ``roi`` = {x, y, w, h} (pixels, or fractions of the image when all <= 1)."""
    if not roi:
        return bgr
    try:
        x, y, w, h = (float(roi[k]) for k in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise BadRequest("roi must contain numeric x, y, w, h") from exc
    H, W = bgr.shape[:2]
    if max(x, y, w, h) <= 1.0:
        x, y, w, h = x * W, y * H, w * W, h * H
    x0, y0 = int(round(max(0, x))), int(round(max(0, y)))
    x1, y1 = int(round(min(W, x + w))), int(round(min(H, y + h)))
    if x1 <= x0 or y1 <= y0:
        raise BadRequest("roi lies outside the image")
    return bgr[y0:y1, x0:x1]


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False
    estimator = load_default_estimator()
    app.extensions["ph_estimator"] = estimator

    model_info = {
        "regression": {
            "name": f"RHb*_{estimator.regression.degree}",
            "variables": list(estimator.regression.variables),
            "degree": estimator.regression.degree,
        },
        "classification": {
            "name": "RGB linear SVM",
            "classes_ph": [c / estimator.class_scale for c in estimator.classifier.classes.tolist()],
        },
        "override_threshold": estimator.threshold,
        "ph_range": list(estimator.ph_range),
        "version": __version__,
    }

    @app.get("/")
    def index():
        return render_template("index.html", version=__version__)

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok", version=__version__)

    @app.get("/api/info")
    def info():
        return jsonify(model_info)

    @app.post("/api/predict")
    def predict():
        try:
            roi = None
            if request.files.get("image"):
                bgr = decode_image(request.files["image"].read())
                roi = _parse_roi_form(request.form)
            else:
                body = request.get_json(silent=True)
                if body is None:
                    body = request.form
                image = body.get("image") or body.get("base64img")
                if not image:
                    raise BadRequest("missing 'image' (data URL or base64)")
                bgr = decode_image(image)
                roi = body.get("roi")
                if isinstance(roi, str):
                    roi = _parse_roi_string(roi)
            bgr = crop_roi(bgr, roi)
            result = estimator.predict(bgr).to_dict()
            result["roi_size"] = [int(bgr.shape[1]), int(bgr.shape[0])]
            return jsonify(result)
        except BadRequest as exc:
            return jsonify(error=str(exc)), 400

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="request too large"), 413

    return app


def _parse_roi_form(form) -> dict[str, Any] | None:
    keys = ("x", "y", "w", "h")
    if not all(k in form for k in keys):
        return None
    return {k: form[k] for k in keys}


def _parse_roi_string(s: str) -> dict[str, Any]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise BadRequest("roi string must be 'x,y,w,h'")
    return dict(zip(("x", "y", "w", "h"), parts))


app = create_app()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the pH Camera development server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--https", action="store_true",
                        help="serve over HTTPS with a self-signed certificate (needed for phone cameras)")
    parser.add_argument("--cert", help="TLS certificate file (PEM); implies HTTPS")
    parser.add_argument("--key", help="TLS private key file (PEM)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    ssl_context = None
    if args.cert:
        ssl_context = (args.cert, args.key) if args.key else args.cert
    elif args.https:
        ssl_context = "adhoc"  # requires the 'cryptography' package

    logging.basicConfig(level=logging.INFO)
    scheme = "https" if ssl_context else "http"
    log.info("pH Camera %s listening on %s://%s:%d", __version__, scheme, args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug, ssl_context=ssl_context, threaded=True)


if __name__ == "__main__":
    main()
