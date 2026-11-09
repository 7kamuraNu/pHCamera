import io

import pytest
from PIL import Image

from tests.conftest import STRIP_COLORS, png_data_url


def test_index_serves_camera_page(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "getUserMedia" not in html  # JS lives in static/app.js
    assert 'id="video"' in html and 'id="measureBtn"' in html
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_healthz_and_info(client):
    assert client.get("/healthz").get_json()["status"] == "ok"
    info = client.get("/api/info").get_json()
    assert info["regression"]["variables"] == ["R", "H", "b*"]
    assert info["classification"]["classes_ph"] == [float(k) for k in range(1, 12)]
    assert info["override_threshold"] == 1.0


@pytest.mark.parametrize("ph,rgb", sorted(STRIP_COLORS.items()))
def test_predict_json_data_url(client, ph, rgb):
    r = client.post("/api/predict", json={"image": png_data_url(rgb)})
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["ph_classification"] == ph
    assert abs(j["ph"] - ph) < 1.0
    assert j["source"] in ("regression", "classification")
    assert j["roi_size"] == [96, 96]
    assert len(j["mean_rgb"]) == 3


def test_predict_accepts_raw_base64_without_data_url_prefix(client):
    url = png_data_url(STRIP_COLORS[5.0])
    r = client.post("/api/predict", json={"image": url.split(",", 1)[1]})
    assert r.status_code == 200


def test_predict_legacy_form_field(client):
    r = client.post("/api/predict", data={"base64img": png_data_url(STRIP_COLORS[7.0])})
    assert r.status_code == 200
    assert r.get_json()["ph_classification"] == 7.0


def test_predict_with_fractional_roi(client):
    r = client.post("/api/predict", json={"image": png_data_url(STRIP_COLORS[4.0], size=400),
                                          "roi": {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}})
    assert r.status_code == 200
    assert r.get_json()["roi_size"] == [80, 80]


def test_predict_with_pixel_roi_string(client):
    r = client.post("/api/predict", data={"image": png_data_url(STRIP_COLORS[4.0], size=200), "roi": "50,50,100,100"})
    assert r.status_code == 200
    assert r.get_json()["roi_size"] == [100, 100]


def test_predict_multipart_upload(client):
    buf = io.BytesIO()
    Image.new("RGB", (60, 40), STRIP_COLORS[9.0]).save(buf, "JPEG", quality=95)
    buf.seek(0)
    r = client.post("/api/predict", data={"image": (buf, "strip.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["roi_size"] == [60, 40]


@pytest.mark.parametrize("payload", [
    {},
    {"image": ""},
    {"image": "data:image/png;base64,AAAA"},
    {"image": "not base64 at all!!"},
])
def test_predict_bad_input_returns_400(client, payload):
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_roi_outside_image_returns_400(client):
    r = client.post("/api/predict", json={"image": png_data_url((1, 2, 3)), "roi": {"x": 500, "y": 500, "w": 10, "h": 10}})
    assert r.status_code == 400
