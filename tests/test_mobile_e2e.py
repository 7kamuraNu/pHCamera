"""End-to-end smartphone emulation test.

Runs the Flask app, opens it in headless Chromium emulating an iPhone 13
(viewport, device pixel ratio, touch, mobile user agent) and feeds the browser
a synthetic camera stream (a pH-5-colored strip on white paper, written as a
Y4M file).  It then checks that the live pH reading appears, that the region
of interest is mapped to the right pixels, and that MEASURE stores a result.

Requirements: ``pip install playwright && playwright install chromium``.
The test is skipped automatically when Playwright or the browser is missing.

Set ``PHCAM_E2E_SCREENSHOT=1`` to save ``docs/e2e_iphone_emulation.png``.
"""

from __future__ import annotations

import os
import re
import socket
import threading
from pathlib import Path

import numpy as np
import pytest
from werkzeug.serving import make_server

from tests.conftest import STRIP_COLORS

playwright_sync = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parent.parent
TARGET_PH = 5.0
TARGET_RGB = STRIP_COLORS[TARGET_PH]


def write_y4m(path: Path, rgb, width=640, height=480, square=320, frames=10):
    """Write a Y4M (I420, BT.601 limited range) clip: colored square centered on white."""
    frame_rgb = np.full((height, width, 3), 255, dtype=np.float64)
    y0, x0 = (height - square) // 2, (width - square) // 2
    frame_rgb[y0 : y0 + square, x0 : x0 + square] = rgb
    r, g, b = (frame_rgb[..., i] / 255.0 for i in range(3))
    y = 16 + 65.481 * r + 128.553 * g + 24.966 * b
    u = 128 - 37.797 * r - 74.203 * g + 112.0 * b
    v = 128 + 112.0 * r - 93.786 * g - 18.214 * b

    def plane(a, sub=False):
        if sub:
            a = a.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
        return np.clip(np.round(a), 0, 255).astype(np.uint8).tobytes()

    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{width} H{height} F30:1 Ip A1:1 C420jpeg\n".encode())
        frame = b"FRAME\n" + plane(y) + plane(u, True) + plane(v, True)
        for _ in range(frames):
            f.write(frame)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url(flask_app):
    port = free_port()
    srv = make_server("127.0.0.1", port, flask_app, threaded=True)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()


@pytest.fixture(scope="module")
def fake_camera(tmp_path_factory):
    path = tmp_path_factory.mktemp("cam") / "strip.y4m"
    write_y4m(path, TARGET_RGB)
    return path


def test_iphone_emulation_live_reading_and_measure(server_url, fake_camera):
    with playwright_sync.sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    f"--use-file-for-fake-video-capture={fake_camera}",
                ]
            )
        except playwright_sync.Error as exc:  # browser binaries not installed
            pytest.skip(f"chromium not available: {exc}")

        device = {k: v for k, v in p.devices["iPhone 13"].items() if k != "default_browser_type"}
        context = browser.new_context(**device, permissions=["camera"], locale="en-US")
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server_url + "/?lang=en")

        # Camera stream started and a live pH value is shown.
        page.wait_for_function("document.getElementById('video').videoWidth > 0", timeout=15000)
        page.wait_for_function(
            "/^\\d+\\.\\d\\d$/.test(document.getElementById('readout').textContent)", timeout=15000
        )
        readout = page.inner_text("#readout")
        assert re.fullmatch(r"\d+\.\d\d", readout)

        # The JSON the page received should describe the colored square, not the white paper.
        last = page.evaluate("window.__phcam.state.last")
        assert last["source"] in ("regression", "classification")
        mean_rgb = np.array(last["mean_rgb"])
        assert np.all(np.abs(mean_rgb - np.array(TARGET_RGB)) < 10), (mean_rgb, TARGET_RGB)
        assert abs(last["ph"] - TARGET_PH) < 1.0, last

        # Viewport is a phone, ROI box lies inside the video frame.
        vp = page.viewport_size
        assert vp["width"] < 500
        roi = page.evaluate("window.__phcam.roiInSource()")
        assert 0 < roi["x"] < 640 and 0 < roi["y"] < 480 and roi["w"] > 20

        # MEASURE stores a swatch in "Previous Results".
        assert page.is_enabled("#measureBtn")
        page.tap("#measureBtn")
        page.wait_for_selector("#results li")
        assert page.locator("#results li").count() == 1
        assert page.inner_text("#results li .val").startswith(f"{round(last['ph'], 1):.1f}")

        if os.environ.get("PHCAM_E2E_SCREENSHOT"):
            page.screenshot(path=str(ROOT / "docs" / "e2e_iphone_emulation.png"))

        assert errors == [], errors
        browser.close()


def test_insecure_context_shows_https_hint(server_url, fake_camera):
    """Over plain http from a non-localhost origin the page must explain the HTTPS requirement."""
    with playwright_sync.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright_sync.Error as exc:
            pytest.skip(f"chromium not available: {exc}")
        device = {k: v for k, v in p.devices["iPhone 13"].items() if k != "default_browser_type"}
        context = browser.new_context(**device, locale="en-US")
        page = context.new_page()
        # Pretend the page is not a secure context (what happens on http://<LAN-ip> from a phone).
        page.add_init_script("Object.defineProperty(window, 'isSecureContext', {get: () => false});")
        page.goto(server_url + "/?lang=en")
        page.wait_for_selector("#notice:not([hidden])")
        assert "HTTPS" in page.inner_text("#notice")
        assert page.is_disabled("#measureBtn")
        assert page.locator("#fileInput").count() == 1  # photo fallback is offered
        browser.close()
