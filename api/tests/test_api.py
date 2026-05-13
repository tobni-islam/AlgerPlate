from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def blank_jpg():
    """Valid JPEG with no plates  tests that the API accepts images."""
    img = np.zeros((480, 640, 3), dtype=np.uint8) + 128
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_model_loaded_true(client):
    data = client.get("/health").json()
    assert data["model_loaded"] is True


def test_metrics_has_required_fields(client):
    data = client.get("/metrics").json()
    assert "request_count" in data
    assert "avg_latency_ms" in data
    assert "total_detections" in data


def test_detect_valid_image_returns_200(client, blank_jpg):
    r = client.post(
        "/detect",
        files={
            "file": (
                "@data/annotated/images/test/0bf13796-img_0789.jpg",
                blank_jpg,
                "image/jpeg",
            )
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "plates" in data
    assert "total_latency_ms" in data
    assert isinstance(data["plates"], list)


def test_detect_non_image_returns_400(client):
    r = client.post(
        "/detect",
        files={
            "file": (
                "@data/annotated/labels/test/0bf13796-img_0789.txt",
                b"not an image",
                "text/plain",
            )
        },
    )
    assert r.status_code == 400


def test_detect_latency_under_1500ms(client, blank_jpg):
    t = time.time()
    r = client.post(
        "/detect",
        files={
            "file": (
                "@data/annotated/images/test/0bf13796-img_0789.jpg",
                blank_jpg,
                "image/jpeg",
            )
        },
    )
    ms = (time.time() - t) * 1000
    assert r.status_code == 200
    assert ms < 1500, f"Latency {ms:.0f}ms exceeded 1500ms limit"
