from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ultralytics import YOLO  # noqa: E402

from ocr.pipeline import run_pipeline  # noqa: E402
from ocr.plate_reader import PlateReader  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return YOLO("models/best_detection_model.pt")


@pytest.fixture(scope="module")
def reader():
    return PlateReader("models/best_digit_model.pth")


def test_pipeline_on_blank_image_returns_empty(model, reader, tmp_path):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    p = str(tmp_path / "blank.jpg")
    cv2.imwrite(p, blank)
    assert run_pipeline(p, model, reader) == []


def test_pipeline_finds_plate_in_real_images(model, reader):
    test_dir = Path("data/annotated/images/test")
    images = sorted(test_dir.iterdir())[:10]
    found = any(
        r.get("parse_success")
        for img in images
        for r in run_pipeline(str(img), model, reader)
    )
    assert found, "Expected at least one parse_success=True in 10 test images"
