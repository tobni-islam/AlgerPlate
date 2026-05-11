from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
from ultralytics import YOLO

from ocr.pipeline import run_pipeline as _ocr_pipeline
from ocr.plate_reader import PlateReader


def run_inference(
    image_bytes: bytes,
    model: YOLO,
    reader: PlateReader,
) -> list[dict]:
    """
    Accept raw image bytes from a FastAPI UploadFile.
    Returns list of plate dicts from the full detection + OCR pipeline.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return []

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        tmp_path = tmp.name

    try:
        return _ocr_pipeline(tmp_path, model, reader)
    finally:
        os.unlink(tmp_path)
