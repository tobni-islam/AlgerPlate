from __future__ import annotations

import time

import cv2
from ultralytics import YOLO

from ocr.plate_reader import PlateReader
from ocr.postprocess import parse_plate_text
from ocr.preprocess import preprocess_plate


def run_pipeline(
    image_path: str,
    model: YOLO,
    reader: PlateReader,
    confidence_threshold: float = 0.4,
    apply_perspective: bool = True,
) -> list[dict]:
    """
    Full pipeline: image path → list of structured plate dicts.
    One dict per detected plate.
    """
    t_start = time.time()
    image = cv2.imread(image_path)
    if image is None:
        return []

    results = model(image_path, verbose=False, conf=confidence_threshold)
    plates = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])

            plate_img = preprocess_plate(
                image,
                (x1, y1, x2, y2),
                apply_perspective=apply_perspective,
            )
            raw_text, ocr_conf = reader.read(plate_img)
            parsed = parse_plate_text(raw_text, ocr_conf)
            parsed["bbox"] = [x1, y1, x2, y2]
            parsed["det_conf"] = round(det_conf, 4)
            parsed["latency_ms"] = round((time.time() - t_start) * 1000, 1)

            plates.append(parsed)

    return plates
