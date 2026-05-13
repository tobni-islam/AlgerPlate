import os
import sys
import time

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


from plate_reader import PlateReader  # noqa: E402
from postprocess import parse_plate_text  # noqa: E402
from preprocess import crop_plate  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def run_pipeline(
    image_path: str,
    model: YOLO,
    reader: PlateReader,
    conf_threshold: float = 0.4,
):
    """
    Full pipeline: image path → list of structured plate dicts.
    One dict per detected plate.
    """
    t_start = time.time()
    image = cv2.imread(image_path)

    if image is None:
        return []

    results = model(image_path, verbose=False, conf=conf_threshold)
    plates = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])

            plate_img = crop_plate(
                image,
                (x1, y1, x2, y2),
            )
            raw_text, avg_conf, min_conf = reader.read(plate_img)
            parsed = parse_plate_text(raw_text, min_conf)

            global_conf = det_conf * min_conf

            parsed["bbox"] = [x1, y1, x2, y2]
            parsed["det_conf"] = round(det_conf, 4)
            parsed["confidence"] = round(global_conf, 4)
            parsed["latency_ms"] = round((time.time() - t_start) * 1000, 1)

            plates.append(parsed)

    return plates
