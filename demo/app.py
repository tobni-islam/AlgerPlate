from __future__ import annotations

import os
import sys
import tempfile

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO  # noqa: E402

from ocr.pipeline import run_pipeline  # noqa: E402
from ocr.plate_reader import PlateReader  # noqa: E402

# Load once at module level — expensive, must not reload per request
_model = YOLO("models/best_detection_model.pt")
_reader = PlateReader("models/best_digit_model.pth")


def _draw_boxes(image_rgb: np.ndarray, plates: list[dict]) -> np.ndarray:
    out = image_rgb.copy()
    for p in plates:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 100), 3)
        label = p["raw_text"] if p["raw_text"] else "plate"
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 100),
            2,
        )
    return out


def detect_plate(image: np.ndarray) -> tuple[np.ndarray, str]:
    if image is None:
        return image, "No image provided."

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, bgr)
        tmp_path = tmp.name
    try:
        plates = run_pipeline(tmp_path, _model, _reader)
    finally:
        os.unlink(tmp_path)

    if not plates:
        return image, "No plate detected in this image."

    annotated = _draw_boxes(image, plates)
    lines = []
    for i, p in enumerate(plates, 1):
        lines.append(
            f"Plate {i}:\n"
            f"  Wilaya : {p['wilaya'] or chr(8212)}\n"
            f"  Serial : {p['serial'] or chr(8212)}\n"
            f"  Year   : {p['year'] or chr(8212)}\n"
            f"  OCR conf : {p['confidence']:.2f} | "
            f"Det conf : {p['det_conf']:.2f} | "
            f"Latency : {p['latency_ms']}ms"
        )
    return annotated, "\n\n".join(lines)


demo = gr.Interface(
    fn=detect_plate,
    inputs=gr.Image(label="Upload plate image (JPEG/PNG)"),
    outputs=[
        gr.Image(label="Detection result"),
        gr.Textbox(label="Extracted plate fields", lines=6),
    ],
    title="AlgerPlate — Algerian License Plate Recognition",
    description=(
        "Upload an image containing an Algerian license plate. "
        "Detects with YOLOv8s (mAP@50: 0.993), "
        "then reads wilaya + serial + year."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
