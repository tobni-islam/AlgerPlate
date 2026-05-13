from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ocr.pipeline import run_pipeline  # noqa: E402
from ocr.plate_reader import PlateReader  # noqa: E402


def char_accuracy(predicted: str, ground_truth: str) -> tuple[int, int]:
    correct = sum(p == g for p, g in zip(predicted, ground_truth))
    total = max(len(predicted), len(ground_truth))
    return correct, total


def main(gt_csv: str, weights: str) -> None:
    model = YOLO(weights)
    reader = PlateReader(
        "/home/islam_tb/Documents/AlgerPlate/models/best_digit_model.pth"
    )
    img_dir = Path("/home/islam_tb/Documents/AlgerPlate/data/raw/unclassified")

    serial_correct = serial_total = 0
    wilaya_correct = wilaya_total = 0
    full_match = 0
    rows_evaluated = 0

    first = True
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            img_path = img_dir / row["filename"]
            if not img_path.exists():
                print(f"SKIP (not found): {row['filename']}")
                continue

            # display the first image
            if first:
                Image.open(img_path).show()
                first = False

            results = run_pipeline(str(img_path), model, reader)
            if not results:
                print(f"NO DETECTION: {row['filename']}")
                continue

            pred = results[0]
            sc, st = char_accuracy(pred["serial"], row["serial"])
            wc, wt = char_accuracy(pred["wilaya"], row["wilaya"])
            serial_correct += sc
            serial_total += st
            wilaya_correct += wc
            wilaya_total += wt
            if pred["serial"] == row["serial"] and pred["wilaya"] == row["wilaya"]:
                full_match += 1
            rows_evaluated += 1
            print(
                f"{row['filename']}: serial={pred['serial']!r}(gt={row['serial']!r}) "
                f"wilaya={pred['wilaya']!r}(gt={row['wilaya']!r})"
            )

    print(f"\n=== OCR ACCURACY ({rows_evaluated} plates) ===")
    if serial_total:
        print(f"Latin serial accuracy:  {serial_correct/serial_total*100:.1f}%")
    if wilaya_total:
        print(f"Arabic wilaya accuracy: {wilaya_correct/wilaya_total*100:.1f}%")
    print(f"Full plate match:       {full_match/rows_evaluated*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument(
        "--weights",
        default="/home/islam_tb/Documents/AlgerPlate/models/best_detection_model.pt",
    )
    args = parser.parse_args()
    main(args.gt, args.weights)
