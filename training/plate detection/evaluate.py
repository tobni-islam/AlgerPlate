from __future__ import annotations

import argparse
import os

import mlflow
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


def evaluate(weights: str = "models/best.pt", data: str = "data/dataset.yaml") -> None:
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("algerplate-detection")

    model = YOLO(weights)
    metrics = model.val(data=data, split="test")

    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    with mlflow.start_run(run_name="test-set-evaluation"):
        mlflow.log_param("weights", weights)
        mlflow.log_param("split", "test")
        mlflow.log_metric("test/mAP50", map50)
        mlflow.log_metric("test/mAP50_95", map50_95)
        mlflow.log_metric("test/precision", precision)
        mlflow.log_metric("test/recall", recall)

    print("\n=== TEST SET RESULTS ===")
    print(f"mAP@50:       {map50:.4f}")
    print(f"mAP@50-95:    {map50_95:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--data", default="data/dataset.yaml")
    args = parser.parse_args()
    evaluate(args.weights, args.data)
