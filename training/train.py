from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


def _setup_mlflow() -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set in .env")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("algerplate-detection")


def train(
    data: str = "data/dataset.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 0.01,
    patience: int = 20,
    optimizer: str = "AdamW",
    name: str = "yolov8s-run",
) -> None:
    _setup_mlflow()
    params = dict(
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        patience=patience,
        optimizer=optimizer,
    )

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param("model_base", "yolov8s.pt")
        mlflow.log_param("dataset", data)

        model = YOLO("yolov8s.pt")
        t0 = time.time()
        results = model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            patience=patience,
            optimizer=optimizer,
            name=name,
            exist_ok=True,
        )
        elapsed = time.time() - t0

        # Log validation metrics from the best epoch
        metrics = results.results_dict
        mlflow.log_metric("val/mAP50", metrics.get("metrics/mAP50(B)", 0.0))
        mlflow.log_metric("val/mAP50_95", metrics.get("metrics/mAP50-95(B)", 0.0))
        mlflow.log_metric("val/precision", metrics.get("metrics/precision(B)", 0.0))
        mlflow.log_metric("val/recall", metrics.get("metrics/recall(B)", 0.0))
        mlflow.log_metric("train_time_sec", elapsed)

        # Log best weights as artifact
        best_pt = Path(f"training/runs/detect/{name}/weights/best.pt")
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
            # Copy to models/ for local use
            import shutil

            shutil.copy(best_pt, "models/best.pt")
            print("best.pt copied to models/best.pt")

        print(f"\nmAP@50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
        print(f"Training time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--name", default="yolov8s-ep50")
    args = parser.parse_args()
    train(**vars(args))
