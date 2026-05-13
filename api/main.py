from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

from api.pipeline import run_inference
from api.schemas import DetectionResponse, HealthResponse, MetricsResponse, PlateResult
from ocr.plate_reader import PlateReader

_state: dict = {
    "model": None,
    "reader": None,
    "request_count": 0,
    "total_latency_ms": 0.0,
    "total_detections": 0,
}

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = YOLO("/app/models/best_detection_model.pt")
    _state["reader"] = PlateReader("/app/models/best_digit_model.pth")
    yield


app = FastAPI(
    title="AlgerPlate API",
    description="Algerian license plate detection and OCR",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)) -> DetectionResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type: {file.content_type}. Use JPEG or PNG.",
        )
    image_bytes = await file.read()
    t_start = time.time()
    plates = run_inference(image_bytes, _state["model"], _state["reader"])
    total_ms = round((time.time() - t_start) * 1000, 1)

    _state["request_count"] += 1
    _state["total_latency_ms"] += total_ms
    _state["total_detections"] += len(plates)

    return DetectionResponse(
        plates=[PlateResult(**p) for p in plates],
        image_name=file.filename or "unknown",
        total_latency_ms=total_ms,
        plate_count=len(plates),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    count = _state["request_count"]
    avg = round(_state["total_latency_ms"] / count, 1) if count > 0 else 0.0
    return MetricsResponse(
        request_count=count,
        avg_latency_ms=avg,
        total_detections=_state["total_detections"],
    )
