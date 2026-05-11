from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from ultralytics import YOLO

from api.schemas import HealthResponse
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
    _state["model"] = YOLO(
        "/home/islam_tb/Documents/AlgerPlate/models/best_detection_model.pt"
    )
    _state["reader"] = PlateReader(
        "/home/islam_tb/Documents/AlgerPlate/models/best_digit_model.pth"
    )
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
