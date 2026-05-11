from __future__ import annotations

from pydantic import BaseModel


class PlateResult(BaseModel):
    wilaya: str
    serial: str
    year: str
    raw_text: str
    confidence: float
    parse_success: bool
    bbox: list[int]
    det_conf: float
    latency_ms: float


class DetectionResponse(BaseModel):
    plates: list[PlateResult]
    image_name: str
    total_latency_ms: float
    plate_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    request_count: int
    avg_latency_ms: float
    total_detections: int
