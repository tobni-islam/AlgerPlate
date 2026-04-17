from __future__ import annotations

import cv2
import numpy as np


def crop_plate(image: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2].copy()


def correct_perspective(plate_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_img
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return plate_img  # can't find 4 corners — return unchanged
    pts = approx.reshape(4, 2).astype(np.float32)
    # Order: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ],
        dtype=np.float32,
    )
    h, w = plate_img.shape[:2]
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(plate_img, M, (w, h))


def binarize(plate_img: np.ndarray) -> np.ndarray:
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
    # Adaptive thresholding handles uneven lighting better than Otsu
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return binary


def resize_for_ocr(plate_img: np.ndarray, target_height: int = 48) -> np.ndarray:
    h, w = plate_img.shape[:2]
    if h == 0:
        return plate_img
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(
        plate_img, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4
    )


def preprocess_plate(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    apply_perspective: bool = True,
) -> np.ndarray:
    plate = crop_plate(image, bbox_xyxy)
    if apply_perspective:
        plate = correct_perspective(plate)
    plate = binarize(plate)
    plate = resize_for_ocr(plate)
    return plate
