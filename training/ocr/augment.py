from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def resize_and_pad_gray(
    img: np.ndarray, target_w: int = 94, target_h: int = 24, pad_value: int = 255
) -> np.ndarray:
    """Resize while preserving aspect ratio, then pad/crop to a fixed canvas.

    LPRNet in the reference repo expects 94x24.
    """
    if img is None or img.size == 0:
        return np.full((target_h, target_w), pad_value, dtype=np.uint8)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((target_h, target_w), pad_value, dtype=np.uint8)

    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(
        img,
        (new_w, target_h),
        interpolation=cv2.INTER_AREA if new_w < w else cv2.INTER_CUBIC,
    )

    if new_w == target_w:
        return resized
    if new_w < target_w:
        pad_left = 0
        pad_right = target_w - new_w
        out = np.pad(
            resized,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )
        return out.astype(np.uint8)

    # If the resized image is wider than the target, center-crop it.
    start = (new_w - target_w) // 2
    return resized[:, start : start + target_w].astype(np.uint8)


def maybe_invert_contrast(
    img: np.ndarray, p: float = 0.2, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if rng.random() >= p:
        return img
    # Plate images are usually dark text on bright background.
    # Some crops may be the opposite after preprocessing.
    return 255 - img


def random_brightness_contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = rng.uniform(0.75, 1.35)  # contrast
    beta = rng.uniform(-25, 25)  # brightness
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def random_gaussian_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(3.0, 18.0)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def random_speckle_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(0.01, 0.05)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + img.astype(np.float32) * noise
    return np.clip(out, 0, 255).astype(np.uint8)


def random_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    choice = rng.integers(0, 3)
    if choice == 0:
        k = int(rng.choice([3, 5]))
        return cv2.GaussianBlur(img, (k, k), 0)
    if choice == 1:
        k = int(rng.choice([3, 5]))
        return cv2.blur(img, (k, k))
    # Motion blur kernel
    k = int(rng.choice([3, 5, 7]))
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def random_jpeg_compression(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    quality = int(rng.integers(25, 95))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(".jpg", img, encode_param)
    if not ok:
        return img
    decoded = cv2.imdecode(
        encoded, cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR
    )
    return decoded if decoded is not None else img


def random_occlusion(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    if h < 4 or w < 4:
        return out

    # 0, 1, or 2 small occlusions
    num_holes = int(rng.integers(0, 3))
    for _ in range(num_holes):
        hole_w = int(rng.integers(max(2, w // 20), max(3, w // 7)))
        hole_h = int(rng.integers(max(2, h // 8), max(3, h // 3)))
        x1 = int(rng.integers(0, max(1, w - hole_w)))
        y1 = int(rng.integers(0, max(1, h - hole_h)))
        color = (
            int(rng.integers(0, 255))
            if out.ndim == 2
            else tuple(int(x) for x in rng.integers(0, 255, size=3))
        )
        out[y1 : y1 + hole_h, x1 : x1 + hole_w] = color
    return out


def random_affine(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild affine transform: rotation, scale, translation."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    angle = float(rng.uniform(-8.0, 8.0))
    scale = float(rng.uniform(0.92, 1.08))
    tx = float(rng.uniform(-0.04 * w, 0.04 * w))
    ty = float(rng.uniform(-0.08 * h, 0.08 * h))

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[:, 2] += [tx, ty]
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255 if img.ndim == 2 else (255, 255, 255),
    )


def random_perspective(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small perspective warp to simulate camera angle."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    dx = w * 0.04
    dy = h * 0.08
    dst = np.float32(
        [
            [rng.uniform(0, dx), rng.uniform(0, dy)],
            [w - 1 - rng.uniform(0, dx), rng.uniform(0, dy)],
            [w - 1 - rng.uniform(0, dx), h - 1 - rng.uniform(0, dy)],
            [rng.uniform(0, dx), h - 1 - rng.uniform(0, dy)],
        ]
    )

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255 if img.ndim == 2 else (255, 255, 255),
    )


def augment_plate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a chain of random augmentations.

    Important: these are mild on purpose. For OCR, too-strong augmentation
    can destroy the character shapes and hurt training.
    """
    out = img.copy()

    # Convert to grayscale early for a stable recognition pipeline.
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # Geometric distortions first.
    if rng.random() < 0.85:
        out = random_affine(out, rng)
    if rng.random() < 0.50:
        out = random_perspective(out, rng)

    # Photometric distortions.
    if rng.random() < 0.80:
        out = random_brightness_contrast(out, rng)
    if rng.random() < 0.45:
        out = random_blur(out, rng)
    if rng.random() < 0.55:
        out = random_gaussian_noise(out, rng)
    if rng.random() < 0.25:
        out = random_speckle_noise(out, rng)
    if rng.random() < 0.30:
        out = random_jpeg_compression(out, rng)
    if rng.random() < 0.35:
        out = random_occlusion(out, rng)
    if rng.random() < 0.15:
        out = maybe_invert_contrast(out, p=1.0, rng=rng)

    return out


def save_augmentation_preview(
    df: pd.DataFrame,
    images_dir: Path,
    out_dir: Path,
    seed: int = 42,
    n: int = 12,
    img_w: int = 94,
    img_h: int = 24,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sample_df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    for i, row in sample_df.iterrows():
        img_path = images_dir / str(row["file_name"])
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        aug = augment_plate(img, rng)
        aug = resize_and_pad_gray(aug, target_w=img_w, target_h=img_h, pad_value=255)
        cv2.imwrite(str(out_dir / f"aug_{i:03d}.jpg"), aug)
