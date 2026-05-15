---
license: cc-by-4.0
task_categories:
  - object-detection
language:
  - ar
tags:
  - license-plate
  - algeria
  - yolov8
  - computer-vision
pretty_name: Algerian License Plates Detection Dataset
size_categories:
  - 1K<n<10K
---

# Algerian License Plates Detection Dataset

First open-source annotated Algerian license plate detection dataset.
1950 images with YOLO-format bounding box annotations, collected from
publicly available Algerian media sources.

## Dataset Description

- **Images:** 1950 JPEG images, resolution >= 480px wide
- **Annotations:** YOLO format (.txt) — class cx cy w h normalized to [0,1]
- **Classes:** 1 (class 0 = plate)
- **Annotation tool:** Label Studio
- **Split:** 80/10/10 train/val/test


## Annotation Format

Each .txt label file: one row per plate
```
0 cx cy w h
```
where cx, cy, w, h are normalized to [0, 1]. Class 0 = plate.

## Model trained on this dataset

YOLOv8s achieves mAP@50 = 0.993.
Live demo: https://tobni-algerplate.hf.space

## Citation

```
@dataset{algerian_license_plates_2026,
  author = {Tobni Mohamed Islam},
  title  = {Algerian License Plates Detection Dataset},
  year   = {2026},
  url    = {https://huggingface.co/datasets/tobni/algerian-license-plates}
}
```