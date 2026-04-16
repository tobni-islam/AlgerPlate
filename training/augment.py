from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np


class AugmentationPipeline:
    def __init__(
        self, output_dir: str | Path, copies_per_image: int = 5, seed: int = 42
    ) -> None:
        self.output_dir = Path(output_dir)
        self.copies_per_image = copies_per_image
        self.seed = seed
        self._transform = self._build_transform()
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

    def _build_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.RandomBrightnessContrast(p=0.6),
                A.MotionBlur(blur_limit=7, p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Perspective(scale=(0.02, 0.08), p=0.4),
                A.HueSaturationValue(p=0.3),
                A.RandomShadow(p=0.2),
                A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),
                A.CLAHE(p=0.2),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.3,
            ),
        )

    def _read_yolo_labels(self, label_path: Path) -> tuple[list, list]:
        class_labels, bboxes = [], []
        if not label_path.exists():
            return class_labels, bboxes
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
        return class_labels, bboxes

    def _write_yolo_labels(
        self, label_path: Path, class_labels: list, bboxes: list
    ) -> None:
        with open(label_path, "w") as f:
            for cls, bbox in zip(class_labels, bboxes):
                f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")

    def augment_image(self, image_path: Path, label_path: Path) -> int:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: Cannot read {image_path}, skipping.")
            return 0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_labels, bboxes = self._read_yolo_labels(label_path)
        if not bboxes:
            print(f"WARNING: No labels for {image_path.name}, skipping.")
            return 0
        written = 0
        for i in range(self.copies_per_image):
            try:
                result = self._transform(
                    image=image, bboxes=bboxes, class_labels=class_labels
                )
                if not result["bboxes"]:
                    continue
                aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
                stem = image_path.stem
                out_img = self.output_dir / "images" / f"{stem}_aug{i:02d}.jpg"
                out_lbl = self.output_dir / "labels" / f"{stem}_aug{i:02d}.txt"
                cv2.imwrite(str(out_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                self._write_yolo_labels(
                    out_lbl, result["class_labels"], result["bboxes"]
                )
                written += 1
            except Exception as e:
                print(f"ERROR on {image_path.name} copy {i}: {e}")
        return written

    def run(self, images_dir: str | Path, labels_dir: str | Path) -> dict:
        images_dir, labels_dir = Path(images_dir), Path(labels_dir)
        np.random.seed(self.seed)
        stats = {"processed": 0, "written": 0, "skipped": 0}
        image_files = sorted(
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for img_path in image_files:
            lbl_path = labels_dir / (img_path.stem + ".txt")
            n = self.augment_image(img_path, lbl_path)
            stats["processed"] += 1
            stats["written"] += n
            if n == 0:
                stats["skipped"] += 1
        print(
            "Done: "
            f"{stats['processed']} processed | "
            f"{stats['written']} written | "
            f"{stats['skipped']} skipped"
        )
        return stats


if __name__ == "__main__":
    pipeline = AugmentationPipeline(
        output_dir="data/augmented", copies_per_image=5, seed=42
    )
    pipeline.run(
        images_dir="data/annotated/images/train",
        labels_dir="data/annotated/labels/train",
    )
