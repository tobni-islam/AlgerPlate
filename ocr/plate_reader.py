from __future__ import annotations

import numpy as np


class PlateReader:
    def __init__(self) -> None:
        # Import here to isolate the slow PaddleOCR import
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang="arabic",
            show_log=False,
            use_gpu=False,
        )

    def read(self, plate_img: np.ndarray) -> tuple[str, float]:
        """Returns (raw_text, confidence). Empty string if nothing detected."""
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
        result = self._ocr.ocr(plate_img, cls=True)
        if not result or not result[0]:
            return "", 0.0
        texts, confs = [], []
        for line in result[0]:
            if line and len(line) >= 2:
                text_info = line[1]
                if text_info and len(text_info) >= 2:
                    texts.append(str(text_info[0]))
                    confs.append(float(text_info[1]))
        if not texts:
            return "", 0.0
        raw = " ".join(texts)
        avg_conf = sum(confs) / len(confs)
        return raw.strip(), avg_conf

    def read_debug(self, plate_img: np.ndarray) -> dict:
        """Extended output for debugging — includes all detected lines."""
        raw_text, confidence = self.read(plate_img)
        result = self._ocr.ocr(plate_img, cls=True)
        lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    lines.append(
                        {"text": line[1][0], "conf": line[1][1], "bbox": line[0]}
                    )
        return {"raw_text": raw_text, "confidence": confidence, "lines": lines}
