import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from preprocess import ensemble_voting_system  # noqa: E402
from torchvision import transforms  # noqa: E402


class DigitClassifierTL(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitClassifierTL, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class PlateReader:
    def __init__(self, model_path, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load Model
        self.model = DigitClassifierTL().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Define Transforms
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _recognize_characters(self, chars_list):
        """Internal helper for digit classification."""
        predicted_text = ""
        confidences = []

        for char_np in chars_list:
            char_pil = Image.fromarray(char_np).convert("RGB")
            input_tensor = self.transforms(char_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred_label = torch.max(probs, dim=1)

                predicted_text += str(pred_label.item())
                confidences.append(conf.item())

        return predicted_text, confidences

    def _apply_smart_repair(self, text_list):
        """Internal logic for Algerian plate semantic rules."""
        try:
            # Wilaya Repair
            wilaya_val = int("".join(text_list[-2:]))
            if not (1 <= wilaya_val <= 58):
                if text_list[-2] == "8":
                    text_list[-2] = "3"
                elif text_list[-2] == "6":
                    text_list[-2] = "0"
                elif text_list[-2] == "7":
                    text_list[-2] = "1"
                elif text_list[-2] == "9":
                    text_list[-2] = "1"
                elif "".join(text_list[-2:]) == "59":
                    text_list[-1] = "1"

            # Year Repair
            year_val = int("".join(text_list[-4:-2]))
            full_year = 1900 + year_val if year_val >= 62 else 2000 + year_val
            if not (1962 <= full_year <= 2026):
                if text_list[-4] == "3":
                    text_list[-4] = "8"
                elif text_list[-4] == "4":
                    text_list[-4] = "0"
                elif text_list[-4] == "5":
                    text_list[-4] = "9"
        except (ValueError, IndexError):
            pass
        return "".join(text_list)

    def read(self, plate_img):
        """Production method: Returns text, avg_conf, min_conf."""
        # Segmentation
        segmented_chars = ensemble_voting_system(plate_img)
        if len(segmented_chars) == 0:
            return "", 0.0, 0.0

        # Recognition
        text, confs = self._recognize_characters(segmented_chars)

        # Edge Case: Length 9 (Aspect Ratio Split)
        if len(text) == 9:
            widths = [c.shape[1] for c in segmented_chars]
            idx = np.argmax(widths)
            h, w = segmented_chars[idx].shape[:2]
            if (w / h) > 0.5:
                mid = w // 2
                t_l, c_l = self._recognize_characters([segmented_chars[idx][:, :mid]])
                t_r, c_r = self._recognize_characters([segmented_chars[idx][:, mid:]])
                text = text[:idx] + t_l + t_r + text[idx + 1 :]
                confs = confs[:idx] + c_l + c_r + confs[idx + 1 :]

        # Edge Case: Length > 11 (Lowest Confidence Drop)
        while len(text) > 11:
            min_idx = np.argmin(confs)
            text = text[:min_idx] + text[min_idx + 1 :]
            confs.pop(min_idx)

        # Integrity Check
        if len(text) < 10:
            return "", 0.0, 0.0

        # Smart Repair
        final_text = self._apply_smart_repair(list(text))

        return final_text, np.mean(confs), np.min(confs)

    def read_debug(self, plate_img):
        """Diagnostic method: Returns detailed dictionary of the process."""
        segmented_chars = ensemble_voting_system(plate_img)
        raw_text, confs = self._recognize_characters(segmented_chars)

        # Capture state before repairs
        debug_info = {
            "initial_segment_count": len(segmented_chars),
            "raw_prediction": raw_text,
            "raw_confidences": confs,
            "final_text": "",
            "status": "Success",
        }

        final_text, avg_c, min_c = self.read(plate_img)

        debug_info["final_text"] = final_text
        debug_info["avg_conf"] = avg_c
        debug_info["min_conf"] = min_c

        if final_text == "" and len(raw_text) > 0:
            debug_info["status"] = "Rejected by Integrity Guard (Length < 10)"
        elif not segmented_chars:
            debug_info["status"] = "Failed: No characters segmented"

        return debug_info
