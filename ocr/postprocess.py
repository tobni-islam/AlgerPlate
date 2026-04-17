from __future__ import annotations

import re
import unicodedata

# Arabic-Indic digits → Latin digits
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
# Extended Arabic-Indic (Persian)
_EXTENDED_ARABIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
# Common OCR substitutions
_OCR_FIXES = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "B": "8", "S": "5"})


def _normalize(text: str) -> str:
    text = text.translate(_ARABIC_INDIC)
    text = text.translate(_EXTENDED_ARABIC)
    # Remove non-breaking spaces and control characters
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C"))
    return text.strip()


def _extract_latin_digits(text: str) -> str:
    """Extract only Latin digit characters from text."""
    return "".join(c for c in text if c.isdigit())


def parse_plate_text(raw_text: str, confidence: float = 0.0) -> dict:
    """
    Parse raw OCR output into structured Algerian plate fields.
    Algerian format: [wilaya 1-2 digits] [serial 6 digits] [year 4 digits]
    Returns dict with: wilaya, serial, year, raw_text, confidence, parse_success
    """
    result = {
        "wilaya": "",
        "serial": "",
        "year": "",
        "raw_text": raw_text,
        "confidence": round(confidence, 4),
        "parse_success": False,
    }
    if not raw_text:
        return result

    normalized = _normalize(raw_text)
    # Apply OCR fixes only to the Latin portion
    latin_fixed = normalized.translate(_OCR_FIXES)
    digits_only = _extract_latin_digits(latin_fixed)

    # Strategy: extract all digit runs, classify by length
    digit_runs = re.findall(r"\d+", latin_fixed)

    year_candidates = [
        r for r in digit_runs if len(r) == 4 and r.startswith(("19", "20"))
    ]
    serial_candidates = [r for r in digit_runs if len(r) == 6]
    wilaya_candidates = [r for r in digit_runs if 1 <= len(r) <= 2]

    if year_candidates:
        result["year"] = year_candidates[-1]  # last 4-digit run
    if serial_candidates:
        result["serial"] = serial_candidates[0]
    if wilaya_candidates:
        result["wilaya"] = wilaya_candidates[0]

    # Fallback: if serial not found but enough digits exist
    if not result["serial"] and len(digits_only) >= 6:
        # Take the longest digit run as serial
        longest = max(digit_runs, key=len, default="")
        if len(longest) >= 6:
            result["serial"] = longest[:6]

    result["parse_success"] = bool(result["serial"])
    return result
