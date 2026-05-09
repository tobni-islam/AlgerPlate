from datetime import datetime

import numpy as np


def parse_plate_text(raw_text, confidences):
    """
    Post-processes the raw digit string based on Algerian constraints.

    Args:
        raw_text (str): The predicted string '12345612216'
        confidences (list): List of floats for each digit's confidence

    Returns:
        dict: Processed results including validation flags
    """
    # Initialize result
    result = {
        "wilaya": None,
        "serial": None,
        "year": None,
        "vehicule_type": None,
        "raw_text": raw_text,
        "seg_confidence": np.mean(confidences) if confidences else 0,
        "parse_success": False,
        "error_msg": "",
    }

    # Length Constraint (10 or 11 digits)
    plate_len = len(raw_text)
    if plate_len not in [10, 11]:
        result["error_msg"] = f"Invalid length: {plate_len}"
        return result

    try:
        # 2. Slicing logic (Work backwards from the end)
        # Last 2 = Wilaya
        # 2 before that = Year
        # 1 before that = Type
        # Everything else = Serial

        raw_wilaya = raw_text[-2:]
        raw_year = raw_text[-4:-2]
        raw_type = raw_text[-5:-4]
        raw_serial = raw_text[:-5]

        # Validation: Wilaya (01 -> 58)
        wilaya_int = int(raw_wilaya)
        if not (1 <= wilaya_int <= 58):
            result["error_msg"] = f"Invalid Wilaya: {raw_wilaya}"
            return result

        # Validation: Year (No car before 1962)
        year_short = int(raw_year)
        current_year = datetime.now().year  # 2026

        # Determine century
        if year_short >= 62:
            full_year = 1900 + year_short
        else:
            full_year = 2000 + year_short

        if full_year > current_year:
            result["error_msg"] = f"Future year detected: {full_year}"
            return result

        # Finalizing Data
        result["wilaya"] = raw_wilaya
        result["year"] = str(full_year)
        result["vehicule_type"] = raw_type
        result["serial"] = raw_serial
        result["parse_success"] = True

    except ValueError:
        result["error_msg"] = "Non-digit characters encountered"

    return result


def format_algerian_plate(raw_text):
    """
    Separates an Algerian license plate into: Serial-TypeYear-Wilaya
    Example: '12345612216' -> '123456-122-16'
    """
    if not raw_text or len(raw_text) < 6:
        return raw_text

    # Wilaya
    wilaya = raw_text[-2:]

    # Type + Year
    type_year = raw_text[-5:-2]

    # Serial Number
    serial = raw_text[:-5]

    return f"{serial}-{type_year}-{wilaya}"
