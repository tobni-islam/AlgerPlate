import cv2
import numpy as np


def crop_plate(image, bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2].copy()


def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lower_width, upper_width, lower_height, upper_height = dimensions

    # Sort contours by area to filter out tiny noise first
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    img_res = []
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # Check if the contour fits the logic of a digit
        if (
            intWidth > lower_width
            and intWidth < upper_width
            and intHeight > lower_height
            and intHeight < upper_height
        ):
            # Extract the character from the BINARY image
            char = img[intY : intY + intHeight, intX : intX + intWidth]

            # Resize with Aspect Ratio preservation
            char = cv2.resize(char, (20, 40))

            # Create the 44x24 canvas (Black background, White digit)
            char_copy = np.zeros((44, 24), dtype=np.uint8)
            char_copy[2:42, 2:22] = char

            img_res.append((intX, char_copy))

    # Sort characters from LEFT to RIGHT based on X position
    img_res = sorted(img_res, key=lambda t: t[0])

    return np.array([x[1] for x in img_res])


def segment_characters(image):
    if image is None or image.size == 0:
        return np.empty((0, 44, 24), dtype=np.uint8)

    img_lp = cv2.resize(image, (333, 75))
    gray = (
        cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY) if len(img_lp.shape) == 3 else img_lp
    )

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 29, 9
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    h, w = binary.shape
    dims = [w * 0.02723, w * 0.18847, h * 0.25341, h * 0.81407]

    return find_contours(dims, binary)


def segment_hybrid_precision(image):
    if image is None or image.size == 0:
        return np.empty((0, 44, 24), dtype=np.uint8)

    img_lp = cv2.resize(image, (333, 75))
    gray = (
        cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY) if len(img_lp.shape) == 3 else img_lp
    )

    # Aggressive Noise Reduction
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive Thresholding with wide block size
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 43, 12
    )

    # Aggressive Frame Removal
    h, w = binary.shape
    binary[0 : int(h * 0.11), :] = 0  # Top 15%
    binary[int(h * 0.89) : h, :] = 0  # Bottom 15%
    binary[:, 0 : int(w * 0.0295)] = 0  # Left 5%
    binary[:, int(w * 0.9705) : w] = 0  # Right 5%

    # Find Contours
    cntrs, _ = cv2.findContours(
        binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Loose Filter to get all potential candidates
    candidates = []
    for cntr in cntrs:
        x, y, w_c, h_c = cv2.boundingRect(cntr)
        if (w_c < w * 0.227) and (h_c > h * 0.268):
            candidates.append({"x": x, "y": y, "w": w_c, "h": h_c, "cntr": cntr})

    if len(candidates) == 0:
        return np.array([])

    # Find the most common Y-center. Digits always form a straight line.
    y_centers = [c["y"] + c["h"] / 2 for c in candidates]
    median_y = np.median(y_centers)
    median_h = np.median([c["h"] for c in candidates])

    final_chars = []
    for c in candidates:
        y_center = c["y"] + c["h"] / 2
        # If it's on the main horizontal band AND has a logical aspect ratio
        if abs(y_center - median_y) < (median_h * 0.588):
            # Split detection: if it's very wide, it's 2 digits
            if c["w"] > (c["h"] * 0.973):
                w_half = c["w"] // 2
                # Split and add both
                for i in range(2):
                    char_crop = binary[
                        c["y"] : c["y"] + c["h"],
                        c["x"] + (i * w_half) : c["x"] + ((i + 1) * w_half),
                    ]
                    final_chars.append((c["x"] + (i * w_half), char_crop))
            else:
                char_crop = binary[c["y"] : c["y"] + c["h"], c["x"] : c["x"] + c["w"]]
                final_chars.append((c["x"], char_crop))

    # Formatting
    final_chars = sorted(final_chars, key=lambda t: t[0])
    output = []
    for _, char_img in final_chars:
        char_res = cv2.resize(char_img, (20, 40))
        canvas = np.zeros((44, 24), dtype=np.uint8)
        canvas[2:42, 2:22] = char_res
        output.append(canvas)

    return np.array(output)


def segment_hybrid_precision_v2(image):
    if image is None or image.size == 0:
        return np.empty((0, 44, 24), dtype=np.uint8)

    # Standardize and Preprocess
    img_lp = cv2.resize(image, (333, 75))
    gray = (
        cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY) if len(img_lp.shape) == 3 else img_lp
    )
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive Thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 43, 12
    )

    # Aggressive Frame Removal
    h, w = binary.shape
    binary[0 : int(h * 0.11), :] = 0
    binary[int(h * 0.89) : h, :] = 0
    binary[:, 0 : int(w * 0.0295)] = 0
    binary[:, int(w * 0.9705) : w] = 0

    # Find Contours
    cntrs, _ = cv2.findContours(
        binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for cntr in cntrs:
        x, y, w_c, h_c = cv2.boundingRect(cntr)

        # Solidity = Area of contour / Area of bounding box
        # Real digits usually have solidity between 0.3 and 0.8
        area = cv2.contourArea(cntr)
        solidity = area / float(w_c * h_c) if w_c * h_c > 0 else 0

        if (w_c < w * 0.227) and (h_c > h * 0.268) and (0.2 < solidity < 0.9):
            candidates.append(
                {"x": x, "y": y, "w": w_c, "h": h_c, "solidity": solidity}
            )

    if not candidates:
        return np.array([])

    # SPATIAL CONSENSUS
    y_centers = [c["y"] + c["h"] / 2 for c in candidates]
    median_y = np.median(y_centers)
    median_h = np.median([c["h"] for c in candidates])

    median_w = np.median([c["w"] for c in candidates])

    final_chars = []
    for c in candidates:
        y_center = c["y"] + c["h"] / 2
        if abs(y_center - median_y) < (median_h * 0.588):
            # Dynamic Split Logic
            # If the blob is significantly wider than the other characters on THIS plate
            if c["w"] > (median_w * 1.6):
                w_half = c["w"] // 2
                for i in range(2):
                    char_crop = binary[
                        c["y"] : c["y"] + c["h"],
                        c["x"] + (i * w_half) : c["x"] + ((i + 1) * w_half),
                    ]
                    final_chars.append((c["x"] + (i * w_half), char_crop))
            else:
                char_crop = binary[c["y"] : c["y"] + c["h"], c["x"] : c["x"] + c["w"]]
                final_chars.append((c["x"], char_crop))

    # Formatting
    final_chars = sorted(final_chars, key=lambda t: t[0])
    output = []
    for _, char_img in final_chars:
        # Final sanity check: skip very thin noise that might have survived
        if char_img.shape[1] < 2:
            continue

        char_res = cv2.resize(char_img, (20, 40))
        canvas = np.zeros((44, 24), dtype=np.uint8)
        canvas[2:42, 2:22] = char_res
        output.append(canvas)

    return np.array(output)


def ensemble_voting_system(image):
    # Get results from all specialists
    chars1 = segment_characters(image)
    chars2 = segment_hybrid_precision(image)
    chars3 = segment_hybrid_precision_v2(image)

    results = [chars1, chars2, chars3]
    lengths = [len(chars1), len(chars2), len(chars3)]
    valid_lengths = [10, 11]

    # Check which functions produced valid lengths
    valid_mask = [length in valid_lengths for length in lengths]

    # LOGIC 1: Multiple Valid Hits
    if sum(valid_mask) > 1:
        if valid_mask[1]:
            return chars2
        elif valid_mask[2]:
            return chars3
        return chars1

    # LOGIC 2: Single Valid Hit
    if sum(valid_mask) == 1:
        # Find the index of the True value
        winner_idx = valid_mask.index(True)
        return results[winner_idx]

    # LOGIC 3: Recovery Mode (No one hit 10 or 11)
    # We find the result closest to the 10-11 range.
    distances = [abs(length - 10.5) for length in lengths]
    min_distance = min(distances)

    # If there's a tie in distance, we prefer Function 3
    # because it has the lowest noise (over-segmentation)
    best_indices = [i for i, d in enumerate(distances) if d == min_distance]

    if 2 in best_indices:
        return chars3
    elif 1 in best_indices:
        return chars2
    else:
        return chars1
