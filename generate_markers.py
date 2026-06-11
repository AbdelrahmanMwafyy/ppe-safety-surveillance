"""
generate_markers.py — Generate ArUco marker images to print

Generates IDs 0-3 from DICT_4X4_50
Saves as high-resolution PNG files ready to print at any size
"""

import cv2
import numpy as np
import os

OUTPUT_DIR   = "aruco_markers"
ARUCO_DICT   = cv2.aruco.DICT_4X4_50
MARKER_IDS   = [0, 1, 2, 3]
PERSONS      = ["Mohamed", "Ahmed", "Khaled", "Youssef"]

# Image size in pixels — 1000px = crisp at any print size
IMG_SIZE     = 1000
BORDER_BITS  = 1   # white border around marker (required)

os.makedirs(OUTPUT_DIR, exist_ok=True)
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

for marker_id, person in zip(MARKER_IDS, PERSONS):
    # Generate marker image
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, IMG_SIZE)

    # Add white border (20px each side)
    border = 60
    img_with_border = cv2.copyMakeBorder(
        marker_img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255
    )

    # Add label below marker
    label_height = 80
    labeled = np.ones((img_with_border.shape[0] + label_height,
                       img_with_border.shape[1]), dtype=np.uint8) * 255
    labeled[:img_with_border.shape[0], :] = img_with_border

    # Put text label
    label = f"ID: {marker_id}  |  {person}"
    cv2.putText(labeled, label,
                (20, img_with_border.shape[0] + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)

    filename = os.path.join(OUTPUT_DIR, f"marker_{marker_id}_{person}.png")
    cv2.imwrite(filename, labeled)
    print(f"Saved: {filename}  ({labeled.shape[1]}x{labeled.shape[0]}px)")

print(f"\nAll markers saved to '{OUTPUT_DIR}/' folder.")
print("Print each image at the size you need (15cm or 20cm recommended).")
print("Make sure to print at actual size — do not scale to fit page.")
