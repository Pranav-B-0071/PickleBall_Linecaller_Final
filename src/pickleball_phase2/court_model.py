"""Canonical pickleball court model - §4.1 of the Technical Disclosure.

Single source of truth for court geometry. Every module imports from here.
Units: FEET, in the canonical court frame:
    origin (0, 0) = far-baseline-left corner
    x: across the court, 0 -> 20 ft
    y: along the court,  0 -> 44 ft   (net at y = 22)

Keypoint index order MUST match Phase 1 (court_calib.py / dataset flip map).
No placeholders in this file.
"""

from __future__ import annotations

import numpy as np

# Court dimensions (regulation)
COURT_W_FT = 20.0
COURT_L_FT = 44.0
NET_Y_FT = 22.0
KITCHEN_FAR_Y_FT = 15.0
KITCHEN_NEAR_Y_FT = 29.0

FT_TO_CM = 30.48

# The 12 canonical keypoints, index -> (name, x_ft, y_ft).  Appendix A order.
KEYPOINTS: list[tuple[str, float, float]] = [
    ("bg_baseline_left", 0.0, 0.0),     # 0  corner
    ("bg_baseline_mid", 10.0, 0.0),     # 1
    ("bg_baseline_right", 20.0, 0.0),   # 2  corner
    ("bg_kitchen_right", 20.0, 15.0),   # 3
    ("bg_kitchen_mid", 10.0, 15.0),     # 4
    ("bg_kitchen_left", 0.0, 15.0),     # 5
    ("fg_kitchen_left", 0.0, 29.0),     # 6
    ("fg_kitchen_mid", 10.0, 29.0),     # 7
    ("fg_kitchen_right", 20.0, 29.0),   # 8
    ("fg_baseline_right", 20.0, 44.0),  # 9  corner
    ("fg_baseline_mid", 10.0, 44.0),    # 10
    ("fg_baseline_left", 0.0, 44.0),    # 11 corner
]

COURT_PTS_FT = np.array([[x, y] for _, x, y in KEYPOINTS], dtype=np.float64)  # (12, 2)
CORNER_IDX = (0, 2, 9, 11)
FLIP_IDX = [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9]  # left-right mirror pairs

# Visibility flag convention (YOLO-pose): 2 = visible, 1 = occluded-but-known, 0 = not labeled
VIS_VISIBLE, VIS_OCCLUDED, VIS_NONE = 2, 1, 0

# Near half (the ROI): net -> near baseline
NEAR_HALF_Y_RANGE = (NET_Y_FT, COURT_L_FT)


def keypoints_3d_ft() -> np.ndarray:
    """12 keypoints as 3-D points on the court plane (Z = 0), for solvePnP. (12, 3)."""
    return np.hstack([COURT_PTS_FT, np.zeros((12, 1))]).astype(np.float64)


def line_call(xy_ft: np.ndarray, zone: str = "near_half", margin_ft: float = 0.0) -> dict:
    """In/out verdict for a bounce point in court coordinates.

    A ball touching a boundary line is IN (the line belongs to the court).
    `margin_ft` widens/narrows the boundary for sensitivity studies (0 = exact).

    Returns dict: verdict ("IN"/"OUT"), distance_to_nearest_line_ft (signed,
    positive = inside), nearest_line (name).
    """
    x, y = float(xy_ft[0]), float(xy_ft[1])
    if zone == "near_half":
        bounds = {"sideline_left": x - 0.0, "sideline_right": COURT_W_FT - x,
                  "net_line": y - NET_Y_FT, "near_baseline": COURT_L_FT - y}
    elif zone == "full_court":
        bounds = {"sideline_left": x - 0.0, "sideline_right": COURT_W_FT - x,
                  "far_baseline": y - 0.0, "near_baseline": COURT_L_FT - y}
    elif zone == "kitchen_near":
        bounds = {"sideline_left": x - 0.0, "sideline_right": COURT_W_FT - x,
                  "net_line": y - NET_Y_FT, "kitchen_line": KITCHEN_NEAR_Y_FT - y}
    else:
        raise ValueError(f"unknown zone: {zone}")

    nearest_line = min(bounds, key=bounds.get)  # type: ignore[arg-type]
    signed = bounds[nearest_line]
    verdict = "IN" if signed >= -margin_ft else "OUT"
    return {"verdict": verdict, "distance_ft": signed,
            "distance_cm": signed * FT_TO_CM, "nearest_line": nearest_line}
