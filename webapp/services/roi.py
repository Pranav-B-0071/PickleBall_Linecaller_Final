"""Half-court ROI + court-line derivation from the 12 detected keypoints.

Two methods:

* **homography (preferred)** - project the EXACT canonical court lines (net,
  kitchen line, near baseline, centerline) through the fitted court->image
  homography ``H_inv``. Because that homography is a RANSAC least-squares fit
  over ALL 12 detected points, this is far more accurate and stable than
  averaging a couple of keypoints: every detection contributes, outliers are
  down-weighted, and the net line is placed at its true court position (y=22)
  rather than guessed from one pair.

* **midpoint-average (fallback)** - when no homography is available, the
  Phase-1 approach: net endpoints = avg of the two baselines' corners down each
  sideline (cross-checked against the kitchen-line pair).

Court frame (feet, court_model): origin far-baseline-left; x 0..20, y 0..44,
net at y=22, near kitchen line at y=29.
"""

from __future__ import annotations

import numpy as np

# Regulation court dimensions in feet (mirror of court_model constants).
COURT_W, COURT_L, NET_Y, KITCHEN_NEAR_Y = 20.0, 44.0, 22.0, 29.0

# Canonical keypoint indices (court_model.KEYPOINTS / Appendix A).
FAR_BL_LEFT, FAR_BL_RIGHT = 0, 2
NEAR_BL_RIGHT, NEAR_BL_LEFT = 9, 11
KITCHEN_FAR_LEFT, KITCHEN_NEAR_LEFT = 5, 6
KITCHEN_FAR_RIGHT, KITCHEN_NEAR_RIGHT = 3, 8


def _xy(kp: np.ndarray, i: int) -> np.ndarray:
    return np.asarray(kp[i][:2], dtype=float)


def _mid(kp: np.ndarray, i: int, j: int) -> np.ndarray:
    return 0.5 * (_xy(kp, i) + _xy(kp, j))


def _round(p) -> list[float]:
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


def derive_half_court_roi(keypoints, homography_inv=None) -> dict:
    """Derive the near-half ROI (image pixels) from 12 keypoints.

    keypoints: (12, 2|3) image points. homography_inv: optional 3x3 court->image
    homography (calib.H_inv); when given, the accurate projection method is used.
    """
    kp = np.asarray(keypoints, dtype=float)
    if kp.shape[0] < 12:
        raise ValueError(f"need 12 keypoints, got {kp.shape[0]}")
    if homography_inv is not None:
        return _roi_from_homography(np.asarray(homography_inv, dtype=float), kp)
    return _roi_from_midpoints(kp)


def _roi_from_homography(h_inv: np.ndarray, kp: np.ndarray) -> dict:
    def P(x, y):
        v = h_inv @ np.array([x, y, 1.0])
        return _round(v[:2] / v[2])

    net_l, net_m, net_r = P(0, NET_Y), P(COURT_W / 2, NET_Y), P(COURT_W, NET_Y)
    base_l, base_m, base_r = P(0, COURT_L), P(COURT_W / 2, COURT_L), P(COURT_W, COURT_L)
    kit_l, kit_r = P(0, KITCHEN_NEAR_Y), P(COURT_W, KITCHEN_NEAR_Y)

    # how far the projected net endpoints sit from the raw keypoint-average
    # estimate (a health signal for the detection/fit)
    consistency_px = float(np.mean([
        np.linalg.norm(np.array(net_l) - _mid(kp, FAR_BL_LEFT, NEAR_BL_LEFT)),
        np.linalg.norm(np.array(net_r) - _mid(kp, FAR_BL_RIGHT, NEAR_BL_RIGHT)),
    ]))
    return {
        "roi_polygon": [net_l, net_r, base_r, base_l],   # net -> near baseline quad
        "net_line": [net_l, net_m, net_r],
        "near_baseline": [base_l, base_m, base_r],
        "kitchen_line": [kit_l, kit_r],
        "center_line": [P(COURT_W / 2, KITCHEN_NEAR_Y), base_m],  # near-half only
        "consistency_px": round(consistency_px, 2),
        "method": "homography",
    }


def _roi_from_midpoints(kp: np.ndarray) -> dict:
    # Net (y=22) is midway between the kitchen lines (y=15 far / y=29 near), so
    # averaging the ADJACENT kitchen-line pairs (5&6 left, 3&8 right) lands on the
    # net; the whole-court baseline average (0&11 / 2&9) does not - perspective
    # foreshortening pushes that estimate well off the net. Baseline pair is kept
    # only as a consistency cross-check.
    net_left = _mid(kp, KITCHEN_FAR_LEFT, KITCHEN_NEAR_LEFT)
    net_right = _mid(kp, KITCHEN_FAR_RIGHT, KITCHEN_NEAR_RIGHT)
    near_left, near_right = _xy(kp, NEAR_BL_LEFT), _xy(kp, NEAR_BL_RIGHT)
    net_left_alt = _mid(kp, FAR_BL_LEFT, NEAR_BL_LEFT)
    net_right_alt = _mid(kp, FAR_BL_RIGHT, NEAR_BL_RIGHT)
    consistency_px = float(np.mean([
        np.linalg.norm(net_left - net_left_alt),
        np.linalg.norm(net_right - net_right_alt),
    ]))
    return {
        "roi_polygon": [_round(net_left), _round(net_right), _round(near_right), _round(near_left)],
        "net_line": [_round(net_left), _round(net_right)],
        "near_baseline": [_round(near_left), _round(near_right)],
        "consistency_px": round(consistency_px, 2),
        "method": "midpoint_average",
    }
