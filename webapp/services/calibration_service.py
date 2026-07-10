"""Court calibration service - real homography fit over (edited) keypoints.

Given the 12 keypoints the user confirmed on Page 1 (auto-detected then dragged),
fit the image->court homography with the project's RANSAC routine and report the
re-projection error (the number shown in the green LOCKED badge). Also derives
the near-half ROI so it can be persisted and reused by Page 3.
"""

from __future__ import annotations

import numpy as np

from ..bootstrap import REPO_ROOT
from pickleball_phase2.calibration import (Intrinsics, fit_court_homography,
                                           solve_camera_pose)
from . import roi as roi_service

INTRINSICS_DIR = REPO_ROOT / "calib" / "intrinsics"


def calibrate(keypoints, ransac_thresh_px: float = 5.0,
              min_inliers: int = 6) -> dict:
    """Fit H from confirmed keypoints. Returns a JSON-friendly calibration dict.

    keypoints: (12, 3) of (u, v, visibility). Raises ValueError/RuntimeError
    (propagated as a 4xx by the route) if the points can't form a homography.
    """
    kpts = np.asarray(keypoints, dtype=np.float64)
    if kpts.shape != (12, 3):
        raise ValueError(f"expected (12, 3) keypoints, got {kpts.shape}")

    calib = fit_court_homography(kpts, ransac_thresh_px, min_inliers)
    # project the exact court lines through the fitted homography (uses all 12
    # detected points, not just a 2-point average) - see services/roi.py
    roi = roi_service.derive_half_court_roi(kpts[:, :2], homography_inv=calib.H_inv)

    return {
        "H": calib.H.tolist(),
        "H_inv": calib.H_inv.tolist(),
        "mean_reproj_px": round(calib.mean_reproj_px, 3),
        "mean_reproj_ft": round(calib.mean_reproj_ft, 4),
        "used_indices": list(calib.used_indices),
        "inlier_mask": calib.inlier_mask.astype(int).tolist(),
        "keypoints": [[round(float(u), 2), round(float(v), 2), int(vis)]
                      for u, v, vis in kpts],
        "roi": roi,
        "locked": bool(calib.mean_reproj_px <= 8.0),   # §4.7 lock threshold
    }


def camera_pose_check(cam: str, keypoints, proj, ransac_thresh_px: float = 5.0,
                      min_inliers: int = 6) -> dict:
    """Recover the camera position via PnP (solve_camera_pose) from the confirmed
    keypoints + the saved intrinsics, and compare to the tape-measured position.

    Returns {available: False, reason} when the intrinsics file is missing or PnP
    fails (so it never blocks calibration), else {available: True, recovered_ft,
    measured_ft?, error_ft?}. This is the Recipe-3 sanity check; the line-call
    itself does not use camera pose.
    """
    intr_path = INTRINSICS_DIR / f"{cam}.yaml"
    if not intr_path.exists():
        return {"available": False, "reason": f"no intrinsics at calib/intrinsics/{cam}.yaml"}
    try:
        kpts = np.asarray(keypoints, dtype=np.float64)
        calib = fit_court_homography(kpts, ransac_thresh_px, min_inliers)
        intr = Intrinsics.load(intr_path)
        calib = solve_camera_pose(kpts, intr, calib)
    except (ValueError, RuntimeError) as exc:
        return {"available": False, "reason": f"PnP failed: {exc}"}

    pos = calib.camera_pos_ft
    above = bool(pos[2] > 0)                       # camera above the court plane?
    out = {"available": True,
           "recovered_ft": [round(float(v), 2) for v in pos],
           "above_court": above,
           "reproj_px": round(calib.mean_reproj_px, 2)}
    if not above:
        # PnP still below the court despite the robust solve -> the correspondence
        # itself is mirrored (model keypoint order / left-right handedness).
        out["warning"] = ("camera resolves BELOW the court - the 12 keypoints look "
                          "mirrored/mis-ordered; check the skeleton and point order")
    measured = (proj.get("camera_positions_ft", {}) or {}).get(cam)
    if measured:
        err = float(np.linalg.norm(np.asarray(pos, float) - np.asarray(measured, float)))
        out["measured_ft"] = [round(float(v), 2) for v in measured]
        out["error_ft"] = round(err, 2)
        # planar PnP + hand-dragged points are noisy; within 5 ft is "consistent"
        out["ok"] = bool(err <= 5.0 and above)
    return out


def store_kitchen_region(points) -> dict:
    """Validate + package CAM3's manual 4-point kitchen box (image pixels).

    CAM3 is not model-calibrated: the user drags a quadrilateral over the
    kitchen. We only sanity-check that the 4 corners form a non-degenerate
    quad (shoelace area), then persist them. Bounces landing inside this
    region are later counted as kitchen bounces (analysis_service).
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape != (4, 2):
        raise ValueError(f"expected 4 (u, v) points, got shape {pts.shape}")
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    if area < 1.0:
        raise ValueError("kitchen box is degenerate - spread the 4 corners apart")
    return {
        "type": "kitchen_box",
        "region_px": [[round(float(u), 2), round(float(v), 2)] for u, v in pts],
        "area_px": round(area, 1),
        "locked": True,
    }
