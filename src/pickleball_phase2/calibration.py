"""Camera calibration - §5.2 (intrinsics/extrinsics) and §5.3 (shared court frame).

Implemented for real: checkerboard intrinsics, RANSAC court homography,
re-projection error (the CORRECT formula: detected pixel vs canonical court
point mapped into the image), PnP pose.

Placeholders: KPT-1 (keypoint-model inference - needs your retrained weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .court_model import COURT_PTS_FT, VIS_OCCLUDED, keypoints_3d_ft

# --------------------------------------------------------------------------
# Intrinsics (§5.2) - Recipe 1
# --------------------------------------------------------------------------


@dataclass
class Intrinsics:
    K: np.ndarray                      # (3, 3) camera matrix
    dist: np.ndarray                   # distortion coefficients (k1 k2 p1 p2 k3)
    image_size: tuple[int, int]        # (w, h) at which K was estimated
    rms: float = 0.0                   # calibration RMS re-projection error (px)

    def save(self, path: str | Path) -> None:
        payload = {"K": self.K.tolist(), "dist": self.dist.ravel().tolist(),
                   "image_size": list(self.image_size), "rms": float(self.rms)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "Intrinsics":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls(K=np.array(d["K"], dtype=np.float64),
                   dist=np.array(d["dist"], dtype=np.float64),
                   image_size=tuple(d["image_size"]), rms=float(d.get("rms", 0.0)))


def calibrate_intrinsics_from_video(
    video_path: str | Path,
    inner_corners: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    max_views: int = 40,
    frame_stride: int = 15,
) -> Intrinsics:
    """Standard checkerboard calibration (Zhang) over a slow sweep video.

    Fully implemented - run it on each phone's checkerboard clip (Recipe 1).
    Uses the settings LOCKED for capture (§5.1); intrinsics are void if
    zoom/resolution change.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    pattern = tuple(inner_corners)
    # Object points for one view: (N, 3) grid in mm, Z = 0
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= float(square_size_mm)

    obj_pts, img_pts = [], []
    size = None
    idx = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    while len(obj_pts) < max_views:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_pts.append(objp)
                img_pts.append(corners)
        idx += 1
    cap.release()

    if len(obj_pts) < 10:
        raise RuntimeError(
            f"only {len(obj_pts)} usable checkerboard views found (need >= 10); "
            "re-shoot: slower sweep, full-frame coverage, good light")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return Intrinsics(K=K, dist=dist, image_size=size, rms=float(rms))


# --------------------------------------------------------------------------
# Court keypoint detection - PLACEHOLDER[KPT-1] (Recipe 2)
# --------------------------------------------------------------------------


def detect_court_keypoints(frame_bgr: np.ndarray, weights_path: str | Path) -> np.ndarray:
    """Run the YOLOv11m-pose court-keypoint model on one frame.

    Must return a (12, 3) array of (u, v, visibility) in the Phase-1 index
    order (court_model.KEYPOINTS).

    PLACEHOLDER[KPT-1]: wire in the model retrained on real phone-view
    frames - the Phase-1 model saw only elevated broadcast views and is NOT
    trusted here. See Cookbook Recipe 2 for the exact ultralytics call.
    """
    raise NotImplementedError(
        "PLACEHOLDER[KPT-1] - see PLACEHOLDER_COOKBOOK.md, Recipe 2")


# --------------------------------------------------------------------------
# Homography to the shared court frame (§5.3) - implemented
# --------------------------------------------------------------------------


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3x3 H to (N, 2) points. Returns (N, 2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    out = (H @ ph.T).T
    return out[:, :2] / out[:, 2:3]


@dataclass
class CourtCalibration:
    """One camera's tie to the canonical court frame."""

    H: np.ndarray                       # image (px) -> court (ft)
    H_inv: np.ndarray                   # court (ft) -> image (px)
    inlier_mask: np.ndarray             # (12,) bool
    mean_reproj_px: float
    mean_reproj_ft: float
    used_indices: list[int] = field(default_factory=list)

    # Optional PnP pose (filled by solve_camera_pose)
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    camera_pos_ft: np.ndarray | None = None   # (3,) position in court frame


def fit_court_homography(
    kpts_uv_vis: np.ndarray,
    ransac_thresh_px: float = 5.0,
    min_inliers: int = 6,
) -> CourtCalibration:
    """RANSAC homography from detected keypoints to the canonical court (§4.5).

    kpts_uv_vis: (12, 3) of (u, v, visibility). Points with visibility >= 1
    (visible or occluded-but-known) participate; vis == 0 are excluded.
    """
    kpts = np.asarray(kpts_uv_vis, dtype=np.float64)
    usable = np.where(kpts[:, 2] >= VIS_OCCLUDED)[0]
    if len(usable) < 4:
        raise ValueError(f"need >= 4 usable keypoints, got {len(usable)}")

    img = kpts[usable, :2].astype(np.float64)
    court = COURT_PTS_FT[usable]

    H, mask = cv2.findHomography(img, court, cv2.RANSAC, ransac_thresh_px)
    if H is None:
        raise RuntimeError("findHomography failed")
    mask = mask.ravel().astype(bool)
    if mask.sum() < min_inliers:
        raise RuntimeError(f"only {int(mask.sum())} RANSAC inliers (< {min_inliers})")

    H_inv = np.linalg.inv(H)

    # Re-projection error, the CORRECT way (fixes the §4.5 typo):
    # where does the canonical court point land in the image vs the detection?
    inl = usable[mask]
    reproj_img = apply_homography(H_inv, COURT_PTS_FT[inl])
    err_px = float(np.mean(np.linalg.norm(reproj_img - kpts[inl, :2], axis=1)))
    court_rt = apply_homography(H, kpts[inl, :2])
    err_ft = float(np.mean(np.linalg.norm(court_rt - COURT_PTS_FT[inl], axis=1)))

    inlier_mask = np.zeros(12, dtype=bool)
    inlier_mask[inl] = True
    return CourtCalibration(H=H, H_inv=H_inv, inlier_mask=inlier_mask,
                            mean_reproj_px=err_px, mean_reproj_ft=err_ft,
                            used_indices=inl.tolist())


def solve_camera_pose(
    kpts_uv_vis: np.ndarray, intr: Intrinsics, calib: CourtCalibration
) -> CourtCalibration:
    """PnP pose (R, t) in the court frame from the 12 keypoints (§5.2-5.3).

    Not needed for the line call itself (homography suffices) but recovers
    the camera position/height used in the separation-signal analysis and
    for sanity-checking against the measured rig positions (§7.1).
    """
    kpts = np.asarray(kpts_uv_vis, dtype=np.float64)
    inl = np.where(calib.inlier_mask)[0]
    obj = keypoints_3d_ft()[inl].astype(np.float64)
    img = kpts[inl, :2].astype(np.float64)

    R, tvec = _solve_planar_pose(obj, img, intr.K, np.asarray(intr.dist, float).reshape(-1))
    cam = (-R.T @ tvec).ravel()
    if cam[2] < 0:
        # FORCE above-court: a coplanar target (the court on Z=0) is inherently
        # 2-fold ambiguous, so PnP can return the mirror twin with the camera
        # BELOW the court (the observed Z ~ -5 ft). The two poses reproject every
        # court point to the SAME pixel, so reflecting to the twin is the correct
        # physical fix - only the one sign the solve cannot observe is flipped.
        R, tvec = _mirror_pose(R, tvec)
        cam = (-R.T @ tvec).ravel()
    calib.R, calib.t = R, tvec
    calib.camera_pos_ft = cam                      # camera centre in court frame
    return calib


def _mirror_pose(R, t):
    """The coplanar 2-fold TWIN of a planar-PnP pose (used to force above-court).

    For a target on Z = 0 the poses (R, t) and ([-r1, -r2, r3], -t) reproject
    every court point to the SAME pixel - their camera-frame vectors are exact
    negatives, identical after the homogeneous divide - while the camera centre's
    height flips sign. [-r1, -r2, r3] stays a proper rotation (det unchanged,
    right-handed). So this returns the pose on the opposite side of the court
    plane: given the below-court solve, its above-court equivalent.
    """
    R_m = np.asarray(R, dtype=np.float64).copy()
    R_m[:, 0] *= -1.0
    R_m[:, 1] *= -1.0
    t_m = -np.asarray(t, dtype=np.float64).reshape(3, 1)
    return R_m, t_m


def _poses_from_homography(H, K):
    """The TWO camera poses consistent with a plane->image homography (the
    coplanar 2-fold ambiguity). H maps court (X, Y) -> image (u, v)."""
    B = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    out = []
    for s in (lam, -lam):                     # +/- resolves in-front vs behind
        r1, r2, t = s * B[:, 0], s * B[:, 1], s * B[:, 2]
        M = np.column_stack([r1, r2, np.cross(r1, r2)])
        U, _, Vt = np.linalg.svd(M)           # nearest valid rotation
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        out.append((R, t))
    return out


def _solve_planar_pose(obj, img, K, dist):
    """PnP for a PLANAR target (the court lies on Z = 0).

    A coplanar point set is inherently 2-fold ambiguous, so a naive solvePnP can
    return the mirror pose that puts the camera BELOW the court (negative height,
    the observed Z ~ -5 ft bug). We instead:

      1. undistort the keypoints and fit the court->image homography on the
         straightened points (robust to large lens distortion);
      2. take the ONE homography pose with the camera above the court and all
         points in front of it;
      3. refine that pose WITH the distortion model (solvePnP seeded from it, so
         it can't flip back to the mirror).

    This resolves the sign ambiguity independently of SOLVEPNP_IPPE (which can
    fail silently on strong-distortion intrinsics and drop to the mirror).
    """
    imgp = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    objxy = obj[:, :2].astype(np.float64)

    seed = None
    und = cv2.undistortPoints(imgp.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    H, _ = cv2.findHomography(objxy, und, 0)
    if H is not None:
        for R, t in _poses_from_homography(H, K):
            depth = (R @ obj.T + t.reshape(3, 1))[2]         # z of points in cam frame
            if np.all(depth > 0) and (-R.T @ t).ravel()[2] > 0:
                seed = (R, t)
                break

    if seed is not None:
        rvec, _ = cv2.Rodrigues(seed[0])
        tvec = seed[1].reshape(3, 1).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, imgp.reshape(-1, 1, 2), K, dist,
                                      rvec.copy(), tvec.copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            return cv2.Rodrigues(rvec)[0], tvec

    ok, rvec, tvec = cv2.solvePnP(obj, imgp.reshape(-1, 1, 2), K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    return cv2.Rodrigues(rvec)[0], tvec


def lock_on(
    frames_kpts: list[np.ndarray],
    ransac_thresh_px: float = 5.0,
    min_inliers: int = 6,
    lock_threshold_px: float = 8.0,
) -> tuple[int, CourtCalibration]:
    """§4.7: over a search window, keep the lowest-error frame; freeze its H.

    frames_kpts: per-frame (12, 3) detections across the ~10 s window.
    Returns (winning frame index, calibration). Raises if nothing locks.
    """
    best: tuple[int, CourtCalibration] | None = None
    for i, k in enumerate(frames_kpts):
        try:
            c = fit_court_homography(k, ransac_thresh_px, min_inliers)
        except (ValueError, RuntimeError):
            continue
        if best is None or c.mean_reproj_px < best[1].mean_reproj_px:
            best = (i, c)
    if best is None or best[1].mean_reproj_px > lock_threshold_px:
        raise RuntimeError("no frame beat the lock threshold - reposition the camera")
    return best
