"""Court keypoint detection - the SINGLE trained-model injection point (KPT-1).

Contract: ``detect(frame_bgr, weights_path)`` returns a ``(12, 3)`` array of
``(u, v, visibility)`` in the canonical index order (court_model.KEYPOINTS),
plus which path produced it.

- **Real** - when ``weights_path`` (models/best.pt) exists AND ``ultralytics``
  is installed, run the YOLO pose model exactly as cookbook Recipe 2 describes.
  Dropping in ``best.pt`` is therefore the ONLY step needed to go live; no route
  or UI code changes.
- **Mock** - otherwise, synthesize a realistic full-court detection by projecting
  the canonical 12 keypoints through a plausible behind-the-baseline perspective
  for this frame size, so the whole UI (edit, confirm, ROI) is functional today.

This mirrors, but does not depend on, the ``pickleball_phase2.calibration``
``detect_court_keypoints`` stub - keeping web glue out of the algorithm package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import bootstrap  # noqa: F401  (ensures src/ is importable)
from pickleball_phase2.court_model import COURT_PTS_FT, CORNER_IDX

_yolo_cache: dict[str, object] = {}


@dataclass(frozen=True)
class Detection:
    keypoints: np.ndarray   # (12, 3) -> (u, v, visibility 0/1/2)
    source: str             # "model" | "mock"
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "keypoints": [[round(float(u), 2), round(float(v), 2), int(vis)]
                          for u, v, vis in self.keypoints],
            "source": self.source,
            "note": self.note,
        }


def detect(frame_bgr: np.ndarray, weights_path: str | Path) -> Detection:
    """Detect the 12 court keypoints in one frame (real model or mock)."""
    weights = Path(weights_path)
    if weights.exists():
        try:
            kpts = _detect_with_model(frame_bgr, weights)
            return Detection(kpts, "model", f"weights={weights.name}")
        except _NoCourt:
            h, w = frame_bgr.shape[:2]
            return mock_from_size(w, h, note="model found no court in this frame - "
                                             "step to a clearer frame, then Re-detect")
        except Exception as exc:  # never break the UI on a model error
            h, w = frame_bgr.shape[:2]
            return mock_from_size(w, h, note=f"model error ({exc}); showing a mock guess")
    h, w = frame_bgr.shape[:2]
    return mock_from_size(w, h, note="no models/best.pt yet - mock detection")


def detect_mock(width: int, height: int) -> Detection:
    """Mock detection from frame size only (no decode) - used when there is no
    trained model, so the UI works even for codecs OpenCV can't decode."""
    return mock_from_size(int(width), int(height),
                          note="no models/best.pt yet - mock detection")


class _NoCourt(Exception):
    """Raised when the model runs fine but finds no court in the frame."""


def _detect_with_model(frame_bgr: np.ndarray, weights: Path) -> np.ndarray:
    """Real YOLO-pose inference (Cookbook Recipe 2, step 5)."""
    from ultralytics import YOLO  # imported lazily; optional dependency

    key = str(weights)
    model = _yolo_cache.get(key) or YOLO(str(weights))
    _yolo_cache[key] = model

    res = model(frame_bgr, verbose=False)[0]
    kp = res.keypoints
    if kp is None or kp.xy is None or kp.xy.shape[0] == 0:
        raise _NoCourt("no detections")

    xy = kp.xy.cpu().numpy()                          # (N, 12, 2)
    conf = (kp.conf.cpu().numpy() if kp.conf is not None
            else np.ones(xy.shape[:2], dtype=float))  # (N, 12)
    best = int(np.argmax(conf.sum(axis=1)))           # strongest court instance
    kxy, kconf = xy[best], conf[best]
    # Three-level visibility, matching the Phase-1 convention (2/1/0): confident
    # -> visible; plausible -> occluded-but-known (still feeds the homography);
    # junk -> not-labeled.
    vis = np.where(kconf > 0.5, 2.0, np.where(kconf > 0.25, 1.0, 0.0))
    return np.hstack([kxy, vis[:, None]])


def mock_from_size(w: int, h: int, note: str) -> Detection:
    """Project the canonical court through a synthetic near-baseline camera."""
    # A trapezoid: far baseline high & narrow, near baseline low & wide - the
    # geometry the phone rigs actually see (behind the near-baseline corners).
    dst = np.array([
        [w * 0.30, h * 0.24],   # far-baseline-left   (kp0)
        [w * 0.70, h * 0.24],   # far-baseline-right  (kp2)
        [w * 0.96, h * 0.82],   # near-baseline-right (kp9)
        [w * 0.04, h * 0.82],   # near-baseline-left  (kp11)
    ], dtype=np.float32)
    src = COURT_PTS_FT[list(CORNER_IDX)].astype(np.float32)
    H_c2i = cv2.getPerspectiveTransform(src, dst)

    pts = COURT_PTS_FT.astype(np.float64)
    ph = np.hstack([pts, np.ones((12, 1))])
    proj = (H_c2i @ ph.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    kpts = np.hstack([uv, np.full((12, 1), 2.0)])    # all visible in the mock
    return Detection(kpts, "mock", note)
