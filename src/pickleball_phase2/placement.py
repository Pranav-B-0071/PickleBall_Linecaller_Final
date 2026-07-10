"""Guided camera placement with audio feedback - §5.4 (proposed, not yet built).

The state machine and readiness score are implemented (pure logic, testable).
Only the actual audio output and the live camera loop are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .court_model import CORNER_IDX, VIS_VISIBLE


class PlacementState(Enum):
    SEARCHING = "searching"   # rapid beeps
    ALIGNING = "aligning"     # beeps slow as readiness rises
    LOCKED = "locked"         # one long tone


@dataclass
class Readiness:
    corners_visible: int          # 0..4  (keypoints 0, 2, 9, 11)
    mean_kpt_conf: float          # 0..1
    reproj_err_px: float | None   # None if homography could not be fitted
    score: float                  # 0..1 combined readiness


def readiness_score(
    kpts_uv_vis: np.ndarray,
    kpt_conf: np.ndarray,
    reproj_err_px: float | None,
    err_good_px: float = 8.0,
) -> Readiness:
    """Per-frame readiness (§5.4): corners seen + confidence + calibration error."""
    corners = int(sum(kpts_uv_vis[i, 2] >= VIS_VISIBLE for i in CORNER_IDX))
    conf = float(np.clip(np.mean(kpt_conf), 0.0, 1.0))
    if reproj_err_px is None:
        err_term = 0.0
    else:
        err_term = float(np.clip(1.0 - reproj_err_px / (2.0 * err_good_px), 0.0, 1.0))
    score = 0.4 * (corners / 4.0) + 0.2 * conf + 0.4 * err_term
    return Readiness(corners, conf, reproj_err_px, score)


def beep_interval_s(score: float, fastest_s: float = 0.12, slowest_s: float = 1.2) -> float:
    """Map readiness -> beep cadence. Low readiness = rapid beeping (§5.4)."""
    return float(fastest_s + (slowest_s - fastest_s) * np.clip(score, 0.0, 1.0))


class PlacementStateMachine:
    """SEARCHING -> ALIGNING -> LOCKED (Figure 8). Pure logic; feed it per-frame
    Readiness values, it returns the state. Lock requires all four corners,
    error under threshold, stable for `stable_frames` consecutive frames.
    """

    def __init__(self, lock_err_px: float = 8.0, stable_frames: int = 30):
        self.lock_err_px = lock_err_px
        self.stable_frames = stable_frames
        self._streak = 0
        self.state = PlacementState.SEARCHING

    def update(self, r: Readiness) -> PlacementState:
        if self.state == PlacementState.LOCKED:
            return self.state
        gate = (r.corners_visible == 4 and r.reproj_err_px is not None
                and r.reproj_err_px <= self.lock_err_px)
        self._streak = self._streak + 1 if gate else 0
        if self._streak >= self.stable_frames:
            self.state = PlacementState.LOCKED
        elif r.corners_visible >= 2:
            self.state = PlacementState.ALIGNING
        else:
            self.state = PlacementState.SEARCHING
        return self.state


def emit_beep(interval_s: float) -> None:
    """PLACEHOLDER[PLC-1]: actually play audio on the phone/laptop.

    Options: `simpleaudio` tone on laptop; on-device via the web app
    (WebAudio API oscillator). See Cookbook Recipe 8.
    """
    raise NotImplementedError("PLACEHOLDER[PLC-1] - see PLACEHOLDER_COOKBOOK.md, Recipe 8")
