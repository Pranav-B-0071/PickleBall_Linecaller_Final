"""Integration tests for the two handoff bundles (July 2026):

  * ball_tracker_handoff  -> TRK-1: GridTrackNetTracker via ONNX runtime
  * bounce_detection_handoff -> BNC-1: classify_bounce_vs_hit (rule-based v1)
    + tracker outlier cleanup (drop_track_outliers)

The ONNX smoke test runs real inference on a tiny synthetic video and is
skipped automatically when onnxruntime or the model file is missing, so the
suite still passes on a bare checkout (Recipe 0 promise).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pickleball_phase2.bounce import classify_bounce_vs_hit, detect_bounces  # noqa: E402
from pickleball_phase2.calibration import apply_homography, fit_court_homography  # noqa: E402
from pickleball_phase2.court_model import COURT_PTS_FT                        # noqa: E402
from pickleball_phase2.tracking import (BallTrack, GridTrackNetTracker,       # noqa: E402
                                        drop_track_outliers)

REPO = Path(__file__).resolve().parents[1]
ONNX_WEIGHTS = REPO / "models" / "model_weights.onnx"

try:
    import onnxruntime  # noqa: F401
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False


# ------------------------------------------------------------- helpers ----

def ballistic_pixel_track(bounce_frame: float = 30.0, n: int = 61) -> BallTrack:
    """Physically-plausible single-view pixel track with a ground bounce:
    pixel-y follows a ballistic V (descend, sharp turn, ascend); pixel-u
    moves steadily across the frame. Scale ~60 px/ft (a court-length view
    at 1080p) so per-frame motion matches real 60 fps footage."""
    frames = np.arange(n, dtype=float)
    t = frames - bounce_frame
    # height above ground (ft), ballistic on each side of contact
    z = np.where(t < 0, -12.0 * t / 60 - 16.0 * (t / 60) ** 2,
                 8.0 * t / 60 - 16.0 * (t / 60) ** 2)
    z = np.maximum(z, 0.0)
    y = 900.0 - 60.0 * z                        # image y: lower ball = larger y
    u = 600.0 + 10.0 * frames
    return BallTrack(frames=frames, uv=np.stack([u, y], axis=1),
                     conf=np.full(n, 0.9), valid=np.ones(n, dtype=bool), fps=60.0)


# --------------------------------------------------------------- tests ----

def test_classifier_scores_bounce_high():
    tr = ballistic_pixel_track()
    assert classify_bounce_vs_hit(tr, 30.0) >= 0.6


def test_classifier_scores_midflight_low():
    """Mid-flight (monotonic descent) must score below a true bounce and
    below the config default filter neighbourhood."""
    tr = ballistic_pixel_track()
    p_bounce = classify_bounce_vs_hit(tr, 30.0)
    p_flight = classify_bounce_vs_hit(tr, 15.0)
    assert p_flight < p_bounce
    assert p_flight <= 0.5


def two_cam_tracks(z_contact: float = 0.0):
    """Two-camera synthetic rig (same geometry as test_smoke): a trajectory
    whose vertical V turns at height `z_contact`. 0 = ground bounce; > 0 =
    paddle/body hit (the ball never reaches the ground). Returns
    (tracks, calibs, true_frame)."""
    import cv2
    rng = np.random.default_rng(7)
    cam_a, cam_b = np.array([-3.0, 50.0, 4.0]), np.array([23.0, 50.0, 4.0])
    true_xy, true_frame = np.array([16.0, 43.75]), 30.4
    frames = np.arange(61, dtype=float)
    t = (frames - true_frame) / 60.0
    # physically-correct bounce: horizontal velocity PRESERVED across contact
    vel_in, vel_out = np.array([8.0, 20.0]), np.array([6.0, 15.0])
    p = np.where(t[:, None] < 0, true_xy + t[:, None] * vel_in,
                 true_xy + t[:, None] * vel_out)
    g, vz_in, vz_out = 32.17, 12.0, 8.0
    z = z_contact + np.maximum(
        np.where(t < 0, -vz_in * t - 0.5 * g * t ** 2,
                 vz_out * t - 0.5 * g * t ** 2), 0.0)
    quads = {"A": [[600, 300], [1400, 280], [1850, 1000], [150, 1020]],
             "B": [[500, 290], [1300, 310], [1800, 1010], [100, 980]]}
    calibs, tracks = {}, {}
    src = np.array([[0, 0], [20, 0], [20, 44], [0, 44]], dtype=np.float32)
    for cam, cpos in (("A", cam_a), ("B", cam_b)):
        H_c2i = cv2.getPerspectiveTransform(src, np.array(quads[cam], dtype=np.float32))
        img12 = apply_homography(H_c2i, COURT_PTS_FT)
        calibs[cam] = fit_court_homography(np.hstack([img12, np.full((12, 1), 2.0)]))
        scale = (z / (cpos[2] - z))[:, None]
        gproj = p + scale * (p - cpos[:2])
        uv = apply_homography(H_c2i, gproj) + rng.normal(0, 0.5, (61, 2))
        tracks[cam] = BallTrack(frames.copy(), uv, np.full(61, 0.9),
                                np.ones(61, dtype=bool), 60.0)
    return tracks, calibs, true_frame


def test_separation_floor_rejects_paddle_hit():
    """A hit at paddle height (Z = 2 ft) produces an s(t) minimum whose
    VALUE stays large (s_min ~ k*Z_contact) - max_separation_ft kills it
    while leaving the true ground bounce untouched."""
    tracks, calibs, _ = two_cam_tracks(z_contact=2.0)
    unfiltered = detect_bounces(tracks["A"], tracks["B"], calibs["A"], calibs["B"],
                                min_prominence_ft=0.3, smooth_window=3)
    assert len(unfiltered) >= 1                      # the false minimum exists...
    filtered = detect_bounces(tracks["A"], tracks["B"], calibs["A"], calibs["B"],
                              min_prominence_ft=0.3, smooth_window=3,
                              max_separation_ft=1.0)
    assert len(filtered) == 0                        # ...and the floor kills it

    tracks_b, calibs_b, true_frame = two_cam_tracks(z_contact=0.0)
    kept = detect_bounces(tracks_b["A"], tracks_b["B"], calibs_b["A"], calibs_b["B"],
                          min_prominence_ft=0.3, smooth_window=3,
                          max_separation_ft=1.0)
    assert any(abs(e.frame - true_frame) < 1.5 for e in kept)  # real bounce survives


def test_classifier_sparse_track_is_neutral():
    """Too few detections around the candidate must NOT veto (returns 0.5)."""
    tr = ballistic_pixel_track()
    tr.valid[24:37] = False                       # occlusion around the event
    assert classify_bounce_vs_hit(tr, 30.0) == 0.5


def test_drop_track_outliers_kills_teleport():
    tr = ballistic_pixel_track()
    tr.uv[40] = [50.0, 100.0]                     # ID-switch onto a player
    cleaned = drop_track_outliers(tr, max_jump_px=60.0)
    assert not cleaned.valid[40]
    assert cleaned.valid.sum() == tr.valid.sum() - 1
    assert cleaned.valid[30] and cleaned.valid[41]  # neighbours untouched


def test_detect_bounces_reports_and_filters_probs():
    """detect_bounces attaches bounce_prob_a/b and min_bounce_prob filters."""
    tracks, calibs, true_frame = two_cam_tracks(z_contact=0.0)
    events = detect_bounces(tracks["A"], tracks["B"], calibs["A"], calibs["B"],
                            min_prominence_ft=0.3, smooth_window=3)
    assert len(events) >= 1
    ev = min(events, key=lambda e: abs(e.frame - true_frame))
    assert 0.0 <= ev.quality["bounce_prob_a"] <= 1.0
    assert 0.0 <= ev.quality["bounce_prob_b"] <= 1.0
    assert max(ev.quality["bounce_prob_a"], ev.quality["bounce_prob_b"]) >= 0.5

    filtered = detect_bounces(tracks["A"], tracks["B"], calibs["A"], calibs["B"],
                              min_prominence_ft=0.3, smooth_window=3,
                              min_bounce_prob=1.01)   # impossible bar
    assert len(filtered) == 0


def test_detect_bounces_survives_zero_detection_track():
    """Regression (reviewer defect): a camera that never saw the ball must
    yield no events, not a zero-size-array crash in separation_signal."""
    tracks, calibs, _ = two_cam_tracks(z_contact=0.0)
    blind = tracks["B"]
    blind.valid[:] = False
    assert detect_bounces(tracks["A"], blind, calibs["A"], calibs["B"]) == []


@pytest.mark.skipif(not (HAVE_ORT and ONNX_WEIGHTS.exists()),
                    reason="onnxruntime or models/model_weights.onnx not available")
def test_onnx_tracker_smoke(tmp_path):
    """Real ONNX inference over a tiny synthetic clip: verifies the video ->
    chunks -> dual-mode interleave -> BallTrack mechanics end to end."""
    import cv2
    video = tmp_path / "synthetic.mp4"
    w, h, n = 640, 360, 25
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 60, (w, h))
    if not vw.isOpened():
        pytest.skip("cv2 mp4v codec not available on this machine")
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(frame, (50 + 20 * i, 180), 4, (255, 255, 255), -1)
        vw.write(frame)
    vw.release()

    # mp4 roundtrips are not guaranteed frame-exact: derive expectations
    # from what the decoder actually returns rather than hardcoding 25
    cap = cv2.VideoCapture(str(video))
    n_dec = 0
    while cap.read()[0]:
        n_dec += 1
    cap.release()
    if n_dec < 10:
        pytest.skip(f"decoder returned only {n_dec} frames")

    tracker = GridTrackNetTracker(ONNX_WEIGHTS, provider="cpu", frame_mode="dual")
    track = tracker.track_video(video, frame_offset=-2.5)
    # dual mode consumes 10 frames per chunk; trailing remainder dropped
    assert len(track.frames) == (n_dec // 10) * 10
    assert track.frames[0] == -2.5                    # offset applied
    assert track.uv.shape == (len(track.frames), 2)
    assert track.valid.dtype == bool
    assert set(np.unique(track.conf)) <= {0.0, 1.0}

    tracker_seq = GridTrackNetTracker(ONNX_WEIGHTS, provider="cpu",
                                      frame_mode="sequential")
    track_seq = tracker_seq.track_video(video)
    assert len(track_seq.frames) == (n_dec // 5) * 5  # chunks of 5
