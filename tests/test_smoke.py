"""Synthetic end-to-end validation - runs WITHOUT any real data or model.

Simulates a 3-D ball trajectory with a known bounce, projects it into two
synthetic cameras using the EXACT displacement law from the derivation doc:

    G = P_xy + (Z / (Cz - Z)) * (P_xy - C_xy)

then checks that the separation-signal pipeline recovers the bounce frame,
position, and IN/OUT verdict. Run:  python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pickleball_phase2.bounce import (GroundTrack, detect_bounces,           # noqa: E402
                                      find_separation_minima,
                                      refine_contact_subframe,
                                      separation_signal, vshape_candidates)
from pickleball_phase2.calibration import apply_homography, fit_court_homography  # noqa: E402
from pickleball_phase2.court_model import COURT_PTS_FT, line_call            # noqa: E402
from pickleball_phase2.fusion import fuse_event                              # noqa: E402
from pickleball_phase2.tracking import BallTrack, MockTracker                # noqa: E402

RNG = np.random.default_rng(7)

# Synthetic rig (court frame, feet): behind the near-baseline corners, 4 ft high
CAM_A = np.array([-3.0, 50.0, 4.0])
CAM_B = np.array([23.0, 50.0, 4.0])

# Ground-truth bounce: 0.25 ft inside the near baseline, right of centre -> IN
TRUE_XY = np.array([16.0, 43.75])
TRUE_FRAME = 30.4          # contact deliberately BETWEEN frames (60 fps test)
FPS = 60.0


def simulate_trajectory(n_frames: int = 61):
    """Incoming ball, bounce at TRUE_FRAME/TRUE_XY, rebound. Returns
    (frames, P_xy, Z) of the true 3-D state."""
    frames = np.arange(n_frames, dtype=float)
    t = (frames - TRUE_FRAME) / FPS                     # seconds from contact
    vel_in = np.array([8.0, 20.0])                      # ft/s horizontal
    vel_out = np.array([6.0, -14.0])                    # after bounce
    p = np.where(t[:, None] < 0, TRUE_XY + t[:, None] * vel_in,
                 TRUE_XY + t[:, None] * vel_out)
    g = 32.17                                           # ft/s^2
    vz_in, vz_out = 12.0, 8.0
    z = np.where(t < 0, -vz_in * t - 0.5 * g * t ** 2, vz_out * t - 0.5 * g * t ** 2)
    return frames, p, np.maximum(z, 0.0)


def ground_projection(p_xy: np.ndarray, z: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """The derivation's displacement law (exact, no camera model needed)."""
    scale = (z / (cam[2] - z))[:, None]
    return p_xy + scale * (p_xy - cam[:2])


def make_ground_tracks(noise_ft: float = 0.02):
    frames, p, z = simulate_trajectory()
    ga = ground_projection(p, z, CAM_A) + RNG.normal(0, noise_ft, (len(frames), 2))
    gb = ground_projection(p, z, CAM_B) + RNG.normal(0, noise_ft, (len(frames), 2))
    conf = np.full(len(frames), 0.9)
    return (GroundTrack(frames, ga, conf), GroundTrack(frames, gb, conf))


# ---------------------------------------------------------------- tests ----

def test_homography_roundtrip():
    """Synthetic camera H: 12 keypoints + noise -> fit -> small reproj error."""
    import cv2
    src = np.array([[0, 0], [20, 0], [20, 44], [0, 44]], dtype=np.float32)
    dst = np.array([[820, 240], [1180, 235], [1700, 980], [230, 990]], dtype=np.float32)
    H_true = cv2.getPerspectiveTransform(src, dst)      # court -> image
    img_pts = apply_homography(H_true, COURT_PTS_FT)
    img_pts += RNG.normal(0, 0.8, img_pts.shape)        # ~1 px detection noise
    kpts = np.hstack([img_pts, np.full((12, 1), 2.0)])
    calib = fit_court_homography(kpts)
    assert calib.mean_reproj_px < 3.0
    court_back = apply_homography(calib.H, img_pts)
    assert np.abs(court_back - COURT_PTS_FT).max() < 0.35   # ft


def test_separation_signal_finds_bounce():
    ga, gb = make_ground_tracks()
    frames, s = separation_signal(ga, gb, smooth_window=3)
    minima = find_separation_minima(frames, s, min_prominence_ft=0.3)
    assert len(minima) >= 1
    best = min(minima, key=lambda i: abs(frames[i] - TRUE_FRAME))
    assert abs(frames[best] - TRUE_FRAME) <= 1.5


def test_subframe_refinement_beats_frame_grid():
    ga, _ = make_ground_tracks(noise_ft=0.01)
    t_c, xy = refine_contact_subframe(ga, frame_hint=round(TRUE_FRAME))
    assert abs(t_c - TRUE_FRAME) < 0.75                 # sub-frame in time
    assert np.linalg.norm(xy - TRUE_XY) < 0.25          # < ~7.5 cm position


def test_end_to_end_with_mock_tracker():
    """Pixels -> homography -> s(t) -> fusion -> verdict, all real code paths."""
    import cv2
    quads = {"A": [[600, 300], [1400, 280], [1850, 1000], [150, 1020]],
             "B": [[500, 290], [1300, 310], [1800, 1010], [100, 980]]}
    calibs, tracks = {}, {}
    frames_t, p, z = simulate_trajectory()
    for cam, quad, cpos in (("A", quads["A"], CAM_A), ("B", quads["B"], CAM_B)):
        src = np.array([[0, 0], [20, 0], [20, 44], [0, 44]], dtype=np.float32)
        H_c2i = cv2.getPerspectiveTransform(src, np.array(quad, dtype=np.float32))
        img12 = apply_homography(H_c2i, COURT_PTS_FT)
        kpts = np.hstack([img12, np.full((12, 1), 2.0)])
        calibs[cam] = fit_court_homography(kpts)
        g = ground_projection(p, z, cpos)
        uv = apply_homography(H_c2i, g) + RNG.normal(0, 0.5, (len(frames_t), 2))
        tracks[cam] = BallTrack(frames=frames_t.copy(), uv=uv,
                                conf=np.full(len(frames_t), 0.9),
                                valid=np.ones(len(frames_t), dtype=bool), fps=FPS)

    events = detect_bounces(tracks["A"], tracks["B"], calibs["A"], calibs["B"],
                            min_prominence_ft=0.3, smooth_window=3)
    assert len(events) >= 1
    ev = min(events, key=lambda e: abs(e.frame - TRUE_FRAME))
    call = fuse_event(ev, {"dispute_threshold_ft": 0.5})
    assert call.verdict == "IN"
    assert np.linalg.norm(np.array(call.xy_ft) - TRUE_XY) < 0.4   # < ~12 cm
    assert abs(ev.frame - TRUE_FRAME) < 1.5


def test_fusion_dispute_prefers_better_camera():
    from pickleball_phase2.bounce import BounceEvent
    ev = BounceEvent(frame=100.0, xy_a_ft=np.array([19.9, 43.0]),
                     xy_b_ft=np.array([20.4, 43.0]),                # A: IN, B: OUT
                     separation_ft=0.5, method="separation",
                     quality={"conf_a": 0.95, "conf_b": 0.40,
                              "reproj_a_px": 2.0, "reproj_b_px": 6.0,
                              "occluded_a": False, "occluded_b": True})
    call = fuse_event(ev, {"dispute_threshold_ft": 0.164})
    assert call.resolved_by == "A"
    assert call.confidence == "two-view-resolved"
    assert call.verdict == "IN"


def test_fusion_occlusion_beats_weight():
    """Regression: an occluded camera must NEVER win arbitration, even with
    the higher evidence weight (reviewer defect #6)."""
    from pickleball_phase2.bounce import BounceEvent
    ev = BounceEvent(frame=50.0, xy_a_ft=np.array([10.0, 43.0]),
                     xy_b_ft=np.array([10.0, 44.5]),
                     separation_ft=1.5, method="separation",
                     quality={"conf_a": 0.99, "conf_b": 0.30,   # A "stronger"...
                              "reproj_a_px": 1.0, "reproj_b_px": 7.0,
                              "occluded_a": True, "occluded_b": False})  # ...but occluded
    call = fuse_event(ev, {"dispute_threshold_ft": 0.164})
    assert call.resolved_by == "B"


def test_line_calls():
    assert line_call(np.array([10.0, 43.0]))["verdict"] == "IN"
    assert line_call(np.array([10.0, 44.0]))["verdict"] == "IN"    # ON the line = IN
    assert line_call(np.array([10.0, 44.3]))["verdict"] == "OUT"
    assert line_call(np.array([-0.1, 30.0]))["verdict"] == "OUT"
    out = line_call(np.array([20.05, 30.0]))
    assert out["verdict"] == "OUT" and out["nearest_line"] == "sideline_right"


def test_vshape_fallback_flags_candidate():
    frames_t, p, z = simulate_trajectory()
    uv_y = 900 - z * 60.0                       # crude image: higher ball = smaller v
    uv = np.stack([np.linspace(600, 1200, len(frames_t)), uv_y], axis=1)
    tr = BallTrack(frames_t, uv, np.full(len(frames_t), 0.9),
                   np.ones(len(frames_t), dtype=bool), FPS)
    cands = vshape_candidates(tr, min_fall_frames=3)
    assert any(abs(c - TRUE_FRAME) <= 2.0 for c in cands)


def test_mock_tracker_offset():
    tr = BallTrack(np.arange(10, dtype=float), np.zeros((10, 2)),
                   np.ones(10), np.ones(10, dtype=bool))
    shifted = MockTracker(tr).track_video("x.mp4", frame_offset=-2.5)
    assert shifted.frames[0] == -2.5
