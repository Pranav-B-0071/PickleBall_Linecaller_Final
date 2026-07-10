"""Bounce (point-of-pitch) detection - §5.6-5.7 + the Dual Ground-Plane
Homography derivation (separation-signal method).

Core idea (from the derivation doc):
  Project each camera's ball pixel through its ground-plane homography.
  While airborne, the two projected ground points DISAGREE; the separation
  s(t) = ||G_A(t) - G_B(t)|| ~ k*Z shrinks to a minimum at contact (Z = 0).

Implemented for real: ground projection, s(t), minimum finding, single-view
V-shape fallback, sub-frame contact refinement (fixes 60 fps undersampling),
and the bounce-vs-paddle-hit classifier (BNC-1, rule-based v1 built on the
bounce_detection_handoff cues - thresholds still need Recipe 6 tuning on
real rallies; TUNE-5 = bounce.min_bounce_prob).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import CourtCalibration, apply_homography
from .tracking import BallTrack


@dataclass
class GroundTrack:
    """A ball track projected to the court plane (Z = 0 assumption)."""

    frames: np.ndarray        # (N,) aligned frame indices (may be fractional)
    xy_ft: np.ndarray         # (N, 2) court coordinates
    conf: np.ndarray          # (N,) tracker confidence carried through


@dataclass
class BounceEvent:
    frame: float                    # sub-frame contact time (aligned timeline)
    xy_a_ft: np.ndarray | None      # camera A's estimate at contact
    xy_b_ft: np.ndarray | None      # camera B's estimate at contact
    separation_ft: float            # s at the detected minimum
    method: str                     # "separation" | "vshape_A" | "vshape_B"
    quality: dict                   # per-camera evidence for fusion/disputes


def project_to_court(track: BallTrack, calib: CourtCalibration) -> GroundTrack:
    """Map every valid ball pixel through H (image -> court ft)."""
    t = track.valid_only()
    return GroundTrack(frames=t.frames.astype(np.float64),
                       xy_ft=apply_homography(calib.H, t.uv),
                       conf=t.conf)


def _interp_to(frames_ref: np.ndarray, g: GroundTrack) -> np.ndarray:
    """Linearly interpolate a ground track onto reference frame times. (N, 2)."""
    x = np.interp(frames_ref, g.frames, g.xy_ft[:, 0])
    y = np.interp(frames_ref, g.frames, g.xy_ft[:, 1])
    return np.stack([x, y], axis=1)


def separation_signal(ga: GroundTrack, gb: GroundTrack,
                      smooth_window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """s(t) = ||G_A - G_B|| on the overlapping frame range.

    Returns (frames, s_ft). Both tracks are interpolated onto a common
    timeline first (handles missed detections and fractional sync offsets).
    """
    lo = max(ga.frames.min(), gb.frames.min())
    hi = min(ga.frames.max(), gb.frames.max())
    if hi <= lo:
        raise ValueError("tracks do not overlap in time")
    frames = np.arange(np.ceil(lo), np.floor(hi) + 1)
    pa, pb = _interp_to(frames, ga), _interp_to(frames, gb)
    s = np.linalg.norm(pa - pb, axis=1)
    if smooth_window > 1 and len(s) >= smooth_window:
        w = smooth_window + (1 - smooth_window % 2)      # force odd
        pad = w // 2
        k = np.ones(w) / w
        # edge-replicated padding: zero-padding ("same") would fake a dip
        # at the clip boundaries of a signal whose absolute level matters
        s = np.convolve(np.pad(s, pad, mode="edge"), k, mode="valid")
    return frames, s


def _side_prominence(s: np.ndarray, i: int, step: int) -> float:
    """Topographic prominence of the minimum s[i] toward one side:
    highest value reached before encountering a point LOWER than s[i]
    (or the boundary). Matches scipy.signal.find_peaks(-s) semantics."""
    peak = s[i]
    j = i + step
    while 0 <= j < len(s) and s[j] >= s[i]:
        peak = max(peak, s[j])
        j += step
    return float(peak - s[i])


def find_separation_minima(frames: np.ndarray, s: np.ndarray,
                           min_prominence_ft: float = 0.5) -> list[int]:
    """Indices of local minima of s(t) prominent enough to be bounce candidates.

    Prominence is measured to the nearest enclosing higher ground on each
    side, stopping at any point lower than the minimum itself - NOT the
    global side maximum (which would over-count on a rising signal).
    Pure numpy; scipy.signal.find_peaks(-s, prominence=...) is equivalent.
    """
    idx = []
    for i in range(1, len(s) - 1):
        if not (s[i] <= s[i - 1] and s[i] <= s[i + 1]):
            continue
        left = _side_prominence(s, i, -1)
        right = _side_prominence(s, i, +1)
        if min(left, right) >= min_prominence_ft:
            idx.append(i)
    return idx


def vshape_candidates(track: BallTrack, min_fall_frames: int = 3) -> list[float]:
    """Single-view fallback (§5.7): the ball's image-space vertical motion
    falls then rises - the V's vertex is a bounce candidate. Used when one
    camera is occluded and s(t) is unavailable."""
    t = track.valid_only()
    if len(t.frames) < 2 * min_fall_frames + 1:
        return []
    v = np.gradient(t.uv[:, 1], t.frames)   # +v = moving DOWN in the image
    out: list[float] = []
    for i in range(min_fall_frames, len(v) - min_fall_frames):
        falling = np.all(v[i - min_fall_frames: i] > 0)
        rising = np.all(v[i: i + min_fall_frames] < 0)
        if falling and rising:
            out.append(float(t.frames[i]))
    return out


def refine_contact_subframe(g: GroundTrack, frame_hint: float,
                            half_window: int = 4) -> tuple[float, np.ndarray]:
    """Sub-frame contact estimate - the 60 fps fix.

    Fit straight lines to the ground track's approach and departure around the
    candidate (the projected ground point moves ~linearly in court space on
    each side of the bounce, with a kink AT the bounce). Their intersection
    in time gives contact time and (X, Y) even if contact fell between frames.
    Falls back to the hinted frame when the window is too sparse.
    """
    m = (g.frames >= frame_hint - half_window) & (g.frames <= frame_hint + half_window)
    f, xy = g.frames[m], g.xy_ft[m]
    pre, post = f < frame_hint, f > frame_hint
    if pre.sum() < 2 or post.sum() < 2:
        return frame_hint, _interp_to(np.array([frame_hint]), g)[0]

    t_star_axes, xy_star = [], []
    for ax in (0, 1):
        a1, b1 = np.polyfit(f[pre], xy[pre, ax], 1)
        a2, b2 = np.polyfit(f[post], xy[post, ax], 1)
        t_star = (b2 - b1) / (a1 - a2) if abs(a1 - a2) > 1e-9 else frame_hint
        t_star_axes.append(t_star)
        xy_star.append((a1, b1))
    # Use the axis with the stronger kink; clamp into the window.
    t_c = float(np.clip(np.median(t_star_axes), f.min(), f.max()))
    contact = np.array([a * t_c + b for a, b in xy_star])
    return t_c, contact


def classify_bounce_vs_hit(track: BallTrack, frame: float,
                           half_window: int = 8) -> float:
    """Probability that the event at `frame` is a surface bounce rather than
    a paddle/body hit (BNC-1 - filled, rule-based v1).

    Hand-tuned combination of the §5.7 image-space SHAPE cues, built on the
    bounce_detection_handoff's signal machinery (single_view_bounce):

      1. concave-down parabola fit of pixel-y around the candidate - a real
         bounce is a ballistic descent/ascent V; R^2 is the shape score
      2. fit vertex lands near the candidate frame
      3. descend-then-ascend pattern (falls before, rises after)

    Deliberately NOT used: horizontal image velocity reversal. With
    near-ground cameras the parallax-displaced pixel path RETRACES at a
    genuine bounce, so reversal fires on bounces too. The discriminative
    contact-height cue lives in detect_bounces instead (max_separation_ft:
    s_min ~ k*Z_contact, so paddle hits leave a large separation floor).

    Returns 0.5 ("no evidence") when the track is too sparse around the
    candidate, so occluded cameras never veto an event. Upgrade path per
    Recipe 6: replace with a small sklearn classifier over these same
    features once ~200 labelled events exist (thresholds = TUNE-5).
    """
    from .single_view_bounce import _smooth as _sv_smooth

    t = track.valid_only()
    m = np.abs(t.frames - frame) <= half_window
    f, y = t.frames[m], t.uv[m, 1]
    pre_m, post_m = f < frame, f > frame
    if len(f) < 5 or pre_m.sum() < 2 or post_m.sum() < 2:
        return 0.5                                   # insufficient evidence

    ys = _sv_smooth(y, 3)
    rel = f - frame
    a, b, c = np.polyfit(rel, ys, 2)
    if a >= 0:                                       # concave-up: no ground contact
        return 0.1
    fit = np.polyval([a, b, c], rel)
    # degenerate-parabola guard: the fitted peak must actually stand out
    # above the fit at the window edges (a near-straight track fits a flat
    # "parabola" with r2 ~ 1 but carries no contact evidence)
    vertex = float(np.clip(-b / (2.0 * a), rel.min(), rel.max()))
    peak_rise = np.polyval([a, b, c], vertex) - max(fit[0], fit[-1])
    if peak_rise < 2.0:                              # px
        return 0.3
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - float(np.sum((ys - fit) ** 2)) / ss_tot if ss_tot > 1e-9 else 0.0
    shape = max(0.0, r2)

    vertex_score = max(0.0, 1.0 - abs(vertex) / half_window)

    # pixel-y grows downward: bounce = y rising before, falling after
    pre_y, post_y = ys[pre_m], ys[post_m]
    v_pattern = 1.0 if (pre_y[-1] >= pre_y[0] and post_y[-1] <= post_y[0]) else 0.0

    score = 0.5 * shape + 0.25 * vertex_score + 0.25 * v_pattern
    return float(np.clip(score, 0.0, 1.0))


def detect_bounces(
    track_a: BallTrack, track_b: BallTrack,
    calib_a: CourtCalibration, calib_b: CourtCalibration,
    min_prominence_ft: float = 0.5,
    smooth_window: int = 5,
    subframe_refine: bool = True,
    max_pair_gap_frames: float = 2.0,
    vshape_min_fall_frames: int = 3,
    min_bounce_prob: float = 0.0,
    max_separation_ft: float | None = None,
) -> list[BounceEvent]:
    """Full §5.6 chain: project both tracks -> s(t) -> minima -> refine.

    Occlusion flagging: if a camera has no REAL detection within
    `max_pair_gap_frames` of a candidate (its value there is pure
    interpolation), that camera is marked occluded for the event - fusion
    then discounts it (§5.8). V-shape corroboration (§5.7) is attached per
    camera as supporting evidence.

    Bounce-vs-hit filtering (BNC-1), two complementary checks:

    * separation floor: s_min ~ k * Z_contact (the derivation). A true
      bounce converges to ~calibration noise; a paddle/body hit at
      paddle height leaves a LARGE separation at its minimum. Candidates
      with s > `max_separation_ft` are dropped (None = off).
    * image shape: every candidate gets a per-camera
      classify_bounce_vs_hit probability (stored in quality as
      bounce_prob_a/b). If `min_bounce_prob` > 0, a candidate is dropped
      when the best NON-OCCLUDED camera scores below it (TUNE-5).
      Occluded cameras are excluded from the vote entirely - they neither
      veto nor rescue; both occluded = no evidence = keep. 0.0 = annotate
      only.
    """
    ga, gb = project_to_court(track_a, calib_a), project_to_court(track_b, calib_b)
    if len(ga.frames) == 0 or len(gb.frames) == 0:
        return []                         # a camera never saw the ball
    frames, s = separation_signal(ga, gb, smooth_window)
    vs_a = vshape_candidates(track_a, vshape_min_fall_frames)
    vs_b = vshape_candidates(track_b, vshape_min_fall_frames)
    events: list[BounceEvent] = []
    for i in find_separation_minima(frames, s, min_prominence_ft):
        if max_separation_ft is not None and s[i] > max_separation_ft:
            continue                          # contact above ground = hit
        hint = float(frames[i])
        occ_a = float(np.min(np.abs(ga.frames - hint))) > max_pair_gap_frames
        occ_b = float(np.min(np.abs(gb.frames - hint))) > max_pair_gap_frames
        prob_a = classify_bounce_vs_hit(track_a, hint)
        prob_b = classify_bounce_vs_hit(track_b, hint)
        # filter on cameras WITH evidence: an occluded camera's neutral 0.5
        # must neither veto nor rescue a candidate (reviewer defect: with the
        # naive max(), 0.5 > threshold always kept single-occlusion events)
        seen = [p for p, occ in ((prob_a, occ_a), (prob_b, occ_b)) if not occ]
        if min_bounce_prob > 0.0 and seen and max(seen) < min_bounce_prob:
            continue                          # paddle/body hit, not a bounce
        if subframe_refine:
            t_a, xy_a = refine_contact_subframe(ga, hint)
            t_b, xy_b = refine_contact_subframe(gb, hint)
            t_c = 0.5 * (t_a + t_b)
        else:
            t_c = hint
            xy_a = _interp_to(np.array([hint]), ga)[0]
            xy_b = _interp_to(np.array([hint]), gb)[0]
        events.append(BounceEvent(
            frame=t_c, xy_a_ft=xy_a, xy_b_ft=xy_b,
            separation_ft=float(s[i]), method="separation",
            quality={
                "conf_a": float(np.interp(hint, ga.frames, ga.conf)),
                "conf_b": float(np.interp(hint, gb.frames, gb.conf)),
                "reproj_a_px": calib_a.mean_reproj_px,
                "reproj_b_px": calib_b.mean_reproj_px,
                "occluded_a": occ_a,
                "occluded_b": occ_b,
                "vshape_a": any(abs(c - hint) <= 2.0 for c in vs_a),
                "vshape_b": any(abs(c - hint) <= 2.0 for c in vs_b),
                "bounce_prob_a": prob_a,
                "bounce_prob_b": prob_b,
            }))
    return events
