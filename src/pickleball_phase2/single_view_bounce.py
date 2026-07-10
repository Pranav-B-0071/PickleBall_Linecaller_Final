"""
single_view_bounce.py - single-camera bounce detection utilities.

VENDORED from the bounce_detection_handoff bundle (June 2026). In THIS repo
the primary bounce detector is the dual-camera separation signal (bounce.py,
per the derivation doc) - do not swap it for this module. What we consume:

  * _is_parabolic_bounce / _smooth  -> image-space bounce-shape cues for
    classify_bounce_vs_hit (BNC-1, bounce.py)
  * reject_velocity_outliers        -> tracker ID-switch cleanup
    (tracking.drop_track_outliers)

The CourtModel / homography / CSV / classification parts duplicate phase-2
functionality (court_model.py, calibration.py, fusion.py) and are unused
here; they are kept verbatim so the module stays diff-able against the
handoff and usable standalone (e.g. StreamingBounceDetector for ARCH-1).

Pure-numpy. Two detectors:
  * detect_bounces(...)            - OFFLINE, centered smoothing over the whole clip.
  * StreamingBounceDetector(...)   - CAUSAL (one-sided Kalman) for live / streaming.

A ball bounces when its pixel-y reaches a LOCAL MAXIMUM (y grows downward, so the
on-screen "lowest point") before rising again.

Offline pipeline:
  1. interpolate_gaps  - linearly fill detection gaps up to max_gap frames
  2. SEGMENT on gaps   - the trajectory is split wherever consecutive valid frames are
                         more than `split_gap` apart, and each segment is smoothed and
                         scanned independently. This stops the smoother + peak scan from
                         bridging an occlusion (the #1 false-bounce source), and a peak
                         landing right at a gap edge (a ball reappearing) is excluded.
  3. _smooth + peak    - box-filter y, local maxima with drop-magnitude guard
  4. lockout           - no new bounce within `lockout_frames` of an accepted one
  5/6. pixel_to_court / classify_bounce - court feet + IN/OUT/CLOSE_IN (needs homography)
  7. _suppress_same_side
  + SURFACE CHECK      - optional: drop bounces above `surface_min_py` (pixel) or more than
                         `max_out_of_bounds_ft` outside the court (occlusion / player FPs).

Defaults preserve the simple behaviour (segmentation is behaviour-neutral on clean data;
surface checks are off until you set them).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class CourtModel:
    width_ft: float
    length_ft: float
    name: str = "court"

    @property
    def corners_ft(self):
        # Order: near-left, near-right, far-right, far-left (clockwise from camera).
        return [(0.0, 0.0), (self.width_ft, 0.0),
                (self.width_ft, self.length_ft), (0.0, self.length_ft)]


PICKLEBALL = CourtModel(20.0, 44.0, "pickleball")
TENNIS_SINGLES = CourtModel(27.0, 78.0, "tennis_singles")
TENNIS_DOUBLES = CourtModel(36.0, 78.0, "tennis_doubles")

COURT_MAP = {
    "pickleball": PICKLEBALL,
    "tennis_singles": TENNIS_SINGLES,
    "tennis_doubles": TENNIS_DOUBLES,
}


@dataclass
class BounceParams:
    max_gap: int = 4             # only interpolate tracker-noise gaps (≤4 frames); longer gaps become segment splits
    smooth_kernel: int = 5       # box-filter width (odd)
    drop_lookback: int = 5       # compare against i-5 and i+5 for wider prominence context
    min_drop_px: float = 10.0    # min y drop vs those neighbours; higher = suppress micro-bounces
    lockout_frames: int = 20     # no new bounce within this many frames after an accepted one
    close_margin_ft: float = 0.5
    suppress_same_side: bool = True
    # --- hardening (opt-in) ---
    split_gap: int | None = None         # split trajectory on gaps > this (default = max_gap)
    surface_min_py: float | None = None  # pixel: drop bounces with px_y < this (too high in frame)
    max_out_of_bounds_ft: float | None = None  # court: drop bounces this far outside the court
    # --- parabolic shape check ---
    require_parabolic: bool = True   # reject candidates that don't fit a concave-down parabola
    parabolic_half_win: int = 5      # half-window around candidate for the poly fit
    parabolic_min_r2: float = 0.65   # minimum R² of the fit; lower = noisier data, set lower if needed
    # --- ascent verification (visible bounces) ---
    min_ascent_frames: int = 3       # ball must rise for this many frames post-peak; 0 = disabled
    # --- occlusion extrapolation (offline detector) ---
    extrap_occlusion: bool = True    # predict bounce through occlusion gaps
    extrap_fit_frames: int = 8       # last N frames of segment used for the parabola fit
    extrap_max_gap: int = 25         # only extrapolate if gap <= this many frames
    # --- streaming gravity model ---
    gravity_px_per_frame: float = 1.5  # apparent pixel-y acceleration per frame while coasting
    # --- velocity outlier rejection ---
    reject_outliers: bool = True       # drop teleport spikes (tracker ID-switches) before peak-finding
    outlier_max_jump_px: float = 60.0  # a point must jump more than this from BOTH neighbours to qualify
    outlier_reversal: bool = True      # ...and the in/out motion must sharply reverse (there-and-back)


@dataclass
class Bounce:
    frame: int
    px_x: float
    px_y: float
    court_x: float | None = None
    court_y: float | None = None
    classification: str = "UNKNOWN"
    side: str | None = None
    suppressed: bool = False
    predicted: bool = False   # True when extrapolated through an occlusion gap
    frame_exact: float | None = None  # sub-frame bounce time (parabola vertex)


def interpolate_gaps(coords, max_gap: int = 20):
    coords = np.asarray(coords, dtype=float)
    xs, ys = coords[:, 0].copy(), coords[:, 1].copy()
    detected = ~((coords[:, 0] == 0) & (coords[:, 1] == 0))
    valid = detected.copy()
    det_idx = np.where(detected)[0]
    for a, b in zip(det_idx[:-1], det_idx[1:]):
        gap = b - a
        if gap <= 1 or gap > max_gap:
            continue
        for f in range(a + 1, b):
            t = (f - a) / gap
            xs[f] = xs[a] + t * (xs[b] - xs[a])
            ys[f] = ys[a] + t * (ys[b] - ys[a])
            valid[f] = True
    return xs, ys, valid


def _smooth(y, kernel: int = 5):
    y = np.asarray(y, dtype=float)
    if kernel <= 1 or y.size < 2:
        return y
    pad = kernel // 2
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, np.ones(kernel) / kernel, mode="valid")


def scale_params_for_fps(params, fps, base_fps=30.0):
    """Return a copy of `params` with FRAME-based thresholds rescaled for `fps`.
    The defaults are tuned for ~30 fps; at 60 fps everything happens over 2x the
    frames. Pixel/feet thresholds (min_drop_px, *_ft, surface_min_py) are unchanged
    because they are physical distances. Per-frame velocity/accel terms scale down."""
    k = float(fps) / float(base_fps)
    if abs(k - 1.0) < 1e-3:
        return params

    def si(v):
        return max(1, int(round(v * k)))

    def odd(v):
        v = max(1, int(round(v * k)))
        return v + 1 if v % 2 == 0 else v

    return replace(
        params,
        max_gap=si(params.max_gap),
        smooth_kernel=odd(params.smooth_kernel),
        drop_lookback=si(params.drop_lookback),
        lockout_frames=si(params.lockout_frames),
        split_gap=(si(params.split_gap) if params.split_gap is not None else None),
        parabolic_half_win=si(params.parabolic_half_win),
        min_ascent_frames=(si(params.min_ascent_frames) if params.min_ascent_frames > 0 else 0),
        extrap_fit_frames=si(params.extrap_fit_frames),
        extrap_max_gap=si(params.extrap_max_gap),
        gravity_px_per_frame=params.gravity_px_per_frame / (k * k),
    )


def reject_velocity_outliers(coords, max_jump_px=60.0, require_reversal=True):
    """Zero out detections that 'teleport' far from BOTH neighbours and snap back
    (tracker ID-switches onto a player/background). Real bounces are preserved:
    they don't jump far between frames and their turn isn't a sharp 180 reversal."""
    coords = [(float(x), float(y)) for (x, y) in coords]
    pts = [(i, x, y) for i, (x, y) in enumerate(coords) if not (x == 0 and y == 0)]
    out = list(coords)
    for k in range(1, len(pts) - 1):
        _, x0, y0 = pts[k - 1]
        i1, x1, y1 = pts[k]
        _, x2, y2 = pts[k + 1]
        din = float(np.hypot(x1 - x0, y1 - y0))
        dout = float(np.hypot(x2 - x1, y2 - y1))
        if din <= max_jump_px or dout <= max_jump_px:
            continue
        if require_reversal:
            vin = np.array([x1 - x0, y1 - y0])
            vout = np.array([x2 - x1, y2 - y1])
            cos = float(vin @ vout) / (din * dout + 1e-9)
            if cos > -0.5:          # not a sharp there-and-back -> consistent fast motion, keep
                continue
        out[i1] = (0.0, 0.0)
    return out


def _subframe_peak(sf, sy, scx, j, half_win=5):
    """Sub-frame bounce time/position from the vertex of a parabola fit around index j.
    Returns (frame_exact, px_x_exact, px_y_exact). Falls back to the integer sample
    if the local shape isn't a clean concave-down arc."""
    lo = max(0, j - half_win)
    hi = min(len(sy), j + half_win + 1)
    if hi - lo < 5:
        return float(sf[j]), float(scx[j]), float(sy[j])
    t = np.arange(lo - j, hi - j, dtype=float)     # centred on j
    seg = np.asarray(sy[lo:hi], dtype=float)
    a, b, c = np.polyfit(t, seg, 2)
    if a >= 0:
        return float(sf[j]), float(scx[j]), float(sy[j])
    tstar = float(np.clip(-b / (2 * a), -half_win, half_win))
    idx = j + tstar
    grid = np.arange(len(sf), dtype=float)
    frame_exact = float(np.interp(idx, grid, np.asarray(sf, dtype=float)))
    px_exact = float(np.interp(idx, grid, np.asarray(scx, dtype=float)))
    py_exact = float(a * tstar * tstar + b * tstar + c)
    return frame_exact, px_exact, py_exact


def _is_parabolic_bounce(sy, j, half_win: int = 5, min_r2: float = 0.65) -> bool:
    """True if the smoothed y-trajectory around index j fits a concave-down parabola.

    A real bounce is a parabolic descent (y rising) followed by parabolic ascent (y falling).
    In pixel-y coords y grows downward, so a bounce peak is concave-down (leading coefficient < 0).
    Net hits and noise spikes don't produce this shape.
    """
    lo = max(0, j - half_win)
    hi = min(len(sy), j + half_win + 1)
    seg = sy[lo:hi]
    if len(seg) < 5:
        return False
    t = np.arange(len(seg), dtype=float)
    coeffs = np.polyfit(t, seg, 2)
    if coeffs[0] >= 0:  # concave-up or flat - not a bounce peak
        return False
    fitted = np.polyval(coeffs, t)
    ss_res = float(np.sum((seg - fitted) ** 2))
    ss_tot = float(np.sum((seg - seg.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return r2 >= min_r2


def _extrapolate_bounce(frames_a, xs_a, ys_a, frames_b, xs_b, ys_b, surface_y, p):
    """Predict a bounce hidden inside an occlusion gap between two trajectory segments.

    Conditions to fire:
    - Segment A must end descending (pixel-y increasing).
    - A parabola fit to the last extrap_fit_frames of A must extrapolate a landing
      within the gap frame range.
    - Segment B must start ascending (pixel-y decreasing) to confirm re-emergence.

    Returns a Bounce(predicted=True) or None.
    """
    n_fit = min(p.extrap_fit_frames, len(frames_a))
    if n_fit < 3:
        return None

    t_a = frames_a[-n_fit:].astype(float)
    y_a = ys_a[-n_fit:]
    x_a = xs_a[-n_fit:]

    # Must be descending at end of segment (pixel-y increasing)
    if y_a[-1] <= y_a[max(0, len(y_a) - 2)]:
        return None

    # Parabola fit on time-relative coordinates
    t0 = t_a[0]
    t_rel = t_a - t0
    coeffs = np.polyfit(t_rel, y_a, 2)
    a_coef, b_coef, c_coef = coeffs

    # a_coef > 0 = concave-up in pixel-y → ball accelerating downward (correct ballistic descent)
    if a_coef <= 0:
        return None

    # Surface y: explicit setting or last known y + small margin
    surf = float(surface_y) if surface_y is not None else float(y_a[-1]) + 8.0

    # Solve a*dt^2 + b*dt + (c - surf) = 0
    disc = b_coef ** 2 - 4 * a_coef * (c_coef - surf)
    if disc < 0:
        return None

    dt1 = (-b_coef + np.sqrt(disc)) / (2 * a_coef)
    dt2 = (-b_coef - np.sqrt(disc)) / (2 * a_coef)
    # Take smallest positive dt beyond the last fitted point
    candidates = [dt for dt in (dt1, dt2) if dt > t_rel[-1]]
    if not candidates:
        return None
    dt_land = min(candidates)
    t_land = t0 + dt_land

    # Landing must fall inside the gap
    t_end_a = float(frames_a[-1])
    t_start_b = float(frames_b[0]) if len(frames_b) > 0 else float("inf")
    if not (t_end_a < t_land < t_start_b):
        return None

    # Segment B must start ascending (ball rising after gap = bounce confirmed)
    if len(frames_b) < 2:
        return None
    n_check = min(4, len(ys_b))
    if n_check >= 2 and ys_b[n_check - 1] >= ys_b[0]:
        return None  # B doesn't show ascent → likely not a bounce

    # Extrapolate x using horizontal velocity from end of A
    vx = (float(x_a[-1]) - float(x_a[max(0, len(x_a) - 2)])) / max(
        1.0, float(t_a[-1]) - float(t_a[max(0, len(t_a) - 2)])
    )
    x_land = float(x_a[-1]) + vx * (dt_land - float(t_rel[-1]))

    return Bounce(
        frame=int(round(t_land)),
        px_x=float(x_land),
        px_y=surf,
        predicted=True,
        frame_exact=float(t_land),
    )


def _segments(frames, split_gap):
    """Yield (start, end) index ranges of runs where consecutive frame gaps <= split_gap."""
    if frames.size == 0:
        return []
    segs, s = [], 0
    for k in range(1, frames.size):
        if frames[k] - frames[k - 1] > split_gap:
            segs.append((s, k)); s = k
    segs.append((s, frames.size))
    return segs


def pixel_to_court(homography, px, py):
    v = np.asarray(homography, dtype=float) @ np.array([px, py, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def classify_bounce(court_x, court_y, court: CourtModel, close_margin: float = 0.5):
    inside = (0.0 <= court_x <= court.width_ft) and (0.0 <= court_y <= court.length_ft)
    if not inside:
        return "OUT"
    margin = min(court_x, court.width_ft - court_x,
                 court_y, court.length_ft - court_y)
    return "CLOSE_IN" if margin < close_margin else "IN"


def _out_of_bounds_ft(cx, cy, court):
    ox = max(0.0, -cx, cx - court.width_ft)
    oy = max(0.0, -cy, cy - court.length_ft)
    return max(ox, oy)


def compute_homography(image_corners, court: CourtModel | None = None, court_corners=None):
    """4-point ground homography pixels -> court feet (numpy DLT).
    image_corners order: near-left, near-right, far-right, far-left."""
    if court_corners is None:
        if court is None:
            raise ValueError("Provide either court or court_corners.")
        court_corners = court.corners_ft
    src = np.asarray(image_corners, dtype=float)
    dst = np.asarray(court_corners, dtype=float)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("Need exactly 4 image corners and 4 court corners.")
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def _suppress_same_side(bounces):
    last_side = None
    for b in bounces:
        if b.classification == "OUT":
            last_side = None
            continue
        if last_side is not None and b.side == last_side:
            b.suppressed = True
        else:
            b.suppressed = False
            last_side = b.side


def _apply_homography_to_bounce(b, homography, court, p):
    """Map a Bounce to court coordinates and classify it in-place."""
    b.court_x, b.court_y = pixel_to_court(homography, b.px_x, b.px_y)
    if p.max_out_of_bounds_ft is not None and \
            _out_of_bounds_ft(b.court_x, b.court_y, court) > p.max_out_of_bounds_ft:
        return False  # drop
    if not b.predicted:
        b.classification = classify_bounce(b.court_x, b.court_y, court, p.close_margin_ft)
    b.side = "near" if b.court_y < court.length_ft / 2 else "far"
    return True


def _finalize(raw, homography, court, p, predicted_bounces=None):
    """raw: list of (frame, px_x, px_y) → map, surface-check, merge predicted, lockout, suppress."""
    mapped = []
    for fr, px, py, fe in raw:
        b = Bounce(frame=fr, px_x=px, px_y=py, frame_exact=fe)
        if homography is not None and court is not None:
            if not _apply_homography_to_bounce(b, homography, court, p):
                continue
        mapped.append(b)

    # Merge in pre-built predicted bounces (from gap extrapolation)
    for b in (predicted_bounces or []):
        if homography is not None and court is not None:
            _apply_homography_to_bounce(b, homography, court, p)
        mapped.append(b)

    mapped.sort(key=lambda b: b.frame)

    kept, last = [], -(10 ** 9)
    for b in mapped:
        if b.frame - last <= p.lockout_frames:
            continue
        last = b.frame
        kept.append(b)
    if homography is not None and court is not None and p.suppress_same_side:
        _suppress_same_side(kept)
    return kept


def detect_bounces(coords, homography=None, court: CourtModel | None = None,
                   params: BounceParams | None = None):
    """OFFLINE ground-bounce detection. Returns list[Bounce] in chronological order."""
    p = params or BounceParams()
    if p.reject_outliers:
        coords = reject_velocity_outliers(coords, p.outlier_max_jump_px, p.outlier_reversal)
    xs, ys, valid = interpolate_gaps(coords, p.max_gap)
    frames = np.where(valid)[0]
    lb = p.drop_lookback
    if frames.size < 2 * lb + 1:
        return []
    cx, cy = xs[frames], ys[frames]
    split_gap = p.split_gap if p.split_gap is not None else p.max_gap
    segs = list(_segments(frames, split_gap))

    raw = []
    for s, e in segs:
        if e - s < 2 * lb + 1:
            continue
        sf, scx, scy = frames[s:e], cx[s:e], cy[s:e]
        sy = _smooth(scy, p.smooth_kernel)
        for j in range(lb, (e - s) - lb):
            if not (sy[j] >= sy[j - 1] and sy[j] >= sy[j + 1]):
                continue
            if sy[j] - min(sy[j - lb], sy[j + lb]) < p.min_drop_px:
                continue
            if p.surface_min_py is not None and scy[j] < p.surface_min_py:
                continue
            if p.require_parabolic and not _is_parabolic_bounce(sy, j, p.parabolic_half_win, p.parabolic_min_r2):
                continue
            # Ascent check: ball must rise for min_ascent_frames after the peak
            if p.min_ascent_frames > 0:
                avail = (e - s) - j - 1
                if avail >= p.min_ascent_frames:
                    if not all(sy[j + k] < sy[j] for k in range(1, p.min_ascent_frames + 1)):
                        continue
            fe, pxx, pyy = _subframe_peak(sf, sy, scx, j, p.parabolic_half_win)
            raw.append((int(round(fe)), float(pxx), float(pyy), float(fe)))

    # Extrapolate bounces hidden in occlusion gaps
    predicted = []
    if p.extrap_occlusion:
        for i in range(len(segs) - 1):
            s0, e0 = segs[i]
            s1, e1 = segs[i + 1]
            gap = int(frames[s1]) - int(frames[e0 - 1])
            if gap > p.extrap_max_gap:
                continue
            b = _extrapolate_bounce(
                frames[s0:e0], cx[s0:e0], cy[s0:e0],
                frames[s1:e1], cx[s1:e1], cy[s1:e1],
                p.surface_min_py, p,
            )
            if b is not None:
                predicted.append(b)

    return _finalize(raw, homography, court, p, predicted)


class StreamingBounceDetector:
    """CAUSAL bounce detector for live / streaming.

    A constant-velocity Kalman filter tracks vertical position y and velocity vy from
    ONLY past frames. Missing detections are handled by predict-only steps (the filter
    coasts on its velocity estimate; no linear interpolation, so no interpolation-induced
    false bounce). A bounce fires when the estimated vy flips from descending (+) to
    ascending (-) at a y-maximum, with a prominence (min_drop), debounce (lockout) and
    optional surface checks. After `max_misses` consecutive gaps the filter resets
    (trajectory split). Feed frames in order with update().
    """

    def __init__(self, params: BounceParams | None = None, homography=None,
                 court: CourtModel | None = None, process_var: float = 3.0,
                 meas_var: float = 4.0, max_misses: int | None = None,
                 min_descent_vy: float = 0.5):
        self.p = params or BounceParams()
        self.homography = homography
        self.court = court
        self.q = float(process_var)
        self.r = float(meas_var)
        self.max_misses = self.p.max_gap if max_misses is None else max_misses
        self.min_descent_vy = float(min_descent_vy)
        self.reset()

    def reset(self):
        self.s = None            # [y, vy]
        self.P = None
        self.prev_vy = None
        self.apex_y = None
        self.last_x = None
        self.last_bounce_frame = -(10 ** 9)
        self.misses = 0
        self.hist = []          # recent measured (frame, x, y) for sub-frame timing

    def _init(self, y):
        self.s = np.array([y, 0.0])
        self.P = np.array([[self.r, 0.0], [0.0, 100.0]])
        self.prev_vy = 0.0
        self.apex_y = y

    def update(self, frame, x, y, detected=True):
        """Feed one frame. Returns a Bounce when one fires this frame, else None."""
        measured = detected and not (x == 0 and y == 0)
        if measured:
            self.last_x = x
            self.misses = 0
            self.hist.append((frame, x, y))
            if len(self.hist) > 64:
                del self.hist[0]
            if self.s is None:
                self._init(y)
                return None
        else:
            self.misses += 1
            if self.s is None:
                return None
            if self.misses > self.max_misses:
                self.reset()
                return None

        # predict (dt = 1); add gravity to vy when coasting through a gap
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = self.q * np.array([[0.25, 0.5], [0.5, 1.0]])
        self.s = F @ self.s
        if not measured:
            self.s[1] += self.p.gravity_px_per_frame  # accelerate downward while coasting
        self.P = F @ self.P @ F.T + Q

        # update
        if measured:
            H = np.array([[1.0, 0.0]])
            innov = y - (H @ self.s)[0]
            S = (H @ self.P @ H.T)[0, 0] + self.r
            K = (self.P @ H.T).flatten() / S
            self.s = self.s + K * innov
            self.P = (np.eye(2) - np.outer(K, H[0])) @ self.P

        fy, vy = self.s[0], self.s[1]
        bounce = None
        if self.prev_vy is not None:
            if self.prev_vy < 0 and vy >= 0:                 # apex (y minimum)
                self.apex_y = fy
            elif self.prev_vy >= self.min_descent_vy and vy < 0:   # bounce (y maximum)
                if self.apex_y is None:
                    self.apex_y = fy
                prominence = fy - self.apex_y
                if prominence >= self.p.min_drop_px and \
                        frame - self.last_bounce_frame > self.p.lockout_frames:
                    bounce = self._emit(frame, fy)
        self.prev_vy = vy
        return bounce

    def _emit(self, frame, fy):
        px = self.last_x if self.last_x is not None else 0.0
        if self.p.surface_min_py is not None and fy < self.p.surface_min_py:
            return None
        is_predicted = self.misses > 0  # fired during a gap = extrapolated
        # For predicted bounces, gate on extrap_max_gap
        if is_predicted and self.misses > self.p.extrap_max_gap:
            return None
        # sub-frame timing: vertex of a parabola fit to recent measured y
        fe = float(frame)
        hw = self.p.parabolic_half_win
        recent = [pt for pt in self.hist if pt[0] >= frame - 2 * hw]
        if not is_predicted and len(recent) >= 5:
            fr = np.array([pt[0] for pt in recent], dtype=float)
            yy = np.array([pt[2] for pt in recent], dtype=float)
            xx = np.array([pt[1] for pt in recent], dtype=float)
            a, bb, cc = np.polyfit(fr - fr[0], yy, 2)
            if a < 0:
                t_exact = fr[0] - bb / (2 * a)
                if fr[0] <= t_exact <= fr[-1] + 1.0:
                    fe = float(t_exact)
                    px = float(np.interp(t_exact, fr, xx))
                    fy = float(np.polyval([a, bb, cc], t_exact - fr[0]))
        b = Bounce(frame=int(frame), px_x=float(px), px_y=float(fy),
                   predicted=is_predicted, frame_exact=fe)
        if self.homography is not None and self.court is not None:
            b.court_x, b.court_y = pixel_to_court(self.homography, px, fy)
            if self.p.max_out_of_bounds_ft is not None and \
                    _out_of_bounds_ft(b.court_x, b.court_y, self.court) > self.p.max_out_of_bounds_ft:
                return None
            b.classification = classify_bounce(b.court_x, b.court_y, self.court, self.p.close_margin_ft)
            b.side = "near" if b.court_y < self.court.length_ft / 2 else "far"
        self.last_bounce_frame = frame
        return b


def detect_bounces_streaming(coords, homography=None, court=None, params=None, **kwargs):
    """Run the causal StreamingBounceDetector over a coords list (offline convenience)."""
    det = StreamingBounceDetector(params=params, homography=homography, court=court, **kwargs)
    out = []
    for i, (x, y) in enumerate(coords):
        b = det.update(i, x, y, detected=not (x == 0 and y == 0))
        if b is not None:
            out.append(b)
    return out


def load_track_csv(path):
    coords = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            x, y = float(row.get("X", 0) or 0), float(row.get("Y", 0) or 0)
            vis = row.get("Visibility")
            if vis is not None and vis != "" and int(float(vis)) == 0:
                x, y = 0.0, 0.0
            coords.append((x, y))
    return coords


def save_bounces_csv(bounces, path, include_suppressed=False):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "frame_exact", "px_x", "px_y", "court_x", "court_y",
                    "classification", "side", "suppressed", "predicted"])
        for b in bounces:
            if b.suppressed and not include_suppressed:
                continue
            w.writerow([b.frame,
                        None if b.frame_exact is None else round(b.frame_exact, 2),
                        round(b.px_x, 1), round(b.px_y, 1),
                        None if b.court_x is None else round(b.court_x, 2),
                        None if b.court_y is None else round(b.court_y, 2),
                        b.classification, b.side, b.suppressed, b.predicted])
