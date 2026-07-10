"""Analysis service (Page 3) - synchronized 3-cam run -> ball track + IN/OUT.

Today this returns a MOCK ball trajectory and bounce events, but:

  * the schema is exactly what the real pipeline yields (a court-space track plus
    ``LineCall``-shaped bounce events), and
  * every bounce verdict is computed with the REAL ``court_model.line_call`` on
    the near-half zone - so IN/OUT, the signed margin, and the nearest line are
    genuine given the (mock) positions.

Swapping in the real system is ``_run_real`` (kept next to the mock and wired to
``pipeline.run_clip_pair`` + the persisted calibration) - no route/UI change.

ROI rule (per spec): the ball is tracked over the WHOLE clip, but a bounce is
only considered for an IN/OUT call when it lands inside the near-half ROI.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .. import bootstrap  # noqa: F401
from pickleball_phase2.court_model import (COURT_L_FT, COURT_W_FT, FT_TO_CM,
                                           KITCHEN_NEAR_Y_FT, NET_Y_FT,
                                           line_call)

# Near-half ROI in canonical feet (net -> near baseline).
ROI_FT = [[0.0, NET_Y_FT], [COURT_W_FT, NET_Y_FT],
          [COURT_W_FT, COURT_L_FT], [0.0, COURT_L_FT]]

# Near-side kitchen (non-volley zone): net -> near kitchen line. CAM3 counts
# bounces here. In the real system the test is point-in-CAM3's-quad on CAM3's
# tracked bounce pixel; here (mock) the equivalent court-ft band is the proxy.
KITCHEN_ZONE_FT = [[0.0, NET_Y_FT], [COURT_W_FT, NET_Y_FT],
                   [COURT_W_FT, KITCHEN_NEAR_Y_FT], [0.0, KITCHEN_NEAR_Y_FT]]


class _NotEligible(Exception):
    """The real pipeline cannot run for this session (missing calibration,
    videos, or tracker weights); the caller falls back to the mock."""


def run_analysis(session_state: dict, fps: float, duration_s: float,
                 seed_key: str = "", store=None, sid: str | None = None,
                 cfg=None) -> dict:
    """Produce the Page-3 analysis payload for the current session.

    When ``store``/``sid``/``cfg`` are supplied AND CAM1+CAM2 are both
    calibrated with their clips uploaded, this runs the REAL pipeline
    (GridTrackNet tracking -> dual ground-plane homography -> separation-signal
    bounce detection -> two-camera fusion). Otherwise, or if that fails, it
    falls back to the deterministic mock so the page is always demonstrable.

    CAM1/CAM2 give the IN/OUT line calls; if CAM3 has a kitchen box, bounces
    inside the kitchen zone are additionally tallied ("kitchen bounces").
    """
    result, note = None, None
    if store is not None and sid is not None and cfg is not None:
        try:
            result = _run_real(session_state, store, sid, cfg, fps)
        except _NotEligible:
            result = None                       # expected -> silent mock fallback
        except Exception as exc:                # real attempted but crashed
            note = f"live analysis failed, showing mock instead: {exc}"
            result = None
    if result is None:
        result = _run_mock(fps=fps, duration_s=duration_s, seed_key=seed_key)
    if note:
        result["note"] = note
    return _finalize(result, session_state, fps, duration_s)


def _finalize(result: dict, session_state: dict, fps: float,
              duration_s: float) -> dict:
    """Attach ROI/kitchen/fps/duration to a bounce payload (real or mock).
    CAM3 (kitchen box), when calibrated, adds the kitchen-bounce tally."""
    result["roi_ft"] = ROI_FT

    cam3 = (session_state.get("calibration") or {}).get("cam3")
    cam3_ok = bool(cam3 and cam3.get("type") == "kitchen_box")
    for b in result["bounces"]:
        b["in_kitchen"] = bool(cam3_ok and b["in_roi"] and _in_kitchen(b["x"], b["y"]))
    result["cam3_calibrated"] = cam3_ok
    result["kitchen_zone_ft"] = KITCHEN_ZONE_FT
    result["summary"]["kitchen_bounces"] = (
        sum(1 for b in result["bounces"] if b["in_kitchen"]) if cam3_ok else None)

    result["fps"] = round(fps, 3)
    result["duration_s"] = round(duration_s, 3)
    return result


def _in_kitchen(x: float, y: float) -> bool:
    # bool(): inputs may be np.float64, whose comparisons yield np.bool_
    # (not JSON-serializable by Flask's provider).
    return bool((0.0 <= x <= COURT_W_FT) and (NET_Y_FT <= y <= KITCHEN_NEAR_Y_FT))


# --------------------------------------------------------------------------
# Mock generator (deterministic per session)
# --------------------------------------------------------------------------

def _rng(seed_key: str) -> np.random.Generator:
    h = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
    return np.random.default_rng(h)


def _run_mock(fps: float, duration_s: float, seed_key: str) -> dict:
    rng = _rng(seed_key or "default")
    fps = fps if fps and fps > 1 else 60.0
    duration_s = duration_s if duration_s and duration_s > 1 else 20.0

    # A rally = alternating bounce points that march up the near half toward the
    # baseline, a few deliberately near a line so verdicts are interesting.
    n_bounces = int(rng.integers(5, 9))
    bounces_ft = []
    for k in range(n_bounces):
        x = float(np.clip(rng.normal(COURT_W_FT / 2, 5.0), -1.5, COURT_W_FT + 1.5))
        y = float(np.clip(NET_Y_FT + 3 + k * (COURT_L_FT - NET_Y_FT - 4) / n_bounces
                          + rng.normal(0, 1.2), NET_Y_FT + 0.5, COURT_L_FT + 1.5))
        bounces_ft.append((x, y))

    # Segment durations, then a straight-line top-view path between bounces
    # (projectile motion projects to a line in the ground plane).
    seg_frames = [int(rng.integers(int(0.5 * fps), int(1.1 * fps)))
                  for _ in range(n_bounces - 1)]
    trajectory, bounce_frames, f = [], [], 0
    for i in range(n_bounces - 1):
        (x0, y0), (x1, y1) = bounces_ft[i], bounces_ft[i + 1]
        nseg = seg_frames[i]
        if i == 0:
            bounce_frames.append(f)
        for s in range(nseg):
            u = s / nseg
            # a gentle lateral wobble so the tail reads as a real flight
            wob = 0.4 * np.sin(u * np.pi) * rng.normal(1.0, 0.15)
            trajectory.append({
                "frame": f, "t": round(f / fps, 4),
                "x": round(float(x0 + u * (x1 - x0) + wob), 3),
                "y": round(float(y0 + u * (y1 - y0)), 3),
            })
            f += 1
        bounce_frames.append(f)
    # tag ROI membership for every sampled point
    for p in trajectory:
        p["in_roi"] = _in_roi(p["x"], p["y"])

    bounces = []
    for idx, bf in enumerate(bounce_frames):
        x, y = bounces_ft[idx]
        in_roi = _in_roi(x, y)
        call = line_call(np.array([x, y]), zone="near_half")
        bounces.append({
            "frame": bf, "t": round(bf / fps, 4),
            "x": round(x, 3), "y": round(y, 3),
            "in_roi": in_roi,
            # Only ROI bounces get an official verdict (per the ROI rule).
            "verdict": call["verdict"] if in_roi else "N/A",
            "margin_ft": round(call["distance_ft"], 3) if in_roi else None,
            "nearest_line": call["nearest_line"] if in_roi else None,
            "confidence": round(float(rng.uniform(0.82, 0.98)), 3),
        })

    considered = [b for b in bounces if b["in_roi"]]
    summary = {
        "total_bounces": len(bounces),
        "in_roi_bounces": len(considered),
        "in": sum(1 for b in considered if b["verdict"] == "IN"),
        "out": sum(1 for b in considered if b["verdict"] == "OUT"),
        "avg_confidence": round(float(np.mean([b["confidence"] for b in bounces])), 3),
    }
    return {"source": "mock", "trajectory": trajectory, "bounces": bounces,
            "summary": summary}


def _in_roi(x: float, y: float) -> bool:
    return bool((0.0 <= x <= COURT_W_FT) and (NET_Y_FT <= y <= COURT_L_FT))


# --------------------------------------------------------------------------
# Real pipeline (§5.5-5.8): GridTrackNet -> dual ground-plane homography ->
# separation-signal bounce detection -> two-camera fusion -> line calls.
# --------------------------------------------------------------------------

# category confidence (LineCall.confidence) -> numeric, for the summary average
_CONF_SCORE = {"high-agreement": 0.95, "two-view-resolved": 0.80, "single-view": 0.65}


def _reconstruct_calib(cal: dict):
    """Rebuild a CourtCalibration from the persisted Page-1 calibration dict.
    Only H (image->court) and mean_reproj_px are needed downstream (ground
    projection + fusion weighting)."""
    from pickleball_phase2.calibration import CourtCalibration
    return CourtCalibration(
        H=np.asarray(cal["H"], dtype=float),
        H_inv=np.asarray(cal["H_inv"], dtype=float),
        inlier_mask=np.asarray(cal.get("inlier_mask", [1] * 12), dtype=bool),
        mean_reproj_px=float(cal.get("mean_reproj_px", 5.0)),
        mean_reproj_ft=float(cal.get("mean_reproj_ft", 0.0)),
        used_indices=list(cal.get("used_indices", [])))


def _run_real(session_state: dict, store, sid: str, cfg, fps: float) -> dict:
    """Run the real clip pair. Raises _NotEligible when prerequisites are absent
    (so the caller falls back to the mock without surfacing an error)."""
    calib_state = session_state.get("calibration") or {}
    ca, cb = calib_state.get("cam1"), calib_state.get("cam2")
    if not (ca and cb and "H" in ca and "H" in cb):
        raise _NotEligible("CAM1 and CAM2 must both be court-calibrated first")
    va, vb = store.path_for(sid, "cam1"), store.path_for(sid, "cam2")
    if va is None or vb is None:
        raise _NotEligible("CAM1 and CAM2 clips are both required")

    from ..bootstrap import REPO_ROOT
    weights = REPO_ROOT / cfg.get("paths.tracker_weights", "models/model_weights.onnx")
    if not weights.exists():
        raise _NotEligible(f"ball-tracker weights not found at {weights}")

    from pickleball_phase2.pipeline import run_clip_pair
    from pickleball_phase2.tracking import GridTrackNetTracker

    fps = fps if fps and fps > 1 else float(cfg.get("capture.fps", 60))
    # aligned = frame_B - offset (the value set/confirmed on the calibration page)
    offset = float(((session_state.get("sync") or {}).get("offsets") or {})
                   .get("cam2", {}).get("offset_frames", 0.0) or 0.0)
    # bound CPU inference to an analysis window (GridTrackNet ~10 fps on CPU)
    window_s = float(cfg.get("analysis.window_s", 20.0))
    max_frames = int(window_s * fps) if window_s and window_s > 0 else None

    # Share the persistent preprocessing cache so a repeat analysis of the same
    # clips reloads GridTrackNet inference instead of recomputing on CPU.
    tracker_cache = None
    if bool(cfg.get("cache.enabled", True)):
        from pickleball_phase2.cache import Cache
        tracker_cache = Cache(REPO_ROOT / cfg.get("cache.dir", "data/cache"))

    tracker = GridTrackNetTracker(
        str(weights),
        provider=cfg.get("tracking.provider", "cpu"),
        min_confidence=cfg.get("tracking.min_confidence", 0.3),
        frame_mode=cfg.get("tracking.frame_mode", "dual"),
        reject_outliers=bool(cfg.get("tracking.reject_outliers", True)),
        outlier_max_jump_px=cfg.get("tracking.outlier_max_jump_px", 60.0),
        max_frames=max_frames,
        cache=tracker_cache)

    calls = run_clip_pair(
        str(va), str(vb), cfg,
        tracker_a=tracker, tracker_b=tracker,
        calib_a=_reconstruct_calib(ca), calib_b=_reconstruct_calib(cb),
        offset_frames=offset)
    payload = _calls_to_payload(calls, fps)
    if max_frames is not None:
        payload["note"] = (f"analyzed the first {window_s:.0f}s; "
                           "raise analysis.window_s (or set it to 0) for full-clip coverage")
    return payload


def analyze_from_tracked(session_state: dict, store, sid: str, cfg, fps: float,
                         duration_s: float, footage_dir) -> dict:
    """Analyze the pre-computed ``Footage/<role>_tracked.csv`` files (written by
    the Calculate step) and produce IN/OUT line calls.

    Detection is SINGLE-VIEW per camera (the ball's on-screen lowest point - a
    down-then-up parabola in image space), because these cameras are mounted low:
    projecting an AIRBORNE ball's pixel through the ground homography explodes
    near the horizon (separation of hundreds/thousands of feet on a 44 ft court),
    which makes the dual-camera separation signal unusable here. Each camera's
    homography is then used ONLY to place the (near-ground) contact point on the
    court and call IN/OUT - a projection that IS stable. The two cameras' bounces
    are merged by aligned frame.

    No re-tracking: the CSVs are used directly. Raises ``ValueError`` with a
    user-facing message when the tracked files or the CAM1/CAM2 calibration
    (needed for IN/OUT) are missing, so the route can surface it as a toast.
    """
    from pathlib import Path
    footage_dir = Path(footage_dir)
    csv_a, csv_b = footage_dir / "cam1_tracked.csv", footage_dir / "cam2_tracked.csv"
    missing = [p.name for p in (csv_a, csv_b) if not p.exists()]
    if missing:
        raise ValueError(f"missing tracked data ({', '.join(missing)}) - run Calculate first")

    calib_state = session_state.get("calibration") or {}
    ca, cb = calib_state.get("cam1"), calib_state.get("cam2")
    if not (ca and cb and "H" in ca and "H" in cb):
        raise ValueError("calibrate CAM 1 and CAM 2 on the Calibration page first "
                         "(their court homographies are needed for IN/OUT)")
    calib_a, calib_b = _reconstruct_calib(ca), _reconstruct_calib(cb)

    fps = fps if fps and fps > 1 else float(cfg.get("capture.fps", 60))
    # aligned = frame_B - offset (the value set/confirmed on the calibration page)
    offset = float(((session_state.get("sync") or {}).get("offsets") or {})
                   .get("cam2", {}).get("offset_frames", 0.0) or 0.0)

    track_a = _track_from_csv(csv_a, fps)   # cam1: aligned = frame (offset 0)
    track_b = _track_from_csv(csv_b, fps)   # cam2: local-indexed; offset applied at merge

    result = _detect_single_view(track_a, track_b, calib_a, calib_b, cfg, fps, offset)
    return _finalize(result, session_state, fps, duration_s)


def _detect_single_view(track_a, track_b, calib_a, calib_b, cfg, fps: float,
                        offset: float) -> dict:
    """Detect bounces per camera in image space, place each on the court via its
    homography, and merge the two cameras. Returns the Page-3 payload (same shape
    as ``_calls_to_payload``)."""
    # image-space contact -> court-ft, then a stable near-ground projection
    off_px = float(cfg.get("bounce.contact_offset_px", 0.0))  # nudge to ball's bottom
    a = _camera_bounces_court(track_a.uv, calib_a, offset=0.0, off_px=off_px, cfg=cfg)
    b = _camera_bounces_court(track_b.uv, calib_b, offset=offset, off_px=off_px, cfg=cfg)
    merged = _merge_camera_bounces(
        a, b, calib_a.mean_reproj_px, calib_b.mean_reproj_px,
        window=float(cfg.get("bounce.sv_merge_window_frames", 10.0)))

    bounces = []
    for frame, xy, matched in merged:
        x, y = float(xy[0]), float(xy[1])
        in_roi = _in_roi(x, y)
        call = line_call(np.array([x, y]), zone="near_half")
        bounces.append({
            "frame": float(frame), "t": round(float(frame) / fps, 4),
            "x": round(x, 3), "y": round(y, 3),
            "in_roi": in_roi,
            "verdict": call["verdict"] if in_roi else "N/A",
            "margin_ft": round(call["distance_ft"], 3) if in_roi else None,
            "nearest_line": call["nearest_line"] if in_roi else None,
            "confidence": 0.9 if matched else 0.7,
        })
    bounces.sort(key=lambda bd: bd["frame"])
    trajectory = _trajectory_from_bounces(bounces, fps)

    considered = [bd for bd in bounces if bd["in_roi"]]
    summary = {
        "total_bounces": len(bounces),
        "in_roi_bounces": len(considered),
        "in": sum(1 for bd in considered if bd["verdict"] == "IN"),
        "out": sum(1 for bd in considered if bd["verdict"] == "OUT"),
        "avg_confidence": round(float(np.mean([bd["confidence"] for bd in bounces])), 3)
                          if bounces else 0.0,
    }
    print(f"[analyze] single-view bounces: cam1={len(a)} cam2={len(b)} "
          f"merged={len(bounces)} (in_roi={len(considered)})")
    return {"source": "gridtracknet", "trajectory": trajectory,
            "bounces": bounces, "summary": summary}


def _camera_bounces_court(uv, calib, offset: float, off_px: float, cfg):
    """Single-view bounce detection on one camera's dense pixel track, mapped to
    court feet. Returns a list of ``(aligned_frame, court_xy)``. ``uv`` is the
    dense (N,2) array (``(0,0)`` = no detection); ``offset`` converts this
    camera's local frame index to the shared timeline (aligned = frame - offset).
    """
    from pickleball_phase2.calibration import apply_homography
    from pickleball_phase2.single_view_bounce import BounceParams
    from pickleball_phase2.single_view_bounce import detect_bounces as sv_detect

    params = BounceParams(
        min_drop_px=float(cfg.get("bounce.sv_min_drop_px", 10.0)),
        lockout_frames=int(cfg.get("bounce.sv_lockout_frames", 15)),
        parabolic_min_r2=float(cfg.get("bounce.sv_parabolic_min_r2", 0.6)))
    out = []
    for bnc in sv_detect(uv, homography=None, court=None, params=params):
        f_local = float(bnc.frame_exact if bnc.frame_exact is not None else bnc.frame)
        # +y is downward in the image, i.e. toward the ball's ground contact
        px = np.array([[bnc.px_x, bnc.px_y + off_px]], dtype=np.float64)
        xy = apply_homography(calib.H, px)[0]
        # A genuine bounce contact projects near the court; a mis-detected
        # airborne point blows up through the ground homography - drop those.
        if not (-15.0 <= xy[0] <= 35.0 and -15.0 <= xy[1] <= 59.0):
            continue
        out.append((f_local - offset, xy))
    return out


def _merge_camera_bounces(a, b, reproj_a: float, reproj_b: float, window: float):
    """Merge two cameras' ``(aligned_frame, court_xy)`` bounce lists. A pair
    within ``window`` frames is one event (position = reprojection-weighted mean,
    lower-error camera trusted more); unmatched bounces are kept as single-view.
    Returns ``(frame, court_xy, matched)`` sorted by frame."""
    wa, wb = 1.0 / max(reproj_a, 1e-3), 1.0 / max(reproj_b, 1e-3)
    used, merged = set(), []
    for fa, xya in a:
        j, best = None, window
        for k, (fb, _) in enumerate(b):
            if k in used:
                continue
            d = abs(fb - fa)
            if d <= best:
                j, best = k, d
        if j is not None:
            used.add(j)
            fb, xyb = b[j]
            merged.append((0.5 * (fa + fb), (wa * xya + wb * xyb) / (wa + wb), True))
        else:
            merged.append((fa, xya, False))
    for k, (fb, xyb) in enumerate(b):
        if k not in used:
            merged.append((fb, xyb, False))
    merged.sort(key=lambda m: m[0])
    return merged


def _track_from_csv(csv_path, fps: float, offset: float = 0.0):
    """Build a ``BallTrack`` from a gridtracknet ``_tracked.csv`` (columns
    ``frame_number,x,y``). Rows are sparse (only tracked frames), so the track is
    densified to one entry per frame with a ``valid`` mask. ``offset`` shifts the
    frame timeline (aligned = frame - offset) so camera B lines up with A."""
    import csv as _csv
    from pickleball_phase2.tracking import BallTrack

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            try:
                rows.append((int(float(r["frame_number"])), float(r["x"]), float(r["y"])))
            except (KeyError, ValueError, TypeError):
                continue
    n = (max(fr for fr, _, _ in rows) + 1) if rows else 0
    uv = np.zeros((n, 2), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for fr, x, y in rows:
        if 0 <= fr < n:
            uv[fr] = (x, y)
            valid[fr] = True
    frames = np.arange(n, dtype=np.float64) - float(offset)
    return BallTrack(frames, uv, valid.astype(np.float64), valid, float(fps))


def _calls_to_payload(calls, fps: float) -> dict:
    """Convert the pipeline's list[LineCall] into the Page-3 schema (same shape
    the mock emits): bounces + a top-view trajectory + a summary."""
    fps = fps if fps and fps > 1 else 60.0
    bounces = []
    for c in calls:
        x, y = float(c.xy_ft[0]), float(c.xy_ft[1])
        in_roi = _in_roi(x, y)
        conf = _CONF_SCORE.get(c.confidence, 0.7)
        bounces.append({
            "frame": float(c.frame), "t": round(float(c.frame) / fps, 4),
            "x": round(x, 3), "y": round(y, 3),
            "in_roi": in_roi,
            # Only ROI bounces get an official verdict (per the ROI rule).
            "verdict": c.verdict if in_roi else "N/A",
            "margin_ft": round(float(c.distance_cm) / FT_TO_CM, 3) if in_roi else None,
            "nearest_line": c.nearest_line if in_roi else None,
            "confidence": round(conf, 3),
            "confidence_label": c.confidence,           # "high-agreement" | ...
            "resolved_by": c.resolved_by,               # which camera won a dispute
            "separation_ft": round(float(c.audit.get("separation_ft", 0.0)), 3),
        })
    bounces.sort(key=lambda b: b["frame"])
    trajectory = _trajectory_from_bounces(bounces, fps)

    considered = [b for b in bounces if b["in_roi"]]
    summary = {
        "total_bounces": len(bounces),
        "in_roi_bounces": len(considered),
        "in": sum(1 for b in considered if b["verdict"] == "IN"),
        "out": sum(1 for b in considered if b["verdict"] == "OUT"),
        "avg_confidence": round(float(np.mean([b["confidence"] for b in bounces])), 3)
                          if bounces else 0.0,
    }
    return {"source": "gridtracknet", "trajectory": trajectory,
            "bounces": bounces, "summary": summary}


def _trajectory_from_bounces(bounces: list, fps: float) -> list:
    """Top-view path = the ball's ground shadow: straight segments between
    consecutive fused bounce points (projectile motion projects to a line on the
    Z=0 court plane). Driven entirely by the real detected bounces."""
    traj = []
    for i in range(len(bounces) - 1):
        b0, b1 = bounces[i], bounces[i + 1]
        f0, f1 = int(round(b0["frame"])), int(round(b1["frame"]))
        n = max(1, f1 - f0)
        for f in range(f0, f1):
            u = (f - f0) / n
            x = b0["x"] + u * (b1["x"] - b0["x"])
            y = b0["y"] + u * (b1["y"] - b0["y"])
            traj.append({"frame": f, "t": round(f / fps, 4),
                         "x": round(float(x), 3), "y": round(float(y), 3),
                         "in_roi": _in_roi(x, y)})
    if bounces:                                         # include the last contact
        last = bounces[-1]
        lf = int(round(last["frame"]))
        traj.append({"frame": lf, "t": round(lf / fps, 4),
                     "x": last["x"], "y": last["y"], "in_roi": last["in_roi"]})
    return traj
