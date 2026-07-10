"""End-to-end offline pipeline: one synchronized clip pair -> line calls.

This is the exact 5-step story:
  1. sync         - clap alignment (sync.py)
  2. find ball    - per-camera tracker (tracking.py)
  3. project      - pixels -> court feet via each homography (bounce.py)
  4. watch gap    - s(t) minima = bounce candidates (bounce.py)
  5. call it      - fuse, arbitrate, compare to lines (fusion.py)

Runs with the ONNX GridTrackNet tracker (TRK-1 filled; tennis weights until
TRK-2) - still needs KPT-1 (keypoint model) for calibration from real clips.
Every part stays injectable (MockTracker etc.) for tests and partial states.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bounce import detect_bounces
from .calibration import (CourtCalibration, detect_court_keypoints,
                          fit_court_homography, lock_on)
from .config import Config
from .fusion import LineCall, fuse_event
from .sync import frame_offset
from .tracking import BaseTracker, GridTrackNetTracker


def calibrate_camera_from_clip(video_path: str | Path, cfg: Config,
                               sample_stride: int = 30) -> CourtCalibration:
    """§4.7 lock-on over a clip: detect keypoints on sampled frames, keep the
    lowest-error homography. Requires KPT-1 (real keypoint model)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video_path}")
    weights = cfg.get("paths.keypoint_weights")
    frames_kpts, idx = [], 0
    max_frames = int(cfg.get("calibration.lock_window_s", 10.0)
                     * cfg.get("capture.fps", 60))
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_stride == 0:
            frames_kpts.append(detect_court_keypoints(frame, weights))  # KPT-1
        idx += 1
    cap.release()
    _, calib = lock_on(frames_kpts,
                       cfg.get("calibration.ransac_thresh_px", 5.0),
                       cfg.get("calibration.min_inliers", 6),
                       cfg.get("calibration.lock_threshold_px", 8.0))
    return calib


def run_clip_pair(
    video_a: str | Path, video_b: str | Path,
    cfg: Config,
    tracker_a: BaseTracker | None = None,
    tracker_b: BaseTracker | None = None,
    calib_a: CourtCalibration | None = None,
    calib_b: CourtCalibration | None = None,
    offset_frames: float | None = None,
) -> list[LineCall]:
    """The main entry point. Trackers/calibrations are injectable so tests
    (and partially-completed placeholder states) can run the full chain.

    ``offset_frames`` overrides step 1's clap detection with a known B-minus-A
    offset (aligned = frame_B - offset_frames); the web app passes the offset the
    user set/confirmed on the calibration page instead of re-running clap sync.
    """
    fps = float(cfg.get("capture.fps", 60))

    # 1. Sync (clap): fractional frame offset applied to camera B (or supplied)
    off = float(offset_frames) if offset_frames is not None \
        else frame_offset(video_a, video_b, fps=fps, cfg=cfg.get("sync", {}))

    # 2. Calibrate each camera to the shared court frame (§5.3)
    calib_a = calib_a or calibrate_camera_from_clip(video_a, cfg)   # needs KPT-1
    calib_b = calib_b or calibrate_camera_from_clip(video_b, cfg)

    # 3. Track the ball in each view (§5.5) - ONNX GridTrackNet (TRK-1)
    if tracker_a is None or tracker_b is None:
        t = GridTrackNetTracker(
            cfg.get("paths.tracker_weights"),
            provider=cfg.get("tracking.provider", "cpu"),
            min_confidence=cfg.get("tracking.min_confidence", 0.3),
            frame_mode=cfg.get("tracking.frame_mode", "dual"),
            reject_outliers=bool(cfg.get("tracking.reject_outliers", True)),
            outlier_max_jump_px=cfg.get("tracking.outlier_max_jump_px", 60.0))
        tracker_a, tracker_b = tracker_a or t, tracker_b or t
    track_a = tracker_a.track_video(video_a, frame_offset=0.0)
    track_b = tracker_b.track_video(video_b, frame_offset=-off)  # aligned = frame_B - off

    # 4. Bounces via the separation signal (§5.6) + sub-frame refinement
    events = detect_bounces(
        track_a, track_b, calib_a, calib_b,
        min_prominence_ft=cfg.get("bounce.min_prominence_ft", 0.5),
        smooth_window=cfg.get("bounce.smooth_window_frames", 5),
        subframe_refine=bool(cfg.get("bounce.subframe_refine", True)),
        max_pair_gap_frames=cfg.get("bounce.max_pair_gap_frames", 2),
        vshape_min_fall_frames=cfg.get("bounce.vshape_min_fall_frames", 3),
        min_bounce_prob=cfg.get("bounce.min_bounce_prob", 0.0),
        max_separation_ft=cfg.get("bounce.max_separation_ft", None))

    # 5. Fuse + arbitrate + line call (§5.8)
    fusion_cfg = {**cfg.get("fusion", {}), "margin_ft": cfg.get("line_call.margin_ft", 0.0)}
    zone = cfg.get("line_call.zone", "near_half")
    return [fuse_event(e, fusion_cfg, zone=zone) for e in events]


def calls_to_json(calls: list[LineCall], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(c) for c in calls], indent=2,
                                     default=lambda o: np.asarray(o).tolist()),
                          encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Clip pair -> line calls")
    ap.add_argument("video_a")
    ap.add_argument("video_b")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="calls.json")
    args = ap.parse_args()
    cfg = Config.load(args.config) if args.config else Config.load()
    result = run_clip_pair(args.video_a, args.video_b, cfg)
    calls_to_json(result, args.out)
    for c in result:
        print(f"frame {c.frame:8.2f}  ({c.xy_ft[0]:5.2f}, {c.xy_ft[1]:5.2f}) ft  "
              f"{c.verdict:3s}  {c.distance_cm:+.1f} cm from {c.nearest_line}  "
              f"[{c.confidence}]")
