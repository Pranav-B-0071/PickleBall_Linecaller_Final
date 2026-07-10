"""Ball detection & tracking - §5.5 (GridTrackNet wrapper).

TRK-1 is FILLED (July 2026): inference runs through the exported ONNX model
(ball_tracker_onnx.py, vendored from the ball_tracker_handoff bundle) - no
TensorFlow and no GridTrackNet source needed at runtime. The weights are the
TENNIS-trained baseline; the pickleball fine-tune is still PLACEHOLDER[TRK-2]
(Cookbook Recipe 5 Part B - that's where the original repo + TF come back in).
`MockTracker` lets the rest of the pipeline run without a model (tests).

GridTrackNet facts (verified from the repo, MIT license):
  - 5 input frames -> 5 output frames, input 768x432, output grid 48x27
  - trained at ~30 fps motion spacing: for 60 fps capture feed even/odd
    frames as two interleaved 30 fps streams ("dual" mode, recommended)
  - (0, 0) output means "no ball"; coords come back in source resolution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BallTrack:
    """Per-camera ball track. Arrays share length N (one entry per frame)."""

    frames: np.ndarray          # (N,) int frame indices (aligned timeline)
    uv: np.ndarray              # (N, 2) float pixel positions
    conf: np.ndarray            # (N,) float detection confidence [0, 1]
    valid: np.ndarray           # (N,) bool - False where no ball was found
    fps: float = 60.0
    meta: dict = field(default_factory=dict)

    def valid_only(self) -> "BallTrack":
        m = self.valid
        return BallTrack(self.frames[m], self.uv[m], self.conf[m],
                         np.ones(int(m.sum()), dtype=bool), self.fps, self.meta)


class BaseTracker:
    def track_video(self, video_path: str | Path, frame_offset: float = 0.0) -> BallTrack:
        raise NotImplementedError


class GridTrackNetTracker(BaseTracker):
    """GridTrackNet inference via ONNX Runtime (TRK-1 - filled).

    Wraps the vendored `ball_tracker_onnx.BallTracker`. Weights are currently
    the TENNIS baseline (models/model_weights.onnx); PLACEHOLDER[TRK-2] =
    fine-tune on labelled pickleball frames and re-export (Recipe 5 Part B).

    frame_mode (60 fps capture, §5.1):
      "dual"       even + odd frames processed as two 30 fps streams and
                   re-interleaved - matches the model's trained motion
                   spacing (recommended, the handoff tool's default).
      "sequential" every frame fed consecutively - the model sees half its
                   trained motion (out-of-distribution); for 30 fps sources.

    Note: the exported model returns only (x, y); grid confidences are not
    exposed, so conf is 1.0 for detections (min_confidence is kept for when
    TRK-2 re-exports with sigmoid scores).
    """

    def __init__(self, weights_path: str | Path, provider: str = "cpu",
                 min_confidence: float = 0.30, frame_mode: str = "dual",
                 reject_outliers: bool = True,
                 outlier_max_jump_px: float = 60.0,
                 max_frames: int | None = None,
                 cache=None):
        if frame_mode not in ("dual", "sequential"):
            raise ValueError(f"frame_mode must be 'dual' or 'sequential', got {frame_mode!r}")
        self.weights_path = Path(weights_path)
        self.provider = provider
        self.min_confidence = min_confidence
        self.frame_mode = frame_mode
        self.reject_outliers = reject_outliers
        self.outlier_max_jump_px = outlier_max_jump_px
        # cap the number of frames read from the clip (None = whole video). Lets
        # the web app bound CPU inference to an analysis window, since GridTrackNet
        # runs at only ~10 fps on CPU (a full 60 fps clip would take minutes).
        self.max_frames = max_frames
        # optional pickleball_phase2.cache.Cache: persists raw inference per clip
        # so a re-run (same video + params) loads instantly instead of decoding
        # and running the model again.
        self.cache = cache
        self._impl = None            # lazy: onnxruntime imported on first use

    def _cache_params(self) -> dict:
        # Deliberately excludes max_frames: a longer precomputed track (e.g. the
        # demo's 50 s) can serve any shorter request by slicing a prefix, so one
        # cache entry per (clip, model, mode) is reused across window sizes.
        return {"weights": self.weights_path.name, "frame_mode": self.frame_mode,
                "reject_outliers": self.reject_outliers,
                "outlier_max_jump_px": self.outlier_max_jump_px}

    def _track_from_arrays(self, z: dict, frame_offset: float) -> "BallTrack" or None:
        frames = np.asarray(z["frames"], dtype=np.float64)
        n = len(frames)
        capped = bool(np.asarray(z["capped"]).ravel()[0]) if "capped" in z else False
        req = self.max_frames
        if req is None:
            if capped:
                return None            # cached prefix can't satisfy a whole-video request
            k = n
        elif n >= req:
            k = req                    # slice the prefix we need
        elif not capped:
            k = n                      # cache holds the whole (shorter) clip
        else:
            return None                # need more frames than the truncated cache has
        return BallTrack(
            frames=frames[:k] + frame_offset,
            uv=np.asarray(z["uv"], dtype=np.float64)[:k],
            conf=np.asarray(z["conf"], dtype=np.float64)[:k],
            valid=np.asarray(z["valid"], dtype=bool)[:k],
            fps=float(np.asarray(z["fps"]).ravel()[0]),
            meta={"cached": True})

    def _tracker(self):
        if self._impl is None:
            from .ball_tracker_onnx import BallTracker   # needs onnxruntime
            self._impl = BallTracker(str(self.weights_path), provider=self.provider)
        return self._impl

    def track_video(self, video_path: str | Path, frame_offset: float = 0.0) -> BallTrack:
        # Cache hit: reload the raw (offset=0) track and re-apply frame_offset.
        if self.cache is not None:
            z = self.cache.load_npz("tracking", video_path, self._cache_params())
            if z is not None:
                hit = self._track_from_arrays(z, frame_offset)
                if hit is not None:
                    return hit

        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        tracker = self._tracker()

        # chunk = one model pass; dual mode needs 10 consecutive frames
        # (5 even + 5 odd), sequential needs 5. Trailing remainders are
        # dropped, mirroring the handoff's predict() contract.
        chunk = 10 if self.frame_mode == "dual" else 5
        coords: list[tuple[int, int]] = []
        buf: list[np.ndarray] = []
        read = 0
        while self.max_frames is None or read < self.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(frame)
            read += 1
            if len(buf) == chunk:
                if self.frame_mode == "dual":
                    # evens then odds -> 2 instances of 5; re-interleave output
                    out = tracker.predict(buf[0::2] + buf[1::2])
                    inter = [None] * chunk
                    inter[0::2], inter[1::2] = out[:5], out[5:]
                    coords.extend(inter)
                else:
                    coords.extend(tracker.predict(buf))
                buf = []
        cap.release()

        n = len(coords)
        uv = np.array(coords, dtype=np.float64).reshape(n, 2)
        valid = ~np.all(uv == 0.0, axis=1)
        track = BallTrack(
            frames=np.arange(n, dtype=np.float64) + frame_offset,
            uv=uv,
            conf=np.where(valid, 1.0, 0.0),
            valid=valid,
            fps=float(fps),
            meta={"weights": str(self.weights_path), "frame_mode": self.frame_mode,
                  "provider": self.provider, "source": str(video_path)})
        if self.reject_outliers:
            track = drop_track_outliers(track, self.outlier_max_jump_px)
        # Persist the offset=0 track so a re-run loads it instantly. capped =
        # inference stopped at max_frames (more video remains), so this entry is
        # a prefix and cannot serve a whole-video request.
        if self.cache is not None:
            capped = self.max_frames is not None and read >= self.max_frames
            self.cache.save_npz("tracking", video_path, {
                "frames": track.frames - frame_offset, "uv": track.uv,
                "conf": track.conf, "valid": track.valid,
                "fps": np.array([track.fps], dtype=np.float64),
                "capped": np.array([1 if capped else 0], dtype=np.int64),
            }, self._cache_params())
        return track


class MockTracker(BaseTracker):
    """Injectable fake for tests/demos: returns a pre-computed track."""

    def __init__(self, track: BallTrack):
        self._track = track

    def track_video(self, video_path: str | Path, frame_offset: float = 0.0) -> BallTrack:
        t = self._track
        return BallTrack(t.frames + frame_offset, t.uv, t.conf, t.valid, t.fps, t.meta)


def drop_track_outliers(track: BallTrack, max_jump_px: float = 60.0) -> BallTrack:
    """Invalidate 'teleport' detections (tracker ID-switches onto a player or
    background) using the handoff's velocity-outlier rule: a detection that
    jumps > max_jump_px from BOTH neighbouring detections AND sharply
    reverses direction is dropped. Real fast rallies are preserved (the
    reversal test), and only `valid` is touched - uv/conf stay aligned."""
    from .single_view_bounce import reject_velocity_outliers
    coords = [tuple(xy) if v else (0.0, 0.0)
              for xy, v in zip(track.uv, track.valid)]
    cleaned = reject_velocity_outliers(coords, max_jump_px)
    still = np.array([not (c[0] == 0.0 and c[1] == 0.0) for c in cleaned])
    return BallTrack(track.frames, track.uv, track.conf,
                     track.valid & still, track.fps, track.meta)


def smooth_track(track: BallTrack, window: int = 5) -> BallTrack:
    """Light moving-average smoothing over valid detections (stand-in for the
    Kalman filter of the derivation doc, step 2 - upgrade freely)."""
    t = track.valid_only()
    if len(t.frames) < window:
        return t
    k = np.ones(window) / window
    uv = np.stack([np.convolve(t.uv[:, 0], k, mode="same"),
                   np.convolve(t.uv[:, 1], k, mode="same")], axis=1)
    return BallTrack(t.frames, uv, t.conf, t.valid, t.fps, t.meta)
