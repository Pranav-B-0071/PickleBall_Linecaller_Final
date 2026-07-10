"""Lightweight video helpers (OpenCV only - no ffprobe dependency).

Used to report clip metadata to the UI and to grab a still frame for the court
keypoint editor / metrics boundary tools.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def probe(video_path: str | Path) -> VideoMeta:
    """Read basic metadata. Raises FileNotFoundError if the clip won't open."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return VideoMeta(w, h, fps, n, n / fps if fps else 0.0)


def grab_frame(video_path: str | Path, frame_index: int = 0):
    """Return one BGR frame (numpy array) or raise if it can't be read."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {frame_index} of {video_path}")
        return frame
    finally:
        cap.release()


def save_first_frame(video_path: str | Path, out_path: str | Path,
                     frame_index: int = 0) -> Path:
    """Extract a still and write it as JPEG (for canvas overlays)."""
    frame = grab_frame(video_path, frame_index)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


# --------------------------------------------------------------------------
# Constant-frame-rate (CFR) normalization
# --------------------------------------------------------------------------
# Phone clips are frequently variable-frame-rate (VFR) and their container FPS
# can be wrong (observed: a 153 s clip reporting 58.46 fps / 109 s). Every
# downstream mechanism - clap->frame sync, GridTrackNet frame indices, and the
# two-camera bounce pairing (max_pair_gap_frames) - assumes a shared constant
# 60 fps clock, so VFR input drifts >=1 frame/second and breaks pairing. We
# recover the TRUE average fps from decoded timestamps and, when a clip is not
# already ~constant 60, re-encode it to CFR before anything indexes frames.


def timing_check(video_path: str | Path, expected_fps: float = 60.0,
                 tol_fps: float = 0.25, sample: int = 200) -> dict:
    """Recover a clip's true frame timing and decide if it needs normalization.

    Returns ``{container_fps, real_fps, real_duration_s, n_frames, is_cfr}``.
    ``is_cfr`` True => the clip is already ~constant ``expected_fps`` and should
    be left alone. The decision is deliberately biased toward normalizing: a
    false "needs normalize" only costs an extra encode, but a false "is_cfr"
    silently breaks frame-index sync.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        container_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # sample the leading frames' presentation timestamps (ms) for the local
        # rate and its jitter (a CFR clip has near-constant inter-frame gaps)
        times: list[float] = []
        for _ in range(sample if n_total <= 0 else min(sample, n_total)):
            ok, _ = cap.read()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        real_fps, real_duration = container_fps, 0.0
        # full-clip average is the most reliable rate: seek to the last frame
        # and read its timestamp (VFR-safe, unlike frame_count / container_fps)
        if n_total > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n_total - 1)
            cap.read()
            end_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if end_ms > 0:
                real_duration = end_ms / 1000.0
                # (n-1) intervals span the first..last frame timestamps
                real_fps = (n_total - 1) / real_duration
        if real_duration <= 0.0 and len(times) >= 2 and times[-1] > times[0]:
            real_fps = (len(times) - 1) * 1000.0 / (times[-1] - times[0])
    finally:
        cap.release()

    # jitter: largest inter-frame-gap deviation relative to the mean gap. A
    # true CFR clip stays tiny (ms rounding only); VFR clips vary a lot.
    jitter_ok = True
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    if len(gaps) >= 4:
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap > 0:
            jitter_ok = max(abs(g - mean_gap) for g in gaps) / mean_gap < 0.5

    is_cfr = abs(real_fps - expected_fps) <= tol_fps and jitter_ok
    return {"container_fps": round(container_fps, 3),
            "real_fps": round(real_fps, 3),
            "real_duration_s": round(real_duration, 3),
            "n_frames": n_total, "is_cfr": bool(is_cfr)}


def normalize_to_cfr(src: str | Path, dst: str | Path, fps: int = 60) -> None:
    """Re-encode ``src`` to constant ``fps`` at ``dst`` (audio preserved).

    Tries NVENC (GPU) first, then falls back to libx264. Raises
    ``subprocess.CalledProcessError`` / ``FileNotFoundError`` if both fail so
    the caller can keep the original upload. ffmpeg is resolved exactly like the
    sync module (bundled imageio-ffmpeg, no system install needed)."""
    from pickleball_phase2.sync import _ffmpeg_exe

    exe = _ffmpeg_exe()
    base = [exe, "-y", "-loglevel", "error", "-i", str(src), "-vf", f"fps={fps}"]
    tail = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)]
    nvenc = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
    x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
    try:
        subprocess.run(base + nvenc + tail, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(base + x264 + tail, check=True, capture_output=True)


def transcode_h264(path: str | Path) -> None:
    """Re-encode a clip to browser-playable H.264 (yuv420p) IN PLACE.

    The gridtracknet trail render writes mp4 with cv2's ``mp4v`` (MPEG-4 Part 2)
    codec, which most browsers will NOT play in a ``<video>`` element. Calculate
    calls this on each ``<role>_tracked.mp4`` so Analyze can play it directly.
    Audio is dropped (``-an``); the tracked render has none anyway. NVENC first,
    then libx264. Reuses the bundled ffmpeg (same resolver as sync)."""
    from pickleball_phase2.sync import _ffmpeg_exe

    src = Path(path)
    exe = _ffmpeg_exe()
    tmp = src.with_name(f"{src.stem}_h264{src.suffix}")
    base = [exe, "-y", "-loglevel", "error", "-i", str(src)]
    tail = ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    nvenc = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
    x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
    try:
        subprocess.run(base + nvenc + tail, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(base + x264 + tail, check=True, capture_output=True)
    os.replace(tmp, src)
