"""Offline demo precompute for the three Footage/ clips.

Runs the expensive, reusable preprocessing ONCE and caches it so every later
demo run loads instantly:

  1. AUTO-SYNC   - detect the clap in the first 15 s of each clip and align
                   Cam2/Cam3 to Cam1 (GCC-PHAT refined). Logs each clap
                   timestamp and the final frame offsets.
  2. TRACKING    - run GridTrackNet over the first ``precompute.tracking_window_s``
                   seconds (50 s) of each clip, on GPU when available, else CPU.
  3. CALIBRATION - best-effort court homography per clip (only if the KPT-1
                   keypoint model is present); otherwise skipped with a note.

Everything is keyed by the clip's content hash under ``data/cache`` (see
cache.py), so re-running this script - or the web app - with the same videos
reloads the cached results instead of recomputing.

    python demo.py                 # precompute + cache
    python demo.py --force         # ignore cache, recompute everything
"""

from __future__ import annotations

import argparse
import sys
import time
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

site_packages = REPO_ROOT.parent / "venv" / "Lib" / "site-packages"

for folder in (
    site_packages / "nvidia" / "cuda_runtime" / "bin",
    site_packages / "nvidia" / "cublas" / "bin",
    site_packages / "nvidia" / "cudnn" / "bin",
):
    if folder.exists():
        os.add_dll_directory(str(folder))

# Preload ALL cudnn DLLs (core first): ort.preload_dlls() predates the
# cuDNN 9.2x sub-libraries (engines_tensor_ir, ext) and misses them, which
# breaks the first GPU convolution. See ball_tracker_onnx.py for details.
import ctypes
_cudnn_bin = site_packages / "nvidia" / "cudnn" / "bin"
if _cudnn_bin.exists():
    for _dll in sorted(_cudnn_bin.glob("cudnn*.dll"),
                       key=lambda p: (p.name != "cudnn64_9.dll", p.name)):
        try:
            ctypes.WinDLL(str(_dll))
        except OSError:
            pass

import onnxruntime as ort
ort.preload_dlls()

CAMS = ("cam1", "cam2", "cam3")
REFERENCE = "cam1"

sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from pickleball_phase2.cache import Cache, video_signature
from pickleball_phase2.config import Config
from pickleball_phase2.sync import frame_offset
from pickleball_phase2.tracking import GridTrackNetTracker


def _log(msg: str) -> None:
    print(f"[demo] {msg}", flush=True)


def _find_clips(footage_dir: Path) -> dict[str, Path]:
    """Map cam1/cam2/cam3 -> the CamN*.mp4 in the footage folder."""
    clips: dict[str, Path] = {}
    for cam in CAMS:
        n = cam[-1]
        matches = sorted(footage_dir.glob(f"Cam{n}*.mp4"))
        if matches:
            clips[cam] = matches[0]
    return clips


def _pick_provider(requested: str) -> str:
    """Use GPU when onnxruntime-gpu exposes CUDA; else fall back to CPU."""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "gpu"
    except Exception:
        pass
    if requested == "gpu":
        _log("GPU requested but CUDAExecutionProvider unavailable -> using CPU")
    return "cpu"


# --------------------------------------------------------------------------
# 1. Auto-sync
# --------------------------------------------------------------------------
def run_sync(clips: dict[str, Path], cfg: Config, cache: Cache, force: bool) -> dict:
    fps = float(cfg.get("capture.fps", 60))
    sync_cfg = cfg.get("sync", {})
    ref = clips[REFERENCE]
    offsets = {REFERENCE: {"offset_frames": 0.0, "clap_s": None, "method": "reference"}}
    _log(f"AUTO-SYNC (reference = {REFERENCE}, clap window = "
         f"{sync_cfg.get('clap_search_window_s', 15.0)}s)")

    for cam in ("cam2", "cam3"):
        if cam not in clips:
            continue
        cached = None if force else cache.load_json("sync", clips[cam], sync_cfg)
        if cached is not None:
            offsets[cam] = cached
            _log(f"  {cam}: CACHED  offset={cached['offset_frames']:.3f} frames "
                 f"(clap {cached.get('clap_b_s')}s vs cam1 {cached.get('clap_a_s')}s)")
            continue
        t0 = time.time()
        _, det = frame_offset(ref, clips[cam], fps=fps, cfg=sync_cfg, return_details=True)
        rec = {"offset_frames": det["offset_frames"], "clap_a_s": det["clap_a_s"],
               "clap_b_s": det["clap_b_s"], "coarse_offset_s": det["coarse_offset_s"],
               "offset_s": det["offset_s"], "method": det["method"]}
        cache.save_json("sync", clips[cam], rec, sync_cfg)
        offsets[cam] = rec
        _log(f"  {cam}: computed in {time.time()-t0:.1f}s  "
             f"clap={det['clap_b_s']}s (cam1 clap={det['clap_a_s']}s)  "
             f"offset={det['offset_frames']:.3f} frames ({det['offset_s']}s)")

    _log("  final offsets (aligned = frame - offset): "
         + ", ".join(f"{c}={o['offset_frames']:.3f}" for c, o in offsets.items()))
    return offsets


# --------------------------------------------------------------------------
# 2. GridTrackNet precompute (first 50 s)
# --------------------------------------------------------------------------
def run_tracking(clips: dict[str, Path], cfg: Config, cache: Cache,
                 force: bool) -> dict:
    fps = float(cfg.get("capture.fps", 60))
    window_s = float(cfg.get("precompute.tracking_window_s", 50.0))
    max_frames = int(window_s * fps)
    weights = REPO_ROOT / cfg.get("paths.tracker_weights", "models/model_weights.onnx")
    provider = _pick_provider(cfg.get("tracking.provider", "cpu"))
    _log(f"TRACKING (GridTrackNet, first {window_s:.0f}s = {max_frames} frames, "
         f"provider={provider})")
    if not weights.exists():
        _log(f"  tracker weights missing at {weights} - skipping tracking")
        return {}

    results = {}
    for cam in CAMS:
        if cam not in clips:
            continue
        tracker = GridTrackNetTracker(
            str(weights), provider=provider,
            frame_mode=cfg.get("tracking.frame_mode", "dual"),
            reject_outliers=bool(cfg.get("tracking.reject_outliers", True)),
            outlier_max_jump_px=cfg.get("tracking.outlier_max_jump_px", 60.0),
            max_frames=max_frames,
            cache=None if force else cache)
        was_cached = (not force) and cache.load_npz(
            "tracking", clips[cam], tracker._cache_params()) is not None
        t0 = time.time()
        track = tracker.track_video(clips[cam])
        n_valid = int(track.valid.sum())
        tag = "CACHED" if was_cached else f"computed in {time.time()-t0:.1f}s"
        _log(f"  {cam}: {tag}  {len(track.frames)} frames, {n_valid} ball detections")
        if force:  # force path bypasses the tracker's own save; persist here
            capped = 1 if len(track.frames) >= max_frames else 0
            cache.save_npz("tracking", clips[cam], {
                "frames": track.frames, "uv": track.uv, "conf": track.conf,
                "valid": track.valid, "fps": np.array([track.fps]),
                "capped": np.array([capped], dtype=np.int64)},
                tracker._cache_params())
        results[cam] = n_valid
    return results


# --------------------------------------------------------------------------
# 3. Calibration (best-effort; needs the KPT-1 keypoint model)
# --------------------------------------------------------------------------
def run_calibration(clips: dict[str, Path], cfg: Config, cache: Cache,
                    force: bool) -> None:
    weights = REPO_ROOT / cfg.get("paths.keypoint_weights", "")
    _log("CALIBRATION (court homography)")
    if not weights.exists():
        _log(f"  keypoint model not found at {weights} - skipping "
             "(handled live by the web app / KPT-1)")
        return
    from pickleball_phase2.pipeline import calibrate_camera_from_clip
    for cam in CAMS:
        if cam not in clips:
            continue
        cached = None if force else cache.load_json("calibration", clips[cam])
        if cached is not None:
            _log(f"  {cam}: CACHED (reproj {cached.get('mean_reproj_px')}px)")
            continue
        try:
            calib = calibrate_camera_from_clip(clips[cam], cfg)
            rec = {"H": calib.H.tolist(), "H_inv": calib.H_inv.tolist(),
                   "mean_reproj_px": float(calib.mean_reproj_px),
                   "used_indices": list(calib.used_indices)}
            cache.save_json("calibration", clips[cam], rec)
            _log(f"  {cam}: computed (reproj {rec['mean_reproj_px']:.2f}px)")
        except Exception as exc:
            _log(f"  {cam}: calibration failed ({exc})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute + cache the demo footage")
    ap.add_argument("--force", action="store_true", help="ignore cache, recompute")
    args = ap.parse_args()

    cfg = Config.load(REPO_ROOT / "config.yaml")
    footage = REPO_ROOT / cfg.get("precompute.footage_dir", "Footage")
    cache = Cache(REPO_ROOT / cfg.get("cache.dir", "data/cache"))
    _log(f"cache root: {cache.root}")

    clips = _find_clips(footage)
    if REFERENCE not in clips:
        _log(f"no Cam1 clip in {footage} - cannot sync; aborting")
        sys.exit(1)
    for cam, p in clips.items():
        _log(f"{cam}: {p.name}  sig={video_signature(p)}")

    run_sync(clips, cfg, cache, args.force)
    run_tracking(clips, cfg, cache, args.force)
    run_calibration(clips, cfg, cache, args.force)
    _log("done. Re-run (without --force) to load everything from cache.")


if __name__ == "__main__":
    main()
