"""Clip synchronization service.

Wraps ``pickleball_phase2.sync.frame_offset`` (clap detection + xcorr refine)
and layers manual override on top. CAM1 is the reference (offset 0); CAM2/CAM3
are aligned to it. Convention (from sync.py): ``aligned_frame = frame - offset``.

Clap sync decodes audio via ffmpeg; when that's unavailable or the clap is too
noisy, the service degrades gracefully to ``status="manual_required"`` instead
of raising, so the UI can prompt for manual offsets.
"""

from __future__ import annotations

from pathlib import Path

from .. import bootstrap  # noqa: F401
from pickleball_phase2.sync import frame_offset

REFERENCE = "cam1"


def auto_sync(paths: dict[str, Path], fps: float, sync_cfg: dict) -> dict:
    """Compute per-camera frame offsets vs CAM1.

    paths: {"cam1": ..., "cam2": ..., "cam3": ...} (missing cams are skipped).
    Returns {"status": "ok"|"partial"|"manual_required",
             "offsets": {cam: {"offset_frames": float, "method": str}},
             "reference": "cam1", "message": str}
    """
    ref = paths.get(REFERENCE)
    if ref is None:
        return _manual_required("upload CAM1 (the sync reference) first", {})

    offsets: dict[str, dict] = {REFERENCE: {"offset_frames": 0.0, "method": "reference"}}
    failures: list[str] = []
    for cam in ("cam2", "cam3"):
        p = paths.get(cam)
        if p is None:
            continue
        try:
            off = float(frame_offset(ref, p, fps=fps, cfg=sync_cfg))
            offsets[cam] = {"offset_frames": round(off, 3), "method": "clap"}
        except Exception as exc:                       # ffmpeg/no-audio/noisy clap
            offsets[cam] = {"offset_frames": 0.0, "method": "manual"}
            failures.append(f"{cam}: {exc}")

    if failures and len(failures) == len([c for c in ("cam2", "cam3") if c in paths]):
        return {"status": "manual_required", "offsets": offsets, "reference": REFERENCE,
                "message": "Automatic clap sync unavailable - set offsets manually. "
                           + " | ".join(failures)}
    status = "partial" if failures else "ok"
    return {"status": status, "offsets": offsets, "reference": REFERENCE,
            "message": "Auto-synced from the clap cue." if not failures
            else "Some cameras need manual sync: " + " | ".join(failures)}


def set_manual_offset(state_sync: dict, cam: str, offset_frames: float) -> dict:
    """Apply a user override to the stored sync block and return the new block."""
    offsets = dict(state_sync.get("offsets", {}))
    offsets[cam] = {"offset_frames": round(float(offset_frames), 3), "method": "manual"}
    return {**state_sync, "offsets": offsets, "reference": REFERENCE}


def _manual_required(message: str, offsets: dict) -> dict:
    return {"status": "manual_required", "offsets": offsets,
            "reference": REFERENCE, "message": message}
