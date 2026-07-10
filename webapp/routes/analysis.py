"""Page 3 API: Calculate (run GridTrackNet per clip) + Analyze (bounces->IN/OUT).

Calculate and Analyze are deliberately decoupled:
  * Calculate shells out to ``gridtracknet_test_improved_safe.py`` once per
    uploaded clip so the tqdm ``tracking``/``rendering`` bars stream to the
    terminal running ``run_web.py``; it writes ``Footage/<role>_tracked.mp4``
    (+ ``.csv``) and transcodes the mp4 to browser-playable H.264.
  * Analyze reads those tracked CSVs directly (no re-tracking) and runs the real
    dual-camera calibrated bounce pipeline to produce IN/OUT line calls.
"""

from __future__ import annotations

import subprocess
import sys

from flask import Blueprint

from ..bootstrap import REPO_ROOT, load_project_config
from ..services import analysis_service, video_utils
from ._helpers import err, ok, with_session

bp = Blueprint("analysis", __name__)

FOOTAGE = REPO_ROOT / "Footage"
TRACK_SCRIPT = REPO_ROOT / "gridtracknet_test_improved_safe.py"
_ROLES = ("cam1", "cam2", "cam3")


@bp.post("/api/analysis/calculate")
@with_session
def calculate(sid, st):
    """Run GridTrackNet on each uploaded clip (cam1/cam2/cam3), sequentially, so
    one clip's progress bars finish before the next starts. The child inherits
    this server's stdout/stderr, so the bars appear in the terminal."""
    proj = load_project_config()
    provider = str(proj.get("tracking.provider", "gpu"))
    FOOTAGE.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for role in _ROLES:
        src = st.path_for(sid, role)
        if src is None:
            skipped.append(role)
            continue
        out = FOOTAGE / f"{role}_tracked.mp4"
        try:
            # inherit stdout/stderr (no capture) -> tqdm bars stream to terminal
            subprocess.run(
                [sys.executable, str(TRACK_SCRIPT), str(src),
                 "--out", str(out), "--provider", provider],
                cwd=str(REPO_ROOT), check=True)
            video_utils.transcode_h264(out)  # cv2 mp4v -> browser-playable H.264
            done.append(role)
        except Exception as exc:
            return err(f"tracking failed for {role}: {exc}", 500)
    if not done:
        return err("no uploaded videos to calculate - upload clips on Page 1 first")
    return ok(done=done, skipped=skipped)


@bp.post("/api/analysis/run")
@with_session
def run(sid, st):
    """Analyze the pre-computed tracked CSVs: dual-camera calibrated bounce
    detection -> IN/OUT. Errors clearly if Calculate hasn't run or CAM1/CAM2
    aren't calibrated."""
    proj = load_project_config()
    fps, duration = _timebase(st, sid, proj)
    try:
        result = analysis_service.analyze_from_tracked(
            st.load_state(sid), st, sid, proj, fps, duration, FOOTAGE)
    except ValueError as exc:  # missing tracked files / calibration -> user toast
        return err(str(exc))
    # persist a compact summary (not the whole trajectory) for session state
    st.update_state(sid, {"analysis": {"summary": result["summary"],
                                       "source": result["source"],
                                       "fps": result["fps"],
                                       "duration_s": result["duration_s"]}})
    return ok(analysis=result)


def _timebase(st, sid, proj) -> tuple[float, float]:
    """Derive fps/duration from CAM1 (fallback: baseline video, then config)."""
    for role in ("cam1", "baseline_video", "cam2"):
        path = st.path_for(sid, role)
        if path is None:
            continue
        try:
            meta = video_utils.probe(path)
            if meta.fps and meta.duration_s:
                return meta.fps, meta.duration_s
        except Exception:
            continue
    return float(proj.get("capture.fps", 60)), 20.0
