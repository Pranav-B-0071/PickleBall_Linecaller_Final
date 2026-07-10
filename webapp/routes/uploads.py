"""Session lifecycle, uploads, and media serving."""

from __future__ import annotations

import os

from flask import Blueprint, request, send_file

from ..bootstrap import REPO_ROOT
from ..services import video_utils
from ..services.storage import VIDEO_ROLES
from ._helpers import cfg, err, ok, store, with_session

bp = Blueprint("uploads", __name__)

FOOTAGE = REPO_ROOT / "Footage"
_TRACKED_ROLES = {"cam1", "cam2", "cam3"}


@bp.post("/api/session/new")
def session_new():
    sid = store().new_session()
    return ok(session_id=sid)


@bp.get("/api/session/state")
@with_session
def session_state(sid, st):
    return ok(session_id=sid, state=st.load_state(sid))


@bp.post("/api/upload")
@with_session
def upload(sid, st):
    role = request.form.get("role", "")
    file = request.files.get("file")
    if file is None or not file.filename:
        return err("no file in request")

    dest = st.save_upload(sid, role, file)
    payload = {"role": role, "url": f"/media/{sid}/video/{role}"}
    if role in VIDEO_ROLES:
        # Normalize VFR / non-60fps clips to constant 60fps IN PLACE (same
        # filename, so persisted state stays valid) before anything indexes
        # frames. Non-fatal: on failure keep the original and warn.
        tmp = dest.with_name(f"{dest.stem}_cfr{dest.suffix}")
        try:
            timing = video_utils.timing_check(dest)
            if not timing["is_cfr"]:
                video_utils.normalize_to_cfr(dest, tmp)
                os.replace(tmp, dest)
                payload["normalized"] = True
                payload["original_fps"] = timing["real_fps"]
        except Exception as exc:  # ffmpeg absent/failed, unreadable timing, etc.
            tmp.unlink(missing_ok=True)  # drop any partial re-encode
            payload["warning"] = f"could not normalize {role} to 60fps: {exc}"

        meta = video_utils.probe(dest)
        try:
            video_utils.save_first_frame(dest, st.frame_path(sid, role, 0))
            payload["frame_url"] = f"/media/{sid}/frame/{role}?frame=0"
        except Exception:  # a corrupt clip still uploads; frame is best-effort
            payload["frame_url"] = None
        payload["meta"] = meta.as_dict()
        st.update_state(sid, {"meta": {role: meta.as_dict()}})  # for frame-step bounds on reload
    return ok(**payload)


@bp.get("/media/<sid>/video/<role>")
def media_video(sid: str, role: str):
    path = store().path_for(sid, role)
    if path is None:
        return err("not found", 404)
    # conditional=True enables HTTP range requests -> <video> seeking works.
    return send_file(path, conditional=True)


@bp.get("/media/<sid>/tracked/<role>")
def media_tracked(sid: str, role: str):
    # The annotated (yellow-trail) clip written by Calculate to Footage/. Shared
    # across sessions by role, so sid is only for a uniform URL shape.
    if role not in _TRACKED_ROLES:
        return err("not found", 404)
    path = FOOTAGE / f"{role}_tracked.mp4"
    if not path.exists():
        return err("not found - run Calculate first", 404)
    return send_file(path, conditional=True)  # range requests -> <video> seeking


@bp.get("/media/<sid>/frame/<role>")
def media_frame(sid: str, role: str):
    # ?frame=N extracts (and caches) that frame on demand. The camera is static,
    # so this lets the user scrub to a frame where all 12 court points are
    # unoccluded before detecting / hand-placing keypoints.
    try:
        frame = max(0, int(request.args.get("frame", 0)))
    except (TypeError, ValueError):
        frame = 0
    st = store()
    path = st.frame_path(sid, role, frame)
    if not path.exists():
        video = st.path_for(sid, role)
        if video is None:
            return err("not found", 404)
        try:
            video_utils.save_first_frame(video, path, frame_index=frame)
        except Exception as exc:
            return err(f"cannot extract frame {frame}: {exc}", 404)
    return send_file(path, mimetype="image/jpeg", conditional=True)
