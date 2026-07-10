"""Page 1 API: clap sync (+ manual override) and court calibration."""

from __future__ import annotations

from flask import Blueprint, request

from ..bootstrap import load_project_config
from ..services import calibration_service, court_detection, roi as roi_service
from ..services import sync_service, video_utils
from ._helpers import cfg, err, ok, with_session

bp = Blueprint("calibration", __name__)

CALIBRATABLE = ("cam1", "cam2")   # CAM3 needs no court calibration (per spec)


@bp.post("/api/sync")
@with_session
def sync(sid, st):
    proj = load_project_config()
    paths = {cam: st.path_for(sid, cam) for cam in ("cam1", "cam2", "cam3")}
    paths = {k: v for k, v in paths.items() if v is not None}
    if "cam1" not in paths:
        return err("upload CAM1 first (it is the sync reference)")

    fps = _fps_of(paths["cam1"], proj)
    result = sync_service.auto_sync(paths, fps=fps, sync_cfg=proj.get("sync", {}))
    st.update_state(sid, {"sync": result})
    return ok(sync=result)


@bp.post("/api/sync/manual")
@with_session
def sync_manual(sid, st):
    body = request.get_json(silent=True) or {}
    cam, offset = body.get("cam"), body.get("offset_frames")
    if cam not in ("cam1", "cam2", "cam3") or offset is None:
        return err("need {cam, offset_frames}")
    new_sync = sync_service.set_manual_offset(st.load_state(sid).get("sync", {}),
                                              cam, offset)
    st.update_state(sid, {"sync": new_sync})
    return ok(sync=new_sync)


@bp.post("/api/calibration/detect")
@with_session
def detect(sid, st):
    body = request.get_json(silent=True) or {}
    cam = body.get("cam")
    if cam not in CALIBRATABLE:
        return err(f"court detection runs on {CALIBRATABLE} only (CAM3 is skipped)")
    path = st.path_for(sid, cam)
    if path is None:
        return err(f"upload {cam} first", 404)

    frame_index = max(0, int(body.get("frame", 0) or 0))   # scrubbed calibration frame
    weights = cfg().court_model_weights

    # Preferred path: the browser captured the frame client-side and sent it as
    # a data URL. We decode that IMAGE (cv2.imdecode always works) and run the
    # model on it -> no OpenCV *video* decode, which fails for some phone codecs.
    image = body.get("image")
    if image:
        frame = _decode_data_url(image)
        if frame is None:
            return err("could not decode the submitted frame image", 400)
        det = court_detection.detect(frame, weights)
    elif weights.exists():
        det = court_detection.detect(video_utils.grab_frame(path, frame_index), weights)
    else:
        w = int(body.get("width") or 0)
        h = int(body.get("height") or 0)
        if not (w and h):
            meta = video_utils.probe(path)
            w, h = meta.width, meta.height
        det = court_detection.detect_mock(w, h)
    return ok(cam=cam, frame=frame_index, **det.as_dict())


@bp.post("/api/calibration/confirm")
@with_session
def confirm(sid, st):
    proj = load_project_config()
    body = request.get_json(silent=True) or {}
    cam, keypoints = body.get("cam"), body.get("keypoints")
    if cam not in CALIBRATABLE or not keypoints:
        return err("need {cam in (cam1, cam2), keypoints:[[u,v,vis]*12]}")

    ransac = proj.get("calibration.ransac_thresh_px", 5.0)
    min_inl = proj.get("calibration.min_inliers", 6)
    try:
        calib = calibration_service.calibrate(keypoints, ransac_thresh_px=ransac,
                                              min_inliers=min_inl)
    except (ValueError, RuntimeError) as exc:
        return err(f"calibration failed: {exc}", 422)

    # PnP sanity check: recover the camera position from these keypoints + the
    # saved intrinsics and compare to the tape-measured camera_positions_ft.
    calib["pnp"] = calibration_service.camera_pose_check(cam, keypoints, proj,
                                                         ransac, min_inl)
    st.update_state(sid, {"calibration": {cam: calib}})
    return ok(cam=cam, calibration=calib)


@bp.post("/api/calibration/kitchen")
@with_session
def kitchen(sid, st):
    """CAM3 manual 4-point kitchen box (no model)."""
    body = request.get_json(silent=True) or {}
    points = body.get("points")
    if not points:
        return err("need {points:[[u,v]*4]}")
    try:
        region = calibration_service.store_kitchen_region(points)
    except ValueError as exc:
        return err(f"invalid kitchen box: {exc}", 422)
    st.update_state(sid, {"calibration": {"cam3": region}})
    return ok(cam="cam3", calibration=region)


@bp.get("/api/calibration/roi")
@with_session
def roi(sid, st):
    cam = request.args.get("cam", "cam1")
    calib = st.load_state(sid).get("calibration", {}).get(cam)
    if calib and "roi" in calib:
        return ok(cam=cam, roi=calib["roi"])
    if calib and "keypoints" in calib:
        return ok(cam=cam, roi=roi_service.derive_half_court_roi(calib["keypoints"]))
    return err(f"{cam} is not calibrated yet", 404)


def _fps_of(path, proj) -> float:
    try:
        fps = video_utils.probe(path).fps
    except Exception:
        fps = 0.0
    return fps or float(proj.get("capture.fps", 60))


def _decode_data_url(data_url: str):
    """Decode a base64 image data URL to a BGR frame (image codec, not video)."""
    import base64

    import cv2
    import numpy as np

    try:
        _, _, b64 = data_url.partition(",")
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
