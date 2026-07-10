"""Smoke tests for the Flask web app: pages render and every API endpoint
responds with the expected shape. Uses the Flask test client against a
temp-dir session store, so it never touches real user data.

Video-dependent endpoints (upload/detect/confirm/roi/sync) are skipped when the
cv2 mp4 codec isn't available; the CSV/analysis paths always run.
"""

from __future__ import annotations

import dataclasses
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))            # import webapp
sys.path.insert(0, str(REPO / "src"))    # import pickleball_phase2

from webapp import create_app                    # noqa: E402
from webapp.config import WebConfig              # noqa: E402


@pytest.fixture()
def client(tmp_path):
    # bogus court model path -> detection uses the (fast) mock path, so tests
    # don't load torch / the real .pt
    cfg = dataclasses.replace(WebConfig.load(), data_root=tmp_path / "webapp",
                              court_model_weights=tmp_path / "no_model.pt")
    app = create_app(cfg)
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture()
def sid(client):
    r = client.post("/api/session/new")
    return r.get_json()["session_id"]


def _hdr(sid):
    return {"X-Session-Id": sid}


def _tiny_mp4(tmp_path) -> bytes | None:
    import cv2

    p = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    if not vw.isOpened():
        return None
    for i in range(15):
        f = np.zeros((180, 320, 3), np.uint8)
        cv2.circle(f, (30 + 15 * i, 90), 4, (255, 255, 255), -1)
        vw.write(f)
    vw.release()
    return p.read_bytes() if p.exists() and p.stat().st_size > 0 else None


CSV = b"throw,call,t,x,y\n1,IN,0.5,10.0,40.0\n2,OUT,1.2,21.0,43.0\n3,IN,2.0,5.0,30.0\n4,OUT,3.1,-1.0,25.0\n"


# ---- pages -------------------------------------------------------------

@pytest.mark.parametrize("path,text", [
    ("/", "Court Calibration"),
    ("/metrics", "Label Metrics"),
    ("/analysis", "Analysis"),
])
def test_pages_render(client, path, text):
    r = client.get(path)
    assert r.status_code == 200
    assert text in r.get_data(as_text=True)


def test_session_lifecycle(client):
    sid = client.post("/api/session/new").get_json()["session_id"]
    assert sid
    r = client.get("/api/session/state", headers=_hdr(sid))
    j = r.get_json()
    assert j["ok"] and "uploads" in j["state"]


def test_state_requires_session(client):
    assert client.get("/api/session/state").status_code == 404


# ---- metrics + analysis (no video needed) ------------------------------

def test_metrics_compare(client, sid):
    up = client.post("/api/upload", headers=_hdr(sid),
                     data={"role": "baseline_csv", "file": (io.BytesIO(CSV), "baseline.csv")},
                     content_type="multipart/form-data")
    assert up.get_json()["ok"]
    r = client.post("/api/metrics/compare", json={"session_id": sid,
                    "boundaries": {"baseline": {"orientation": "horizontal", "position": 22, "in_side": "above"}}},
                    headers=_hdr(sid))
    j = r.get_json()
    assert j["ok"]
    assert j["panels"]["baseline"]["available"] is True
    assert j["panels"]["baseline"]["count"] == 4
    assert any(c["label"] == "Accuracy" for c in j["cards"])


def test_analysis_run(client, sid):
    r = client.post("/api/analysis/run", json={"session_id": sid}, headers=_hdr(sid))
    j = r.get_json()
    assert j["ok"]
    a = j["analysis"]
    assert a["trajectory"] and a["bounces"]
    assert set(a["roi_ft"][0]) == {0.0, 22.0}   # near-half ROI corner
    assert a["summary"]["total_bounces"] == len(a["bounces"])
    # every in-ROI bounce carries a real IN/OUT verdict
    for b in a["bounces"]:
        if b["in_roi"]:
            assert b["verdict"] in ("IN", "OUT")
    # CAM3 not calibrated in this session -> no kitchen tally
    assert a["cam3_calibrated"] is False
    assert a["summary"]["kitchen_bounces"] is None


def test_kitchen_calibration_and_bounce_count(client, sid):
    box = [[100, 200], [500, 200], [520, 420], [80, 420]]
    r = client.post("/api/calibration/kitchen", json={"session_id": sid, "points": box},
                    headers=_hdr(sid)).get_json()
    assert r["ok"] and r["calibration"]["type"] == "kitchen_box"
    assert len(r["calibration"]["region_px"]) == 4

    # a degenerate (collapsed) box is rejected
    bad = client.post("/api/calibration/kitchen",
                      json={"session_id": sid, "points": [[0, 0]] * 4}, headers=_hdr(sid))
    assert bad.status_code == 422

    # with CAM3 calibrated, analysis now tallies kitchen bounces
    a = client.post("/api/analysis/run", json={"session_id": sid},
                    headers=_hdr(sid)).get_json()["analysis"]
    assert a["cam3_calibrated"] is True
    assert set(a["kitchen_zone_ft"][3]) == {0.0, 29.0}      # kitchen line corner
    assert isinstance(a["summary"]["kitchen_bounces"], int)
    # every kitchen bounce is inside the near kitchen band (net..kitchen line)
    for b in a["bounces"]:
        if b["in_kitchen"]:
            assert 22.0 <= b["y"] <= 29.0


def test_analysis_payload_is_json_native():
    """Regression: the mock analysis must never leak numpy scalars (np.float64
    / np.bool_), which Flask's JSON provider rejects. Sweep many seeds since
    the offending values were RNG-dependent (out-of-court wobble points)."""
    from webapp.services import analysis_service

    def assert_native(o, path="root"):
        if isinstance(o, dict):
            for k, v in o.items():
                assert type(k) is str, f"{path}: non-str key {k!r}"
                assert_native(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                assert_native(v, f"{path}[{i}]")
        else:
            assert type(o) in (str, int, float, bool, type(None)), \
                f"{path}: non-native {type(o).__module__}.{type(o).__name__}"

    for seed in (str(n) for n in range(40)):
        state = {"calibration": {"cam3": {"type": "kitchen_box"}}} if int(seed) % 2 else {}
        assert_native(analysis_service.run_analysis(state, fps=60, duration_s=20, seed_key=seed))


# ---- video-dependent flow ---------------------------------------------

def test_calibration_flow(client, sid, tmp_path):
    mp4 = _tiny_mp4(tmp_path)
    if mp4 is None:
        pytest.skip("cv2 mp4 codec unavailable")
    for role in ("cam1", "cam2"):
        up = client.post("/api/upload", headers=_hdr(sid),
                         data={"role": role, "file": (io.BytesIO(mp4), f"{role}.mp4")},
                         content_type="multipart/form-data")
        assert up.get_json()["ok"], up.get_json()

    # detect on a scrubbed frame index (camera static -> any frame works)
    det = client.post("/api/calibration/detect", json={"session_id": sid, "cam": "cam1", "frame": 3},
                      headers=_hdr(sid)).get_json()
    assert det["ok"] and len(det["keypoints"]) == 12 and det["source"] == "mock"
    assert det["frame"] == 3

    conf = client.post("/api/calibration/confirm",
                       json={"session_id": sid, "cam": "cam1", "keypoints": det["keypoints"]},
                       headers=_hdr(sid)).get_json()
    assert conf["ok"]
    assert conf["calibration"]["mean_reproj_px"] < 8.0
    assert "roi" in conf["calibration"]
    # PnP sanity-check block is always present; available since intrinsics exist
    pnp = conf["calibration"]["pnp"]
    assert "available" in pnp
    if pnp["available"]:
        assert len(pnp["recovered_ft"]) == 3

    roi = client.get("/api/calibration/roi?cam=cam1", headers=_hdr(sid)).get_json()
    assert roi["ok"] and len(roi["roi"]["roi_polygon"]) == 4


def test_pnp_recovers_camera_above_court():
    """Regression: planar PnP must return the camera ABOVE the court (height>0),
    not the mirror solution (the bug that gave Z ~ -5 ft)."""
    import cv2
    from pickleball_phase2.calibration import (Intrinsics, fit_court_homography,
                                               solve_camera_pose)
    from pickleball_phase2.court_model import keypoints_3d_ft

    C = np.array([10.0, 60.0, 8.0])                 # 8 ft above, behind baseline
    K = np.array([[1400, 0, 960], [0, 1400, 540], [0, 0, 1.0]])
    fwd = np.array([10.0, 33.0, 0.0]) - C; fwd /= np.linalg.norm(fwd)
    right = np.cross([0, 0, 1.0], fwd); right /= np.linalg.norm(right)
    R = np.vstack([right, np.cross(fwd, right), fwd])
    t = -R @ C
    obj = keypoints_3d_ft()
    proj = (K @ (R @ obj.T + t[:, None])).T
    uv = proj[:, :2] / proj[:, 2:3] + np.random.default_rng(0).normal(0, 0.4, (12, 2))
    kpts = np.hstack([uv, np.full((12, 1), 2.0)])
    intr = Intrinsics(K=K, dist=np.zeros(5), image_size=(1920, 1080))

    calib = solve_camera_pose(kpts, intr, fit_court_homography(kpts))
    rec = calib.camera_pos_ft
    assert rec[2] > 0                                # camera above the court
    assert np.linalg.norm(rec - C) < 1.0            # within a foot of truth


def test_real_analysis_payload_from_line_calls():
    """The real pipeline's list[LineCall] must convert to the SAME Page-3 schema
    the mock emits: bounces (with in/out verdicts gated on the ROI), a top-view
    trajectory interpolated between bounces, and a summary - all JSON-native."""
    from webapp.services import analysis_service as A
    from pickleball_phase2.fusion import LineCall

    calls = [
        LineCall(frame=30.0, xy_ft=(10.0, 25.0), verdict="IN", distance_cm=91.4,
                 nearest_line="net_line", confidence="high-agreement",
                 resolved_by=None, audit={"separation_ft": 0.12}),
        LineCall(frame=95.0, xy_ft=(19.7, 40.0), verdict="OUT", distance_cm=-9.1,
                 nearest_line="sideline_right", confidence="two-view-resolved",
                 resolved_by="A", audit={"separation_ft": 0.31}),
    ]
    pay = A._calls_to_payload(calls, fps=60.0)
    assert pay["source"] == "gridtracknet"
    assert len(pay["bounces"]) == 2
    assert pay["summary"]["total_bounces"] == 2
    assert pay["summary"]["in"] == 1 and pay["summary"]["out"] == 1
    # trajectory spans the two bounce frames (30..95) as the ground shadow
    assert pay["trajectory"] and pay["trajectory"][0]["frame"] == 30
    assert pay["trajectory"][-1]["frame"] == 95
    # every bounce has a native margin/verdict and no numpy scalars leak
    import json
    json.dumps(pay)                     # raises if a np.float64/np.bool_ slipped in
    for b in pay["bounces"]:
        assert isinstance(b["x"], float) and b["verdict"] in ("IN", "OUT")


def test_mirror_pose_forces_camera_above_court():
    """The forced-flip twin: reflecting a below-court pose must (a) put the
    camera above with the SAME (x, y), and (b) reproject every court point to
    the identical pixel - i.e. it is the legitimate coplanar twin, not a fudge."""
    import cv2
    from pickleball_phase2.calibration import _mirror_pose
    from pickleball_phase2.court_model import keypoints_3d_ft

    C = np.array([7.0, 55.0, 9.0])                  # a valid above-court camera
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1.0]])
    fwd = np.array([10.0, 22.0, 0.0]) - C; fwd /= np.linalg.norm(fwd)
    right = np.cross([0, 0, 1.0], fwd); right /= np.linalg.norm(right)
    R = np.vstack([right, np.cross(fwd, right), fwd])
    t = (-R @ C).reshape(3, 1)
    obj = keypoints_3d_ft()

    Rm, tm = _mirror_pose(R, t)
    assert abs(np.linalg.det(Rm) - 1.0) < 1e-9      # still a proper rotation
    cam_m = (-Rm.T @ tm).ravel()
    assert np.allclose(cam_m[:2], C[:2], atol=1e-6) # x, y unchanged
    assert np.isclose(cam_m[2], -C[2])              # height sign flipped

    def px(Rx, tx):
        p = (K @ (Rx @ obj.T + tx.reshape(3, 1))).T
        return p[:, :2] / p[:, 2:3]
    assert np.allclose(px(R, t), px(Rm, tm), atol=1e-6)   # identical reprojection


def test_roi_uses_homography_when_available():
    from pickleball_phase2.calibration import fit_court_homography
    import cv2
    from pickleball_phase2.court_model import COURT_PTS_FT
    from webapp.services import roi as roi_service
    src = np.array([[0, 0], [20, 0], [20, 44], [0, 44]], dtype=np.float32)
    dst = np.array([[600, 300], [1400, 280], [1850, 1000], [150, 1020]], dtype=np.float32)
    H_c2i = cv2.getPerspectiveTransform(src, dst)
    img12 = (lambda H, p: ((H @ np.hstack([p, np.ones((12, 1))]).T).T)[:, :2]
             / ((H @ np.hstack([p, np.ones((12, 1))]).T).T)[:, 2:3])(H_c2i, COURT_PTS_FT)
    calib = fit_court_homography(np.hstack([img12, np.full((12, 1), 2.0)]))
    roi = roi_service.derive_half_court_roi(img12, homography_inv=calib.H_inv)
    assert roi["method"] == "homography"
    assert len(roi["roi_polygon"]) == 4 and "kitchen_line" in roi and "center_line" in roi


def test_frame_stepping(client, sid, tmp_path):
    mp4 = _tiny_mp4(tmp_path)
    if mp4 is None:
        pytest.skip("cv2 mp4 codec unavailable")
    client.post("/api/upload", headers=_hdr(sid),
                data={"role": "cam1", "file": (io.BytesIO(mp4), "cam1.mp4")},
                content_type="multipart/form-data")
    # frame 0 and a later frame both extract + serve as JPEG
    for frame in (0, 4):
        r = client.get(f"/media/{sid}/video/cam1")  # ensure video is stored
        assert r.status_code == 200
        f = client.get(f"/media/{sid}/frame/cam1?frame={frame}", headers=_hdr(sid))
        assert f.status_code == 200
        assert f.mimetype == "image/jpeg"
    # meta was persisted so the UI can bound the stepper on reload
    state = client.get("/api/session/state", headers=_hdr(sid)).get_json()["state"]
    assert state["meta"]["cam1"]["frame_count"] >= 1


def test_sync_endpoint(client, sid, tmp_path):
    mp4 = _tiny_mp4(tmp_path)
    if mp4 is None:
        pytest.skip("cv2 mp4 codec unavailable")
    for role in ("cam1", "cam2"):
        client.post("/api/upload", headers=_hdr(sid),
                    data={"role": role, "file": (io.BytesIO(mp4), f"{role}.mp4")},
                    content_type="multipart/form-data")
    r = client.post("/api/sync", json={"session_id": sid}, headers=_hdr(sid)).get_json()
    assert r["ok"]
    # ffmpeg may be absent -> manual_required; either way CAM1 is the reference.
    assert r["sync"]["reference"] == "cam1"
    assert r["sync"]["status"] in ("ok", "partial", "manual_required")
