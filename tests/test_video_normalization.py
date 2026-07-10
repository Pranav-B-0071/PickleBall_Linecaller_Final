"""Unit tests for upload-time CFR normalization (video_utils + upload route).

All hermetic: cv2.VideoCapture and ffmpeg/subprocess are stubbed, so nothing
decodes or re-encodes a real file. Covers (1) timing_check's CFR decision on
constant vs variable frame timing, (2) normalize_to_cfr's NVENC->x264 fallback,
and (3) the upload route's normalize-vs-skip-vs-non-fatal-failure branch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from webapp.services import video_utils   # noqa: E402


class _FakeCap:
    """Minimal cv2.VideoCapture stand-in driven by a list of frame PTS (ms)."""

    def __init__(self, times_ms, container_fps):
        self._t = list(times_ms)
        self._fps = container_fps
        self._cur = 0            # index of the next frame read() returns
        self._last = 0.0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self._t)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self._last
        return 0.0

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._cur = int(val)
        return True

    def read(self):
        if 0 <= self._cur < len(self._t):
            self._last = self._t[self._cur]
            self._cur += 1
            return True, object()
        return False, None

    def release(self):
        pass


def _patch_cap(monkeypatch, times_ms, container_fps):
    monkeypatch.setattr(video_utils.cv2, "VideoCapture",
                        lambda *_a, **_k: _FakeCap(times_ms, container_fps))


def test_timing_check_constant_60_is_cfr(monkeypatch):
    times = [i * 1000.0 / 60.0 for i in range(300)]       # exact CFR 60fps
    _patch_cap(monkeypatch, times, container_fps=60.0)
    res = video_utils.timing_check("x.mp4")
    assert res["is_cfr"] is True
    assert abs(res["real_fps"] - 60.0) < 0.05


def test_timing_check_wrong_average_needs_normalize(monkeypatch):
    # container LIES (says 58.46) but the true average is ~41.8fps (cam1 case)
    times = [i * 1000.0 / 41.8 for i in range(300)]
    _patch_cap(monkeypatch, times, container_fps=58.46)
    res = video_utils.timing_check("x.mp4")
    assert res["is_cfr"] is False
    assert 41.0 < res["real_fps"] < 43.0


def test_timing_check_variable_jitter_needs_normalize(monkeypatch):
    # average ~60fps but wildly variable inter-frame gaps -> jitter catches it
    times, t = [], 0.0
    for i in range(300):
        t += (8.0 if i % 2 else 25.0)                     # alternating gaps
        times.append(t)
    _patch_cap(monkeypatch, times, container_fps=60.0)
    res = video_utils.timing_check("x.mp4")
    assert res["is_cfr"] is False


def test_normalize_falls_back_to_x264(monkeypatch):
    monkeypatch.setattr("pickleball_phase2.sync._ffmpeg_exe", lambda: "ffmpeg")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "h264_nvenc" in cmd:                           # GPU encoder "fails"
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)
    video_utils.normalize_to_cfr("in.mp4", "out.mp4")
    assert len(calls) == 2                                # nvenc tried, then x264
    assert "h264_nvenc" in calls[0] and "libx264" in calls[1]


def test_normalize_uses_nvenc_when_it_works(monkeypatch):
    monkeypatch.setattr("pickleball_phase2.sync._ffmpeg_exe", lambda: "ffmpeg")
    calls = []
    monkeypatch.setattr(video_utils.subprocess, "run",
                        lambda cmd, **kw: (calls.append(cmd),
                                           subprocess.CompletedProcess(cmd, 0))[1])
    video_utils.normalize_to_cfr("in.mp4", "out.mp4")
    assert len(calls) == 1 and "h264_nvenc" in calls[0]   # no fallback needed


# ---- upload route wiring (Flask client, stubbed normalization) ---------

import dataclasses          # noqa: E402
import io                   # noqa: E402

import numpy as np          # noqa: E402

from webapp import create_app            # noqa: E402
from webapp.config import WebConfig      # noqa: E402


@pytest.fixture()
def client(tmp_path):
    cfg = dataclasses.replace(WebConfig.load(), data_root=tmp_path / "webapp",
                              court_model_weights=tmp_path / "no_model.pt")
    app = create_app(cfg)
    app.config.update(TESTING=True)
    return app.test_client()


def _tiny_mp4(tmp_path):
    p = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    if not vw.isOpened():
        return None
    for i in range(12):
        f = np.zeros((180, 320, 3), np.uint8)
        cv2.circle(f, (30 + 12 * i, 90), 4, (255, 255, 255), -1)
        vw.write(f)
    vw.release()
    return p.read_bytes() if p.exists() and p.stat().st_size > 0 else None


def _upload(client, sid, mp4):
    return client.post("/api/upload", headers={"X-Session-Id": sid},
                       data={"role": "cam1", "file": (io.BytesIO(mp4), "cam1.mp4")},
                       content_type="multipart/form-data").get_json()


def test_upload_normalizes_non_cfr(client, tmp_path, monkeypatch):
    mp4 = _tiny_mp4(tmp_path)
    if mp4 is None:
        pytest.skip("cv2 mp4 codec unavailable")
    sid = client.post("/api/session/new").get_json()["session_id"]

    # force the "needs normalize" branch; stub the encode to a plain copy so no
    # real ffmpeg runs but the replaced file is still a valid, probeable clip
    import webapp.routes.uploads as up
    monkeypatch.setattr(up.video_utils, "timing_check",
                        lambda *_a, **_k: {"is_cfr": False, "real_fps": 41.8})
    monkeypatch.setattr(up.video_utils, "normalize_to_cfr",
                        lambda src, dst, **_k: Path(dst).write_bytes(Path(src).read_bytes()))
    j = _upload(client, sid, mp4)
    assert j["ok"] and j.get("normalized") is True
    assert j["original_fps"] == 41.8
    assert j["meta"]["frame_count"] >= 1                  # replaced file still valid


def test_upload_skips_cfr(client, tmp_path, monkeypatch):
    mp4 = _tiny_mp4(tmp_path)
    if mp4 is None:
        pytest.skip("cv2 mp4 codec unavailable")
    sid = client.post("/api/session/new").get_json()["session_id"]
    import webapp.routes.uploads as up
    monkeypatch.setattr(up.video_utils, "timing_check",
                        lambda *_a, **_k: {"is_cfr": True, "real_fps": 60.0})
    called = []
    monkeypatch.setattr(up.video_utils, "normalize_to_cfr",
                        lambda *a, **k: called.append(a))
    j = _upload(client, sid, mp4)
    assert j["ok"] and "normalized" not in j and not called


def test_upload_normalization_failure_is_non_fatal(client, tmp_path, monkeypatch):
    mp4 = _tiny_mp4(tmp_path)
    if mp4 is None:
        pytest.skip("cv2 mp4 codec unavailable")
    sid = client.post("/api/session/new").get_json()["session_id"]
    import webapp.routes.uploads as up
    monkeypatch.setattr(up.video_utils, "timing_check",
                        lambda *_a, **_k: {"is_cfr": False, "real_fps": 41.8})

    def boom(*_a, **_k):
        raise RuntimeError("ffmpeg missing")

    monkeypatch.setattr(up.video_utils, "normalize_to_cfr", boom)
    j = _upload(client, sid, mp4)
    assert j["ok"] and "warning" in j                     # upload still succeeds
    assert j["meta"]["frame_count"] >= 1                  # original clip preserved
