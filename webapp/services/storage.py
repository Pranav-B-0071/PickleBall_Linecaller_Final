"""Session storage: uploads + JSON state on disk.

One folder per session under ``data/webapp/<session_id>/``:

    cam1.mp4  cam2.mp4  cam3.mp4                 (Page 1 uploads)
    baseline_video.mp4  sideline_video.mp4       (Page 2 uploads)
    baseline_csv.csv    sideline_csv.csv
    frames/<role>.jpg                            (extracted first frames)
    state.json                                   (sync/calibration/metrics/analysis)

Roles are a fixed vocabulary so paths are predictable and safe (no user-supplied
filenames touch the filesystem - only the extension is preserved).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage

from ..config import WebConfig

VIDEO_ROLES = ("cam1", "cam2", "cam3", "baseline_video", "sideline_video")
CSV_ROLES = ("baseline_csv", "sideline_csv")
ALL_ROLES = VIDEO_ROLES + CSV_ROLES


class StorageError(Exception):
    """Raised on invalid role / extension / missing session."""


class SessionStore:
    """Filesystem-backed session store bound to one ``WebConfig``."""

    def __init__(self, cfg: WebConfig):
        self.cfg = cfg
        self.root = cfg.data_root

    # -- sessions -----------------------------------------------------------
    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:12]
        (self.session_dir(sid) / "frames").mkdir(parents=True, exist_ok=True)
        self.save_state(sid, _empty_state())
        return sid

    def session_dir(self, sid: str) -> Path:
        _validate_sid(sid)
        return self.root / sid

    def exists(self, sid: str) -> bool:
        return (self.root / sid).is_dir() if _is_sid(sid) else False

    def require(self, sid: str) -> Path:
        d = self.session_dir(sid)
        if not d.is_dir():
            raise StorageError(f"unknown session: {sid}")
        return d

    # -- uploads ------------------------------------------------------------
    def save_upload(self, sid: str, role: str, file: FileStorage) -> Path:
        if role not in ALL_ROLES:
            raise StorageError(f"unknown upload role: {role}")
        ext = Path(file.filename or "").suffix.lower()
        if role in VIDEO_ROLES and not self.cfg.is_video(file.filename or ""):
            raise StorageError(f"{role}: expected {self.cfg.allowed_video_ext}, got {ext!r}")
        if role in CSV_ROLES and not self.cfg.is_csv(file.filename or ""):
            raise StorageError(f"{role}: expected a .csv, got {ext!r}")

        self.require(sid)
        dest = self.session_dir(sid) / f"{role}{ext}"
        for stale in self.session_dir(sid).glob(f"{role}.*"):
            stale.unlink()  # a re-upload replaces the previous file/ext
        file.save(dest)
        # A re-upload is a NEW clip: drop this role's cached stills and any
        # calibration tied to the old clip so nothing downstream mixes them.
        for stale in (self.session_dir(sid) / "frames").glob(f"{role}_f*.jpg"):
            stale.unlink()
        state = self.load_state(sid)
        state.setdefault("uploads", {})[role] = dest.name
        state.get("calibration", {}).pop(role, None)
        self.save_state(sid, state)
        return dest

    def path_for(self, sid: str, role: str) -> Path | None:
        if not self.exists(sid):
            return None
        matches = sorted(self.session_dir(sid).glob(f"{role}.*"))
        matches = [m for m in matches if m.parent == self.session_dir(sid)]
        return matches[0] if matches else None

    def frame_path(self, sid: str, role: str, frame: int = 0) -> Path:
        # one cached still per (role, frame index) so scrubbing is cheap
        return self.session_dir(sid) / "frames" / f"{role}_f{int(frame)}.jpg"

    # -- JSON state ---------------------------------------------------------
    def _state_file(self, sid: str) -> Path:
        return self.session_dir(sid) / "state.json"

    def load_state(self, sid: str) -> dict[str, Any]:
        f = self._state_file(sid)
        if not f.exists():
            return _empty_state()
        return json.loads(f.read_text(encoding="utf-8"))

    def save_state(self, sid: str, state: dict[str, Any]) -> None:
        self.session_dir(sid).mkdir(parents=True, exist_ok=True)
        self._state_file(sid).write_text(
            json.dumps(state, indent=2, default=_json_default), encoding="utf-8")

    def update_state(self, sid: str, patch: dict[str, Any]) -> dict[str, Any]:
        state = self.load_state(sid)
        _deep_merge(state, patch)
        self.save_state(sid, state)
        return state


def _empty_state() -> dict[str, Any]:
    return {"uploads": {}, "sync": {}, "calibration": {}, "metrics": {}, "analysis": {}}


def _deep_merge(base: dict, patch: dict) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _is_sid(sid: str) -> bool:
    return isinstance(sid, str) and sid.isalnum() and 0 < len(sid) <= 32


def _validate_sid(sid: str) -> None:
    if not _is_sid(sid):
        raise StorageError(f"invalid session id: {sid!r}")
