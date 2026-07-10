"""Persistent, content-addressed cache for expensive preprocessing.

Every entry is keyed by a *content signature* of the source video plus the
parameters that produced it, so:

  * the cache PERSISTS across runs (it is a plain folder on disk),
  * it LOADS automatically when the same video + params are seen again, and
  * it INVALIDATES only when the video's bytes change (re-encode, re-record) or
    the parameters change - a different clip with the same name can never hit a
    stale entry.

Signature = SHA-256 over (file size, mtime_ns, and small byte samples from the
head/middle/tail of the file). Sampling a few KB keeps it O(1) instead of
hashing multi-hundred-MB clips, while still catching any real content change.

Namespaces separate the stages: ``sync``, ``tracking``, ``calibration``, ...
Payloads are stored as JSON (small dicts) or ``.npz`` (numpy arrays, e.g. ball
tracks). Nothing here is web- or pipeline-specific; both the demo and the Flask
app can share one cache root (``data/cache`` by default).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

_SAMPLE_BYTES = 65536  # head/mid/tail sample size for the content signature


def video_signature(video_path: str | Path) -> str:
    """Short, stable content fingerprint of a video file (16 hex chars).

    Changes iff the file's size, mtime, or sampled bytes change; identical for
    the same file across runs. Cheap: reads at most ~192 KB regardless of size.
    """
    p = Path(video_path)
    st = p.stat()
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    h.update(str(st.st_mtime_ns).encode())
    with p.open("rb") as f:
        for pos in (0, max(0, st.st_size // 2), max(0, st.st_size - _SAMPLE_BYTES)):
            f.seek(pos)
            h.update(f.read(_SAMPLE_BYTES))
    return h.hexdigest()[:16]


def _params_tag(params: dict | None) -> str:
    if not params:
        return "default"
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


class Cache:
    """A folder-backed cache rooted at ``root`` (created on demand)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def enabled(self) -> bool:
        return True

    def _entry(self, namespace: str, video_path: str | Path,
               params: dict | None, ext: str) -> Path:
        sig = video_signature(video_path)
        stem = Path(video_path).stem
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{stem}.{sig}.{_params_tag(params)}.{ext}"

    # -- JSON payloads (offsets, calib dicts, clap timestamps) --------------
    def load_json(self, namespace: str, video_path: str | Path,
                  params: dict | None = None) -> dict | None:
        f = self._entry(namespace, video_path, params, "json")
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save_json(self, namespace: str, video_path: str | Path, obj: dict,
                  params: dict | None = None) -> Path:
        f = self._entry(namespace, video_path, params, "json")
        f.write_text(json.dumps(obj, indent=2, default=_json_default),
                     encoding="utf-8")
        return f

    # -- ndarray payloads (ball tracks) ------------------------------------
    def load_npz(self, namespace: str, video_path: str | Path,
                 params: dict | None = None) -> dict | None:
        f = self._entry(namespace, video_path, params, "npz")
        if not f.exists():
            return None
        try:
            with np.load(f, allow_pickle=True) as z:
                return {k: z[k] for k in z.files}
        except (OSError, ValueError):
            return None

    def save_npz(self, namespace: str, video_path: str | Path,
                 arrays: dict, params: dict | None = None) -> Path:
        f = self._entry(namespace, video_path, params, "npz")
        np.savez_compressed(f, **arrays)
        return f


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")
