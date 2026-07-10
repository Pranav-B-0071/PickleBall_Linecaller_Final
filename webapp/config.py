"""Web-layer configuration.

Thin wrapper over the project's existing ``pickleball_phase2.Config`` (which
reads ``config.yaml``). Everything the Flask layer needs - upload dirs, size
limits, allowed extensions, the court-model path - is derived from the
``webapp:`` section there, so nothing is hard-coded in the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .bootstrap import REPO_ROOT, load_project_config


@dataclass(frozen=True)
class WebConfig:
    """Resolved, absolute web settings for one running app."""

    host: str
    port: int
    max_upload_mb: int
    allowed_video_ext: tuple[str, ...]
    allowed_csv_ext: tuple[str, ...]
    data_root: Path                 # data/webapp - session folders live here
    court_model_weights: Path       # models/best.pt (may not exist yet -> mock)
    bounce_marker_ttl_s: int
    repo_root: Path = field(default=REPO_ROOT)

    @property
    def max_content_length(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def is_video(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.allowed_video_ext

    def is_csv(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.allowed_csv_ext

    @classmethod
    def load(cls) -> "WebConfig":
        cfg = load_project_config()
        w = cfg.get("webapp", {}) or {}

        def rel(p: str) -> Path:
            path = Path(p)
            return path if path.is_absolute() else REPO_ROOT / path

        data_root = rel("data") / w.get("data_subdir", "webapp")
        return cls(
            host=w.get("host", "127.0.0.1"),
            port=int(w.get("port", 5001)),
            max_upload_mb=int(w.get("max_upload_mb", 512)),
            allowed_video_ext=tuple(w.get("allowed_video_ext", [".mp4", ".mov"])),
            allowed_csv_ext=tuple(w.get("allowed_csv_ext", [".csv"])),
            data_root=data_root,
            court_model_weights=rel(w.get("court_model_weights", "models/best.pt")),
            bounce_marker_ttl_s=int(w.get("bounce_marker_ttl_s", 10)),
        )
