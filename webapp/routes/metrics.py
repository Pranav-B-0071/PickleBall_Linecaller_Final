"""Page 2 API: label-metrics compare (uploads reuse the generic /api/upload)."""

from __future__ import annotations

from flask import Blueprint, request

from ..services import metrics_service
from ._helpers import err, ok, with_session

bp = Blueprint("metrics", __name__)


@bp.post("/api/metrics/compare")
@with_session
def compare(sid, st):
    body = request.get_json(silent=True) or {}
    boundaries = body.get("boundaries")   # {"baseline": {...}, "sideline": {...}}
    csv_paths = {
        "baseline": st.path_for(sid, "baseline_csv"),
        "sideline": st.path_for(sid, "sideline_csv"),
    }
    if not any(csv_paths.values()):
        return err("upload at least one CSV (baseline or sideline) first", 404)

    dashboard = metrics_service.build_dashboard(csv_paths, boundaries)
    st.update_state(sid, {"metrics": {"boundaries": boundaries or {},
                                      "cards": dashboard["cards"]}})
    return ok(**dashboard)
