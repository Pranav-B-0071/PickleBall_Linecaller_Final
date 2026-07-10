"""Pickleball Linecaller web application (Flask app-factory).

A 3-page UI + JSON API layered over the ``pickleball_phase2`` algorithm package:

    Page 1  /          upload + clap-sync + court calibration (near-half ROI)
    Page 2  /metrics   label-metrics: model vs manual IN/OUT accuracy
    Page 3  /analysis  synchronized 3-cam analysis + top-view homography

Blueprints are thin; all real work lives in ``webapp.services`` (the only place
that imports the algorithm package). The trained models are behind swappable
service modules, so dropping in ``models/best.pt`` needs no route/UI changes.
"""

from __future__ import annotations

from flask import Flask

from .config import WebConfig


def create_app(config: WebConfig | None = None) -> Flask:
    """Application factory. Pass a ``WebConfig`` in tests; else it's loaded."""
    cfg = config or WebConfig.load()
    cfg.data_root.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["WEB"] = cfg
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_content_length
    app.config["JSON_SORT_KEYS"] = False

    from .routes.pages import bp as pages_bp
    from .routes.uploads import bp as uploads_bp
    from .routes.calibration import bp as calibration_bp
    from .routes.metrics import bp as metrics_bp
    from .routes.analysis import bp as analysis_bp
    from .routes.errors import register_error_handlers

    for bp in (pages_bp, uploads_bp, calibration_bp, metrics_bp, analysis_bp):
        app.register_blueprint(bp)
    register_error_handlers(app)

    return app
