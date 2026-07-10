"""HTML page routes (the three-page workflow)."""

from __future__ import annotations

from flask import Blueprint, render_template

bp = Blueprint("pages", __name__)

# (step key, label) - the shared workflow stepper in base.html.
STEPS = [
    ("upload", "Upload"),
    ("sync", "Sync"),
    ("calibrate", "Calibrate"),
    ("metrics", "Metrics"),
    ("analysis", "Run Analysis"),
]


@bp.route("/")
def calibration():
    return render_template("page1_calibration.html", steps=STEPS, active="upload",
                           page_title="Calibration")


@bp.route("/metrics")
def metrics():
    return render_template("page2_metrics.html", steps=STEPS, active="metrics",
                           page_title="Label Metrics")


@bp.route("/analysis")
def analysis():
    return render_template("page3_analysis.html", steps=STEPS, active="analysis",
                           page_title="Analysis")
