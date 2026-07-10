"""Fusion of the two cameras' estimates + dispute resolution - §5.8.

Fully implemented logic; the NUMBERS (dispute threshold, weight scale) are
TUNE-* config values to be calibrated on ground-truth data (Recipe 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bounce import BounceEvent
from .court_model import line_call


@dataclass
class LineCall:
    """The system's output for one bounce - with a full audit trail."""

    frame: float
    xy_ft: tuple[float, float]
    verdict: str                     # "IN" | "OUT"
    distance_cm: float               # signed distance to the nearest line
    nearest_line: str
    confidence: str                  # "high-agreement" | "two-view-resolved" | "single-view"
    resolved_by: str | None = None   # "A" | "B" | None (fused)
    audit: dict = field(default_factory=dict)


def camera_weight(conf: float, reproj_err_px: float,
                  reproj_scale_px: float = 3.0, floor: float = 0.05) -> float:
    """Evidence weight for one camera: detection confidence, discounted by
    calibration error (exponential decay with scale TUNE-4)."""
    w = float(conf) * float(np.exp(-reproj_err_px / reproj_scale_px))
    return max(w, floor)


def fuse_event(event: BounceEvent, cfg: dict | None = None,
               zone: str = "near_half") -> LineCall:
    """§5.8 in code:
    1. Weight each camera's (X, Y) by confidence and calibration quality.
    2. If the two estimates agree (< dispute threshold): fused weighted mean,
       high confidence.
    3. If they disagree: occlusion decides first - a camera flagged occluded
       never wins arbitration against a non-occluded one; only when both or
       neither are occluded does the evidence weight decide. Flagged as
       two-view-resolved, both estimates kept in the audit trail.
    """
    cfg = cfg or {}
    thresh_ft = cfg.get("dispute_threshold_ft", 0.164)     # TUNE-3 (5 cm)
    scale = cfg.get("reproj_err_scale_px", 3.0)            # TUNE-4
    floor = cfg.get("weight_floor", 0.05)
    q = event.quality

    xa, xb = event.xy_a_ft, event.xy_b_ft
    wa = camera_weight(q.get("conf_a", 0.5), q.get("reproj_a_px", 5.0), scale, floor)
    wb = camera_weight(q.get("conf_b", 0.5), q.get("reproj_b_px", 5.0), scale, floor)
    if q.get("occluded_a"):
        wa = floor
    if q.get("occluded_b"):
        wb = floor

    audit = {"xy_a_ft": None if xa is None else list(map(float, xa)),
             "xy_b_ft": None if xb is None else list(map(float, xb)),
             "w_a": wa, "w_b": wb, "separation_ft": event.separation_ft,
             "method": event.method, **q}

    if xa is None and xb is None:
        raise ValueError("bounce event carries no position estimate")
    occ_a, occ_b = bool(q.get("occluded_a")), bool(q.get("occluded_b"))
    if xa is None or xb is None:                       # one camera lost it
        xy = xa if xb is None else xb
        resolved_by = "A" if xb is None else "B"
        confidence = "single-view"
    else:
        disagreement = float(np.linalg.norm(np.asarray(xa) - np.asarray(xb)))
        audit["disagreement_ft"] = disagreement
        if disagreement <= thresh_ft:                  # agreement -> fuse
            xy = (wa * np.asarray(xa) + wb * np.asarray(xb)) / (wa + wb)
            resolved_by, confidence = None, "high-agreement"
        elif occ_a != occ_b:                           # occlusion decides FIRST (§5.8)
            resolved_by = "B" if occ_a else "A"
            xy = np.asarray(xb) if occ_a else np.asarray(xa)
            confidence = "two-view-resolved"
        else:                                          # then evidence weight
            xy = np.asarray(xa) if wa >= wb else np.asarray(xb)
            resolved_by = "A" if wa >= wb else "B"
            confidence = "two-view-resolved"

    call = line_call(np.asarray(xy, dtype=float), zone=zone,
                     margin_ft=cfg.get("margin_ft", 0.0))
    return LineCall(frame=event.frame, xy_ft=(float(xy[0]), float(xy[1])),
                    verdict=call["verdict"], distance_cm=call["distance_cm"],
                    nearest_line=call["nearest_line"], confidence=confidence,
                    resolved_by=resolved_by, audit=audit)
