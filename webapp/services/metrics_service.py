"""Label-metrics service (Page 2) - model calls vs your manual calls.

The manual/ground-truth calls come from the uploaded CSV (``throw,call`` schema
from page2_example, with optional ``t`` and ``x,y`` columns). The "model" call
for each throw is derived from the boundary line the user draws over the court:

    Baseline   above the line -> IN,  below -> OUT
    Sideline   left  of  line -> IN,  right -> OUT

That derivation is REAL (given the CSV court coordinates), so accuracy / the
confusion matrix / precision-recall are all genuine today. When the bounce model
lands, the court coordinates come from the model instead of the CSV - the compare
math is unchanged. Timing/FPS/confidence cards that need the model are labelled
placeholders.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

# Vocabulary. ON_LINE / CLOSE_IN both count as IN for the binary verdict
# (a ball touching the line is IN), matching the page2_example footnote.
IN_LIKE = {"IN", "ON_LINE", "CLOSE_IN"}
TRUTH_CLASSES = ("IN", "OUT", "ON_LINE")
PRED_CLASSES = ("IN", "OUT", "CLOSE_IN", "UNKNOWN", "MISSED")

DEFAULT_BOUNDARY = {
    "baseline": {"orientation": "horizontal", "position": 22.0, "in_side": "above"},
    "sideline": {"orientation": "vertical", "position": 10.0, "in_side": "left"},
}


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_calls_csv(path: str | Path) -> list[dict]:
    """Parse a manual-calls CSV. Tolerant of column naming/casing."""
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        norm = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}

        def col(*names):
            for n in names:
                if n in norm:
                    return norm[n]
            return None

        c_call = col("call", "truth", "label", "verdict")
        c_t = col("t", "t_s", "time", "t(s)")
        c_x = col("x", "court_x", "cx")
        c_y = col("y", "court_y", "cy")
        c_throw = col("throw", "#", "idx", "id")
        for i, r in enumerate(reader, start=1):
            call = str(r.get(c_call, "") or "").strip().upper() or "UNKNOWN"
            rows.append({
                "i": int(_num(r.get(c_throw), i)),
                "t": _num(r.get(c_t)),
                "x": _num(r.get(c_x)),
                "y": _num(r.get(c_y)),
                "call": call,
            })
    return rows


def _derived_call(x, y, boundary: dict) -> str:
    """Boundary-rule prediction from a court point; UNKNOWN if no coordinates."""
    if boundary["orientation"] == "horizontal":
        coord = y
        in_when_less = boundary["in_side"] == "above"
    else:
        coord = x
        in_when_less = boundary["in_side"] == "left"
    if coord is None:
        return "UNKNOWN"
    inside = (coord <= boundary["position"]) if in_when_less else (coord >= boundary["position"])
    return "IN" if inside else "OUT"


def _binary(call: str) -> str:
    return "IN" if call in IN_LIKE else "OUT"


def build_panel(rows: list[dict], boundary: dict) -> dict:
    """One panel (baseline or sideline): per-throw table + accuracy + matrix."""
    table, confusion = [], {t: {p: 0 for p in PRED_CLASSES} for t in TRUTH_CLASSES}
    correct = 0
    tp = fp = fn = 0
    for r in rows:
        pred = _derived_call(r["x"], r["y"], boundary)
        truth = r["call"] if r["call"] in TRUTH_CLASSES else _binary(r["call"])
        if truth not in confusion:
            truth = _binary(truth)
        confusion[truth][pred if pred in PRED_CLASSES else "UNKNOWN"] += 1

        tb, pb = _binary(r["call"]), _binary(pred) if pred in IN_LIKE | {"OUT"} else None
        if pb is not None and tb == pb:
            correct += 1
        if pb == "IN" and tb == "IN":
            tp += 1
        elif pb == "IN" and tb == "OUT":
            fp += 1
        elif pb == "OUT" and tb == "IN":
            fn += 1
        table.append({**r, "pred": pred})

    n = len(rows)
    graded = sum(1 for r in rows if _derived_call(r["x"], r["y"], boundary) != "UNKNOWN")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "rows": table,
        "count": n,
        "in_count": sum(1 for r in rows if _binary(r["call"]) == "IN"),
        "out_count": sum(1 for r in rows if _binary(r["call"]) == "OUT"),
        "accuracy": round(100 * correct / graded, 1) if graded else 0.0,
        "graded": graded,
        "correct": correct,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives": fp,
        "false_negatives": fn,
        "confusion": confusion,
        "boundary": boundary,
    }


def build_dashboard(csv_paths: dict[str, Path], boundaries: dict | None = None) -> dict:
    """Build both panels + the combined summary cards.

    csv_paths: {"baseline": Path|None, "sideline": Path|None}
    boundaries: {"baseline": {...}, "sideline": {...}} or None for defaults.
    """
    t0 = time.perf_counter()
    boundaries = boundaries or {}
    panels: dict[str, dict] = {}
    for panel_name in ("baseline", "sideline"):
        path = csv_paths.get(panel_name)
        boundary = boundaries.get(panel_name, DEFAULT_BOUNDARY[panel_name])
        if path is None:
            panels[panel_name] = {"rows": [], "count": 0, "available": False,
                                  "boundary": boundary}
            continue
        panels[panel_name] = {**build_panel(parse_calls_csv(path), boundary),
                              "available": True}

    cards = _summary_cards(panels, elapsed_s=time.perf_counter() - t0)
    return {"panels": panels, "cards": cards}


def _summary_cards(panels: dict, elapsed_s: float) -> dict:
    """Combined dashboard cards. Model-dependent ones are marked placeholder."""
    graded = [p for p in panels.values() if p.get("available")]
    total = sum(p["count"] for p in graded)
    in_c = sum(p.get("in_count", 0) for p in graded)
    out_c = sum(p.get("out_count", 0) for p in graded)
    corr = sum(p.get("correct", 0) for p in graded)
    gd = sum(p.get("graded", 0) for p in graded)
    fp = sum(p.get("false_positives", 0) for p in graded)
    fn = sum(p.get("false_negatives", 0) for p in graded)
    prec = _avg([p["precision"] for p in graded if "precision" in p])
    rec = _avg([p["recall"] for p in graded if "recall" in p])
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    def card(label, value, unit="", placeholder=False):
        return {"label": label, "value": value, "unit": unit, "placeholder": placeholder}

    return [
        card("Total Bounces", total),
        card("IN", in_c),
        card("OUT", out_c),
        card("Accuracy", round(100 * corr / gd, 1) if gd else 0.0, "%"),
        card("Precision", round(prec, 3)),
        card("Recall", round(rec, 3)),
        card("F1 Score", round(f1, 3)),
        card("False Positives", fp),
        card("False Negatives", fn),
        card("Processing Time", round(elapsed_s * 1000, 1), "ms"),
        card("FPS", "-", "", placeholder=True),
        card("Confidence", "-", "", placeholder=True),
        card("Average Error", "-", "cm", placeholder=True),
    ]


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
