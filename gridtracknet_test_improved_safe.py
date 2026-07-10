"""gridtracknet_test.py - simple GridTrackNet visualizer with a smooth trail.

Give it a video (a bare filename is looked up in Footage/), it runs GridTrackNet
on the GPU and writes an annotated copy to Footage/<name>_tracked.mp4: a thin
yellow trail that follows the ball. The raw per-frame detections are jittery, so
the trail is smoothed with a local quadratic fit (ball flight IS a parabola), so
each airborne arc reads as a clean parabola rather than a shaky line.

    python gridtracknet_test.py Cam1_evening_redcourt_part1.mp4
    python gridtracknet_test.py Cam1_evening_redcourt_part1.mp4 --seconds 30
    python gridtracknet_test.py path/to/any.mp4 --provider cpu
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
FOOTAGE = REPO / "Footage"
sys.path.insert(0, str(REPO / "src"))

from pickleball_phase2.ball_tracker_onnx import BallTracker
from pickleball_phase2.tracking import BallTrack, drop_track_outliers

class _NullBar:
    def update(self, *_): pass
    def close(self): pass


try:
    from tqdm import tqdm
except ImportError:                       # progress bar is optional
    def tqdm(iterable=None, **_kw):
        return iterable if iterable is not None else _NullBar()


def resolve_input(p: str) -> Path:
    """Accept a full path or a bare filename living in Footage/."""
    cand = Path(p)
    if cand.exists():
        return cand
    if (FOOTAGE / cand.name).exists():
        return FOOTAGE / cand.name
    raise FileNotFoundError(f"{p} (not found, and not in {FOOTAGE})")


def track_with_progress(src: Path, weights: Path, provider: str,
                        frame_mode: str, max_frames: int | None, fps: float):
    """Run GridTrackNet frame-by-frame with a progress bar; return (uv, valid).

    Mirrors GridTrackNetTracker.track_video (dual 60fps even/odd interleave +
    teleport-outlier rejection) but drives the loop here so we can show a bar
    over the slow GPU inference."""
    bt = BallTracker(str(weights), provider=provider)
    chunk = 10 if frame_mode == "dual" else 5

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total = min(total, max_frames) if total else max_frames

    coords: list = []
    buf: list = []
    read = 0
    bar = tqdm(total=total or None, desc="tracking", unit="frames")
    while max_frames is None or read < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        read += 1
        if len(buf) == chunk:
            if frame_mode == "dual":
                out = bt.predict(buf[0::2] + buf[1::2])
                inter = [None] * chunk
                inter[0::2], inter[1::2] = out[:5], out[5:]
                coords.extend(inter)
            else:
                coords.extend(bt.predict(buf))
            bar.update(len(buf))
            buf = []
    cap.release()
    bar.close()

    uv = np.array(coords, dtype=np.float64).reshape(-1, 2)
    valid = ~np.all(uv == 0.0, axis=1)
    # reuse the repo's teleport/ID-switch rejection
    track = drop_track_outliers(
        BallTrack(np.arange(len(uv), dtype=np.float64), uv,
                  np.where(valid, 1.0, 0.0), valid, fps), 60.0)
    return track.uv, track.valid


def _local_poly_smooth(y: np.ndarray, win: int, poly: int = 2) -> np.ndarray:
    """Local polynomial (quadratic) smoother - preserves parabolic arcs while
    removing per-frame jitter. Pure numpy (no scipy)."""
    n = len(y)
    if n < 3:
        return y
    win = min(win, n if n % 2 else n - 1)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    half = win // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        xx = np.arange(lo, hi) - i
        c = np.polyfit(xx, y[lo:hi], min(poly, hi - lo - 1))
        out[i] = np.polyval(c, 0.0)
    return out


def smooth_trajectory(uv, valid, gap_fill: int = 4, win: int = 11,
                      max_jump: float = 140.0) -> list:
    """Per-frame smoothed (x, y) or None. Splits the track into flight segments
    (breaking on long gaps / big jumps = a bounce or a new rally), fills tiny
    gaps, then quadratic-smooths each segment so arcs come out as parabolas."""
    n = len(valid)
    smoothed: list = [None] * n
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return smoothed

    seg, segs = [idx[0]], []
    for prev, cur in zip(idx, idx[1:]):
        jump = float(np.hypot(*(uv[cur] - uv[prev])))
        if (cur - prev) <= gap_fill + 1 and jump <= max_jump:
            seg.append(cur)
        else:
            segs.append(seg)
            seg = [cur]
    segs.append(seg)

    for seg in segs:
        seg = np.array(seg)
        if len(seg) < 3:                          # too short to smooth: pass through
            for f in seg:
                smoothed[f] = (int(uv[f, 0]), int(uv[f, 1]))
            continue
        grid = np.arange(seg[0], seg[-1] + 1)     # dense frame range (fills gaps)
        xi = np.interp(grid, seg, uv[seg, 0])
        yi = np.interp(grid, seg, uv[seg, 1])
        xs, ys = _local_poly_smooth(xi, win), _local_poly_smooth(yi, win)
        for f, xx, yy in zip(grid, xs, ys):
            smoothed[f] = (int(round(xx)), int(round(yy)))
    return smoothed


def draw_trail(frame, trail: list) -> None:
    """Thin yellow poly-trail, brighter toward the newest point; breaks on gaps."""
    m = len(trail)
    for j in range(1, m):
        a, b = trail[j - 1], trail[j]
        if a is None or b is None:
            continue
        f = j / m                                 # 0 = oldest, 1 = newest
        col = (0, int(120 + 135 * f), int(120 + 135 * f))   # BGR dim -> bright yellow
        cv2.line(frame, a, b, col, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GridTrackNet on a clip and draw a smooth ball trail.")
    ap.add_argument("video", help="video path or a filename inside Footage/")
    ap.add_argument("--provider", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--frame-mode", default="dual", choices=["dual", "sequential"],
                    help="dual = 60fps even/odd interleave (default); sequential = 30fps")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="only process the first N seconds (0 = whole clip)")
    ap.add_argument("--trail", type=int, default=36, help="trail length in frames")
    ap.add_argument("--out", default=None, help="output path (default Footage/<name>_tracked.mp4)")
    args = ap.parse_args()

    src = resolve_input(args.video)
    weights = REPO / "models" / "model_weights.onnx"
    if not weights.exists():
        sys.exit(f"missing tracker weights: {weights}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    max_frames = int(args.seconds * fps) if args.seconds > 0 else None

    print(f"[track] {src.name}  {w}x{h}@{fps:.2f}  provider={args.provider}  mode={args.frame_mode}")
    uv, valid = track_with_progress(src, weights, args.provider, args.frame_mode, max_frames, fps)
    n = len(uv)
    print(f"[track] {n} frames, {int(valid.sum())} detections ({100 * valid.mean():.0f}%)")

    smoothed = smooth_trajectory(uv, valid)

    out_path = Path(args.out) if args.out else FOOTAGE / f"{src.stem}_tracked.mp4"

    # CSV of the (smoothed) ball position, one row per frame it was tracked
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "x", "y"])
        for i, pt in enumerate(smoothed):
            if pt is not None:
                writer.writerow([i, pt[0], pt[1]])
    print(f"[done] wrote {csv_path}")
    cap = cv2.VideoCapture(str(src))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail = deque(maxlen=args.trail)
    i = 0
    bar = tqdm(total=n or None, desc="rendering", unit="frames")
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and i >= max_frames):
            break
        trail.append(smoothed[i] if i < n else None)
        draw_trail(frame, list(trail))
        vw.write(frame)
        i += 1
        bar.update(1)
    cap.release()
    vw.release()
    bar.close()
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
