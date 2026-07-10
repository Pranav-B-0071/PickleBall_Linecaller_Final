"""Real-time streaming architecture - §5.9 (skeleton only, by design).

The offline pipeline (pipeline.py) is the deliverable for this hand-off;
this file pins down the intended real-time shape so a future intern doesn't
have to re-derive it. Everything here is PLACEHOLDER[ARCH-1] (Recipe 9).

Design targets (§5.9): homography computed once per camera and cached;
tracking on ROI-cropped frames; fusion triggered only on bounce events;
the two streams batched on one GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamConfig:
    camera_id: str            # "A" | "B" | "K"
    source_url: str           # e.g. RTMP/WebRTC endpoint from the phone
    fps: float = 60.0


class RealTimeServer:
    """PLACEHOLDER[ARCH-1] - target shape:

    ingest (per camera thread/async task)
        -> ring buffer of frames (aligned via sync offset)
        -> cached CourtCalibration per camera (recomputed only on drift alarm)
        -> BallTracker on ROI crops, batched A+B on the GPU
        -> separation-signal monitor (bounce.py) on the two ground tracks
        -> on bounce event: fusion.fuse_event -> LineCall
        -> broadcast call + overlay to clients (websocket)

    Suggested stack: FastAPI + websockets, one asyncio task per stream,
    a single inference worker consuming a frame queue.
    """

    def __init__(self, cam_a: StreamConfig, cam_b: StreamConfig,
                 cam_k: StreamConfig | None = None):
        self.cams = [c for c in (cam_a, cam_b, cam_k) if c is not None]

    def run(self) -> None:
        raise NotImplementedError(
            "PLACEHOLDER[ARCH-1] - see PLACEHOLDER_COOKBOOK.md, Recipe 9")
