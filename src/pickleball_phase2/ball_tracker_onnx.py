"""
ball_tracker_onnx.py - standalone GridTrackNet ball tracker (ONNX Runtime).

VENDORED VERBATIM from the ball_tracker_handoff bundle (June 2026) - do not
"improve" the preprocessing here; it deliberately mirrors the original
GridTrackNet inference so predictions match the exported model. Consumed by
tracking.GridTrackNetTracker (TRK-1). Weights: models/model_weights.onnx.

A drop-in, framework-agnostic ball tracker extracted from GridTrackNet
(https://github.com/asigatchov/GridTrackNet). It depends ONLY on:
    onnxruntime  (or onnxruntime-gpu),  numpy,  opencv-python
No TensorFlow / Keras / GridTrackNet.py required.

It loads a GridTrackNet ONNX model (produced by `export_onnx.py`) and returns
the (x, y) pixel location of the ball for each input frame. Frames are processed
in groups of 5 (the model's temporal window). A coordinate of (0, 0) means
"no ball detected" in that frame.

The grid math, thresholds, and the (deliberately verbatim) two-step colour
conversion all mirror the original inference code, so predictions are identical
to running Predict.py / inference_onnx.py.

IMPORTANT: the shipped weights were trained on TENNIS footage. On pickleball the
model may still track the ball (it is a similar small, fast object) but accuracy
is NOT guaranteed. Validate on your own clips and, if needed, fine-tune/retrain
on labelled pickleball data. See README_INTEGRATION.md.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Find site-packages (None on CPU-only installs without the nvidia wheels)
site_packages = next(
    (p for p in map(Path, sys.path) if (p / "nvidia").exists()),
    None,
)

if site_packages is not None:
    for dll_dir in (
        site_packages / "nvidia" / "cuda_runtime" / "bin",
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
    ):
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]

    # cuDNN 9.2x ships sub-libraries (engines_tensor_ir, ext) that
    # onnxruntime.preload_dlls() predates. cuDNN dlopens them by bare name at
    # the first convolution; that lookup misses unless the DLL is already in
    # the process, and the miss surfaces as CUDNN_STATUS_SUBLIBRARY_LOADING_
    # FAILED -> silent CPU fallback. Preload every cudnn DLL (core first).
    _cudnn_bin = site_packages / "nvidia" / "cudnn" / "bin"
    if _cudnn_bin.exists():
        for _dll in sorted(_cudnn_bin.glob("cudnn*.dll"),
                           key=lambda p: (p.name != "cudnn64_9.dll", p.name)):
            try:
                ctypes.WinDLL(str(_dll))
            except OSError:
                pass

import onnxruntime as ort

if site_packages is not None:
    ort.preload_dlls()

class BallTracker:
    # --- GridTrackNet constants. Do NOT change; they define the model I/O. ---
    WIDTH = 768
    HEIGHT = 432
    IMGS_PER_INSTANCE = 5            # frames consumed (and produced) per inference
    GRID_COLS = 48
    GRID_ROWS = 27
    GRID_SIZE_COL = WIDTH / GRID_COLS    # 16.0 px per grid cell (horizontal)
    GRID_SIZE_ROW = HEIGHT / GRID_ROWS   # 16.0 px per grid cell (vertical)
    CONF_THRESHOLD = 0.5            # >= -> ball present; < -> output (0, 0)

    def __init__(self, model_path: str = "model_weights.onnx", provider: str = "gpu"):
        available = ort.get_available_providers()

        if provider == "gpu":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider is not available. Install onnxruntime-gpu "
                    "with CUDA 12 / cuDNN 9 on PATH, or construct with provider='cpu'."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)

        print("ONNX Runtime providers:", self.session.get_providers())

        self.input_name = self.session.get_inputs()[0].name
        self.active_providers = self.session.get_providers()

    def predict(self, frames, is_bgr: bool = True):
        """
        Run the tracker over a list of consecutive video frames.

        frames : list of HxWx3 uint8 images (consecutive frames, any resolution).
                 Length should be a multiple of 5; a trailing remainder of < 5
                 frames is ignored (same behaviour as the original repo).
                 Pass is_bgr=True for OpenCV frames (the default).
        returns: list of (x, y) int tuples, ONE PER PROCESSED FRAME, expressed in
                 the ORIGINAL frame's pixel coordinates. (0, 0) means no ball.
        """
        if len(frames) < self.IMGS_PER_INSTANCE:
            return []

        output_height = frames[0].shape[0]
        output_width = frames[0].shape[1]

        # ---- group frames into instances of 5 ----
        batches = []
        for i in range(0, len(frames), self.IMGS_PER_INSTANCE):
            batch = frames[i:i + self.IMGS_PER_INSTANCE]
            if len(batch) == self.IMGS_PER_INSTANCE:
                batches.append(batch)

        # ---- preprocess (verbatim from GridTrackNet; produces (N, 15, 432, 768)) ----
        units = []
        for batch in batches:
            unit = []
            for frame in batch:
                if is_bgr:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # intentional 2nd pass
                frame = np.moveaxis(frame, -1, 0)                # HWC -> CHW
                unit.append(frame[0])
                unit.append(frame[1])
                unit.append(frame[2])
            units.append(unit)
        units = np.asarray(units, dtype=np.float32)
        units /= 255.0

        # ---- inference ----
        y_pred = self.session.run(None, {self.input_name: units})[0]   # (N, 15, 27, 48)

        # ---- reshape 15 channels -> 5 frames x (conf, x_off, y_off) ----
        y_pred = np.split(y_pred, self.IMGS_PER_INSTANCE, axis=1)
        y_pred = np.stack(y_pred, axis=2)
        y_pred = np.moveaxis(y_pred, 1, -1)
        conf_grid, x_off_grid, y_off_grid = np.split(y_pred, 3, axis=-1)
        conf_grid = np.squeeze(conf_grid, axis=-1)
        x_off_grid = np.squeeze(x_off_grid, axis=-1)
        y_off_grid = np.squeeze(y_off_grid, axis=-1)

        # ---- decode each frame's grid to a pixel coordinate ----
        coords = []
        for i in range(conf_grid.shape[0]):          # instances
            for j in range(conf_grid.shape[1]):      # 5 frames within the instance
                cg = conf_grid[i][j]
                max_conf = np.max(cg)
                row, col = np.unravel_index(np.argmax(cg), cg.shape)

                if max_conf >= self.CONF_THRESHOLD:
                    x_off = x_off_grid[i][j][row][col]
                    y_off = y_off_grid[i][j][row][col]
                    x_pred = int((x_off + col) * self.GRID_SIZE_COL)
                    y_pred = int((y_off + row) * self.GRID_SIZE_ROW)
                    coords.append((
                        int((x_pred / self.WIDTH) * output_width),
                        int((y_pred / self.HEIGHT) * output_height),
                    ))
                else:
                    coords.append((0, 0))
        return coords
