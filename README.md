# Pickleball Linecaller
# ALSO READ THE EXPLANATION.MD FOR BETTER UNDERSTANDING
## How to run

This project needs Python 3.10. A virtual environment already exists at the
repository root (`../venv`); activate it, then from this folder:

```
pip install -r requirements.txt
python run_web.py
```

Open http://127.0.0.1:5001 and work through the three pages in order:
Calibration, Metrics, Analysis. To run the tests (no data required):

```
python -m pytest tests/ -q
```

---

A computer-vision system that watches a pickleball rally from two phone cameras
(plus an optional third camera for the kitchen zone) and calls each ball bounce
IN or OUT relative to the near-half court lines. This repository is the Phase 2
implementation: it turns recorded clips into synchronized playback, ball
tracking, bounce detection, and IN/OUT verdicts, all through a local web app.

For a plain-language walkthrough of how the whole thing fits together, read
[Explanation.md](Explanation.md). This file is the reference: what every folder
and file is, how to run it, and the current status of each part.

---

## What it does

- Calibrates each camera to the court. A YOLO-pose model finds 12 court
  keypoints in the video, and a homography maps image pixels to court feet, so
  every camera shares one coordinate frame.
- Tracks the ball in each camera with GridTrackNet (run through ONNX Runtime,
  on GPU when available).
- Synchronizes the cameras from a hand clap near the start of each clip (audio
  cross-correlation), so all views share one timeline.
- Detects bounces and calls them IN or OUT near the near-half lines.
- Presents all of this in a 3-page web app: calibration, metrics, and a
  synchronized analysis dashboard with a top-view court and a bounce feed.

---

## The three web pages

1. Calibration (`/`)
   - Upload the CAM1, CAM2, and CAM3 clips.
   - Auto clap-sync, with manual offset controls as a fallback.
   - "Detect Court" runs the keypoint model on CAM1 and CAM2; drag, zoom, undo,
     and redo the 12 keypoints, then "Confirm" to fit the homography and derive
     the near-half ROI.
   - CAM3 is calibrated by hand: draw a 4-point kitchen box. Bounces inside it
     are tallied as kitchen bounces.

2. Metrics (`/metrics`)
   - Upload a baseline or sideline video plus a labels CSV, draw the boundary
     line, and compare predicted calls against the labels (accuracy, confusion
     matrix, dashboard cards).

3. Analysis (`/analysis`)
   - Synchronized playback of all three cameras with a top-view court, ball
     trajectory, and bounce markers. IN/OUT is called only inside the near-half
     ROI. This page uses the two-button Calculate / Analyze flow described next.

---

## The Calculate / Analyze flow (Page 3)

Page 3 splits the work into two buttons so tracking (slow) is separate from
viewing (fast).

- Calculate
  - Runs `gridtracknet_test_improved_safe.py` as a subprocess, once per uploaded
    clip, so the tracking and rendering progress bars stream to the terminal
    that is running `run_web.py`.
  - Writes `Footage/<role>_tracked.mp4` (annotated with a yellow ball trail) and
    `Footage/<role>_tracked.csv` (one row per tracked frame: `frame_number,x,y`)
    for cam1, cam2, and cam3.
  - Transcodes each tracked mp4 to H.264 so browsers can play it.

- Analyze
  - Reads the tracked CSVs directly (no re-tracking).
  - Detects bounces per camera in image space, places each contact point on the
    court via that camera's homography, calls IN or OUT, and merges the two
    cameras by aligned frame.
  - Swaps each video slot to the annotated tracked clip and plays all three in
    sync, with the bounce numbers on the side.
  - If the tracked files are missing it asks you to run Calculate first; if CAM1
    or CAM2 is not calibrated it asks you to calibrate first.

---

## How bounce detection works (and why there are two methods)

There are two bounce detectors in this repo. They exist for different reasons.

- Single-view detection (used by the web app Analyze path).
  Each camera's ball track is scanned in image space for the ball's on-screen
  lowest point (pixel-y reaches a local maximum, then rises) - a down-then-up
  parabola. That is a bounce. Only the contact point (which sits on or near the
  ground) is projected through the homography for the IN/OUT call, so the
  projection stays stable. The two cameras' bounces are then merged by aligned
  frame. This is robust on low, near-ground phone cameras.
  Code: `webapp/services/analysis_service.py` (`analyze_from_tracked`) built on
  `src/pickleball_phase2/single_view_bounce.py`.

- Dual-camera separation signal (used by the offline pipeline).
  Both cameras' ball pixels are projected onto the court plane; while the ball is
  airborne the two projections disagree, and the separation shrinks to a minimum
  at contact. This is the method from the technical disclosure. It depends on
  accurate calibration and works best with cameras that are not mounted too low
  (projecting an airborne ball through the ground homography can blow up near the
  horizon on low cameras).
  Code: `src/pickleball_phase2/bounce.py`, driven by
  `src/pickleball_phase2/pipeline.py`.

CAM3 is auxiliary. It is not used for IN/OUT; its only current role is counting
bounces inside the manually drawn kitchen box (future scope).

---

## Project structure

```
PICKELBALL_3D-LINECALLER_REAL_WORLD_IMPLEMENTATION/
  run_web.py                    Start the Flask web app (http://127.0.0.1:5001)
  demo.py                       Offline runner / precompute over the demo footage
  gridtracknet_test_improved_safe.py
                                Standalone tracker visualizer: clip -> annotated
                                mp4 + CSV (used by the Calculate button)
  config.yaml                   Single source of truth: paths and thresholds
  requirements.txt              Python dependencies
  CLAUDE.md                     Instructions for the AI agent team on this repo
  Explanation.md                Plain-language overview of how the repo works
  README.md                     This file

  src/pickleball_phase2/        The algorithm package (pure Python, importable)
    court_model.py              Canonical court, 12 keypoints, line_call (IN/OUT)
    config.py                   Typed config access over config.yaml
    calibration.py              Intrinsics, homography fit, PnP camera pose
    sync.py                     Clap detection + GCC-PHAT offset refinement
    tracking.py                 GridTrackNetTracker wrapper + BallTrack type
    ball_tracker_onnx.py        Vendored ONNX loader/decoder (do not edit)
    bounce.py                   Dual-camera separation-signal bounce detection
    single_view_bounce.py       Single-camera bounce detection + streaming
    fusion.py                   Two-camera weighting, dispute rule, audit trail
    pipeline.py                 Offline chain: clip pair -> list of LineCalls
    placement.py                Camera-placement readiness score + state machine
    server.py                   Real-time server skeleton (future work)
    cache.py                    Content-hash cache for sync and tracking results

  webapp/                       Flask app layered over the algorithm package
    __init__.py                 create_app() factory + blueprint registration
    bootstrap.py                Puts src/ on sys.path; loads the project config
    config.py                   WebConfig (upload dirs, limits, model path)
    routes/                     Thin HTTP blueprints (see below)
    services/                   The only layer that imports pickleball_phase2
    templates/                  Jinja pages: base + 3 pages + partials
    static/                     css/, js/, img/ for the front end

  models/
    best.pt                     Court-keypoint model (YOLO-pose, 12 points) - live
    small_model_best.pt         Smaller keypoint variant (present, not wired)
    model_weights.onnx          GridTrackNet ball tracker exported to ONNX

  calib/intrinsics/             Per-camera intrinsics: cam1.yaml, cam2.yaml, cam3.yaml
  Footage/                      Original demo clips + _tracked.mp4 / _tracked.csv
  data/                         Runtime data (see the data section below)
  tests/                        Synthetic end-to-end and web smoke tests

  GridTrackNet-essentials/      GridTrackNet training tooling (for fine-tuning)
  ball_tracker_handoff/         The ONNX inference bundle that was vendored into src

  Pickleball_Linecaller_Technical_Disclosure (1).pdf   Design document (reference)
```

---

## Component reference

### Algorithm package (`src/pickleball_phase2/`)

- `court_model.py` - the canonical court geometry. Defines the 12 keypoints in a
  fixed index order, the court dimensions (20 ft by 44 ft, net at y = 22), and
  `line_call(xy, zone)` which returns the IN/OUT verdict, the signed distance to
  the nearest line, and the nearest line name. A ball on a line is IN.
- `config.py` - loads `config.yaml` and provides dotted-key access, for example
  `cfg.get("tracking.provider", "gpu")`.
- `calibration.py` - camera intrinsics loading, RANSAC homography fitting from
  keypoints (image to court), and PnP camera-pose recovery (a diagnostic, not
  used for the line call).
- `sync.py` - extracts audio, finds the loudest short burst (the clap), and
  refines the offset between two clips with GCC-PHAT. Convention:
  aligned_frame = frame_B - offset.
- `tracking.py` - `GridTrackNetTracker` runs the ONNX model over a clip and
  returns a `BallTrack` (per-frame pixel positions plus a valid mask). 60 fps
  input is fed as even/odd 30 fps streams (dual mode), matching the model's
  trained motion spacing.
- `ball_tracker_onnx.py` - the vendored ONNX preprocessing and grid decoder. It
  mirrors the original GridTrackNet inference exactly and should not be changed.
- `bounce.py` - the dual-camera separation-signal detector: project both tracks
  to the court, compute the separation signal, find prominent minima, refine to
  sub-frame contact, and classify bounce versus paddle hit.
- `single_view_bounce.py` - single-camera detection from the pixel track (local
  maximum of image-y with a parabola guard, outlier rejection, gap handling,
  sub-frame vertex), plus a causal streaming detector. The web app uses this for
  detection.
- `fusion.py` - combines the two cameras' estimates into one `LineCall` with a
  confidence label, a dispute rule when the cameras disagree, and an audit trail.
- `pipeline.py` - `run_clip_pair` runs the full offline chain (sync, calibrate,
  track, detect, fuse) and returns a list of `LineCall`. Every part is
  injectable, so mocks and partial states still run end to end.
- `placement.py` - readiness score and state machine for guiding camera
  placement (sound output is not implemented).
- `server.py` - a skeleton for a future real-time server.
- `cache.py` - a content-hash cache so repeated sync and tracking work is reused.

### Web layer (`webapp/`)

Routes (`webapp/routes/`), thin HTTP handlers:
- `pages.py` - serves the three HTML pages.
- `uploads.py` - session lifecycle, video upload (with automatic 60 fps
  normalization), media serving (originals and the tracked clips), and frame
  extraction for the calibration editor.
- `calibration.py` - court keypoint detection, homography confirm, and the CAM3
  kitchen box.
- `metrics.py` - the Page 2 compare endpoints.
- `analysis.py` - the Page 3 Calculate and Analyze endpoints.
- `errors.py` - JSON error handlers.
- `_helpers.py` - shared request/response helpers (`ok`, `err`, `with_session`).

Services (`webapp/services/`), the only code that imports the algorithm package:
- `storage.py` - per-session file layout under `data/webapp/<session_id>/`.
  Uploads are stored by role (`cam1.mp4`, `cam2.mp4`, `cam3.mp4`), not by their
  original filename.
- `video_utils.py` - clip metadata, first-frame extraction, constant-frame-rate
  normalization on upload, and H.264 transcoding of the tracked clips.
- `sync_service.py` - three-camera sync wrapper (CAM1 is the reference).
- `court_detection.py` - runs the `best.pt` keypoint model (or a realistic mock
  when the model or ultralytics is unavailable).
- `calibration_service.py` - fits the homography from confirmed keypoints and
  reports the reprojection error; also the PnP sanity check and the kitchen box.
- `roi.py` - derives the near-half ROI from the calibration.
- `metrics_service.py` - accuracy, confusion matrix, and dashboard cards.
- `analysis_service.py` - the Page 3 engine. `analyze_from_tracked` reads the
  tracked CSVs, runs single-view detection, places bounces on the court, calls
  IN/OUT, and merges the two cameras. A deterministic mock path exists as a
  fallback.

Front end (`webapp/static/js/`):
- `api.js` - session-aware fetch, upload, and toast helpers.
- `app.js` - the workflow stepper.
- `court-overlay.js`, `keypoint-editor.js` - the calibration canvas and editor.
- `video-sync.js` - `SyncedPlayerGroup` drives several videos on one timeline.
- `topview.js` - the top-down court with the trajectory and bounce markers.
- `page1.js`, `page2.js`, `page3.js` - per-page controllers.

### Top-level scripts

- `run_web.py` - starts the Flask app; wipes `data/webapp` on each fresh launch.
- `demo.py` - runs the offline pipeline over the demo footage and can precompute
  caches.
- `gridtracknet_test_improved_safe.py` - takes a clip and writes an annotated
  mp4 with a smoothed ball trail plus a CSV of the ball positions. This is the
  script the Calculate button runs once per camera.

### Models (`models/`)

- `best.pt` - the court-keypoint model (YOLO-pose, 12 points). Live: Page 1 runs
  it through ultralytics.
- `small_model_best.pt` - a smaller keypoint variant; present but not wired in.
- `model_weights.onnx` - the GridTrackNet ball tracker exported to ONNX. Used
  for all ball tracking.

### Support bundles

- `GridTrackNet-essentials/` - the GridTrackNet training tooling (`Train.py`,
  `DataGen.py`, `LabellingTool.py`, `Predict.py`, and the `model_weights.h5`
  checkpoint). Used to train or extend the ball tracker. Training needs
  TensorFlow, which has no native Windows GPU support, so train in WSL2 or the
  cloud and re-export to ONNX. Inference stays on ONNX Runtime.
- `ball_tracker_handoff/` - the original self-contained ONNX inference bundle
  that was vendored into `src/pickleball_phase2/ball_tracker_onnx.py`.

---

## Configuration (`config.yaml`)

`config.yaml` is the single source of truth for paths and thresholds. Key
sections:

- `capture` - frame rate and capture settings (60 fps).
- `paths` - model and intrinsics locations.
- `tracking` - GridTrackNet settings: `provider` (gpu or cpu), `frame_mode`
  (dual), outlier rejection.
- `sync` - clap-sync search window settings.
- `bounce` - detection thresholds. The web app single-view path also reads
  optional keys `sv_min_drop_px`, `sv_lockout_frames`, `sv_parabolic_min_r2`,
  `sv_merge_window_frames`, and `contact_offset_px` (all have code defaults, so
  no config change is needed to run).
- `fusion` and `line_call` - dual-camera weighting and the IN/OUT zone.
- `analysis` - the analysis window length.

---

## Data and outputs (`data/`, `Footage/`)

- `Footage/` - the original demo clips (`Cam1/2/3_evening_redcourt_part1.mp4`)
  plus the tracked outputs produced by the Calculate step
  (`<role>_tracked.mp4` and `<role>_tracked.csv`).
- `data/webapp/<session_id>/` - per-session uploads (`cam1.mp4`, `cam2.mp4`,
  `cam3.mp4`), extracted frames, and `state.json` (uploads, sync, calibration,
  metrics, analysis). Cleared on each fresh web launch.
- `data/cache/` - content-hash caches for sync and tracking, so repeat runs of
  the same clips are fast.
- `data/calib_frames/` - throwaway diagnostic images from calibration.
- `calib/intrinsics/` - per-camera intrinsics YAML files.

---

## Status

- Court keypoints: live in the web app via `best.pt` and ultralytics.
- Ball tracker: done. GridTrackNet runs through ONNX Runtime on GPU.
- Sync: done. Audio clap plus GCC-PHAT, verified on the demo footage.
- Bounce detection: single-view path is live in the web app; the dual-camera
  separation method exists but is limited by the low camera geometry. Threshold
  tuning on labelled rallies is pending.
- Real-time server: future work.

---

## Notes

- GPU inference is enabled by default (`tracking.provider: gpu`) and requires
  `onnxruntime-gpu` with the CUDA 12 and cuDNN 9 runtime. It falls back to CPU
  if CUDA is unavailable.
- Uploaded clips are automatically normalized to constant 60 fps on upload,
  because sync, tracking, and bounce pairing all assume a shared 60 fps clock.
- The tracked mp4 files are silent, so audio clap-sync always runs on the
  original clips, not the tracked ones.
