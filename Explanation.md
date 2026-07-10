# Explanation - how this project works, in plain terms

This is the friendly overview. If you want the exact file-by-file reference,
see [README.md](README.md). Here the goal is just to give you the mental model:
what the system is trying to do, the path a video takes through it, and where to
look when you want to change something.

---

## The big picture in one minute

You set up two phones behind the near baseline of a pickleball court, both
pointing at the same court. A rally is played. The system watches both videos
and, for each time the ball bounces near the lines, says IN or OUT. A third
phone can watch the kitchen (the non-volley zone) as an extra.

Three ideas make this possible:

1. Each camera learns where the court is. A model finds the court line
   intersections (12 known points), and from those points we compute a
   "homography" - a formula that converts any pixel in the video into a real
   position on the court, measured in feet. After this, both cameras describe
   the court in the same units.

2. Each camera follows the ball. A tracker called GridTrackNet marks the ball's
   pixel position in every frame.

3. The videos are lined up in time. Someone claps near the start of each clip.
   The system hears the clap in both videos and shifts one so both play on the
   same timeline.

Once the court, the ball, and the timing are known, finding a bounce and calling
it IN or OUT is the last step.

---

## The journey of a video, step by step

1. Upload. You drop the three clips into the Calibration page. The app stores
   them by role (cam1, cam2, cam3) and, if a clip is not a clean 60 frames per
   second, re-encodes it so everything runs on the same clock.

2. Sync. The app listens for the clap in each clip and computes how many frames
   apart the cameras are. CAM1 is the reference; the others are shifted to match.

3. Calibrate. On CAM1 and CAM2 you press "Detect Court". The keypoint model
   marks the 12 court points; you can drag them to fix any that are off. When you
   confirm, the app fits the homography and shows a reprojection error (how many
   pixels off the fit is). A low number, locked, means good calibration. CAM3 is
   simpler: you just draw a 4-point box around the kitchen.

4. Track (the Calculate button on the Analysis page). The app runs GridTrackNet
   on each clip. You watch the progress bars in the terminal. For each camera it
   saves two files in the Footage folder: an annotated video with a yellow ball
   trail, and a CSV listing the ball position in every frame it was seen.

5. Analyze (the Analyze button). The app reads those CSVs and looks, in each
   camera separately, for the moment the ball reaches its lowest point on screen
   and then rises again. That is a bounce. It converts the bounce's contact point
   to court feet using the homography, decides IN or OUT, and combines what the
   two cameras saw. Then it plays the three annotated videos in sync and shows
   the bounce count and verdicts on the side.

---

## Why bounces are found "per camera" and not by comparing cameras

There is an older, more elegant method in the code (in `bounce.py`): project the
ball from both cameras onto the ground and watch the two projections come
together at the moment of contact. It is mathematically nice, but it only works
when the cameras are high enough. These phones sit low, about 5 feet up. When the
ball is in the air and low cameras look at it, converting that high pixel to a
ground position produces wild, meaningless numbers (hundreds or thousands of
feet on a 44 foot court). So the comparison never settles down.

The reliable approach on low cameras is simpler: in each single video, a bounce
looks like the ball falling and then rising - a little V shape. That V is easy to
see in one camera and does not depend on the other camera at all. We only use the
homography for the one thing it is good at here: placing the contact point (which
is on the ground) onto the court to decide IN or OUT. This is why the web app
uses the single-view method.

---

## The two ways to run it

- The web app is the main way. Start it with `python run_web.py` and open the
  browser. Everything above (upload, sync, calibrate, Calculate, Analyze) happens
  through the three pages.

- The offline pipeline is for scripts and experiments. `demo.py` and
  `src/pickleball_phase2/pipeline.py` run the whole chain on a clip pair and
  return the verdicts as data, without the browser. This path still uses the
  dual-camera separation method.

---

## How the repository is organized

Think of the repo as a few clear zones. Here is what each top-level thing is
for, in plain terms.

- `config.yaml` - the settings file. Paths to the models, the frame rate, and
  the detection thresholds all live here. When in doubt about a number the code
  uses, it came from here.
- Three entry points (the things you actually run):
  - `run_web.py` - starts the web app. This is the normal way to use the project.
  - `demo.py` - runs the whole pipeline over the demo clips from the command
    line, without a browser.
  - `gridtracknet_test_improved_safe.py` - takes one clip and writes an
    annotated video with a ball trail plus a CSV of ball positions. The
    Calculate button runs this once per camera.
- `src/pickleball_phase2/` - the brain. Pure Python that does the vision and
  geometry (finding the court, tracking the ball, detecting bounces, calling
  IN/OUT). It has no idea a web app exists, which means you can also use it from
  a plain script.
- `webapp/` - the Flask app wrapped around the brain. This is everything you see
  and click in the browser.
- `models/` - the trained model files (the court-keypoint model and the ball
  tracker).
- `calib/intrinsics/` - one lens-calibration file per camera (its focal length
  and distortion).
- `Footage/` - the original demo videos, and the `_tracked.mp4` and
  `_tracked.csv` files the Calculate step produces.
- `data/` - runtime scratch space: each browser session's uploads live in
  `data/webapp/<session>/`, and `data/cache/` holds saved sync and tracking
  results so repeat runs are fast.
- `tests/` - automated checks that run on synthetic data, so they work even with
  no videos present.
- `GridTrackNet-essentials/` and `ball_tracker_handoff/` - support bundles for
  the ball tracker: the first is the training tooling, the second is the
  self-contained inference code that was copied into `src/`.
- `README.md`, `Explanation.md`, `CLAUDE.md` - the docs (CLAUDE.md is guidance
  for the AI agents that work on this repo).

---

## How the code is structured

The code is built in two layers, and keeping them separate is the main design
idea.

1. The algorithm package (`src/pickleball_phase2/`) is a plain Python library.
   Each file does one job: `court_model.py` (the court and the IN/OUT rule),
   `calibration.py` (pixels to court feet), `sync.py` (clap alignment),
   `tracking.py` (follow the ball), `bounce.py` and `single_view_bounce.py`
   (find bounces), `fusion.py` (combine the two cameras), and `pipeline.py`
   (run all of it in order). Nothing here knows about HTTP, sessions, or HTML.

2. The web app (`webapp/`) is a thin shell around that library, and it is split
   by responsibility:
   - `routes/` - the URL handlers. When the browser calls an address like
     `/api/analysis/run`, a small function in `routes/` receives it. These stay
     thin: they read the request, call a worker, and return the answer.
   - `services/` - the workers, and the only place allowed to import the
     algorithm package. A route hands work to a service; the service calls the
     brain and shapes the result back into simple data. For example,
     `analysis_service.py` reads the tracked CSVs, runs bounce detection, and
     returns the bounce list.
   - `templates/` - the HTML pages (Jinja), one per page plus shared partials.
   - `static/` - the browser-side code: the CSS and the JavaScript that make the
     pages interactive (dragging keypoints, playing the three videos in sync,
     drawing the top-view court).
   - `bootstrap.py`, `config.py`, `__init__.py` - the wiring that puts `src/` on
     the path, loads `config.yaml`, and registers the routes.

The flow of a single click looks like this: browser JavaScript (`static/js`)
calls an endpoint -> a handler in `routes/` receives it -> it asks a worker in
`services/` -> the worker calls the algorithm package in `src/` -> the answer
travels back up as JSON -> the JavaScript updates the page. The rule "only
services import the brain" is what keeps the web layer and the vision code from
tangling together.

Where things get stored while this runs: uploaded clips are saved by role
(cam1, cam2, cam3) under `data/webapp/<session>/`, the Calculate step writes the
tracked videos and CSVs into `Footage/`, and every path and threshold traces
back to `config.yaml`.

---

## Where the main pieces live

If you want to change one part, this is where to look:

- The court definition and the IN/OUT rule: `src/pickleball_phase2/court_model.py`.
- How pixels become court feet (calibration): `src/pickleball_phase2/calibration.py`
  and, in the web app, `webapp/services/calibration_service.py` plus
  `webapp/services/court_detection.py`.
- Ball tracking: `src/pickleball_phase2/tracking.py` (and the vendored ONNX
  decoder in `ball_tracker_onnx.py`), plus the standalone
  `gridtracknet_test_improved_safe.py` that the Calculate button runs.
- Time alignment (clap sync): `src/pickleball_phase2/sync.py` and
  `webapp/services/sync_service.py`.
- Bounce detection: single-view in `src/pickleball_phase2/single_view_bounce.py`
  (used by the web app), dual-camera in `src/pickleball_phase2/bounce.py`
  (used by the offline pipeline).
- The Page 3 Calculate and Analyze logic: `webapp/routes/analysis.py` and
  `webapp/services/analysis_service.py`.
- Settings and thresholds: `config.yaml`.

---

## A small glossary

- Keypoints: the 12 court line-intersection points the model looks for. They pin
  the video to the real court.
- Homography: the formula that turns a video pixel into a court position in feet.
  It is accurate for points on the ground and unreliable for points high in the
  air.
- Sync offset: how many frames one camera is ahead of or behind another, found
  from the clap.
- ROI (region of interest): the near half of the court. IN/OUT verdicts are only
  made for bounces that land inside it.
- Kitchen: the non-volley zone next to the net. CAM3 optionally counts bounces
  there; it is not part of the IN/OUT call.
- Tracked files: the `<role>_tracked.mp4` (annotated video) and
  `<role>_tracked.csv` (ball positions) that the Calculate step writes into the
  Footage folder.
