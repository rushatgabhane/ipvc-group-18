# HIGH NOON — Duck & Cover

A two-player webcam duel shooter for the IPCV 2026-1A group project (University of Twente).
Players aim with an outstretched arm, fire with a recoil flick, and duck behind cover to dodge.
The only controller is computer vision on a single webcam.

## Team and task ownership

| Task | Owner | Main code |
|---|---|---|
| T1 Face tracking & augmented effects | _TBD_ | `highnoon/vision/perception.py` (face), `highnoon/render/` (face effects) |
| T2 Body pose & motion tracking | _TBD_ | `highnoon/vision/perception.py` (pose), `highnoon/vision/motion.py`, `highnoon/core/filters.py` |
| T3 Multiplayer identity tracking | _TBD_ | `highnoon/tracking/identity.py` |
| T4 Player interaction & game control | _TBD_ | `highnoon/game/` (to be added) |
| T5 Scene processing & integration | _TBD_ | `highnoon/app.py`, `highnoon/render/`, `highnoon/core/` |

## Setup (tested on macOS, Apple Silicon)

Requires **Python 3.12**. mediapipe has no wheels for 3.14 yet.

```bash
# with uv (recommended)
uv venv --python 3.12 .venv
uv pip install -r requirements-dev.txt

# or with plain pip
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

# download model weights (~19 MB, verified by sha256)
.venv/bin/python tools/download_models.py
```

## Run

```bash
.venv/bin/python -m highnoon                 # webcam 0, 1280x720
.venv/bin/python -m highnoon --source 1      # another camera
.venv/bin/python -m highnoon --help          # all options
```

Keys: `q`/`esc` quit, `d` toggle debug overlay, `b` toggle background replacement, `f` fullscreen.

On first run macOS will ask for camera permission for your terminal app.

## Testing and benchmarking

```bash
.venv/bin/python -m pytest                    # unit tests (no camera needed)
.venv/bin/ruff check . && .venv/bin/ruff format --check .

# record a test clip, then replay it through the full pipeline (repeatable)
.venv/bin/python tools/record.py clips/crossing.mp4 --seconds 20
.venv/bin/python -m highnoon --source clips/crossing.mp4 --metrics-csv results/crossing.csv
.venv/bin/python -m highnoon --source clips/crossing.mp4 --headless --fast   # max throughput

# compare model/threading variants in isolation
.venv/bin/python tools/bench_models.py --source clips/crossing.mp4
```

`--metrics-csv` writes one row per frame with every stage's time, FPS and capture-to-screen
latency, ready for the plots in the report. `docs/PERFORMANCE.md` records all measurements so
far and the decisions taken from them.

## Architecture

```
 capture thread          perception thread                  main thread
 ──────────────          ─────────────────                  ───────────
 webcam ──► newest  ──►  BGR->RGB ─┬─► pose (+seg mask) ─┐   T3 identity  ──► T2 filtering
            frame                  └─► face  (parallel) ─┴─► ──► T4 game ──► T5 render ──► display
            (1 slot)                                         (newest result only)
```

- **Three threads, newest-frame-only hand-off.** When any stage is slow, frames are dropped,
  never queued, so latency stays bounded. Rendering frame N overlaps inference of frame N+1.
- **`highnoon/contracts.py`** defines the data passed between tasks (`Frame`, `Perception`,
  `PoseObservation`, `FaceObservation`, `Player`). Each task module depends only on these, so
  it can be unit-tested with synthetic inputs (see `tests/test_identity.py`).
- **`highnoon/config.py`** holds every tunable. Each default records the measurement it came from.
- All coordinates are pixels in the **mirrored** frame, so what you see is what the game uses.

## File structure

```
highnoon/
  __main__.py          CLI entry point
  app.py               main loop, thread orchestration (T5)
  config.py            all parameters
  contracts.py         data passed between tasks
  core/
    capture.py         threaded camera / video file sources
    filters.py         vectorised One Euro filter
    profiler.py        per-stage timing, FPS, latency, CSV export
  vision/
    perception.py      MediaPipe pose + face inference, parallel (T1/T2)
    motion.py          per-player keypoint smoothing (T2)
  tracking/
    identity.py        persistent player ids, face-to-player assignment (T3)
  render/
    renderer.py        background compositing, overlays, HUD (T5)
    display.py         pygame / OpenCV window output
tools/
  download_models.py   fetch + verify model weights
  record.py            record raw webcam clips for repeatable tests
  bench_models.py      model / threading benchmark
tests/                 pytest unit tests
docs/PERFORMANCE.md    measurements and design decisions
```

## Dependencies

Pinned in `requirements.txt`: mediapipe 0.10.35 (1.0.x crashes on macOS), opencv-contrib-python
5.0.0, numpy, scipy, pygame-ce. Model weights are not in the repo. `tools/download_models.py`
fetches them from Google's MediaPipe model storage and checks their sha256.

## AI usage

This foundation was scaffolded with Claude Code (Anthropic). The design choices and every
number in `docs/PERFORMANCE.md` were measured on our hardware. The full AI and external-tools
statement will be in the report appendix.
