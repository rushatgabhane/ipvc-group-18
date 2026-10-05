# Performance log

Every design decision below was taken from a measurement. Re-run the numbers on your
machine with the commands given, and add a row when you change something.

**Test machine:** MacBook Pro M1 Pro, macOS 26, Python 3.12, mediapipe 0.10.35,
OpenCV 5.0.0, 1280x720 webcam at 30 fps. CPU inference.
**Caveat:** all numbers so far have **one** person in view. Pose cost with two people still
has to be measured (see TODO at the bottom).

## Frame budget

At 30 fps the camera delivers a frame every **33.3 ms**. A stage only limits throughput if it
is slower than that, because inference and rendering run on different threads (pipelined).
Latency is the sum of the stages one frame passes through.

| Stage (thread)              | Mean    | p95     | Notes |
|-----------------------------|---------|---------|-------|
| capture (capture)           | –       | –       | keeps only the newest frame; no queueing |
| BGR->RGB convert (perception) | 0.3 ms | 0.3 ms | |
| pose lite, num_poses=2 (perception) | 26.0 ms | 26.3 ms | **critical path** |
| face, num_faces=2 (perception, parallel) | 11.2 ms | 11.4 ms | hidden behind pose |
| postprocess (perception)    | 0.4 ms  | 0.4 ms  | landmarks -> numpy, mask copy |
| identity tracking (main)    | 0.1 ms  | 0.1 ms  | Hungarian, 2x2 |
| One Euro filtering (main)   | <0.1 ms | <0.1 ms | vectorised over 33 keypoints |
| game logic (main)           | 0.02 ms | 0.02 ms | actions, ray casts, rules |
| render (main)               | 1.5 ms  | 1.6 ms  | background composite + game layer + HUD |
| display, pyglet (main)      | 2.4 ms  | 2.5 ms  | |
| **capture -> on screen**    | **30.7 ms** | **31.3 ms** | 300 frames, 0 dropped |
| **throughput**              | **30.0 fps** | | camera-bound; 36 fps unpaced on a clip |

Reproduce: `python -m highnoon --max-frames 1800 --metrics-csv results/soak.csv`

## Decisions and the measurements behind them

### 1. Pose lite instead of full
| Model | Mean | p95 |
|---|---|---|
| pose_landmarker_lite | 25.4 ms | 26.0 ms |
| pose_landmarker_full | 32.9 ms | 33.5 ms |

Full uses the entire 33 ms budget by itself. Lite leaves ~7 ms headroom. Revisit if lite's
keypoints turn out too noisy for aiming (`--pose-model full`).

### 2. Pose and face in parallel
| | Per frame |
|---|---|
| sequential | 36.7 ms (over budget) |
| parallel (2 threads) | **25.7 ms** |

MediaPipe releases the GIL inside its C++ graph, so plain Python threads give real parallelism.

### 3. No input downscaling
Passing 0.5x frames changed nothing (lite 25.4 -> 25.3 ms, face 11.3 -> 11.1 ms): both
models resize to their own input size internally. Full 720p is kept because faces at 2.5-3 m
are small and lose landmark accuracy when downscaled.

### 4. Segmentation mask from the pose model
`output_segmentation_masks=True` costs **+0.3 ms** (25.3 -> 25.6 ms). No separate selfie
segmentation model is needed for background replacement.

### 5. Blending with OpenCV, not numpy
| Method (720p, float mask) | Time |
|---|---|
| numpy `f*m + bg*(1-m)` | 12.7 ms |
| uint8 cv2.multiply/add | 2.2 ms |
| **cv2.blendLinear** | **0.35 ms** |

### 6. pyglet (OpenGL) for display instead of cv2.imshow
| Backend | Per frame |
|---|---|
| cv2.imshow + waitKey(1) | 16.3 ms (waitKey alone ~15 ms, Cocoa event pump; same in every window mode, and with pollKey) |
| pygame blit+flip, vsync on | 16.5 ms |
| pygame blit+flip, vsync off | 2.9 ms |
| **pyglet texture upload + flip, vsync off** | **2.2 ms** |

Measured in the app: latency went from **44.5 ms (imshow) to ~30 ms**. pygame was used first,
but it ships its own SDL2, which clashes with the SDL2 bundled in every OpenCV wheel (macOS
warns "Class SDL... is implemented in both"). pyglet has no SDL and is slightly faster.
Texture upload is ~1.4 ms in every variant tried (pyglet ImageData, raw glTexSubImage2D,
BGR or native BGRA): that is the cost of moving 2.7 MB to the GPU. Possible tearing is the
trade-off (`--vsync` removes it at the cost of up to one refresh).

### 7. Hand tracking on its own thread
The fist trigger needs MediaPipe's hand landmarker. Its cost scales with the number of hands:

| num_hands | Per frame |
|---|---|
| 1 | 15.4 ms |
| 2 | 31.3 ms (slower than pose) |
| 4 | 49.5 ms |

Running it inline would make it the critical path (about 20 fps with 4 hands). Instead it runs on a
separate thread. Each frame is submitted without waiting, and the pipeline reads the newest
finished hand result. Measured in the app: hand 17.9 ms per run, results 0 frames behind
(`hand_lag`), still 30 fps with 0 dropped frames and 29.4 ms latency. `max_hands = 2` (one trigger
hand per player) keeps the cost bounded.

### 8. Newest-frame-only capture
The capture thread overwrites a single slot instead of filling a queue. If any stage gets slow,
frames are dropped (logged as `dropped`) rather than delayed, so latency cannot accumulate.

## Platform issues found

- **GPU delegate:** aborts the process on macOS at graph creation, so macOS uses CPU. MediaPipe
  documents GPU support for Python tasks as "limited to Ubuntu". `--delegate auto` (the default)
  therefore tests the GPU in a subprocess on Windows/Linux and only uses it if that succeeds. On
  this Mac the probe aborted as expected; the game fell back to CPU without crashing, and later
  runs answered from the cache. **Not yet measured on a Windows machine.**
- **mediapipe 1.0.1:** aborts on macOS even with the CPU delegate
  (`graph_service.h: Check failed: service_ Service is unavailable`), so it is pinned to 0.10.35.
- **Python 3.14:** no mediapipe wheels, so the project uses 3.12.
- **Duplicate SDL2 with pygame:** every OpenCV wheel (4.x, 5.x, headless) bundles SDL2 through
  FFmpeg. Using pygame as well loads two SDL copies (`Class SDL... is implemented in both`),
  which is one reason the display moved to pyglet.
- **MediaPipe telemetry:** 0.10.35 tries to upload usage logs ("clearcut uploader") and
  logs an error when it fails. No opt-out was found; it does not affect results.

## TODO (measure next)

- [ ] **Two players in frame**: pose cost per extra person (the landmark model runs once per body).
- [ ] The exhibition laptop, if different, plus behaviour on battery power.
- [ ] Glass-to-glass latency (camera sensor + display) with a phone slow-motion video of a
      flashing timer. The `latency` metric only covers capture -> on screen in software.
- [ ] The cost of real game rendering (sprites, particles) once T4/T5 add it.
