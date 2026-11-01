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
| render (main)               | 4.8 ms  | 5.7 ms  | backdrop composite, wall, glow layer, sprites, HUD, vignette |
| display, pyglet (main)      | 2.4 ms  | 2.5 ms  | |
| **capture -> on screen**    | **33.4 ms** | **35.6 ms** | 300 frames, 0 dropped |
| **throughput**              | **30.0 fps** | | camera-bound; 36 fps unpaced on a clip |

Reproduce: `python -m highnoon --max-frames 1800 --metrics-csv results/soak.csv`

## Decisions and the measurements behind them

### 1. Pose lite instead of full
| Model | Mean | p95 |
|---|---|---|
| pose_landmarker_lite | 25.4 ms | 26.0 ms |
| pose_landmarker_full | 32.9 ms | 33.5 ms |

Full uses the entire 33 ms budget by itself. Lite leaves ~7 ms headroom. Revisit if lite's
keypoints turn out too noisy for aiming (the full model is no longer downloaded or selectable).

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

### 8. Polished rendering within budget
The visual pass (Western backdrop, plank wall, TrueType HUD, glow, particles, vignette, shake)
raised render time from 1.5 to 4.8 ms. It runs on the main thread in parallel with inference, so
throughput is unchanged (30 fps) and latency rose about 3 ms. The rules that keep it cheap:
- Backdrop, wall, shadow ramp and vignette are generated once per resolution.
- Text is rendered once per distinct string with Pillow and cached as sprites (LRU). A HUD blit
  costs under 0.1 ms; Pillow rendering every frame would cost milliseconds.
- All glow (lasers, tracers, muzzle flashes) goes into one layer applied with a single `cv2.add`.
- Alpha blending only touches each sprite's own region, never the full frame.
- The flying powerup (token sprite built once, rotated with `warpAffine` on its ~60 px box, plus a
  3-circle trail, glow and label) costs 0.56 ms median / 0.64 ms p95 per frame, and only while one
  is on screen. Its position is a closed-form parabola, so the game logic costs nothing measurable.

### 8b. Render cleanup pass (measured on a synthetic duel frame, M1 Pro)
Profiling the game layer showed the translucent HUD panels were the biggest cost (0.8 ms each, 3 per
frame) because they blended in float numpy every frame. Changes and results:
- **Sprites carry precomputed blend weights** (`assets.Sprite`) and blit with `cv2.blendLinear`:
  11 us per HUD-sized blit instead of 129 us (float numpy).
- **Panels are cached sprites** per size instead of being rebuilt and blended in numpy each frame.
- **Vignette baked into the backdrop and wall** once, instead of a full-frame multiply per frame
  (0.68 ms saved).
- **Screen shake moved to the GPU**: the display offsets and slightly zooms the texture instead of
  `warpAffine` on the whole frame.
- **Rotated rocket/grenade/powerup sprites are cached** in 4 degree steps.
- **Big announcement texts are rendered once at startup** (BOOM!, GRENADE! DUCK!, winner names,
  countdowns), so the first explosion no longer stalls a frame (39 ms -> 11 ms worst frame).
- **Hit testing is vectorised**: every shot, rocket step and the whole 60-step rocket arc preview are
  tested against the wall, both hitboxes and the powerup in one numpy batch (segments vs circles and
  polygons). Game update: 0.38 ms median with a rocket preview active.

| Scenario (render per frame) | median | p95 | worst |
|---|---|---|---|
| Duel, bullets only, before | 6.2 ms | 7.0 ms | |
| Duel, bullets only, after | 2.4 ms | 3.5 ms | 5.0 ms |
| Rocket in flight + explosion | 2.5 ms | 4.8 ms | 7.8 ms |
| Grenade + explosion + round end | 3.0 ms | 5.4 ms | 10.9 ms |

On `clips/smoke.mp4` the full pipeline render stage went from 7.4 ms to 3.1 ms mean.

Removed to keep the code small: the flick trigger, the OpenCV window backend (16 ms/frame vs 2.3 ms
for pyglet), and the benchmark-only switches (`--sequential`, `--pose-model full`, `--no-face`,
`--no-seg`, inline hands, `tools/bench_models.py`). The measurements they produced are kept above.

### 8c. Hand tracking fixed from a recorded two-player session
A 149 s two-player session was recorded (`--record`, `--signals-csv`) and replayed with hand
detection on every frame. Findings: the hand model saw 2 hands in most frames, its limit with
`max_hands = 2`, so with 4 hands in view P2's shooting hand was often not tracked (63% of aiming
frames, gaps up to 7.7 s). Open hands clustered at 1.6-1.8 openness with many at 1.5-1.6, just under
the old 1.55 "open" threshold, and the 0.15 s open-hand hold rejected 53 fists.

| Replay of the session | P1 tracked | P2 tracked | P1 fired | P2 fired | hand model |
|---|---|---|---|---|---|
| max_hands 2, open > 1.55, open hold 0.15 s | 92% | 63% | 46 | 39 | 30 ms |
| max_hands 4, open > 1.45, no open hold | 96% | 81% | 67 | 66 | 50 ms |

Real-time replay with the new settings: 29.9 fps, 0 dropped frames, render 3.3 ms, hand results
2.1 frames old on average (was 1.0). Rapid-fire spam is handled by the cooldown and recoil spread.

### 9. Newest-frame-only capture
The capture thread overwrites a single slot instead of filling a queue. If any stage gets slow,
frames are dropped (logged as `dropped`) rather than delayed, so latency cannot accumulate.

### 10. Faces looked for in head crops, not the whole frame (T1)
Live play showed the hat and face damage missing most of the time. On a recorded two-player session
(`clips/session_20261101_1041.mp4`, players about 2 m away) the face landmarker found **0 faces** on a
full 1280x720 frame and **1 face in each half** of the same frame. Its detector is a short-range model
that shrinks the input to 128x128, so at play distance a face is only a few pixels wide. Faces are now
looked for in a square around each body's head, from the previous frame's pose (`head_box()` in
`vision/perception.py`).

| Same 900 frames, two players | Faces found for visible players | Face step (mean / p95) |
|---|---|---|
| Whole frame, VIDEO mode (before) | 33.5% | 10.9 / 19.0 ms |
| **Head crops, IMAGE mode (after)** | **97.4%** | 22.0 / 23.1 ms |

The face step runs in parallel with pose inference (39 ms), so the extra cost does not lengthen the
frame. Face ID (SFace) needs a face, so identity checks were starved too before this fix.

### 11. Hat anchor smoothed instead of every landmark (T1)
The hat used to be placed from all 478 landmarks, each smoothed by its own One Euro filter with its
own noisy speed, which bent the face shape and made the tilt wobble; a missed face reset the filter,
so the hat jumped. Now the hat's centre, tilt and width are measured once per frame and only those are
smoothed (`vision/head.py`), and the filters survive a missed face for 0.5 s. Unit test on a noisy
synthetic face: the smoothed centre has less than half the jitter of the raw one. Jitter and lag on
real footage: see section 14 (tilt jitter down 91%, size 93%).

### 12. Identity takeover fixed (T3)
In a live test, a third person stepped into Player 1's spot while Player 1's body was missed for
0.2 s, and got Player 1's ID: position tracking bridged gaps of up to `track_gap_s = 5` s, and the face
check only released the ID after 0.3 s of mismatches. Changes: position alone only bridges 0.15 s
(after that the face must confirm), the other player's face releases an ID at once, and two clearly
different faces in a row (below `face_reject`) release it within about 0.07 s. Replayed with identical
detections through both versions:

| Recording | Tracker | Time an ID showed the wrong face | Episodes | Longest | Body in view without an ID |
|---|---|---|---|---|---|
| `session_20261101_1053` (third person walks in) | old | 1.60 s | 13 | 0.33 s | 428 player-frames |
| | **new** | **0.27 s** | 8 | **0.03 s** | 522 player-frames |
| `session_20261101_1041` (two players) | old | 0 s | 0 | – | 8 |
| | **new** | 0 s | 0 | – | 20 |

"Wrong face" means the face on an ID scored below 0.2 against its owner. The cost is a few more frames
in which a returning player waits to be recognised (12 extra player-frames, 0.4 s, in a 105 s session).
Live re-test (`session_20261101_1106`, 183 s, 3 games): 5 wrong-face frames in about 7,100 face checks,
longest 0.1 s, and no IDs handed out after enrolment.

### 13. Live sessions, two players (1 Nov 2026)
Whole-session averages from `--metrics-csv` (`tools/task_latency.py`). The summary the game prints at
exit only covers the last 120 frames, which were idle; an earlier version of this table quoted those
(30 fps, 41 ms for `1053`) by mistake.

| Session | Length | FPS | Latency mean / p95 | Pose | Face | Face ID | Hands | Render | Dropped |
|---|---|---|---|---|---|---|---|---|---|
| `1041` (whole-frame face search) | 125 s | 25.1 | 70.5 / 87.4 ms | 37.9 ms | 7.8 ms | 3.0 ms | 51.7 ms | 2.2 ms | 0.21 fr |
| `1053` (head crops) | 86 s | 24.2 | 70.9 / 91.8 ms | 36.9 ms | 21.8 ms | 15.3 ms | 31.3 ms | 3.7 ms | 0.26 fr |
| `1106` (identity fix) | 181 s | 23.3 | 75.5 / 93.5 ms | 38.7 ms | 22.4 ms | 16.2 ms | 41.6 ms | 4.2 ms | 0.30 fr |

With two players the game runs at **23-25 fps with 71-76 ms from capture to screen**, below the
camera's 30 fps. Pose (37-39 ms) is over the 33 ms budget on its own. Since the head crops find faces,
face detection plus face ID (22 + 16 = 38 ms, one after the other) is as slow as pose, so both
inference paths are now critical. The hand model (31-52 ms, asynchronous) competes with both for the
CPU. Face and face ID were cheap in `1041` only because faces were mostly not found.

### 14. Per-task evaluation of a recorded session (`tools/evaluate_clip.py`)
`session_20261101_1053` (69 s, two players, a third person walking in), replayed frame by frame with
the identity fix. Full output: `results/session_20261101_1053_eval.md`. Replayed offline every frame is
processed (no frame dropping), so module times are a little higher than live.

**Module cost (ms, mean / p95):** pose 37.7 / 47.6, face (head crops) 21.7 / 31.3, face ID 16.1 / 27.1,
hands (own thread) 32.4 / 65.3, tracking 0.26 / 0.35, filtering 0.25 / 0.36, face effects 2.0 / 3.1.
Pose and face run in parallel, so inference is 40.1 / 56.6.

**Detection:** a face was found for 96.8% (P1) and 94.7% (P2) of the frames the player was visible.
Only 55% of the 33 pose keypoints are above confidence 0.5, because the legs and hips are below the
picture or behind the wall; the controls use shoulders, arms and hands, which are kept.

**Smoothing (jitter = per-frame noise after removing smooth motion; lag = delay of the filtered
signal):**

| Signal | Raw jitter | Smoothed | Reduction | Lag |
|---|---|---|---|---|
| Shoulder centre (P1 / P2) | 1.6 / 1.2 px | 0.6 / 0.4 px | 64-69% | 0-4 ms |
| Right wrist (P1 / P2) | 5.4-6.6 / 2.5 px | 3.1-3.8 / 1.3 px | 43-49% | 0 ms |
| Hat centre | 0.4-0.7 px | 0.07-0.18 px | 75-84% | 0-16 ms |
| Hat tilt | 0.30-0.34° | 0.03° | 91-92% | 26-27 ms |
| Hat width | 0.5-0.6 px | 0.03-0.04 px | 93-94% | 59-63 ms |

The wrist is smoothed least on purpose: One Euro raises its cutoff with speed, so a fast aim is not
delayed. The hat's size is smoothed hardest, because it changes slowly and its jitter is most visible
as "breathing"; its 60 ms lag only shows when walking towards the camera.

**Identity:** against each player's faces from their first 4 s, the ID agreed with the face in 100%
(P1, 1844 checks) and 99.0% (P2, 1245) of frames, with no disagreement lasting 0.3 s or more. 24 gaps
(12 under 0.5 s, detection dropouts); median re-identification wait 0.17 s, p95 1.75 s. The 10.7 s
gap is P2 leaving the picture. Face match scores: same person mean 0.81 (5th percentile 0.62), best
other person mean 0.09 (95th percentile 0.19), so `face_accept = 0.36` rejects 0.8% of real matches
and accepts 0.1% of other people, and `face_reject = 0.2` sits just above 95% of other people.

**Position (metres, from the torso):** median distance P1 1.59 m, P2 1.85 m; distance jitter 0.5 cm.
The players' real distances were not measured in this session, so the error is still to be checked at
taped distances (recording protocol, step 5). Known limit: the estimate assumes a 0.50 m torso, so it
is off by the same fraction as a player's torso differs (about ±10% between adults), and leaning
towards the camera makes the torso look shorter and the player look further away.

### 15. Latency per task (`tools/task_latency.py`)
Stage times from the longest session, `session_20261101_1106` (181 s, two players, 3 games, final
code). Pose runs on its own thread, face detection and face ID one after the other on the perception
thread, hands asynchronously on a third, and the game, filtering and rendering on the main thread
while the next frame is inferred. Stage times therefore do not add up to the latency: inference is
the slower of the pose and face paths, and latency runs from camera capture to the frame on screen.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops | 22.4 | 25.9 |
| | face_fx | face damage + hat drawing | 2.3 | 4.3 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask | 38.7 | 41.2 |
| | hand | hand landmarks, 4 hands (asynchronous) | 41.6 | 63.7 |
| | postprocess | landmarks to arrays, mask copy | 0.4 | 0.5 |
| | filter | One Euro filtering (keypoints, aim, face, hat), control signals | 0.4 | 0.7 |
| T3 Identity | face_id | SFace embedding per face | 16.2 | 20.0 |
| | tracking | assignment, face verification, re-identification | 0.3 | 0.4 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.3 | 0.5 |
| T5 Integration & scene | convert | BGR to RGB for the models | 1.0 | 1.6 |
| | inference | pose and face paths in parallel (the slower one) | 41.3 | 45.8 |
| | camera | background replacement + wall onto the camera frame | 9.2 | 11.1 |
| | render | game layer, HUD, effects (includes face_fx) | 4.2 | 6.3 |
| | display | OpenGL upload + flip | 2.5 | 3.0 |
| **End to end** | latency | camera capture to frame on screen | **75.5** | **93.5** |
| | FPS | | **23.3** | |

Dropped frames 0.30 per frame shown; hand results on average 0.9 frames old (p95 3).

Per task, the real-time cost is: T1 25 ms (face 22.4 + effects 2.3), T2 39 ms on the pose thread plus
the asynchronous hand model, T3 16.5 ms (face ID 16.2 + tracking 0.3), T4 0.3 ms, T5 17 ms on the main
thread (camera 9.2 + render 4.2 + display 2.5 + convert 1.0). The camera stage (background replacement)
costs more than the whole render pass and is a candidate for optimisation, along with the face path
(face ID every frame) and pose.

Reproduce: `python -m highnoon --metrics-csv results/session_metrics.csv`, play, then
`python tools/task_latency.py results/session_metrics.csv`.

## Report evaluation checklist

The project description (section 4.2.1) asks every task section of the report for **technical
evaluation, experiment results, measured frame rate or latency, accuracy, and significant failure
cases**, and every task owner to state what does not work reliably (3.2.6). Status per task:

| Task | Required evidence | Where it comes from | Status |
|---|---|---|---|
| T1 Face | ≥2 faces detected under movement, turning, distance | `evaluate_clip.py` detection table; section 10 | done on recordings |
| | Jitter raw vs smoothed and the lag it adds (hat) | `evaluate_clip.py` smoothing table | done on recordings |
| | Effects never switch players | identity tables (T3), section 12 | done |
| | Behaviour on short face loss | hat filter kept 0.5 s; face crops | done; show in video |
| | Cost | face 22 ms, face effects 2.3 ms (section 15) | done |
| T2 Pose | Pose cost with two people | sections 13-14 | done |
| | Keypoint jitter raw vs filtered, filter delay | `evaluate_clip.py` smoothing table | done on recordings |
| | Unreliable keypoints (confidence < 0.5) | `evaluate_clip.py` detection table | done |
| | Missing keypoints do not trigger actions | demo: cover a hand / arm; `games.csv` ignored fists | **to record** |
| | Duck and fist accuracy | `games.csv` over ~10 games, scripted clip with known ducks/shots | **to record** |
| T3 Identity | Consistency, switches, crossing, occlusion, re-entry | `evaluate_clip.py` identity tables, section 12 | done; add scripted clips |
| | Thresholds justified by score distributions | `evaluate_clip.py` face match scores | done |
| | Position in metres and its reliability | `tracking/spatial.py`, `evaluate_clip.py` position table | **measure error at taped 2 m / 3 m** |
| | Module cost | tracking 0.3 ms, face ID 16 ms (section 15) | done |
| T4 Game | Action recognition accuracy, cooldowns | `results/games.csv` | **play ~10 games** |
| | Simultaneous actions, fist-to-shot delay | scripted clip | **to record** |
| T5 Integration | End-to-end FPS and latency, two players; latency per task | sections 13, 15 | done |
| | Bottlenecks and what was done | sections 1-13 | done |
| | Module failures handled | `--fail-module hand` / `face` / `tracking` run | **to run and note** |
| | Glass-to-glass latency | phone slow-motion of the screen | **to measure** |
| All | Owners named in README and report | README "Team and task ownership" | **fill in names** |

### Recording protocol for the remaining clips
Record each with `--record clips/<name>.mp4 --identity-csv results/<name>_identity.csv`, then run
`tools/evaluate_clip.py` on it. Say the step out loud so the times can be checked in the video.

1. **crossing** (60 s): both players stand still 5 s, swap places 5 times at walking speed, then once
   with one passing behind the other.
2. **occlusion** (40 s): one player steps in front of the other for 2 s, 5 times.
3. **re-entry** (60 s): P2 leaves for 2 s, then 6 s, then 12 s, and comes back each time.
4. **stranger** (60 s): during a duel a third person walks in while P2 is away, stands in P2's spot,
   then leaves; then P2 returns.
5. **distance** (40 s): each player stands 5 s at taped 1.5 m, 2.0 m, 3.0 m from the camera (for the
   position error and face detection by distance).
6. **controls** (60 s): each player ducks 10 times and fires 10 times on a count, then both fire
   together 5 times.
7. **failure** (30 s each): `--fail-module hand`, `--fail-module face`, `--fail-module tracking:0.2`.

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

- [x] **Two players in frame**: see section 13.
- [ ] The exhibition laptop, if different, plus behaviour on battery power.
- [ ] Glass-to-glass latency (camera sensor + display) with a phone slow-motion video of a
      flashing timer. The `latency` metric only covers capture -> on screen in software.
- [x] The cost of real game rendering: 2.1-3.6 ms with two players (section 13).

## 8d. Wrist crops for the hand model (tested, not adopted)

Idea: the hand model's palm detector sees the whole 1280x720 frame shrunk to 192x192, so hands are
small. Cropping a square around each wrist (from the pose) should make hands bigger and easier to
find. Replayed `clips/hand_session.mp4` (149 s, two players) with synchronous hand detection; three
runs in parallel, so absolute times are inflated but comparable.

| Variant | Hand model time (median) | P1 tracked while aiming | P2 tracked | P1 fired | P2 fired |
|---|---|---|---|---|---|
| Full frame, 4 hands (current) | 40 ms | 96% | 81% | 67 | 66 |
| Wrist crops, side 2.5x forearm | 93 ms | 93% | 89% | 66 | 74 |
| Wrist crops, side 3.5x forearm | 93 ms | 89% | 87% | 65 | 77 |

Crops track the farther player better (+8 points) but cost 2.3x the time, because every crop runs
the palm detector again (up to 4 crops per frame, no tracking between frames). With the hand model
already running asynchronously at 40-60 ms, doubling it would double the hand lag. Kept full frame.
