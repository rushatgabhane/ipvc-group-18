### `session_20261101_1148_metrics.csv`

56 frames over 2.3 s: **24.1 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 19.9 | 24.9 |
|  | face_fx | face damage + hat drawing (render) | 0.0 | 0.0 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 39.9 | 44.5 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 37.7 | 63.1 |
|  | postprocess | landmarks to arrays, mask copy | 0.5 | 0.6 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.0 | 0.0 |
| T3 Identity | face_id | SFace embedding per face (perception thread, after face) | 14.4 | 20.7 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.2 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.0 | 0.0 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.6 | 0.8 |
|  | inference | pose and face paths in parallel (the slower one) | 42.0 | 48.8 |
|  | wait | result waiting for the main loop | 1.0 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 29.4 | 9.1 |
|  | render | game layer, HUD, effects (includes face_fx) | 2.1 | 2.3 |
|  | display | OpenGL upload + flip | 5.2 | 4.8 |
| **End to end** | latency | camera capture to frame on screen | **97.1** | **102.9** |

Per frame: dropped 0.95 frames (p95 1), hand_lag 0.96 frames (p95 3).

