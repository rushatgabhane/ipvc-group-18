### `session_20261101_122227_metrics.csv`

2069 frames over 53.3 s: **38.8 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 18.3 | 21.8 |
|  | face_fx | face damage + hat drawing (render) | 1.8 | 3.8 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 20.9 | 22.7 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 34.7 | 60.3 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.3 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.3 | 0.5 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 24.0 | 32.6 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 7.6 | 8.7 |
|  | render | game layer, HUD, effects (includes face_fx) | 3.6 | 5.8 |
| **End to end** | latency | camera capture to frame on screen | **36.6** | **46.5** |

Per frame: dropped 0.01 frames (p95 0), hand_lag 1.38 frames (p95 3).

