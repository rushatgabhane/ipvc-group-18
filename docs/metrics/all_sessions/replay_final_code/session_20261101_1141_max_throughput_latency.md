### `session_20261101_1141_metrics.csv`

2198 frames over 55.6 s: **39.5 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 14.4 | 21.2 |
|  | face_fx | face damage + hat drawing (render) | 0.4 | 1.0 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 22.0 | 29.8 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 26.2 | 48.9 |
|  | postprocess | landmarks to arrays, mask copy | 0.3 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.3 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.1 | 0.4 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 23.8 | 31.6 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 7.2 | 8.3 |
|  | render | game layer, HUD, effects (includes face_fx) | 2.0 | 3.9 |
| **End to end** | latency | camera capture to frame on screen | **34.3** | **43.7** |

Per frame: dropped 0.01 frames (p95 0), hand_lag 0.74 frames (p95 2).

