### `session_20261101_1143_metrics.csv`

1634 frames over 24.3 s: **67.2 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 2.8 | 9.1 |
|  | face_fx | face damage + hat drawing (render) | 0.0 | 0.4 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 13.9 | 20.7 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 19.1 | 37.5 |
|  | postprocess | landmarks to arrays, mask copy | 0.1 | 0.2 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.0 | 0.1 |
|  | tracking | assignment, face verification, re-identification | 0.0 | 0.1 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.0 | 0.1 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 13.9 | 20.8 |
|  | wait | result waiting for the main loop | 0.0 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 6.7 | 6.9 |
|  | render | game layer, HUD, effects (includes face_fx) | 1.6 | 2.4 |
| **End to end** | latency | camera capture to frame on screen | **22.7** | **30.6** |

Per frame: dropped 0.02 frames (p95 0), hand_lag 0.45 frames (p95 3).

