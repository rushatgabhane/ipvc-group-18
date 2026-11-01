### `session_20261101_121659_metrics.csv`

1912 frames over 53.5 s: **35.7 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 19.7 | 28.5 |
|  | face_fx | face damage + hat drawing (render) | 1.6 | 3.4 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 22.7 | 31.6 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 33.9 | 60.5 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.6 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.3 | 0.5 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.5 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.5 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.3 | 0.4 |
|  | inference | pose and face paths in parallel (the slower one) | 26.1 | 38.3 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 8.2 | 11.2 |
|  | render | game layer, HUD, effects (includes face_fx) | 3.3 | 6.0 |
| **End to end** | latency | camera capture to frame on screen | **39.1** | **55.6** |

Per frame: dropped 0.01 frames (p95 0), hand_lag 1.12 frames (p95 3).

