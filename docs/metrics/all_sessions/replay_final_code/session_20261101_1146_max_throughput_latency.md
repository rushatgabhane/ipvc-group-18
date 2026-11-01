### `session_20261101_1146_metrics.csv`

1918 frames over 45.8 s: **41.8 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 13.3 | 19.1 |
|  | face_fx | face damage + hat drawing (render) | 1.1 | 3.2 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 20.2 | 21.2 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 30.2 | 59.7 |
|  | postprocess | landmarks to arrays, mask copy | 0.3 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.3 |
|  | tracking | assignment, face verification, re-identification | 0.1 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.4 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 22.4 | 30.8 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 7.2 | 7.9 |
|  | render | game layer, HUD, effects (includes face_fx) | 2.7 | 5.1 |
| **End to end** | latency | camera capture to frame on screen | **33.5** | **43.5** |

Per frame: dropped 0.01 frames (p95 0), hand_lag 1.18 frames (p95 3).

