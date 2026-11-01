### `session_20261101_1053_metrics.csv`

2055 frames over 51.5 s: **39.9 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 17.1 | 21.8 |
|  | face_fx | face damage + hat drawing (render) | 1.3 | 2.4 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 20.6 | 22.4 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 25.1 | 49.4 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.3 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.4 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 23.5 | 31.1 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 7.5 | 8.7 |
|  | render | game layer, HUD, effects (includes face_fx) | 2.9 | 4.8 |
| **End to end** | latency | camera capture to frame on screen | **35.2** | **44.8** |

Per frame: dropped 0.01 frames (p95 0), hand_lag 0.73 frames (p95 3).

