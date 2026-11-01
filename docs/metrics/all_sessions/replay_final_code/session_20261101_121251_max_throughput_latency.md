### `session_20261101_121251_metrics.csv`

799 frames over 22.7 s: **35.2 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 20.3 | 26.8 |
|  | face_fx | face damage + hat drawing (render) | 1.8 | 4.4 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 22.7 | 28.4 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 52.1 | 73.5 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.6 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.3 | 0.6 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.5 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.3 | 0.7 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.3 | 0.4 |
|  | inference | pose and face paths in parallel (the slower one) | 26.4 | 38.6 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 8.8 | 11.3 |
|  | render | game layer, HUD, effects (includes face_fx) | 4.2 | 8.2 |
| **End to end** | latency | camera capture to frame on screen | **41.2** | **59.4** |

Per frame: dropped 0.03 frames (p95 0), hand_lag 2.25 frames (p95 4).

