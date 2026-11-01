### `session_20261101_1106_metrics.csv`

4204 frames over 106.5 s: **39.5 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 17.5 | 21.0 |
|  | face_fx | face damage + hat drawing (render) | 1.5 | 3.0 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 20.8 | 22.9 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 33.4 | 55.9 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.3 | 0.4 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.3 | 0.5 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 23.6 | 32.1 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 7.4 | 8.8 |
|  | render | game layer, HUD, effects (includes face_fx) | 3.4 | 5.4 |
| **End to end** | latency | camera capture to frame on screen | **35.8** | **45.9** |

Per frame: dropped 0.00 frames (p95 0), hand_lag 1.41 frames (p95 3).

