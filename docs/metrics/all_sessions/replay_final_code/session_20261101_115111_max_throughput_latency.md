### `session_20261101_115111_metrics.csv`

1393 frames over 34.5 s: **40.4 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 17.1 | 19.1 |
|  | face_fx | face damage + hat drawing (render) | 1.5 | 3.4 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 20.1 | 21.0 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 36.2 | 60.6 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.4 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.3 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.4 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 23.2 | 30.6 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 7.4 | 8.3 |
|  | render | game layer, HUD, effects (includes face_fx) | 3.2 | 5.7 |
| **End to end** | latency | camera capture to frame on screen | **35.1** | **43.8** |

Per frame: dropped 0.03 frames (p95 0), hand_lag 1.46 frames (p95 4).

