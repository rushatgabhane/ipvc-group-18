### `session_20261101_121216_metrics.csv`

522 frames over 12.8 s: **40.6 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 17.7 | 18.8 |
|  | face_fx | face damage + hat drawing (render) | 0.7 | 1.2 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 20.1 | 21.0 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 45.6 | 60.9 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.3 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.4 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 23.1 | 30.5 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 8.3 | 7.5 |
|  | render | game layer, HUD, effects (includes face_fx) | 2.3 | 3.9 |
| **End to end** | latency | camera capture to frame on screen | **35.0** | **42.0** |

Per frame: dropped 0.04 frames (p95 0), hand_lag 2.23 frames (p95 4).

