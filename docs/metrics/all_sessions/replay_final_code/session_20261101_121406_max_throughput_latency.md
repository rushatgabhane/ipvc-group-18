### `session_20261101_121406_metrics.csv`

448 frames over 12.1 s: **37.0 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 19.2 | 24.9 |
|  | face_fx | face damage + hat drawing (render) | 0.8 | 2.0 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 21.9 | 27.9 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 51.1 | 72.8 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.6 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.5 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.4 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.5 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.3 | 0.4 |
|  | inference | pose and face paths in parallel (the slower one) | 25.2 | 35.3 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 9.4 | 10.2 |
|  | render | game layer, HUD, effects (includes face_fx) | 2.7 | 5.7 |
| **End to end** | latency | camera capture to frame on screen | **38.8** | **52.4** |

Per frame: dropped 0.04 frames (p95 0), hand_lag 2.33 frames (p95 4).

