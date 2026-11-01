### `session_20261101_122227_metrics.csv`

1860 frames over 68.7 s: **27.1 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 19.4 | 24.6 |
|  | face_fx | face damage + hat drawing (render) | 1.8 | 3.4 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 31.8 | 34.4 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 34.8 | 53.3 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.3 | 0.5 |
| T3 Identity | face_id | SFace embedding per face (perception thread, after face) | 14.6 | 17.4 |
|  | tracking | assignment, face verification, re-identification | 0.3 | 0.4 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.4 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.9 | 1.4 |
|  | inference | pose and face paths in parallel (the slower one) | 35.3 | 40.4 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 7.7 | 9.0 |
|  | render | game layer, HUD, effects (includes face_fx) | 3.6 | 5.5 |
|  | display | OpenGL upload + flip | 2.3 | 2.7 |
| **End to end** | latency | camera capture to frame on screen | **66.6** | **83.9** |

Per frame: dropped 0.12 frames (p95 1), hand_lag 0.76 frames (p95 2).

