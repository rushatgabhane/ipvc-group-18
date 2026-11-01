### `session_20261101_1146_metrics.csv`

1558 frames over 63.5 s: **24.5 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 16.6 | 27.4 |
|  | face_fx | face damage + hat drawing (render) | 1.5 | 4.3 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 35.4 | 42.1 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 37.2 | 73.1 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.3 | 0.6 |
| T3 Identity | face_id | SFace embedding per face (perception thread, after face) | 11.7 | 20.9 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.4 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.5 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.7 | 1.6 |
|  | inference | pose and face paths in parallel (the slower one) | 37.7 | 48.7 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 9.4 | 10.8 |
|  | render | game layer, HUD, effects (includes face_fx) | 3.2 | 6.6 |
|  | display | OpenGL upload + flip | 2.6 | 3.2 |
| **End to end** | latency | camera capture to frame on screen | **66.1** | **93.6** |

Per frame: dropped 0.25 frames (p95 1), hand_lag 0.77 frames (p95 3).

