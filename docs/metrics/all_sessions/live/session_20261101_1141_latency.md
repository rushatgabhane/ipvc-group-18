### `session_20261101_1141_metrics.csv`

1850 frames over 72.5 s: **25.5 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 17.0 | 25.4 |
|  | face_fx | face damage + hat drawing (render) | 0.6 | 1.4 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 34.9 | 40.9 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 32.7 | 61.6 |
|  | postprocess | landmarks to arrays, mask copy | 0.3 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.6 |
| T3 Identity | face_id | SFace embedding per face (perception thread, after face) | 11.2 | 19.1 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.4 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.5 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.7 | 1.4 |
|  | inference | pose and face paths in parallel (the slower one) | 36.2 | 48.2 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 9.1 | 10.4 |
|  | render | game layer, HUD, effects (includes face_fx) | 2.1 | 4.4 |
|  | display | OpenGL upload + flip | 2.7 | 3.2 |
| **End to end** | latency | camera capture to frame on screen | **64.2** | **89.6** |

Per frame: dropped 0.20 frames (p95 1), hand_lag 0.53 frames (p95 2).

