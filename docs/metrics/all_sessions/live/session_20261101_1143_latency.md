### `session_20261101_1143_metrics.csv`

1621 frames over 54.1 s: **30.0 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 3.6 | 11.7 |
|  | face_fx | face damage + hat drawing (render) | 0.0 | 0.4 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 18.2 | 26.9 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 24.3 | 48.1 |
|  | postprocess | landmarks to arrays, mask copy | 0.1 | 0.3 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.0 | 0.2 |
| T3 Identity | face_id | SFace embedding per face (perception thread, after face) | 1.1 | 9.6 |
|  | tracking | assignment, face verification, re-identification | 0.1 | 0.2 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.0 | 0.1 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.3 | 0.4 |
|  | inference | pose and face paths in parallel (the slower one) | 18.2 | 27.1 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 7.9 | 8.4 |
|  | render | game layer, HUD, effects (includes face_fx) | 1.9 | 2.8 |
|  | display | OpenGL upload + flip | 2.5 | 2.8 |
| **End to end** | latency | camera capture to frame on screen | **31.8** | **43.9** |

Per frame: dropped 0.02 frames (p95 0), hand_lag 0.33 frames (p95 2).

