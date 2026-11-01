### `session_20261101_114917_metrics.csv`

637 frames over 15.7 s: **40.5 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 17.4 | 19.0 |
|  | face_fx | face damage + hat drawing (render) | 1.1 | 2.4 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 20.3 | 21.1 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 37.2 | 60.7 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.3 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.2 | 0.4 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 23.1 | 30.9 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 8.3 | 8.6 |
|  | render | game layer, HUD, effects (includes face_fx) | 3.1 | 5.5 |
| **End to end** | latency | camera capture to frame on screen | **35.8** | **43.5** |

Per frame: dropped 0.04 frames (p95 0), hand_lag 1.65 frames (p95 3).

