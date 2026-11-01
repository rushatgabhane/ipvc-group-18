### `session_20261101_1041_metrics.csv`

3121 frames over 77.6 s: **40.2 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 17.4 | 18.9 |
|  | face_fx | face damage + hat drawing (render) | 1.4 | 3.1 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 20.2 | 21.2 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 41.2 | 60.7 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.5 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.3 | 0.3 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.3 | 0.4 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 23.2 | 30.7 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 7.2 | 8.5 |
|  | render | game layer, HUD, effects (includes face_fx) | 3.3 | 5.1 |
| **End to end** | latency | camera capture to frame on screen | **35.0** | **43.6** |

Per frame: dropped 0.01 frames (p95 0), hand_lag 1.96 frames (p95 4).

