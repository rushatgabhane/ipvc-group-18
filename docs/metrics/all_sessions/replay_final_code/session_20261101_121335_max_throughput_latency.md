### `session_20261101_121335_metrics.csv`

380 frames over 9.4 s: **40.1 fps** average.

| Task | Stage | What it covers | mean ms | p95 ms |
|---|---|---|---|---|
| T1 Face tracking & effects | face | face landmarks in head crops (perception thread) | 17.8 | 19.0 |
|  | face_fx | face damage + hat drawing (render) | 0.5 | 1.1 |
| T2 Pose & motion | pose | pose landmarks + segmentation mask (pose thread) | 20.4 | 21.9 |
|  | hand | hand landmarks, 4 hands (own thread, asynchronous) | 50.8 | 61.2 |
|  | postprocess | landmarks to arrays, mask copy | 0.4 | 0.4 |
|  | filter | One Euro filtering: keypoints, aim, face, hat; control signals | 0.2 | 0.3 |
|  | tracking | assignment, face verification, re-identification | 0.2 | 0.3 |
| T4 Game control | game | actions, shots, hits, powerups, rules | 0.1 | 0.4 |
| T5 Integration & scene | convert | BGR to RGB for the models | 0.2 | 0.3 |
|  | inference | pose and face paths in parallel (the slower one) | 23.4 | 30.8 |
|  | wait | result waiting for the main loop | 0.1 | 0.1 |
|  | camera | background replacement + wall onto the camera frame | 9.2 | 7.8 |
|  | render | game layer, HUD, effects (includes face_fx) | 2.3 | 4.3 |
| **End to end** | latency | camera capture to frame on screen | **36.1** | **43.0** |

Per frame: dropped 0.07 frames (p95 0), hand_lag 2.59 frames (p95 4).

