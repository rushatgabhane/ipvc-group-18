# Measured results across all recordings, current code (e86b5f9)

Every raw camera recording of 1 Nov 2026 replayed through the current pipeline. `session_20261101_1148` (3.6 s) is left out. Repeated frames (written by the recorder to keep real time) are skipped for jitter and detection.

## T1 Face tracking and effects

| Clip | unique frames | both players visible | face found (visible player) | both faces (both visible) | wrong-face checks |
|---|---|---|---|---|---|
| 1041 | 3141 | 95.9% | 94.6% | 92.5% | 0 / 1927 |
| 1053 | 2082 | 63.0% | 96.3% | 92.9% | 6 / 1077 |
| 1106 | 4225 | 90.9% | 91.8% | 84.0% | 2 / 2449 |
| 1141 | 1852 | 33.5% | 85.1% | 80.5% | 0 / 628 |
| 1143 | 1585 | 0.3% | 0.0% | 0.0% | 0 / 0 |
| 1146 | 1552 | 59.8% | 73.4% | 89.4% | 0 / 569 |
| 114917 | 468 | 80.3% | 93.9% | 88.6% | 0 / 213 |
| 115111 | 1087 | 69.4% | 92.8% | 89.9% | 1 / 556 |
| 121216 | 432 | 72.5% | 96.7% | 93.6% | 0 / 187 |
| 121251 | 691 | 84.4% | 94.3% | 88.9% | 0 / 363 |
| 121335 | 329 | 65.0% | 95.1% | 90.7% | 0 / 131 |
| 121406 | 383 | 66.6% | 97.1% | 94.5% | 0 / 167 |
| 121659 | 1742 | 73.7% | 91.9% | 85.7% | 3 / 867 |
| 122227 | 1841 | 84.8% | 93.8% | 89.8% | 2 / 1005 |
| **all (without 1143)** | 19825 | 75.9% | 91.3% | 88.2% | 14 / 10139 |

Jitter (frame-to-frame noise after removing smooth motion, px or deg per frame), median over every clip and player; still = shoulders < 1.5 px/frame, moving = > 4 px/frame.

| Signal | still raw → smoothed | reduction | moving raw → smoothed | reduction | filter lag |
|---|---|---|---|---|---|
| T1 hat centre (px) | 0.47 → 0.08 (n=26) | 82.5% | 1.51 → 0.93 (n=26) | 38.5% | 2 ms |
| T1 hat tilt (deg) | 0.27 → 0.03 (n=26) | 89.5% | 0.51 → 0.07 (n=26) | 86.9% | 4 ms |
| T2 shoulder centre (px) | 1.27 → 0.30 (n=26) | 76.1% | 3.12 → 2.18 (n=27) | 30.2% | 0 ms |
| T2 right wrist (px) | 5.65 → 2.24 (n=26) | 60.4% | 9.24 → 6.41 (n=27) | 30.7% | 0 ms |

Hat alignment error (smoothed hat vs raw face mesh in the same frame):

| | mean px | p95 px | share of face width |
|---|---|---|---|
| still | 2.2 | 5.6 | 2.2% |
| moving | 10.4 | 20.4 | 9.7% |

## T2 Pose and motion

| Clip | upper-body keypoints confident | hand found with wrist raised |
|---|---|---|
| 1041 | 91.9% | 88.7% |
| 1053 | 80.5% | 73.7% |
| 1106 | 89.0% | 78.0% |
| 1141 | 84.4% | 39.6% |
| 1143 | 95.7% | 40.4% |
| 1146 | 92.6% | 56.2% |
| 114917 | 86.8% | 86.7% |
| 115111 | 88.1% | 75.0% |
| 121216 | 93.7% | 90.5% |
| 121251 | 95.2% | 96.1% |
| 121335 | 96.6% | 86.3% |
| 121406 | 97.6% | 96.0% |
| 121659 | 82.7% | 75.8% |
| 122227 | 86.9% | 81.2% |
| **all (without 1143)** | 88.2% | 75.5% |

Hand tracking against right-wrist speed (all clips, normal light):

| Wrist speed | frames | pose wrist confident | hand landmarks found |
|---|---|---|---|
| 0.5-1 m/s | 4610 | 100% | 75% |
| 1-2 m/s | 3812 | 100% | 75% |
| 2-3 m/s | 1433 | 100% | 73% |
| > 3 m/s | 2107 | 100% | 61% |

## T3 Identity

| Clip | length | face-reference agreement P1 / P2 | switches ≥ 0.3 s | side swaps (wrong-face checks within 1.5 s) | gaps (≥ 0.5 s) | re-entries recovered | re-id wait median / p95 / max |
|---|---|---|---|---|---|---|---|
| 1041 | 105 s | 100.0% (939) / 100.0% (988) | 0 | none | 1 | 1 / 1 | 0.17 / 0.17 / 0.17 s |
| 1053 | 69 s | 100.0% (629) / 99.6% (450) | 0 | none | 7 | 6 / 7 | 0.40 / 1.42 / 1.70 s |
| 1106 | 141 s | 100.0% (1207) / 99.9% (1241) | 0 | none | 6 | 6 / 6 | 0.17 / 0.25 / 0.27 s |
| 1141 | 74 s | 100.0% (482) / 99.0% (313) | 0 | none | 22 | 21 / 22 | 0.17 / 0.20 / 0.57 s |
| 1143 | 55 s |  | 0 | none | 8 | 7 / 8 | 0.00 / 1.80 / 2.57 s |
| 1146 | 65 s | 100.0% (435) / 100.0% (436) | 0 | none | 5 | 3 / 5 | 0.17 / 0.65 / 0.70 s |
| 114917 | 22 s | 100.0% (190) / 100.0% (199) | 0 | none | 0 | 0 / 0 | - |
| 115111 | 48 s |  | 0 | 35.7 s (0) | 6 | 5 / 6 | 0.73 / 4.10 / 4.10 s |
| 121216 | 18 s | 100.0% (169) / 100.0% (178) | 0 | none | 0 | 0 / 0 | - |
| 121251 | 27 s | 100.0% (250) / 100.0% (271) | 0 | none | 0 | 0 / 0 | - |
| 121335 | 14 s | 100.0% (127) / 100.0% (124) | 0 | none | 0 | 0 / 0 | - |
| 121406 | 16 s | 100.0% (145) / 100.0% (154) | 0 | none | 1 | 0 / 1 | - |
| 121659 | 64 s | 98.2% (545) / 100.0% (541) | 1 | 33.1 s (0) | 6 | 6 / 6 | 0.35 / 1.79 / 2.20 s |
| 122227 | 70 s | 99.8% (614) / 98.7% (632) | 0 | 57.3 s (0) | 3 | 3 / 3 | 0.17 / 0.44 / 0.47 s |

**All clips:** identity agreement 99.8% of 11259 face-checked player-frames; 1 switches ≥ 0.3 s in 13.1 min (0.08 per minute); side swaps with both identities preserved (no wrong-face check within 1.5 s): 3 / 3; re-entries after ≥ 0.5 s out of view recovered: 58 / 65, re-identification median 0.17 s, p95 2.26 s, max 4.10 s.
Distance estimate: medians 0.93-2.31 m across clips, distance jitter 0.8 cm (median); no tape-measured reference.

## T4 Game control (live logs of every session: results/games.csv, results/*_signals.csv)

19 rounds (574 s of duel). Fists 481: fired 308 (243 shots, 65 empty-gun clicks), cooldown 68, not aiming 32, ducked 1, aim settling 52. Fired share of non-cooldown fists: 75% (308 / 413). Bullet hits 111 (headshots 60), blocked 21, ducks 100, reloads 66, powerups 17.

| Round | clip time | result | P1 fists / fired % | P2 fists / fired % | P1 hand tracked | P2 hand tracked |
|---|---|---|---|---|---|---|
| 1 | 2026-11-01T10:42:07 | ko (28.2 s) | 9 / 89% | 13 / 82% | 98% | 99% |
| 2 | 2026-11-01T10:42:51 | ko (34.0 s) | 14 / 85% | 26 / 94% | 93% | 99% |
| 3 | 2026-11-01T10:43:20 | ko (19.3 s) | 13 / 100% | 13 / 91% | 87% | 97% |
| 1 | 2026-11-01T10:54:22 | ko (60.5 s) | 19 / 75% | 27 / 90% | 86% | 91% |
| 1 | 2026-11-01T11:07:35 | ko (47.7 s) | 19 / 88% | 26 / 81% | 94% | 94% |
| 2 | 2026-11-01T11:08:14 | ko (28.9 s) | 14 / 54% | 20 / 81% | 91% | 94% |
| 3 | 2026-11-01T11:09:32 | ko (67.2 s) | 23 / 70% | 30 / 67% | 95% | 98% |
| 1 | 2026-11-01T11:43:04 | quit (56.2 s) | 16 / 73% | 12 / 33% | 76% | 97% |
| 1 | 2026-11-01T11:47:43 | quit (53.8 s) | 18 / 53% | 18 / 82% | 90% | 91% |
| 1 | 2026-11-01T11:49:43 | quit (11.0 s) | 6 / 50% | 4 / 75% | 93% | 89% |
| 1 | 2026-11-01T11:51:56 | ko (29.6 s) | 12 / 64% | 12 / 83% | 92% | 93% |
| 1 | 2026-11-01T12:12:38 | quit (9.4 s) | 5 / 50% | 7 / 71% | 100% | 96% |
| 1 | 2026-11-01T12:13:17 | ko (14.8 s) | 9 / 100% | 8 / 100% | 94% | 97% |
| 1 | 2026-11-01T12:13:52 | quit (4.9 s) | 2 / 100% | 2 / 50% | 83% | 96% |
| 1 | 2026-11-01T12:14:25 | quit (7.0 s) | 1 / 100% | 3 / 100% | 95% | 98% |
| 1 | 2026-11-01T12:17:50 | ko (39.4 s) | 12 / 82% | 9 / 100% | 90% | 96% |
| 2 | 2026-11-01T12:18:07 | quit (10.1 s) | 11 / 75% | 8 / 100% | 82% | 62% |
| 1 | 2026-11-01T12:22:54 | ko (14.6 s) | 3 / 100% | 9 / 100% | 97% | 96% |
| 2 | 2026-11-01T12:23:37 | ko (37.8 s) | 8 / 100% | 20 / 60% | 88% | 90% |

Simultaneous shots (both players fired within 0.3 s, both registered): 48.

## T5 Integration

Real-time replay (paced like the live camera, headless), and maximum throughput (unpaced):

| Clip | real-time FPS mean / p5 / p1 | latency mean / p95 / p99 ms | dropped per frame | longest frame | CPU | memory mean / peak | max throughput fps | pose / face / face ID / hand ms |
|---|---|---|---|---|---|---|---|---|
| 1041 | 30.0 / 21.4 / 19.5 | 35.4 / 46.4 / 56.8 | 0.00 | 109 ms | 321% | 763 / 860 MB | 40.2 | 20.6 / 17.8 / nan / 41.8 |
| 1053 | 30.0 / 22.0 / 18.2 | 34.3 / 44.9 / 57.1 | 0.00 | 107 ms | 290% | 766 / 826 MB | 39.8 | 20.7 / 17.2 / nan / 24.9 |
| 1106 | 30.0 / 20.8 / 16.6 | 37.0 / 50.8 / 71.3 | 0.00 | 146 ms | 313% | 692 / 755 MB | 39.4 | 21.7 / 18.4 / nan / 34.3 |
| 1141 | 30.1 / 21.8 / 20.6 | 32.3 / 40.8 / 43.7 | 0.00 | 83 ms | 269% | 714 / 756 MB | 39.6 | 21.7 / 14.0 / nan / 25.5 |
| 1143 | 30.0 / 26.3 / 22.0 | 22.1 / 30.6 / 36.2 | 0.00 | 74 ms | 144% | 638 / 710 MB | 67.7 | 14.1 / 2.9 / nan / 19.2 |
| 1146 | 30.0 / 21.6 / 20.2 | 33.1 / 44.1 / 50.1 | 0.00 | 80 ms | 274% | 755 / 790 MB | 41.8 | 20.3 / 13.4 / nan / 30.0 |
| 114917 | 30.0 / 21.4 / 20.1 | 34.4 / 44.7 / 47.3 | 0.00 | 80 ms | 315% | 755 / 806 MB | 40.3 | 20.4 / 17.8 / nan / 37.0 |
| 115111 | 30.0 / 21.3 / 20.0 | 34.3 / 44.2 / 51.1 | 0.00 | 70 ms | 302% | 749 / 797 MB | 40.2 | 20.3 / 17.2 / nan / 36.1 |
| 121216 | 30.1 / 21.7 / 20.7 | 33.4 / 42.9 / 45.3 | 0.00 | 71 ms | 322% | 745 / 787 MB | 40.5 | 20.2 / 17.9 / nan / 44.5 |
| 121251 | 30.0 / 21.4 / 20.1 | 35.2 / 45.7 / 49.4 | 0.00 | 87 ms | 331% | 767 / 822 MB | 35.1 | 20.4 / 17.9 / nan / 46.6 |
| 121335 | 30.0 / 21.9 / 20.8 | 33.4 / 42.4 / 44.0 | 0.00 | 60 ms | 313% | 725 / 761 MB | 40.0 | 20.2 / 17.9 / nan / 51.5 |
| 121406 | 30.0 / 21.6 / 19.4 | 34.5 / 45.5 / 54.2 | 0.00 | 87 ms | 310% | 654 / 757 MB | 36.8 | 20.7 / 18.2 / nan / 48.9 |
| 121659 | 30.0 / 21.6 / 20.4 | 34.1 / 44.3 / 46.8 | 0.00 | 101 ms | 304% | 838 / 925 MB | 35.7 | 20.4 / 17.5 / nan / 30.1 |
| 122227 | 30.0 / 21.3 / 17.4 | 36.1 / 47.1 / 69.4 | 0.00 | 126 ms | 315% | 736 / 809 MB | 38.8 | 21.1 / 18.3 / nan / 35.0 |
| **all (frame-weighted)** | 30.0 | 34.8 / 45.8 | 0.00 | | 295% | 736 / 925 MB | 39.3 | |

## Lighting (current code)

| Clip | condition | both ids visible | face found | upper-body keypoints | hand with wrist raised | wrong-face checks | own-face score |
|---|---|---|---|---|---|---|---|
| 1106 | normal | 90.9% | 91.8% | 89.0% | 78.0% | 2 / 2449 | 0.94 |
| 1106 | 50% | 91.7% | 91.3% | 88.5% | 77.9% | 2 / 2443 | 0.94 |
| 1106 | 30% | 92.9% | 90.7% | 87.3% | 76.7% | 2 / 2445 | 0.95 |
| 1106 | 30% + noise | 90.7% | 91.3% | 86.8% | 79.0% | 1 / 2437 | 0.91 |
| 1106 | 15% | 88.2% | 90.1% | 86.0% | 45.5% | 1 / 2363 | 0.95 |
| 1106 | 15% + noise | 85.8% | 91.1% | 85.9% | 6.4% | 0 / 2353 | 0.89 |
| 1053 | normal | 63.0% | 96.3% | 80.5% | 73.7% | 6 / 1077 | 0.93 |
| 1053 | 50% | 64.7% | 95.9% | 80.9% | 70.3% | 4 / 1087 | 0.94 |
| 1053 | 30% | 63.4% | 95.9% | 80.5% | 66.6% | 3 / 1080 | 0.95 |
| 1053 | 30% + noise | 59.0% | 96.5% | 81.3% | 60.8% | 3 / 1053 | 0.92 |
| 1053 | 15% | 63.0% | 95.1% | 80.3% | 35.2% | 4 / 1071 | 0.95 |
| 1053 | 15% + noise | 61.2% | 94.7% | 78.4% | 1.4% | 3 / 1055 | 0.88 |

## Impact factor: repetition (spread across sessions)

Mean ± standard deviation over the sessions (each session one repetition; sessions shorter than 20 s left out).

| Metric | mean ± SD | min - max | sessions |
|---|---|---|---|
| T1 face found per visible player | 90.8% ± 6.8% | 73.4% - 96.3% | 10 |
| T1 both faces when both visible | 88.2% ± 3.8% | 80.5% - 92.9% | 10 |
| T2 upper-body keypoints confident | 87.8% ± 4.6% | 80.5% - 95.2% | 10 |
| T2 hand found with wrist raised | 75.1% ± 16.4% | 39.6% - 96.1% | 10 |
| T3 face-reference agreement | 99.7% ± 0.4% | 99.1% - 100.0% | 9 |
| T5 real-time FPS | 30.0 ± 0.0 | 30.0 - 30.1 | 10 |
| T5 latency mean (ms) | 34.6 ± 1.4 | 32.3 - 37.0 | 10 |
| T5 max throughput (fps) | 39.1 ± 2.1 | 35.1 - 41.8 | 10 |

## Impact factor: distance from the camera

Distance from the torso length (0.5 m torso, 70° field of view, 1280 px wide); all sessions, normal light.

| Distance | player-frames | face found | upper-body keypoints | hand with wrist raised | wrong-face checks |
|---|---|---|---|---|---|
| < 1.25 m | 4287 | 74.3% | 92.9% | 60.3% | 4 / 1032 |
| 1.25-1.75 m | 21001 | 97.8% | 92.3% | 82.8% | 8 / 6750 |
| 1.75-2.25 m | 5993 | 93.8% | 79.5% | 77.3% | 0 / 1857 |
| > 2.25 m | 2495 | 61.7% | 68.5% | 42.2% | 2 / 500 |
