# Demo session metrics (1 Nov 2026, 12:22)

Logs of the live session recorded as the demo video (`demo/high_noon_demo.mp4`): two players, best of 3,
68.7 s, recorded with `--record-output --demo-timer --metrics-csv --signals-csv --identity-csv`.

| File | What it is |
|---|---|
| `latency.md` | Per-task (T1-T5) stage times, FPS and capture-to-screen latency (`tools/task_latency.py metrics.csv`) |
| `metrics.csv` | One row per frame: every stage's time in ms, FPS, latency, dropped frames, hand lag |
| `signals.csv` | One row per player per frame: hand openness, fist, aim, duck, ammo, HP, events (SHOT, HIT, ...) |
| `identity.csv` | One row per player per frame: visible, new id, position (px and m), face match to the id's owner |
| `games.csv` | The two rounds of this match from `results/games.csv`: result, duration, FPS, latency, per-player stats |

**Summary:** 27.1 fps, latency 66.6 ms mean / 83.9 ms p95, 0.12 dropped frames per frame shown.
Pose 31.8 ms, face (head crops) 19.4 ms, face ID 14.6 ms, hands 34.8 ms (asynchronous), render 3.6 ms.
25 shots (10 body hits, 6 headshots, 3 blocked), 3 explosions; identity: 4 wrong-face frames in 3,116
face checks, no ids handed out after the match started. Round 1 won by P2 by K.O. after 14.6 s of duel,
round 2 by P1 after 37.8 s.
