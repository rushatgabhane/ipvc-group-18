# All test sessions (1 Nov 2026, 10:41-12:22)

Logs of every test session, and the same sessions' camera recordings replayed through the final code
(`e86b5f9`: two tracked bodies, face ID every 3rd frame). `summary.md` has the per-task tables used in
the report ("all sessions" figures). The camera recordings themselves are not committed (personal
footage).

| Folder / file | What it is |
|---|---|
| `summary.md` | Per-task results over all sessions: detection, jitter, hat alignment, identity, re-entry, side swaps, game control, FPS, latency, CPU and memory, lighting, spread between sessions, distance |
| `live/` | Logs written live while playing: `games.csv` (one row per round, all 19 rounds), and per session `*_metrics.csv` (stage times per frame), `*_signals.csv` (control signals and game events), `*_identity.csv` (ids and face checks), `*_latency.md` (per-task latency) |
| `replay_final_code/*_eval.md` | `tools/evaluate_clip.py` on each recording: module times, detection, smoothing, identity agreement, gaps and re-entry, positions in metres |
| `replay_final_code/*_identity.csv` | Identity log of that replay |
| `replay_final_code/*_realtime_metrics.csv` | Replay paced like the live camera (headless): FPS, latency, dropped frames, stage times |
| `replay_final_code/*_realtime_cpu_mem.txt` | `ps` samples of that replay once per second: CPU % of one core, resident memory in KB |
| `replay_final_code/*_max_throughput_*` | Replay as fast as possible (`--fast`): the pipeline's maximum frame rate |

Game control numbers (T4) come from the live logs only: in a replay the players react to the live game,
not to the replayed one, so the replay's shots, ammo and cooldowns do not match what happened.
`session_20261101_1148` (3.6 s) is too short to replay; `session_20261101_1143` (an aborted take with the
faces mostly out of view) is shown but not counted in the totals.

## Reproduce

```bash
# per recording: identity report, max-throughput replay, real-time replay
.venv/bin/python tools/evaluate_clip.py clips/<session>.mp4 --out results/latest_e86b5f9
.venv/bin/python -m highnoon --source clips/<session>.mp4 --headless --fast --mute --no-games-csv \
    --metrics-csv results/latest_e86b5f9/<session>_metrics.csv
.venv/bin/python -m highnoon --source clips/<session>.mp4 --headless --mute --no-games-csv \
    --metrics-csv results/latest_e86b5f9/exp/<session>_paced_metrics.csv
# per-frame perception log (gain 1.0 = normal light; 0.5/0.3/0.15 and noise 6 for the lighting tests)
.venv/bin/python tools/perception_log.py clips/<session>.mp4 1.0 0 results/latest_e86b5f9/exp/<session>_g1.0.json
# tables
.venv/bin/python tools/summarise_sessions.py results/latest_e86b5f9 > docs/metrics/all_sessions/summary.md
```
