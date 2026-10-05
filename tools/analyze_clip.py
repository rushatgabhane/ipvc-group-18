"""Replay a recorded clip through the full game and report what was detected, frame by frame.

    python tools/analyze_clip.py clips/moves.mp4

Writes:
  results/<clip>_signals.csv   per frame, per player: signals, action states, events
  results/<clip>_annotated.mp4 the rendered game with the debug overlay
and prints a timeline of game events (shots, hits, ducks, reloads).

Processing is sequential (no frame dropping), so the same clip always gives the same
result. Use it to tune GameConfig thresholds and as a regression check.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from highnoon.config import Config  # noqa: E402
from highnoon.core.capture import open_source  # noqa: E402
from highnoon.core.profiler import Profiler  # noqa: E402
from highnoon.game.game import Game  # noqa: E402
from highnoon.render.renderer import Renderer  # noqa: E402
from highnoon.tracking.identity import IdentityTracker  # noqa: E402
from highnoon.vision.motion import MotionFilter  # noqa: E402
from highnoon.vision.perception import Perceiver  # noqa: E402

COLUMNS = [
    "frame", "t", "player", "visible", "extension", "elevation_deg", "aim_valid", "aiming",
    "duck_drop", "ducked", "arms_down", "hand_openness", "hand_closed", "ammo", "hp", "event",
]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    cfg = Config()
    cfg.capture.source = str(a.clip)
    cfg.capture.realtime = False
    cfg.perception.hands_async = False  # deterministic: hands for every frame, inline
    source = open_source(cfg.capture)
    source.start()
    perceiver = Perceiver(cfg.perception)
    tracker = IdentityTracker(cfg.tracking, cfg.perception.max_players)
    motion = MotionFilter(cfg.filters)
    renderer = Renderer(cfg.display)
    profiler = Profiler()

    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.clip.stem
    csv_file = open(a.out / f"{stem}_signals.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(COLUMNS)
    video = None
    events: list[tuple[float, str]] = []
    prev_ducked: dict[int, bool] = {}

    game = None

    while True:
        frame = source.read(0)
        if frame is None:
            break
        if game is None:
            game = Game(cfg.game, frame.size)
        # Use the clip's own timeline so timing rules behave as in real time.
        t = frame.timestamp_ms / 1000.0
        frame.t_capture = t
        perception = perceiver(frame)
        players = tracker.update(perception)
        motion.update(players, t, tracker.reassigned)
        n_effects = len(game.effects)
        prev_ammo = {pid: st.ammo for pid, st in game.states.items()}
        game.update(players, t, tracker.reassigned)

        frame_events: dict[int, list[str]] = {}
        for e in game.effects[n_effects:] if len(game.effects) >= n_effects else game.effects:
            label = {"muzzle": "SHOT"}.get(e.kind, e.kind.upper())
            frame_events.setdefault(e.player, []).append(label)
            events.append((t, f"P{e.player} {label}"))
        for p in players:
            st = game.states.get(p.id)
            if st is None:
                continue
            if st.frame.reload and st.ammo > prev_ammo.get(p.id, st.ammo):
                frame_events.setdefault(p.id, []).append("RELOAD")
                events.append((t, f"P{p.id} RELOAD"))
            if st.frame.ducked != prev_ducked.get(p.id, False):
                label = "DUCK" if st.frame.ducked else "STAND"
                frame_events.setdefault(p.id, []).append(label)
                events.append((t, f"P{p.id} {label}"))
                prev_ducked[p.id] = st.frame.ducked
            s = p.signals
            writer.writerow([
                frame.id, f"{t:.3f}", p.id, int(p.visible),
                f"{s.extension:.3f}" if s else "", f"{np.degrees(s.elevation):.1f}" if s else "",
                int(s.aim_valid) if s else "", int(st.frame.aiming), f"{st.frame.duck_drop:.3f}",
                int(st.frame.ducked), int(s.arms_down) if s else "",
                f"{s.hand_openness:.3f}" if s and s.hand_openness is not None else "",
                "" if st.frame.hand_closed is None else int(st.frame.hand_closed), st.ammo, st.hp,
                " ".join(frame_events.get(p.id, [])),
            ])  # fmt: skip

        if not a.no_video:
            out = renderer.render(perception, players, game, profiler)
            if video is None:
                h, w = out.shape[:2]
                video = cv2.VideoWriter(
                    str(a.out / f"{stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                )
            cv2.putText(out, f"t={t:5.2f}s  frame {frame.id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip
            video.write(out)

    perceiver.close()
    source.stop()
    csv_file.close()
    if video is not None:
        video.release()

    print(f"\nTimeline ({a.clip}):")
    for t, text in events:
        print(f"  {t:6.2f}s  {text}")
    counts: dict[str, int] = {}
    for _, text in events:
        kind = text.split(" ", 1)[1]
        counts[kind] = counts.get(kind, 0) + 1
    print("\nTotals:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no events")
    print(f"Wrote {a.out / (stem + '_signals.csv')}" + ("" if a.no_video else f" and {stem}_annotated.mp4"))


if __name__ == "__main__":
    main()
