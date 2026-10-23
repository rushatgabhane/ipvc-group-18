from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from highnoon.config import Config
from highnoon.core.capture import open_source
from highnoon.core.profiler import Profiler
from highnoon.game.game import Game
from highnoon.render.renderer import Renderer
from highnoon.tracking.identity import IdentityTracker
from highnoon.vision.motion import MotionFilter
from highnoon.vision.perception import Perceiver

COLUMNS = [
    "frame",
    "t",
    "player",
    "visible",
    "extension",
    "elevation_deg",
    "aim_valid",
    "aiming",
    "head_pos",
    "ducked",
    "hand_openness",
    "hand_closed",
    "ammo",
    "hp",
    "event",
]
EVENT_LABELS = {"muzzle": "SHOT"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.capture.source = str(args.clip)
    cfg.capture.realtime = False
    source = open_source(cfg.capture)
    source.start()
    perceiver = Perceiver(cfg.perception)
    tracker = IdentityTracker(cfg.tracking, cfg.perception.max_players)
    motion = MotionFilter(cfg.filters)
    renderer = Renderer(cfg.display)
    profiler = Profiler()

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"{args.clip.stem}_signals.csv"
    video_path = args.out / f"{args.clip.stem}_annotated.mp4"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(COLUMNS)
    video = None
    timeline: list[tuple[float, str]] = []
    was_ducked: dict[int, bool] = {}
    game = None

    while (frame := source.read(0)) is not None:
        game = game or Game(cfg.game, frame.size)
        t = frame.timestamp_ms / 1000.0
        frame.t_capture = t
        perception = perceiver(frame)
        players = tracker.update(perception)
        motion.update(players, t, tracker.reassigned)
        effects_before = len(game.effects)
        ammo_before = {pid: state.ammo for pid, state in game.states.items()}
        game.update(players, t, tracker.reassigned)

        events: dict[int, list[str]] = {}
        log = EventLog(t, events, timeline)

        new_effects = game.effects[effects_before:] if len(game.effects) >= effects_before else game.effects
        for effect in new_effects:
            log(effect.player, EVENT_LABELS.get(effect.kind, effect.kind.upper()))
        for player in players:
            state = game.states.get(player.id)
            if state is None:
                continue
            if state.frame.ignored_fist is not None:
                log(player.id, f"FIST IGNORED ({state.frame.ignored_fist})")
            if state.frame.reload and state.ammo > ammo_before.get(player.id, state.ammo):
                log(player.id, "RELOAD")
            if state.frame.ducked != was_ducked.get(player.id, False):
                log(player.id, "DUCK" if state.frame.ducked else "STAND")
                was_ducked[player.id] = state.frame.ducked
            writer.writerow(signal_row(frame.id, t, player, state, events.get(player.id, [])))

        if not args.no_video:
            output = renderer.render(perception, players, game, profiler)
            if video is None:
                height, width = output.shape[:2]
                video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
            label = f"t={t:5.2f}s  frame {frame.id}"
            cv2.putText(
                output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
            video.write(output)

    perceiver.close()
    source.stop()
    csv_file.close()
    if video is not None:
        video.release()
    print_timeline(args.clip, timeline)
    print(f"Wrote {csv_path}" + ("" if args.no_video else f" and {video_path}"))


class EventLog:
    def __init__(self, t: float, events: dict[int, list[str]], timeline: list[tuple[float, str]]):
        self.t, self.events, self.timeline = t, events, timeline

    def __call__(self, player_id: int, label: str) -> None:
        self.events.setdefault(player_id, []).append(label)
        self.timeline.append((self.t, f"P{player_id} {label}"))


def signal_row(frame_id: int, t: float, player, state, events: list[str]) -> list:
    signals = player.signals
    hand = signals.hand_openness if signals else None
    return [
        frame_id,
        f"{t:.3f}",
        player.id,
        int(player.visible),
        f"{signals.extension:.3f}" if signals else "",
        f"{np.degrees(signals.elevation):.1f}" if signals else "",
        int(signals.aim_valid) if signals else "",
        int(state.frame.aiming),
        f"{state.frame.head_pos:.3f}",
        int(state.frame.ducked),
        f"{hand:.3f}" if hand is not None else "",
        "" if state.frame.hand_closed is None else int(state.frame.hand_closed),
        state.ammo,
        state.hp,
        " ".join(events),
    ]


def print_timeline(clip: Path, timeline: list[tuple[float, str]]) -> None:
    print(f"\nTimeline ({clip}):")
    for t, text in timeline:
        print(f"  {t:6.2f}s  {text}")
    totals = Counter(text.split(" ", 1)[1] for _, text in timeline)
    print("\nTotals:", ", ".join(f"{kind} {count}" for kind, count in sorted(totals.items())) or "no events")


if __name__ == "__main__":
    main()
