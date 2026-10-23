from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

COLUMNS = [
    "frame",
    "t",
    "player",
    "visible",
    "hands_in_frame",
    "hand_tracked",
    "hand_openness",
    "hand_closed",
    "aim_valid",
    "aiming",
    "ducked",
    "extension",
    "elevation_deg",
    "head_pos",
    "ammo",
    "hp",
    "event",
]
EVENT_LABELS = {"muzzle": "SHOT"}


class SignalLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        self._effects_seen: set[int] = set()

    def record(self, frame_id: int, t: float, players, game, hands_in_frame: int) -> dict[int, list[str]]:
        events = self._new_events(game)
        for player in players:
            state = game.states.get(player.id)
            if state is None:
                continue
            if state.frame.ignored_fist is not None:
                events.setdefault(player.id, []).append(f"FIST IGNORED ({state.frame.ignored_fist})")
            row = signal_row(frame_id, t, player, state, events.get(player.id, []), hands_in_frame)
            self._writer.writerow(row)
        return events

    def _new_events(self, game) -> dict[int, list[str]]:
        events: dict[int, list[str]] = {}
        current = {id(effect) for effect in game.effects}
        for effect in game.effects:
            if id(effect) not in self._effects_seen:
                events.setdefault(effect.player, []).append(
                    EVENT_LABELS.get(effect.kind, effect.kind.upper())
                )
        self._effects_seen = current
        return events

    def close(self) -> None:
        self._file.close()


def signal_row(frame_id: int, t: float, player, state, events: list[str], hands_in_frame: int = 0) -> list:
    signals = player.signals
    hand = signals.hand_openness if signals else None
    return [
        frame_id,
        f"{t:.3f}",
        player.id,
        int(player.visible),
        hands_in_frame,
        int(hand is not None),
        f"{hand:.3f}" if hand is not None else "",
        "" if state.frame.hand_closed is None else int(state.frame.hand_closed),
        int(signals.aim_valid) if signals else "",
        int(state.frame.aiming),
        int(state.frame.ducked),
        f"{signals.extension:.3f}" if signals else "",
        f"{np.degrees(signals.elevation):.1f}" if signals else "",
        f"{state.frame.head_pos:.3f}",
        state.ammo,
        state.hp,
        " ".join(events),
    ]
