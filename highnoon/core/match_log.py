from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

PLAYER_IDS = (1, 2)
IGNORED_REASONS = ("not aiming", "ducked", "aim settling", "cooldown", "no ammo")
PLAYER_COLUMNS = [
    "shots",
    "bullet_hits",
    "headshots",
    "accuracy",
    "rockets",
    "grenades",
    "damage_dealt",
    "camping_damage_taken",
    "blocked",
    "powerups",
    "reloads",
    "dry_fires",
    "ducks",
    "ducked_s",
    "aim_s",
    "hand_tracked_pct",
    "fists",
    "fists_fired_pct",
    *[f"ignored_{reason.replace(' ', '_')}" for reason in IGNORED_REASONS],
]
TIMING_COLUMNS = ("latency", "inference", "hand", "hand_lag", "render", "dropped")
COLUMNS = [
    "ended_at",
    "session_round",
    "result",
    "winner",
    "duration_s",
    "frames",
    "fps",
    *[f"{name}_{stat}" for name in TIMING_COLUMNS for stat in ("mean", "p95")],
    *[f"p{player_id}_{column}" for player_id in PLAYER_IDS for column in PLAYER_COLUMNS],
]


@dataclass
class PlayerTally:
    counts: Counter = field(default_factory=Counter)
    ducked_s: float = 0.0
    aim_s: float = 0.0
    tracked_aim_s: float = 0.0
    was_ducked: bool = False
    was_closed: bool | None = None


class MatchLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.rounds = 0
        self._start_t: float | None = None
        self._last_t = 0.0
        self._tallies: dict[int, PlayerTally] = {}
        self._timings: dict[str, list[float]] = {}
        self._seen: set[int] = set()
        self._had_winner = False

    def update(self, game, players, t: float, timings: dict[str, float]) -> None:
        live = game.mode == "duel"
        if game.round_live and self._start_t is None:
            self._begin(t)
        if self._start_t is not None:
            dt = t - self._last_t
            self._count_effects(game)
            for player in players:
                self._count_player(game, player, dt)
            for name in TIMING_COLUMNS:
                if name in timings:
                    self._timings.setdefault(name, []).append(timings[name])
            if game.winner is not None and not self._had_winner:
                self._finish(t, "ko", game.winner)
            elif not live:
                self._finish(t, "abandoned", None)
        self._had_winner = game.winner is not None
        self._last_t = t

    def _begin(self, t: float) -> None:
        self._start_t = t
        self._tallies = {player_id: PlayerTally() for player_id in PLAYER_IDS}
        self._timings = {}

    def _count_effects(self, game) -> None:
        current = {id(effect) for effect in game.effects}
        for effect in game.effects:
            if id(effect) not in self._seen and effect.player in self._tallies:
                count_effect(self._tallies, effect)
        self._seen = current

    def _count_player(self, game, player, dt: float) -> None:
        tally, state = self._tallies.get(player.id), game.states.get(player.id)
        if tally is None or state is None or not player.visible:
            return
        frame, counts = state.frame, tally.counts
        if frame.ducked:
            tally.ducked_s += dt
            counts["ducks"] += not tally.was_ducked
        tally.was_ducked = frame.ducked
        if frame.aiming:
            tally.aim_s += dt
            tally.tracked_aim_s += dt * (
                player.signals is not None and player.signals.hand_openness is not None
            )
        if frame.hand_closed and tally.was_closed is False:
            counts["fists"] += 1
        tally.was_closed = frame.hand_closed
        if frame.ignored_fist is not None:
            counts[f"ignored_{frame.ignored_fist.replace(' ', '_')}"] += 1

    def _finish(self, t: float, result: str, winner: int | None) -> None:
        self.rounds += 1
        row = {
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "session_round": self.rounds,
            "result": result,
            "winner": winner or "",
            "duration_s": f"{t - self._start_t:.1f}",
            "frames": len(self._timings.get("latency", [])),
        }
        row["fps"] = f"{row['frames'] / max(1e-6, t - self._start_t):.1f}"
        row.update(timing_stats(self._timings))
        for player_id, tally in self._tallies.items():
            row.update({f"p{player_id}_{key}": value for key, value in player_stats(tally).items()})
        self._append(row)
        self._start_t = None

    def close(self) -> None:
        if self._start_t is not None:
            self._finish(self._last_t, "quit", None)

    def _append(self, row: dict) -> None:
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=COLUMNS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def count_effect(tallies: dict[int, PlayerTally], effect) -> None:
    counts = tallies[effect.player].counts
    if effect.kind == "muzzle":
        counts[{"rocket": "rockets", "grenade": "grenades"}.get(effect.detail, "shots")] += 1
    elif effect.kind in ("hit", "headshot"):
        counts["damage_dealt"] += effect.amount
        counts["bullet_hits"] += effect.detail == "bullet"
        counts["headshots"] += effect.kind == "headshot"
    elif effect.kind == "camping":
        counts["damage_dealt"] += effect.amount
        target = next((pid for pid in tallies if pid != effect.player), None)
        if target is not None:
            tallies[target].counts["camping_damage_taken"] += effect.amount
    elif effect.kind == "pickup":
        counts["powerups"] += 1
    elif effect.kind in ("blocked", "reload", "dry_fire"):
        counts[{"reload": "reloads", "dry_fire": "dry_fires"}.get(effect.kind, effect.kind)] += 1


def player_stats(tally: PlayerTally) -> dict:
    counts = tally.counts
    fired = counts["shots"] + counts["rockets"] + counts["grenades"] + counts["dry_fires"]
    stats = {column: counts[column] for column in PLAYER_COLUMNS if column in counts}
    stats.update(
        {
            "accuracy": f"{counts['bullet_hits'] / counts['shots']:.2f}" if counts["shots"] else "",
            "ducked_s": f"{tally.ducked_s:.1f}",
            "aim_s": f"{tally.aim_s:.1f}",
            "hand_tracked_pct": f"{100 * tally.tracked_aim_s / tally.aim_s:.0f}" if tally.aim_s else "",
            "fists_fired_pct": f"{100 * min(fired, counts['fists']) / counts['fists']:.0f}"
            if counts["fists"]
            else "",
        }
    )
    return {column: stats.get(column, 0) for column in PLAYER_COLUMNS}


def timing_stats(timings: dict[str, list[float]]) -> dict[str, str]:
    stats = {}
    for name in TIMING_COLUMNS:
        values = np.asarray(timings.get(name, []), np.float64)
        if values.size:
            stats[f"{name}_mean"] = f"{values.mean():.1f}"
            stats[f"{name}_p95"] = f"{np.percentile(values, 95):.1f}"
    return stats
