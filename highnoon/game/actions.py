from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from highnoon.config import GameConfig
from highnoon.contracts import MotionSignals


@dataclass(slots=True)
class Shot:
    origin: np.ndarray
    direction: np.ndarray


@dataclass(slots=True)
class ActionFrame:
    aiming: bool = False
    ducked: bool = False
    shot: Shot | None = None
    reload: bool = False
    reload_progress: float = 0.0
    ducked_for: float = 0.0
    head_pos: float = 0.0
    hand_closed: bool | None = None
    ignored_fist: str | None = None
    spread_deg: float = 0.0


@dataclass(slots=True)
class AimSample:
    t: float
    origin: np.ndarray
    direction: np.ndarray


class PlayerActions:
    def __init__(self, cfg: GameConfig, seed: int = 0):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._recoil_deg = 0.0
        self._recoil_t = 0.0
        self._recent_aims: deque[AimSample] = deque()
        self._aiming_since: float | None = None
        self._last_aim_t: float | None = None
        self._pending_fist_t: float | None = None
        self._pending_reason: str | None = None
        self._last_shot_t = -math.inf
        self._hand_closed: bool | None = None
        self._ducked_since: float | None = None
        self._reloaded_this_duck = False
        self.ducked = False

    def update(self, signals: MotionSignals | None, t: float, frame_height: int) -> ActionFrame:
        if signals is None:
            self._stop_aiming()
            self._hand_closed = None
            self._pending_fist_t = None
            return ActionFrame(ducked=self.ducked)
        head_pos = self._update_duck(signals.head_y / frame_height)
        reload, reload_progress = self._update_reload(t)
        aiming, shot, ignored_fist = self._update_trigger(signals, t)
        return ActionFrame(
            spread_deg=self.spread_at(t),
            ignored_fist=ignored_fist,
            aiming=aiming,
            ducked=self.ducked,
            shot=shot,
            reload=reload,
            reload_progress=reload_progress,
            ducked_for=t - self._ducked_since if self._ducked_since is not None else 0.0,
            head_pos=head_pos,
            hand_closed=self._hand_closed,
        )

    def _update_reload(self, t: float) -> tuple[bool, float]:
        if not self.ducked:
            self._ducked_since = None
            self._reloaded_this_duck = False
            return False, 0.0
        if self._ducked_since is None:
            self._ducked_since = t
        progress = min(1.0, (t - self._ducked_since) / self.cfg.reload_s)
        if progress >= 1.0 and not self._reloaded_this_duck:
            self._reloaded_this_duck = True
            return True, progress
        return False, progress

    def _update_duck(self, head_pos: float) -> float:
        wall_top = 1.0 - self.cfg.cover_height
        if self.ducked and head_pos < wall_top - self.cfg.duck_margin:
            self.ducked = False
        elif not self.ducked and head_pos > wall_top:
            self.ducked = True
        return head_pos

    def _stop_aiming(self) -> None:
        self._recent_aims.clear()
        self._aiming_since = None
        self._last_aim_t = None

    def _update_trigger(self, signals: MotionSignals, t: float) -> tuple[bool, Shot | None, str | None]:
        if self._update_hand(signals.hand_openness):
            self._pending_fist_t, self._pending_reason = t, None

        if signals.aim_valid:
            self._last_aim_t = t
            if self._aiming_since is None:
                self._aiming_since = t
            if not self.ducked:
                self._remember_aim(AimSample(t, signals.aim_origin, signals.aim_dir))
        elif not self._in_aim_grace(t):
            self._stop_aiming()
        if self.ducked:
            self._recent_aims.clear()

        aiming = signals.aim_valid and not self.ducked
        if self._pending_fist_t is None:
            return aiming, None, None
        blocker = self._fire_blocker(t)
        if blocker is None:
            self._pending_fist_t = None
            self._last_shot_t = t
            steady_aim = self._recent_aims[0]
            return aiming, Shot(steady_aim.origin.copy(), self._apply_recoil(steady_aim.direction, t)), None
        # Only a fist made while the aim settles is worth waiting for; the others are dropped at once.
        waiting = (
            blocker == "aim settling"
            and self._hand_closed
            and t - self._pending_fist_t <= self.cfg.fist_queue_s
        )
        if waiting:
            self._pending_reason = blocker
            return aiming, None, None
        self._pending_fist_t = None
        return aiming, None, blocker

    def _in_aim_grace(self, t: float) -> bool:
        return (
            self._last_aim_t is not None
            and t - self._last_aim_t <= self.cfg.aim_grace_s
            and bool(self._recent_aims)
        )

    def spread_at(self, t: float) -> float:
        recovery_per_s = self.cfg.recoil_per_shot_deg / self.cfg.recoil_recovery_s
        return max(0.0, self._recoil_deg - (t - self._recoil_t) * recovery_per_s)

    def _apply_recoil(self, direction: np.ndarray, t: float) -> np.ndarray:
        spread = self.spread_at(t)
        self._recoil_deg = min(self.cfg.recoil_max_deg, spread + self.cfg.recoil_per_shot_deg)
        self._recoil_t = t
        return rotate(direction, math.radians(self._rng.uniform(-spread, spread)))

    def _fire_blocker(self, t: float) -> str | None:
        if self.ducked:
            return "ducked"
        if self._aiming_since is None or not self._recent_aims:
            return "not aiming"
        if t - self._last_shot_t < self.cfg.fire_cooldown_s:
            return "cooldown"
        if t - self._aiming_since < self.cfg.armed_after_s:
            return "aim settling"
        return None

    def _remember_aim(self, sample: AimSample) -> None:
        self._recent_aims.append(sample)
        while sample.t - self._recent_aims[0].t > self.cfg.aim_lookback_s:
            self._recent_aims.popleft()

    def _update_hand(self, openness: float | None) -> bool:
        if openness is None:
            return False
        was_closed = self._hand_closed
        if openness < self.cfg.fist_closed:
            self._hand_closed = True
        elif openness > self.cfg.fist_open:
            self._hand_closed = False
        return was_closed is False and self._hand_closed is True


def rotate(direction: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    x, y = float(direction[0]), float(direction[1])
    return np.array([x * c - y * s, x * s + y * c], np.float32)
