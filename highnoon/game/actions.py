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


@dataclass(slots=True)
class AimSample:
    t: float
    origin: np.ndarray
    direction: np.ndarray


class PlayerActions:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self._recent_aims: deque[AimSample] = deque()
        self._aiming_since: float | None = None
        self._last_shot_t = -math.inf
        self._hand_closed: bool | None = None
        self._hand_open_since: float | None = None
        self._ducked_since: float | None = None
        self._reloaded_this_duck = False
        self.ducked = False

    def update(self, signals: MotionSignals | None, t: float, frame_height: int) -> ActionFrame:
        if signals is None:
            self._stop_aiming()
            self._hand_closed = None
            return ActionFrame(ducked=self.ducked)
        head_pos = self._update_duck(signals.head_y / frame_height)
        reload, reload_progress = self._update_reload(t)
        aiming, shot = self._update_trigger(signals, t)
        return ActionFrame(
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

    def _update_trigger(self, signals: MotionSignals, t: float) -> tuple[bool, Shot | None]:
        fist_just_closed = self._update_hand(signals.hand_openness, t)
        if not signals.aim_valid or self.ducked:
            self._stop_aiming()
            return False, None
        if self._aiming_since is None:
            self._aiming_since = t
        self._remember_aim(AimSample(t, signals.aim_origin, signals.aim_dir))

        armed = t - self._aiming_since >= self.cfg.armed_after_s
        cooled_down = t - self._last_shot_t >= self.cfg.fire_cooldown_s
        if not (fist_just_closed and armed and cooled_down):
            return True, None
        self._last_shot_t = t
        steady_aim = self._recent_aims[0]
        return True, Shot(steady_aim.origin.copy(), steady_aim.direction.copy())

    def _remember_aim(self, sample: AimSample) -> None:
        self._recent_aims.append(sample)
        while sample.t - self._recent_aims[0].t > self.cfg.aim_lookback_s:
            self._recent_aims.popleft()

    def _update_hand(self, openness: float | None, t: float) -> bool:
        if openness is None:
            return False
        was_closed = self._hand_closed
        if openness < self.cfg.fist_closed:
            self._hand_closed = True
        elif openness > self.cfg.fist_open:
            self._hand_closed = False
        if self._hand_closed is False and was_closed is not False:
            self._hand_open_since = t
        held_open = self._hand_open_since is not None and t - self._hand_open_since >= self.cfg.open_hand_s
        just_closed = was_closed is False and self._hand_closed is True
        if self._hand_closed:
            self._hand_open_since = None
        return just_closed and held_open
