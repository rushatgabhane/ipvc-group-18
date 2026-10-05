"""T4: recognise player actions from T2 motion signals.

Each action is a small state machine with explicit timing rules, so a noisy frame cannot
trigger an action by itself:

  fire    trigger "fist": AIMING (steady >= armed_after_s) --hand open->closed--> FIRED.
          The hand must open again before the next shot (hysteresis between fist_closed
          and fist_open), so holding a fist fires once, not continuously.
          trigger "flick": AIMING --quick upward arm flick--> FIRED.
          Both use the aim from slightly *before* the trigger: closing the hand or flicking
          moves the arm a little, which would otherwise pull the shot off target.
  duck    STANDING --drop > duck_enter--> DUCKED --drop < duck_exit--> STANDING
          drop = shoulder fall relative to the player's own standing height, in torso lengths.
  reload  both arms down for reload_hold_s -> one reload event (re-armed when the arms come up)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from highnoon.config import GameConfig
from highnoon.contracts import MotionSignals

STAND_REF_UP = 0.3  # standing reference follows upward moves quickly (player stands up straight) ...
STAND_REF_DOWN = 0.02  # ... and downward moves slowly, so a duck is not absorbed into the reference


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
    duck_drop: float = 0.0
    hand_closed: bool | None = None  # None = aiming hand not tracked


class PlayerActions:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self._flick_rise = math.radians(cfg.flick_rise_deg)
        self._history: deque[tuple[float, float, np.ndarray, np.ndarray]] = deque()
        self._aim_since: float | None = None
        self._last_shot = -math.inf
        self.ducked = False
        self._ref_shoulder: float | None = None
        self._ref_torso: float | None = None
        self._arms_down_since: float | None = None
        self._reload_done = False
        self._hand_closed: bool | None = None

    @property
    def ref_shoulder(self) -> float | None:
        return self._ref_shoulder

    @property
    def ref_torso(self) -> float | None:
        return self._ref_torso

    def update(self, sig: MotionSignals | None, t: float) -> ActionFrame:
        if sig is None:
            self._history.clear()
            self._aim_since = None
            self._arms_down_since = None
            self._hand_closed = None
            return ActionFrame(ducked=self.ducked)
        out = ActionFrame()
        out.duck_drop = self._update_duck(sig)
        out.ducked = self.ducked
        out.reload = self._update_reload(sig, t)
        out.aiming, out.shot = self._update_fire(sig, t)
        out.hand_closed = self._hand_closed
        return out

    def _update_duck(self, sig: MotionSignals) -> float:
        if self._ref_shoulder is None:
            self._ref_shoulder, self._ref_torso = sig.shoulder_y, sig.torso_len
        drop = (sig.shoulder_y - self._ref_shoulder) / self._ref_torso
        if self.ducked:
            if drop < self.cfg.duck_exit:
                self.ducked = False
        elif drop > self.cfg.duck_enter:
            self.ducked = True
        if not self.ducked:  # the reference is frozen while ducked
            rate = STAND_REF_UP if sig.shoulder_y < self._ref_shoulder else STAND_REF_DOWN
            self._ref_shoulder += rate * (sig.shoulder_y - self._ref_shoulder)
            self._ref_torso += 0.05 * (sig.torso_len - self._ref_torso)
        return drop

    def _update_reload(self, sig: MotionSignals, t: float) -> bool:
        if not sig.arms_down:
            self._arms_down_since = None
            self._reload_done = False
            return False
        if self._arms_down_since is None:
            self._arms_down_since = t
        if not self._reload_done and t - self._arms_down_since >= self.cfg.reload_hold_s:
            self._reload_done = True
            return True
        return False

    def _update_fire(self, sig: MotionSignals, t: float) -> tuple[bool, Shot | None]:
        just_closed = self._update_hand(sig.hand_openness)
        if not sig.aim_valid or self.ducked:
            self._history.clear()
            self._aim_since = None
            return False, None
        if self._aim_since is None:
            self._aim_since = t
        self._history.append((t, sig.elevation, sig.aim_origin, sig.aim_dir))
        keep = max(self.cfg.flick_window_s, self.cfg.aim_lookback_s)
        while self._history and t - self._history[0][0] > keep:
            self._history.popleft()

        cooled = t - self._last_shot >= self.cfg.fire_cooldown_s
        if self.cfg.trigger == "fist":
            armed = t - self._aim_since >= self.cfg.armed_after_s
            if not (just_closed and armed and cooled):
                return True, None
            # Aim from aim_lookback_s ago: the oldest sample still inside the lookback window.
            before = next(h for h in self._history if t - h[0] <= self.cfg.aim_lookback_s)
        else:
            armed = t - self._aim_since >= self.cfg.armed_after_s + self.cfg.flick_window_s
            if not (armed and cooled):
                return True, None
            window = [h for h in self._history if t - h[0] <= self.cfg.flick_window_s]
            before = min(window, key=lambda h: h[1])
            if sig.elevation - before[1] < self._flick_rise:
                return True, None
            self._history.clear()
        self._last_shot = t
        return True, Shot(origin=before[2].copy(), direction=before[3].copy())

    def _update_hand(self, openness: float | None) -> bool:
        """Track open/closed with hysteresis. Returns True on the open -> closed transition."""
        if openness is None:
            return False  # hand not tracked this frame: keep the last known state
        was_closed = self._hand_closed
        if openness < self.cfg.fist_closed:
            self._hand_closed = True
        elif openness > self.cfg.fist_open:
            self._hand_closed = False
        return was_closed is False and self._hand_closed is True
