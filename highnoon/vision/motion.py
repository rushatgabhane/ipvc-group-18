"""T2: turns raw keypoints into smooth, stable per-player signals.

Filters are keyed by player id, so they need T3's identities. When a slot gets a new
identity, its filter is reset so a newcomer's pose is not blended with the previous one.
Keypoints below `min_visibility` are frozen at their last filtered value instead of
following noisy guesses.

Signals are expressed relative to the player's torso length where possible, so thresholds
work the same whether a player stands 1.5 m or 3 m from the camera.
"""

from __future__ import annotations

import math

import numpy as np

from highnoon.config import FilterConfig
from highnoon.contracts import MotionSignals, Player
from highnoon.core.filters import OneEuroFilter

L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP = 23, 24
ARMS = ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST))

MIN_EXTENSION = 0.8  # shoulder-wrist distance in torso lengths for an arm to count as aiming
ARM_SWITCH_MARGIN = 0.25  # the other arm must be this much more extended to take over aiming
ELEVATION_RANGE = (math.radians(-50), math.radians(70))
ARMS_DOWN_DROP = 0.75  # wrists this many torso lengths below the shoulders = gun lowered


class MotionFilter:
    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        self._filters: dict[int, OneEuroFilter] = {}
        self._aim_arm: dict[int, int] = {}

    def _filter(self, player_id: int) -> OneEuroFilter:
        f = self._filters.get(player_id)
        if f is None:
            f = OneEuroFilter(self.cfg.min_cutoff, self.cfg.beta, self.cfg.d_cutoff)
            self._filters[player_id] = f
        return f

    def update(self, players: list[Player], t: float, reassigned: set[int]) -> None:
        for p in players:
            f = self._filter(p.id)
            if p.id in reassigned:
                f.reset()
                self._aim_arm.pop(p.id, None)
            if p.pose is None:
                p.signals = None
                continue
            reliable = p.pose.visibility >= self.cfg.min_visibility
            p.smoothed = f(p.pose.keypoints, t, mask=reliable)
            p.signals = self._signals(p.id, p.smoothed, reliable)

    def _signals(self, player_id: int, kp: np.ndarray, reliable: np.ndarray) -> MotionSignals:
        shoulder_mid = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2
        hip_mid = (kp[L_HIP] + kp[R_HIP]) / 2
        torso_len = max(float(np.linalg.norm(hip_mid - shoulder_mid)), 1.0)

        extension = [
            float(np.linalg.norm(kp[w] - kp[s])) / torso_len if reliable[[s, e, w]].all() else 0.0
            for s, e, w in ARMS
        ]
        arm = self._aim_arm.get(player_id, int(np.argmax(extension)))
        other = 1 - arm
        if extension[other] > extension[arm] + ARM_SWITCH_MARGIN:
            arm = other
        self._aim_arm[player_id] = arm

        s, _, w = ARMS[arm]
        vec = kp[w] - kp[s]
        norm = float(np.linalg.norm(vec))
        direction = vec / norm if norm > 1e-6 else np.array([1.0, 0.0], np.float32)
        elevation = math.atan2(-direction[1], abs(direction[0]))  # image y points down
        aim_valid = extension[arm] >= MIN_EXTENSION and ELEVATION_RANGE[0] <= elevation <= ELEVATION_RANGE[1]

        drop = ARMS_DOWN_DROP * torso_len
        arms_down = bool(kp[L_WRIST][1] > shoulder_mid[1] + drop and kp[R_WRIST][1] > shoulder_mid[1] + drop)
        return MotionSignals(
            torso_len=torso_len,
            shoulder_y=float(shoulder_mid[1]),
            aim_valid=aim_valid,
            aim_origin=kp[w].copy(),
            aim_dir=direction.astype(np.float32),
            elevation=elevation,
            arms_down=arms_down,
        )
