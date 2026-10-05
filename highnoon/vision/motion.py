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

STRAIGHT_ARM = 0.8  # shoulder-wrist distance (torso lengths) above which the arm counts as straight
MIN_FOREARM = 0.3  # shorter forearms are pointing at the camera (foreshortened) and give no direction
ELEVATION_RANGE = (math.radians(-50), math.radians(80))
SHOULDERS_PER_TORSO = 0.8  # typical shoulder width / torso length, used when the hips are out of frame
ARMS_DOWN_DROP = 0.75  # wrists this many torso lengths below the shoulders = gun lowered


class MotionFilter:
    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        self._filters: dict[int, OneEuroFilter] = {}
        self._aim_arm: dict[int, int] = {}
        self._scale: dict[int, float] = {}

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
                self._scale.pop(p.id, None)
            if p.pose is None:
                p.signals = None
                continue
            reliable = p.pose.visibility >= self.cfg.min_visibility
            p.smoothed = f(p.pose.keypoints, t, mask=reliable)
            p.signals = self._signals(p.id, p.smoothed, reliable)
            hand = p.hands.get(ARMS[self._aim_arm[p.id]][2])
            p.signals.hand_openness = hand.openness if hand is not None else None

    def _signals(self, player_id: int, kp: np.ndarray, reliable: np.ndarray) -> MotionSignals:
        shoulder_mid = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2
        torso_len = self._body_scale(player_id, kp, reliable, shoulder_mid)

        # Aim ray per arm. A straight arm uses shoulder -> wrist (long baseline, steadiest);
        # a bent arm, or one whose shoulder is out of frame (sitting close to the camera),
        # uses the forearm, elbow -> wrist.
        candidates: list[tuple[np.ndarray, float, float, bool] | None] = []
        for s, e, w in ARMS:
            if not (reliable[e] and reliable[w]):
                candidates.append(None)
                continue
            forearm = kp[w] - kp[e]
            reach = float(np.linalg.norm(kp[w] - kp[s])) / torso_len if reliable[s] else 0.0
            vec = kp[w] - kp[s] if reach >= STRAIGHT_ARM else forearm
            norm = float(np.linalg.norm(vec))
            if norm < 1e-6 or float(np.linalg.norm(forearm)) / torso_len < MIN_FOREARM:
                candidates.append(None)
                continue
            direction = vec / norm
            elevation = math.atan2(-direction[1], abs(direction[0]))  # image y points down
            valid = ELEVATION_RANGE[0] <= elevation <= ELEVATION_RANGE[1]
            extension = reach if reliable[s] else float(np.linalg.norm(forearm)) / torso_len
            candidates.append((direction, elevation, extension, valid))

        # Keep the current aiming arm while it is valid; switch only when it stops being usable.
        arm = self._aim_arm.get(player_id, 1)
        if not (candidates[arm] and candidates[arm][3]) and candidates[1 - arm] and candidates[1 - arm][3]:
            arm = 1 - arm
        self._aim_arm[player_id] = arm

        wrist = ARMS[arm][2]
        if candidates[arm] is not None:
            direction, elevation, extension, aim_valid = candidates[arm]
        else:
            direction, elevation, extension, aim_valid = np.array([1.0, 0.0], np.float32), 0.0, 0.0, False

        drop = ARMS_DOWN_DROP * torso_len
        arms_down = bool(kp[L_WRIST][1] > shoulder_mid[1] + drop and kp[R_WRIST][1] > shoulder_mid[1] + drop)
        return MotionSignals(
            torso_len=torso_len,
            shoulder_y=float(shoulder_mid[1]),
            aim_valid=aim_valid,
            extension=extension,
            aim_origin=kp[wrist].copy(),
            aim_dir=np.asarray(direction, np.float32),
            elevation=elevation,
            arms_down=arms_down,
        )

    def _body_scale(
        self, player_id: int, kp: np.ndarray, reliable: np.ndarray, shoulder_mid: np.ndarray
    ) -> float:
        """Torso length in px: from the hips when visible, else estimated from shoulder width."""
        if reliable[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].all():
            scale = float(np.linalg.norm((kp[L_HIP] + kp[R_HIP]) / 2 - shoulder_mid))
        elif reliable[[L_SHOULDER, R_SHOULDER]].all():
            scale = float(np.linalg.norm(kp[L_SHOULDER] - kp[R_SHOULDER])) / SHOULDERS_PER_TORSO
        else:
            scale = self._scale.get(player_id, 200.0)
        scale = max(scale, 1.0)
        self._scale[player_id] = scale
        return scale
