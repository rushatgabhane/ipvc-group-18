from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from highnoon.config import FilterConfig
from highnoon.contracts import HandObservation, MotionSignals, Player
from highnoon.core.filters import OneEuroFilter

NOSE = 0
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16
L_HIP, R_HIP = 23, 24
WRISTS = (L_WRIST, R_WRIST)
RIGHT_HAND = 1

# The palm points from the wrist to the knuckles, which stay put when the fingers close into a fist.
HAND_WRIST = 0
HAND_KNUCKLES = [5, 9, 13, 17]
MIN_PALM_LENGTH = 0.08
HAND_HOLD_S = 0.3
ELEVATION_RANGE = (math.radians(-50), math.radians(80))
SHOULDER_WIDTH_PER_TORSO = 0.8
DEFAULT_TORSO_PX = 200.0
AIM_MIN_CUTOFF = 0.8
AIM_BETA = 0.6
FACE_MIN_CUTOFF = 1.0
FACE_BETA = 0.08


@dataclass
class PalmAim:
    direction: np.ndarray
    elevation: float
    extension: float
    valid: bool


@dataclass
class SmoothedAim:
    hand: int
    angle_filter: OneEuroFilter
    angle: float


class MotionFilter:
    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        self._keypoint_filters: dict[int, OneEuroFilter] = {}
        self._face_filters: dict[int, OneEuroFilter] = {}
        self._aiming_hand: dict[int, int] = {}
        self._torso_len: dict[int, float] = {}
        self._aim: dict[int, SmoothedAim] = {}
        self._last_palm: dict[tuple[int, int], tuple[float, PalmAim]] = {}

    def update(self, players: list[Player], t: float, reassigned: set[int]) -> None:
        for player in players:
            if player.id in reassigned:
                self._forget(player.id)
            self._smooth_face(player, t)
            if player.pose is None:
                player.signals = None
                continue
            reliable = player.pose.visibility >= self.cfg.min_visibility
            player.smoothed = self._keypoint_filter(player.id)(player.pose.keypoints, t, mask=reliable)
            player.signals = self._signals(player, player.smoothed, reliable, t)
            player.signals.hand_openness = self._aiming_hand_openness(player)
            player.signals.head_y = head_height(player)
            self._smooth_aim(player.id, player.signals, t)

    def _forget(self, player_id: int) -> None:
        self._keypoint_filter(player_id).reset()
        self._face_filters.pop(player_id, None)
        self._aiming_hand.pop(player_id, None)
        self._torso_len.pop(player_id, None)
        self._aim.pop(player_id, None)
        for wrist in WRISTS:
            self._last_palm.pop((player_id, wrist), None)

    def _smooth_face(self, player: Player, t: float) -> None:
        if player.face is None:
            self._face_filters.pop(player.id, None)
            player.face_smoothed = None
            return
        face_filter = self._face_filters.setdefault(player.id, OneEuroFilter(FACE_MIN_CUTOFF, FACE_BETA))
        player.face_smoothed = face_filter(player.face.landmarks, t)

    def _keypoint_filter(self, player_id: int) -> OneEuroFilter:
        if player_id not in self._keypoint_filters:
            cfg = self.cfg
            self._keypoint_filters[player_id] = OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
        return self._keypoint_filters[player_id]

    def _aiming_hand_openness(self, player: Player) -> float | None:
        hand = player.hands.get(WRISTS[self._aiming_hand[player.id]])
        return hand.openness if hand is not None else None

    def _smooth_aim(self, player_id: int, signals: MotionSignals, t: float) -> None:
        if not signals.aim_valid:
            self._aim.pop(player_id, None)
            return
        hand = self._aiming_hand[player_id]
        raw_angle = math.atan2(float(signals.aim_dir[1]), float(signals.aim_dir[0]))
        aim = self._aim.get(player_id)
        if aim is None or aim.hand != hand:
            aim = SmoothedAim(hand, OneEuroFilter(AIM_MIN_CUTOFF, AIM_BETA, 1.0), raw_angle)
        unwrapped = aim.angle + math.remainder(raw_angle - aim.angle, math.tau)
        aim.angle = float(aim.angle_filter(np.array([unwrapped], np.float32), t)[0])
        self._aim[player_id] = aim
        signals.aim_dir = np.array([math.cos(aim.angle), math.sin(aim.angle)], np.float32)

    def _signals(
        self, player: Player, keypoints: np.ndarray, reliable: np.ndarray, t: float
    ) -> MotionSignals:
        torso_len = self._estimate_torso_len(player.id, keypoints, reliable)
        aims = [self._held_palm_aim(player, wrist, torso_len, t) for wrist in WRISTS]
        hand = self._choose_aiming_hand(player.id, aims)
        aim = aims[hand] or PalmAim(np.array([1.0, 0.0], np.float32), 0.0, 0.0, False)
        return MotionSignals(
            torso_len=torso_len,
            aim_valid=aim.valid,
            extension=aim.extension,
            aim_origin=keypoints[WRISTS[hand]].copy(),
            aim_dir=np.asarray(aim.direction, np.float32),
            elevation=aim.elevation,
        )

    def _held_palm_aim(self, player: Player, wrist: int, torso_len: float, t: float) -> PalmAim | None:
        """The palm's aim, or its last one for a moment, so a tracking dropout does not stop the aim."""
        key = (player.id, wrist)
        hand = player.hands.get(wrist)
        aim = palm_aim(hand, torso_len) if hand is not None else None
        if aim is not None:
            self._last_palm[key] = (t, aim)
            return aim
        last = self._last_palm.get(key)
        if last is not None and t - last[0] <= HAND_HOLD_S:
            return last[1]
        return None

    def _choose_aiming_hand(self, player_id: int, aims: list[PalmAim | None]) -> int:
        hand = self._aiming_hand.get(player_id, RIGHT_HAND)
        other = 1 - hand
        current_usable = aims[hand] is not None and aims[hand].valid
        other_usable = aims[other] is not None and aims[other].valid
        if not current_usable and other_usable:
            hand = other
        self._aiming_hand[player_id] = hand
        return hand

    def _estimate_torso_len(self, player_id: int, keypoints: np.ndarray, reliable: np.ndarray) -> float:
        shoulder_mid = (keypoints[L_SHOULDER] + keypoints[R_SHOULDER]) / 2
        if reliable[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].all():
            hip_mid = (keypoints[L_HIP] + keypoints[R_HIP]) / 2
            torso_len = float(np.linalg.norm(hip_mid - shoulder_mid))
        elif reliable[[L_SHOULDER, R_SHOULDER]].all():
            shoulder_width = float(np.linalg.norm(keypoints[L_SHOULDER] - keypoints[R_SHOULDER]))
            torso_len = shoulder_width / SHOULDER_WIDTH_PER_TORSO
        else:
            torso_len = self._torso_len.get(player_id, DEFAULT_TORSO_PX)
        torso_len = max(torso_len, 1.0)
        self._torso_len[player_id] = torso_len
        return torso_len


def palm_aim(hand: HandObservation, torso_len: float) -> PalmAim | None:
    pointer = hand.landmarks[HAND_KNUCKLES].mean(axis=0) - hand.landmarks[HAND_WRIST]
    pointer_len = float(np.linalg.norm(pointer))
    # A short palm on screen is one pointing at the camera, which has no usable direction.
    if pointer_len / torso_len < MIN_PALM_LENGTH:
        return None
    direction = pointer / pointer_len
    elevation = math.atan2(-direction[1], abs(direction[0]))
    valid = ELEVATION_RANGE[0] <= elevation <= ELEVATION_RANGE[1]
    return PalmAim(direction, elevation, pointer_len / torso_len, valid)


def head_height(player: Player) -> float:
    if player.face is not None:
        return float((player.face.bbox[1] + player.face.bbox[3]) / 2)
    return float(player.smoothed[NOSE][1])
