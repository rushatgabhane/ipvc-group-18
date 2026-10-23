from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from highnoon.config import FilterConfig
from highnoon.contracts import MotionSignals, Player
from highnoon.core.filters import OneEuroFilter

NOSE = 0
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP = 23, 24
ARMS = ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST))
RIGHT_ARM = 1

STRAIGHT_ARM_REACH = 0.8
MIN_FOREARM_LENGTH = 0.3
ELEVATION_RANGE = (math.radians(-50), math.radians(80))
SHOULDER_WIDTH_PER_TORSO = 0.8
DEFAULT_TORSO_PX = 200.0
AIM_MIN_CUTOFF = 0.8
AIM_BETA = 0.6


@dataclass
class ArmAim:
    direction: np.ndarray
    elevation: float
    extension: float
    valid: bool


@dataclass
class SmoothedAim:
    arm: int
    angle_filter: OneEuroFilter
    angle: float


class MotionFilter:
    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        self._keypoint_filters: dict[int, OneEuroFilter] = {}
        self._aiming_arm: dict[int, int] = {}
        self._torso_len: dict[int, float] = {}
        self._aim: dict[int, SmoothedAim] = {}

    def update(self, players: list[Player], t: float, reassigned: set[int]) -> None:
        for player in players:
            if player.id in reassigned:
                self._forget(player.id)
            if player.pose is None:
                player.signals = None
                continue
            reliable = player.pose.visibility >= self.cfg.min_visibility
            player.smoothed = self._keypoint_filter(player.id)(player.pose.keypoints, t, mask=reliable)
            player.signals = self._signals(player.id, player.smoothed, reliable)
            player.signals.hand_openness = self._aiming_hand_openness(player)
            player.signals.head_y = head_height(player)
            self._smooth_aim(player.id, player.signals, t)

    def _forget(self, player_id: int) -> None:
        self._keypoint_filter(player_id).reset()
        self._aiming_arm.pop(player_id, None)
        self._torso_len.pop(player_id, None)
        self._aim.pop(player_id, None)

    def _keypoint_filter(self, player_id: int) -> OneEuroFilter:
        if player_id not in self._keypoint_filters:
            cfg = self.cfg
            self._keypoint_filters[player_id] = OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
        return self._keypoint_filters[player_id]

    def _aiming_hand_openness(self, player: Player) -> float | None:
        wrist = ARMS[self._aiming_arm[player.id]][2]
        hand = player.hands.get(wrist)
        return hand.openness if hand is not None else None

    def _smooth_aim(self, player_id: int, signals: MotionSignals, t: float) -> None:
        if not signals.aim_valid:
            self._aim.pop(player_id, None)
            return
        arm = self._aiming_arm[player_id]
        raw_angle = math.atan2(float(signals.aim_dir[1]), float(signals.aim_dir[0]))
        aim = self._aim.get(player_id)
        if aim is None or aim.arm != arm:
            aim = SmoothedAim(arm, OneEuroFilter(AIM_MIN_CUTOFF, AIM_BETA, 1.0), raw_angle)
        unwrapped = aim.angle + math.remainder(raw_angle - aim.angle, math.tau)
        aim.angle = float(aim.angle_filter(np.array([unwrapped], np.float32), t)[0])
        self._aim[player_id] = aim
        signals.aim_dir = np.array([math.cos(aim.angle), math.sin(aim.angle)], np.float32)

    def _signals(self, player_id: int, keypoints: np.ndarray, reliable: np.ndarray) -> MotionSignals:
        torso_len = self._estimate_torso_len(player_id, keypoints, reliable)
        aims = [arm_aim(keypoints, reliable, joints, torso_len) for joints in ARMS]
        arm = self._choose_aiming_arm(player_id, aims)
        aim = aims[arm] or ArmAim(np.array([1.0, 0.0], np.float32), 0.0, 0.0, False)
        return MotionSignals(
            torso_len=torso_len,
            aim_valid=aim.valid,
            extension=aim.extension,
            aim_origin=keypoints[ARMS[arm][2]].copy(),
            aim_dir=np.asarray(aim.direction, np.float32),
            elevation=aim.elevation,
        )

    def _choose_aiming_arm(self, player_id: int, aims: list[ArmAim | None]) -> int:
        arm = self._aiming_arm.get(player_id, RIGHT_ARM)
        other = 1 - arm
        current_usable = aims[arm] is not None and aims[arm].valid
        other_usable = aims[other] is not None and aims[other].valid
        if not current_usable and other_usable:
            arm = other
        self._aiming_arm[player_id] = arm
        return arm

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


def arm_aim(
    keypoints: np.ndarray, reliable: np.ndarray, joints: tuple[int, int, int], torso_len: float
) -> ArmAim | None:
    shoulder, elbow, wrist = joints
    if not (reliable[elbow] and reliable[wrist]):
        return None
    forearm = keypoints[wrist] - keypoints[elbow]
    forearm_len = float(np.linalg.norm(forearm)) / torso_len
    reach = (
        float(np.linalg.norm(keypoints[wrist] - keypoints[shoulder])) / torso_len
        if reliable[shoulder]
        else 0.0
    )
    pointer = keypoints[wrist] - keypoints[shoulder] if reach >= STRAIGHT_ARM_REACH else forearm
    pointer_len = float(np.linalg.norm(pointer))
    if pointer_len < 1e-6 or forearm_len < MIN_FOREARM_LENGTH:
        return None
    direction = pointer / pointer_len
    elevation = math.atan2(-direction[1], abs(direction[0]))
    valid = ELEVATION_RANGE[0] <= elevation <= ELEVATION_RANGE[1]
    extension = reach if reliable[shoulder] else forearm_len
    return ArmAim(direction, elevation, extension, valid)


def head_height(player: Player) -> float:
    if player.face is not None:
        return float((player.face.bbox[1] + player.face.bbox[3]) / 2)
    return float(player.smoothed[NOSE][1])
