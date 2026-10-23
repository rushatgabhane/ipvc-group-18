from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from highnoon.config import TrackingConfig
from highnoon.contracts import Perception, Player

NOSE = 0
SHOULDERS = [11, 12]
POSE_WRISTS = (15, 16)
HAND_WRIST = 0
MIN_HAND_GATE_PX = 60.0
HAND_GATE_PER_SHOULDER_WIDTH = 0.8
NO_MATCH = 1e6


class IdentityTracker:
    def __init__(self, cfg: TrackingConfig, max_players: int = 2):
        self.cfg = cfg
        self.players = [Player(id=i + 1) for i in range(max_players)]
        self._active = [False] * max_players
        self._last_center: list[np.ndarray | None] = [None] * max_players
        self.reassigned: set[int] = set()

    def update(self, perception: Perception) -> list[Player]:
        t = perception.frame.t_capture
        self.reassigned = set()
        self._release_lost_players(t)

        centers = [pose.center() for pose in perception.poses]
        assignment = self._match_bodies_to_slots(centers, perception.frame.size[0])
        for player in self.players:
            player.visible, player.pose, player.face, player.hands = False, None, None, {}
        for detection, slot in assignment.items():
            self._claim_slot(slot, perception.poses[detection], centers[detection], t)

        self._attach_faces(perception)
        self._attach_hands(perception)
        return self.players

    def _release_lost_players(self, t: float) -> None:
        for slot, player in enumerate(self.players):
            lost_too_long = not player.visible and t - player.last_seen > self.cfg.lost_timeout_s
            if self._active[slot] and lost_too_long:
                self._active[slot] = False
                self._last_center[slot] = None

    def _claim_slot(self, slot: int, pose, center: np.ndarray, t: float) -> None:
        player = self.players[slot]
        if not self._active[slot]:
            self.reassigned.add(player.id)
            self._active[slot] = True
        player.pose = pose
        player.visible = True
        player.last_seen = t
        self._last_center[slot] = center

    def _match_bodies_to_slots(self, centers: list[np.ndarray], width: int) -> dict[int, int]:
        if not centers:
            return {}
        cost = self._distance_cost(centers, self.cfg.max_match_distance * width)
        rows, cols = linear_sum_assignment(cost)
        assignment = {r: c for r, c in zip(rows, cols, strict=True) if cost[r, c] < NO_MATCH}

        taken = set(assignment.values())
        free_slots = [s for s in range(len(self.players)) if not self._active[s] and s not in taken]
        newcomers = sorted(
            (r for r in range(len(centers)) if r not in assignment), key=lambda r: centers[r][0]
        )
        for detection, slot in zip(newcomers, free_slots, strict=False):
            assignment[detection] = slot
        return assignment

    def _distance_cost(self, centers: list[np.ndarray], max_distance: float) -> np.ndarray:
        cost = np.full((len(centers), len(self.players)), NO_MATCH)
        for slot in range(len(self.players)):
            last = self._last_center[slot]
            if self._active[slot] and last is not None:
                distance = np.linalg.norm(np.asarray(centers) - last, axis=1)
                cost[:, slot] = np.where(distance <= max_distance, distance, NO_MATCH)
        return cost

    def _attach_faces(self, perception: Perception) -> None:
        visible = [p for p in self.players if p.visible]
        if not visible or not perception.faces:
            return
        noses = np.array([p.pose.keypoints[NOSE] for p in visible])
        face_centers = np.array([f.center() for f in perception.faces])
        cost = np.linalg.norm(face_centers[:, None, :] - noses[None, :, :], axis=2)
        for r, c in zip(*linear_sum_assignment(cost), strict=True):
            face = perception.faces[r]
            face_size = max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
            if cost[r, c] <= face_size:
                visible[c].face = face

    def _attach_hands(self, perception: Perception) -> None:
        wrist_slots = [(p, wrist) for p in self.players if p.visible for wrist in POSE_WRISTS]
        if not wrist_slots or not perception.hands:
            return
        pose_wrists = np.array([p.pose.keypoints[wrist] for p, wrist in wrist_slots])
        hand_wrists = np.array([h.landmarks[HAND_WRIST] for h in perception.hands])
        cost = np.linalg.norm(hand_wrists[:, None, :] - pose_wrists[None, :, :], axis=2)
        for r, c in zip(*linear_sum_assignment(cost), strict=True):
            player, wrist = wrist_slots[c]
            if cost[r, c] <= hand_gate(player):
                player.hands[wrist] = perception.hands[r]


def hand_gate(player: Player) -> float:
    left, right = player.pose.keypoints[SHOULDERS]
    shoulder_width = float(np.linalg.norm(left - right))
    return max(MIN_HAND_GATE_PX, HAND_GATE_PER_SHOULDER_WIDTH * shoulder_width)
