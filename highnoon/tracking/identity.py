"""T3: persistent player identities.

Baseline: optimal (Hungarian) assignment of detected bodies to player slots by torso-centre
distance, gated by a maximum jump. A slot survives short detection losses for
`lost_timeout_s`, so a briefly occluded player gets the same id back. Faces are then assigned
to players by matching the face centre to the pose nose keypoint.

Known weakness to work on (T3 owner): distance-only matching can swap ids when players
cross. Next steps are a constant-velocity Kalman prediction and an appearance cue
(e.g. torso HSV histogram) added to the cost matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from highnoon.config import TrackingConfig
from highnoon.contracts import Perception, Player

NOSE = 0
_INF_COST = 1e6


class IdentityTracker:
    def __init__(self, cfg: TrackingConfig, max_players: int = 2):
        self.cfg = cfg
        self.players = [Player(id=i + 1) for i in range(max_players)]
        self._active = [False] * max_players  # slot holds an identity (seen within timeout)
        self._last_center: list[np.ndarray | None] = [None] * max_players
        self.reassigned: set[int] = set()  # player ids that got a new identity this frame

    def update(self, perception: Perception) -> list[Player]:
        frame = perception.frame
        t = frame.t_capture
        width = frame.size[0]
        self.reassigned = set()

        # Release identities that have been lost for too long.
        for i, p in enumerate(self.players):
            if self._active[i] and not p.visible and t - p.last_seen > self.cfg.lost_timeout_s:
                self._active[i] = False
                self._last_center[i] = None

        centers = [pose.center() for pose in perception.poses]
        assigned = self._assign(centers, width)

        for p in self.players:
            p.visible = False
            p.pose = None
            p.face = None
        for det_idx, slot in assigned.items():
            p = self.players[slot]
            if not self._active[slot]:
                self.reassigned.add(p.id)
                self._active[slot] = True
            p.pose = perception.poses[det_idx]
            p.visible = True
            p.last_seen = t
            self._last_center[slot] = centers[det_idx]

        self._assign_faces(perception)
        return self.players

    def _assign(self, centers: list[np.ndarray], width: int) -> dict[int, int]:
        """Map detection index -> player slot index."""
        if not centers:
            return {}
        max_dist = self.cfg.max_match_distance * width
        n_slots = len(self.players)
        cost = np.full((len(centers), n_slots), _INF_COST)
        for s in range(n_slots):
            if self._active[s] and self._last_center[s] is not None:
                d = np.linalg.norm(np.asarray(centers) - self._last_center[s], axis=1)
                cost[:, s] = np.where(d <= max_dist, d, _INF_COST)

        rows, cols = linear_sum_assignment(cost)
        result = {r: c for r, c in zip(rows, cols, strict=True) if cost[r, c] < _INF_COST}

        # New detections take free slots, left-to-right so Player 1 is the leftmost newcomer.
        free = [s for s in range(n_slots) if not self._active[s] and s not in result.values()]
        new = sorted((r for r in range(len(centers)) if r not in result), key=lambda r: centers[r][0])
        for r, s in zip(new, free, strict=False):
            result[r] = s
        return result

    def _assign_faces(self, perception: Perception) -> None:
        visible = [p for p in self.players if p.visible]
        if not visible or not perception.faces:
            return
        noses = np.array([p.pose.keypoints[NOSE] for p in visible])
        face_centers = np.array([f.center() for f in perception.faces])
        cost = np.linalg.norm(face_centers[:, None, :] - noses[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols, strict=True):
            face = perception.faces[r]
            # The nose must lie within ~one face-size of the face centre.
            size = max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
            if cost[r, c] <= size:
                visible[c].face = face
