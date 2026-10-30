from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from highnoon.config import TrackingConfig
from highnoon.contracts import FaceObservation, Perception, Player, PoseObservation

NOSE = 0
SHOULDERS = [11, 12]
POSE_WRISTS = (15, 16)
HAND_WRIST = 0
MIN_HAND_GATE_PX = 60.0
HAND_GATE_PER_SHOULDER_WIDTH = 0.8
NO_MATCH = 1e6
UPDATE_MIN_GAP_PER_WIDTH = 0.12
PENDING_GAP_S = 0.4
# Past this, a missing player can only come back by face: a body near where they left could be anyone.
TRACK_GAP_S = 1.5
RECENT_FACE_SCORES = 3
CANDIDATE_SAMPLES = 8
FACE_TOP_K = 3


@dataclass(eq=False)
class Candidate:
    """Someone without an id, followed by position while evidence of who they are builds up."""

    center: np.ndarray
    first_t: float
    last_t: float
    embeddings: deque = field(default_factory=lambda: deque(maxlen=CANDIDATE_SAMPLES))


@dataclass
class Doubt:
    first_t: float
    last_t: float


@dataclass
class Verdict:
    cost: float
    plausible: bool
    confirmed: bool


class IdentityTracker:
    """Slot-based identity: the people seen before `lock()` are enrolled and nobody else ever gets an id.

    Before `lock()` (the enrolment window) a slot is kept for `lost_timeout_s` after its player leaves,
    while collecting face samples. After it, a slot stays reserved for the rest of the game. Either way a
    body is followed by position only while it stays in view; a returning person gets their slot back once
    their face matches its owner's, clearly better than any other player's. Someone whose face cannot be
    seen waits without an id. A slot that is being tracked by position is also checked against its owner's
    face, so a stranger who steps into the spot a player just left, or two players whose ids swapped while
    crossing, are released and sorted out again.
    """

    def __init__(self, cfg: TrackingConfig, max_players: int = 2):
        self.cfg = cfg
        self.players = [Player(id=i + 1) for i in range(max_players)]
        self._active = [False] * max_players
        self._last_center: list[np.ndarray | None] = [None] * max_players
        self._claimed = [False] * max_players
        self._face_samples: list[list[np.ndarray]] = [[] for _ in range(max_players)]
        self._gallery: list[np.ndarray | None] = [None] * max_players
        self._recent_faces = [deque(maxlen=cfg.face_recent_size) for _ in range(max_players)]
        self._candidates: list[Candidate] = []
        self._doubts: dict[int, Doubt] = {}
        self.locked = False
        self.reassigned: set[int] = set()

    def lock(self) -> None:
        """End enrolment: freeze each player's face gallery from what was seen."""
        if self.locked:
            return
        self.locked = True
        for slot, samples in enumerate(self._face_samples):
            self._gallery[slot] = spread(samples, self.cfg.face_gallery_size)
        self._face_samples = [[] for _ in self.players]

    def update(self, perception: Perception) -> list[Player]:
        t = perception.frame.t_capture
        width = perception.frame.size[0]
        self.reassigned = set()
        self._release_lost_players(t)

        poses = perception.poses
        centers = [pose.center() for pose in poses]
        faces = match_faces(poses, perception.faces)
        embeddings = [face.embedding if face is not None else None for face in faces]
        assignment = self._match_bodies_to_slots(centers, embeddings, width, t)
        for player in self.players:
            player.visible, player.pose, player.face, player.hands = False, None, None, {}
        apart = well_separated(centers, UPDATE_MIN_GAP_PER_WIDTH * width)
        for detection, slot in assignment.items():
            self._claim_slot(slot, poses[detection], centers[detection], t)
            self.players[slot].face = faces[detection]
            if apart and embeddings[detection] is not None:
                self._learn(slot, embeddings[detection])

        self._attach_hands(perception)
        return self.players

    def _release_lost_players(self, t: float) -> None:
        for slot, player in enumerate(self.players):
            lost_too_long = not player.visible and t - player.last_seen > self.cfg.lost_timeout_s
            if self._active[slot] and lost_too_long:
                self._active[slot] = False
                self._last_center[slot] = None
                self._doubts.pop(slot, None)
                if not self.locked:
                    self._claimed[slot] = False
                    self._face_samples[slot] = []

    def _claim_slot(self, slot: int, pose, center: np.ndarray, t: float) -> None:
        player = self.players[slot]
        if not self._claimed[slot]:
            self.reassigned.add(player.id)
            self._claimed[slot] = True
        self._active[slot] = True
        player.pose = pose
        player.visible = True
        player.last_seen = t
        self._last_center[slot] = center

    def _learn(self, slot: int, embedding: np.ndarray) -> None:
        if not self.locked:
            samples = self._face_samples[slot]
            samples.append(embedding)
            del samples[: -self.cfg.enrol_max_samples]
        elif (
            not self._has_face(slot)
            or self._face_similarity(slot, embedding[None])[0] >= self.cfg.face_accept
        ):
            self._recent_faces[slot].append(embedding)

    def _match_bodies_to_slots(
        self, centers: list[np.ndarray], embeddings: list[np.ndarray | None], width: int, t: float
    ) -> dict[int, int]:
        if not centers:
            return {}
        cost = self._distance_cost(centers, self.cfg.max_match_distance * width, t)
        rows, cols = linear_sum_assignment(cost)
        assignment = {r: c for r, c in zip(rows, cols, strict=True) if cost[r, c] < NO_MATCH}
        if self.locked:
            self._verify_owners(assignment, embeddings, t)

        newcomers = [r for r in range(len(centers)) if r not in assignment]
        held = self._return_known_players(newcomers, centers, embeddings, assignment, width, t)

        taken = set(assignment.values())
        free_slots = [s for s in range(len(self.players)) if not self._claimed[s] and s not in taken]
        newcomers = sorted(
            (r for r in newcomers if r not in assignment and r not in held), key=lambda r: centers[r][0]
        )
        for detection, slot in zip(newcomers, free_slots, strict=False):
            assignment[detection] = slot
        return assignment

    def _verify_owners(
        self, assignment: dict[int, int], embeddings: list[np.ndarray | None], t: float
    ) -> None:
        """Release a slot whose tracked body keeps showing a face that is not its owner's."""
        for detection, slot in list(assignment.items()):
            embedding = embeddings[detection]
            if embedding is None or not self._has_face(slot):
                continue
            similarity = self._face_similarity(slot, embedding[None])[0]
            if similarity >= self.cfg.face_accept:
                self._doubts.pop(slot, None)
                continue
            if similarity >= self.cfg.face_reject:
                continue
            doubt = self._doubts.get(slot)
            if doubt is None or t - doubt.last_t > PENDING_GAP_S:
                doubt = self._doubts[slot] = Doubt(t, t)
            doubt.last_t = t
            if t - doubt.first_t >= self.cfg.verify_confirm_s:
                del assignment[detection]
                del self._doubts[slot]
                self._active[slot] = False
                self._last_center[slot] = None

    def _return_known_players(
        self,
        newcomers: list[int],
        centers: list[np.ndarray],
        embeddings: list[np.ndarray | None],
        assignment: dict[int, int],
        width: int,
        t: float,
    ) -> set[int]:
        """Hand a slot back to someone who left the frame, once their face clearly matches its owner's.

        Returns the detections held back (no id yet) so they do not turn into new players meanwhile.
        """
        candidates = self._track_candidates(newcomers, centers, embeddings, width, t)
        taken = set(assignment.values())
        away = [s for s in range(len(self.players)) if self._claimed[s] and s not in taken]
        if not (newcomers and away):
            return set()
        verdicts = [[self._judge(candidates[d], slot, t) for slot in away] for d in newcomers]
        held = {d for d, row in zip(newcomers, verdicts, strict=True) if any(v.plausible for v in row)}
        cost = np.array([[v.cost for v in row] for row in verdicts])
        for i, j in zip(*linear_sum_assignment(cost), strict=True):
            if verdicts[i][j].confirmed:
                detection = newcomers[i]
                assignment[detection] = away[j]
                held.discard(detection)
                self._candidates.remove(candidates[detection])
        return held

    def _track_candidates(
        self,
        newcomers: list[int],
        centers: list[np.ndarray],
        embeddings: list[np.ndarray | None],
        width: int,
        t: float,
    ) -> dict[int, Candidate]:
        self._candidates = [c for c in self._candidates if t - c.last_t <= PENDING_GAP_S]
        matched: dict[int, Candidate] = {}
        if self._candidates and newcomers:
            last = np.array([c.center for c in self._candidates])
            cost = np.linalg.norm(np.array([centers[d] for d in newcomers])[:, None] - last[None], axis=2)
            for i, j in zip(*linear_sum_assignment(cost), strict=True):
                if cost[i, j] <= self.cfg.max_match_distance * width:
                    matched[newcomers[i]] = self._candidates[j]
        for detection in newcomers:
            candidate = matched.get(detection)
            if candidate is None:
                candidate = matched[detection] = Candidate(centers[detection], t, t)
                self._candidates.append(candidate)
            candidate.center, candidate.last_t = centers[detection], t
            if embeddings[detection] is not None:
                candidate.embeddings.append(embeddings[detection])
        return matched

    def _judge(self, candidate: Candidate, slot: int, t: float) -> Verdict:
        """How well a candidate's face fits an away slot. Until a face is seen they could be anyone."""
        if not candidate.embeddings or not self._has_face(slot):
            return Verdict(NO_MATCH, True, False)
        seen = np.array(candidate.embeddings)[-RECENT_FACE_SCORES:]
        score = float(self._face_similarity(slot, seen).mean())
        rivals = [s for s in range(len(self.players)) if s != slot and self._claimed[s] and self._has_face(s)]
        clearly_best = all(
            score - float(self._face_similarity(s, seen).mean()) >= self.cfg.face_margin for s in rivals
        )
        confirmed = (
            score >= self.cfg.face_accept
            and clearly_best
            and len(candidate.embeddings) >= self.cfg.face_min_samples
            and t - candidate.first_t >= self.cfg.reid_confirm_s
        )
        return Verdict(1.0 - score, score >= self.cfg.face_reject, confirmed)

    def _known_faces(self, slot: int) -> list[np.ndarray]:
        known = [self._gallery[slot]] if self._gallery[slot] is not None else []
        for faces in (self._recent_faces[slot], self._face_samples[slot]):
            if faces:
                known.append(np.array(faces))
        return known

    def _has_face(self, slot: int) -> bool:
        return bool(self._known_faces(slot))

    def _face_similarity(self, slot: int, embeddings: np.ndarray) -> np.ndarray:
        """Per embedding: cosine similarity to the closest few of the owner's known faces."""
        known = self._known_faces(slot)
        similarities = embeddings @ np.concatenate(known).T
        top = min(FACE_TOP_K, similarities.shape[1])
        return np.sort(similarities, axis=1)[:, -top:].mean(axis=1)

    def _distance_cost(self, centers: list[np.ndarray], max_distance: float, t: float) -> np.ndarray:
        cost = np.full((len(centers), len(self.players)), NO_MATCH)
        for slot, player in enumerate(self.players):
            last = self._last_center[slot]
            if self._active[slot] and last is not None and t - player.last_seen <= TRACK_GAP_S:
                distance = np.linalg.norm(np.asarray(centers) - last, axis=1)
                cost[:, slot] = np.where(distance <= max_distance, distance, NO_MATCH)
        return cost

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


def match_faces(poses: list[PoseObservation], faces: list[FaceObservation]) -> list[FaceObservation | None]:
    """The face belonging to each body (nearest to its nose, within one face size), or None."""
    matched: list[FaceObservation | None] = [None] * len(poses)
    if not poses or not faces:
        return matched
    noses = np.array([pose.keypoints[NOSE] for pose in poses])
    face_centers = np.array([f.center() for f in faces])
    cost = np.linalg.norm(face_centers[:, None, :] - noses[None, :, :], axis=2)
    for r, c in zip(*linear_sum_assignment(cost), strict=True):
        face = faces[r]
        face_size = max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
        if cost[r, c] <= face_size:
            matched[c] = face
    return matched


def spread(samples: list[np.ndarray], count: int) -> np.ndarray | None:
    """Up to `count` samples evenly spaced over the enrolment, so the gallery covers every pose seen."""
    if not samples:
        return None
    picks = np.unique(np.linspace(0, len(samples) - 1, min(count, len(samples))).round().astype(int))
    return np.array([samples[i] for i in picks])


def hand_gate(player: Player) -> float:
    left, right = player.pose.keypoints[SHOULDERS]
    shoulder_width = float(np.linalg.norm(left - right))
    return max(MIN_HAND_GATE_PX, HAND_GATE_PER_SHOULDER_WIDTH * shoulder_width)


def well_separated(centers: list[np.ndarray], gap: float) -> bool:
    return all(np.linalg.norm(a - b) >= gap for i, a in enumerate(centers) for b in centers[i + 1 :])
