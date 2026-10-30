from __future__ import annotations

from dataclasses import dataclass

import cv2
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
TORSO_CORNERS = [11, 12, 24, 23]
HIST_BINS = [3, 8, 8]
HIST_RANGES = [0, 256, 0, 256, 0, 256]
TORSO_SHRINK = 0.7
MIN_HALF_PIXELS = 80
MIN_CORNER_VISIBILITY = 0.5
UPDATE_MIN_GAP_PER_WIDTH = 0.12
PENDING_GAP_S = 0.4
CHEST_DEPTH_PER_SHOULDER_WIDTH = 0.7


@dataclass
class Pending:
    first_t: float
    last_t: float
    center: np.ndarray


class IdentityTracker:
    """Slot-based identity: the people seen before `lock()` are enrolled and nobody else ever gets an id.

    Before `lock()` (the enrolment window) slots behave as plain position trackers and collect appearance
    samples. After it, a slot stays reserved for the rest of the game even when its player leaves the frame,
    and a returning person gets it back only by matching the enrolled clothing signature.
    """

    def __init__(self, cfg: TrackingConfig, max_players: int = 2):
        self.cfg = cfg
        self.players = [Player(id=i + 1) for i in range(max_players)]
        self._active = [False] * max_players
        self._last_center: list[np.ndarray | None] = [None] * max_players
        self._claimed = [False] * max_players
        self._samples: list[list[np.ndarray]] = [[] for _ in range(max_players)]
        self._reference: list[np.ndarray | None] = [None] * max_players
        self._pending: dict[int, Pending] = {}
        self.locked = False
        self.reassigned: set[int] = set()

    def lock(self) -> None:
        """End enrolment: freeze each player's signature as the average of what was seen."""
        if self.locked:
            return
        self.locked = True
        for slot, samples in enumerate(self._samples):
            self._reference[slot] = mean_signature(samples) if samples else None
        self._samples = [[] for _ in self.players]

    def update(self, perception: Perception) -> list[Player]:
        t = perception.frame.t_capture
        width = perception.frame.size[0]
        self.reassigned = set()
        self._release_lost_players(t)

        centers = [pose.center() for pose in perception.poses]
        signatures = [torso_signature(perception.frame.image, pose) for pose in perception.poses]
        assignment = self._match_bodies_to_slots(centers, signatures, width, t)
        for player in self.players:
            player.visible, player.pose, player.face, player.hands = False, None, None, {}
        apart = well_separated(centers, UPDATE_MIN_GAP_PER_WIDTH * width)
        for detection, slot in assignment.items():
            self._claim_slot(slot, perception.poses[detection], centers[detection], t)
            self._learn(slot, signatures[detection], apart)

        self._attach_faces(perception)
        self._attach_hands(perception)
        return self.players

    def _release_lost_players(self, t: float) -> None:
        for slot, player in enumerate(self.players):
            lost_too_long = not player.visible and t - player.last_seen > self.cfg.lost_timeout_s
            if self._active[slot] and lost_too_long:
                self._active[slot] = False
                self._last_center[slot] = None
                if not self.locked:
                    self._claimed[slot] = False
                    self._samples[slot] = []

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

    def _learn(self, slot: int, signature: np.ndarray | None, apart: bool) -> None:
        if signature is None:
            return
        if not self.locked:
            samples = self._samples[slot]
            samples.append(signature)
            del samples[: -self.cfg.enrol_max_samples]
            return
        reference = self._reference[slot]
        if reference is None:
            self._reference[slot] = signature
        elif apart:
            self._reference[slot] = blend_signature(reference, signature, self.cfg.reid_blend)

    def _match_bodies_to_slots(
        self, centers: list[np.ndarray], signatures: list[np.ndarray | None], width: int, t: float
    ) -> dict[int, int]:
        if not centers:
            return {}
        cost = self._distance_cost(centers, self.cfg.max_match_distance * width)
        rows, cols = linear_sum_assignment(cost)
        assignment = {r: c for r, c in zip(rows, cols, strict=True) if cost[r, c] < NO_MATCH}

        newcomers = [r for r in range(len(centers)) if r not in assignment]
        held = self._return_known_players(newcomers, centers, signatures, assignment, width, t)

        taken = set(assignment.values())
        free_slots = [s for s in range(len(self.players)) if not self._claimed[s] and s not in taken]
        newcomers = sorted(
            (r for r in newcomers if r not in assignment and r not in held), key=lambda r: centers[r][0]
        )
        for detection, slot in zip(newcomers, free_slots, strict=False):
            assignment[detection] = slot
        return assignment

    def _return_known_players(
        self,
        newcomers: list[int],
        centers: list[np.ndarray],
        signatures: list[np.ndarray | None],
        assignment: dict[int, int],
        width: int,
        t: float,
    ) -> set[int]:
        """Hand a slot back to someone who left the frame, once they clearly look like its owner for a while.

        Returns the detections held back (no id yet) so they do not turn into new players meanwhile.
        """
        taken = set(assignment.values())
        away = [s for s in range(len(self.players)) if self._claimed[s] and not self._active[s] and s not in taken]
        self._pending = {s: p for s, p in self._pending.items() if t - p.last_t <= PENDING_GAP_S and s in away}
        if not (newcomers and away):
            return set()
        cost = np.full((len(newcomers), len(away)), NO_MATCH)
        for i, detection in enumerate(newcomers):
            for j, slot in enumerate(away):
                if self._reference[slot] is not None and signatures[detection] is not None:
                    cost[i, j] = signature_distance(self._reference[slot], signatures[detection])
        held: set[int] = set()
        for i, j in zip(*linear_sum_assignment(cost), strict=True):
            detection, slot = newcomers[i], away[j]
            if cost[i, j] > self.cfg.reid_max_distance:
                continue
            held.add(detection)
            if not self._clearly_best(slot, cost[i, j], signatures[detection]):
                continue
            pending = self._pending.get(slot)
            continues = pending is not None and (
                np.linalg.norm(centers[detection] - pending.center) <= self.cfg.max_match_distance * width
            )
            pending = pending if continues else Pending(t, t, centers[detection])
            pending.last_t, pending.center = t, centers[detection]
            self._pending[slot] = pending
            if t - pending.first_t >= self.cfg.reid_confirm_s:
                assignment[detection] = slot
                held.discard(detection)
                del self._pending[slot]
        return held

    def _clearly_best(self, slot: int, distance: float, signature: np.ndarray) -> bool:
        """The person must look much more like this slot's owner than like any other enrolled player."""
        for other, reference in enumerate(self._reference):
            if other != slot and self._claimed[other] and reference is not None:
                if signature_distance(reference, signature) - distance < self.cfg.reid_margin:
                    return False
        return True

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


def well_separated(centers: list[np.ndarray], gap: float) -> bool:
    return all(np.linalg.norm(a - b) >= gap for i, a in enumerate(centers) for b in centers[i + 1 :])


def has_half(signature: np.ndarray, half: int) -> bool:
    return bool(signature[half].any())


def mean_signature(samples: list[np.ndarray]) -> np.ndarray:
    """Average per torso half, using only the samples in which that half was seen."""
    mean = np.zeros_like(samples[0])
    for half in range(len(mean)):
        seen = [s[half] for s in samples if has_half(s, half)]
        if seen:
            mean[half] = np.mean(seen, axis=0)
    return mean


def blend_signature(reference: np.ndarray, signature: np.ndarray, rate: float) -> np.ndarray:
    blended = reference.copy()
    for half in range(len(blended)):
        if not has_half(signature, half):
            continue
        known = reference[half] if has_half(reference, half) else signature[half]
        blended[half] = known + rate * (signature[half] - known)
    return blended


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Bhattacharyya distance over the torso halves both signatures have (0 = identical, 1 = worst)."""
    shared = [h for h in range(len(a)) if has_half(a, h) and has_half(b, h)]
    if not shared:
        return 1.0
    return float(np.mean([cv2.compareHist(a[h], b[h], cv2.HISTCMP_BHATTACHARYYA) for h in shared]))


def torso_quads(pose, width: int, height: int) -> list[np.ndarray | None] | None:
    """Upper and lower torso quads; the lower is None when the hips cannot be trusted.

    Someone entering from the side or standing close often has no usable hips, so the upper half alone
    (shoulders and a patch of chest below them) is still enough to recognise them.
    """

    def inside(points: np.ndarray) -> bool:
        return bool((points >= 0).all() and (points[:, 0] < width).all() and (points[:, 1] < height).all())

    def seen(indices: list[int]) -> bool:
        return bool((pose.visibility[indices] >= MIN_CORNER_VISIBILITY).all())

    corners = pose.keypoints[TORSO_CORNERS].astype(np.float32)
    left_sh, right_sh, right_hip, left_hip = corners
    if seen(TORSO_CORNERS) and inside(corners):
        mid_left, mid_right = (left_sh + left_hip) / 2, (right_sh + right_hip) / 2
        return [
            np.array([left_sh, right_sh, mid_right, mid_left]),
            np.array([mid_left, mid_right, right_hip, left_hip]),
        ]
    if not (seen(SHOULDERS) and inside(corners[:2])):
        return None
    across = right_sh - left_sh
    down = np.array([-across[1], across[0]], np.float32)
    if down[1] < 0:
        down = -down
    down *= CHEST_DEPTH_PER_SHOULDER_WIDTH
    return [np.array([left_sh, right_sh, right_sh + down, left_sh + down]), None]


def torso_signature(image: np.ndarray, pose) -> np.ndarray | None:
    """Lab colour histograms of the upper and lower torso (an all-zero half means it was not seen).

    Clothing colour survives leaving the frame, unlike position. Splitting the torso keeps a red top over
    blue trousers distinct from the reverse.
    """
    height, width = image.shape[:2]
    quads = torso_quads(pose, width, height)
    if quads is None:
        return None
    points = np.concatenate([q for q in quads if q is not None])
    x0, y0 = np.maximum(points.min(axis=0), 0).astype(int)
    x1, y1 = np.minimum(points.max(axis=0).astype(int) + 1, (width, height))
    if x1 <= x0 or y1 <= y0:
        return None
    lab = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
    histograms = np.zeros((len(quads), int(np.prod(HIST_BINS))), np.float32)
    for half, quad in enumerate(quads):
        if quad is None:
            continue
        middle = quad.mean(axis=0)
        polygon = np.rint(middle + (quad - middle) * TORSO_SHRINK - (x0, y0)).astype(np.int32)
        mask = np.zeros(lab.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        if cv2.countNonZero(mask) >= MIN_HALF_PIXELS:
            hist = cv2.calcHist([lab], [0, 1, 2], mask, HIST_BINS, HIST_RANGES)
            histograms[half] = cv2.normalize(hist, None, 1.0, 0.0, cv2.NORM_L1).flatten()
    return histograms if histograms.any() else None
