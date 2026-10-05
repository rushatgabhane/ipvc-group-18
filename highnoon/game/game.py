"""T4: game rules. Turns recognised actions into game state: shots, hits, HP, rounds.

Modes (chosen from how many players T3 currently sees):
  idle      nobody in frame
  practice  one player: shoot bottles that spawn on the other side of the screen
  duel      two players: shoot each other; first to 0 HP loses the round

A shot is resolved when its bullet arrives, `bullet_travel_s` after firing, against the
target's geometry *at arrival time*. That delay is what makes ducking a real defence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from highnoon.config import GameConfig
from highnoon.contracts import Player
from highnoon.game.actions import ActionFrame, PlayerActions, Shot
from highnoon.game.geometry import ray_circle, ray_exit_distance, ray_polygon, rect_polygon

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, NOSE = 11, 12, 23, 24, 0
COVER_FOLLOW = 0.15  # crate x follows the player with this lerp factor per frame


@dataclass
class Cover:
    x0: float
    y0: float
    x1: float
    y1: float

    def polygon(self) -> np.ndarray:
        return rect_polygon(self.x0, self.y0, self.x1, self.y1)


@dataclass
class PlayerState:
    actions: PlayerActions
    hp: int
    ammo: int
    frame: ActionFrame = field(default_factory=ActionFrame)
    cover: Cover | None = None
    last_hit_t: float = -math.inf


@dataclass
class Bullet:
    shooter: int
    origin: np.ndarray
    direction: np.ndarray
    length: float  # px to where it was heading at fire time (for drawing the tracer)
    t_fire: float
    t_arrive: float


@dataclass
class Target:
    center: np.ndarray
    radius: float
    alive: bool = True
    respawn_t: float = 0.0


@dataclass
class Effect:
    kind: str  # "hit", "headshot", "blocked", "shatter", "muzzle"
    pos: np.ndarray
    t: float
    player: int  # shooter


@dataclass
class Aim:
    """What the laser sight of a player currently points at (for rendering)."""

    origin: np.ndarray
    end: np.ndarray
    on_target: bool


class Game:
    def __init__(self, cfg: GameConfig, frame_size: tuple[int, int], seed: int = 0):
        self.cfg = cfg
        self.width, self.height = frame_size
        self.states: dict[int, PlayerState] = {}
        self.bullets: list[Bullet] = []
        self.effects: list[Effect] = []
        self.targets: list[Target] = []
        self.aims: dict[int, Aim] = {}
        self.mode = "idle"
        self.winner: int | None = None
        self.round_over_t: float | None = None
        self.scores: dict[int, int] = {}
        self._rng = np.random.default_rng(seed)
        self._players: dict[int, Player] = {}

    # ---------------------------------------------------------------- update
    def update(self, players: list[Player], t: float, reassigned: set[int]) -> None:
        self._players = {p.id: p for p in players}
        for p in players:
            if p.id in reassigned or (p.visible and p.id not in self.states):
                self.states[p.id] = self._new_state()

        visible = [p for p in players if p.visible]
        self.mode = {0: "idle", 1: "practice"}.get(len(visible), "duel")
        if self.mode == "practice":
            self._update_targets(visible[0], t)
        else:
            self.targets.clear()

        if self.round_over_t is not None and t - self.round_over_t >= self.cfg.round_end_s:
            self.reset_round()

        for p in players:
            st = self.states.get(p.id)
            if st is None:  # never seen yet
                continue
            st.frame = st.actions.update(p.signals if p.visible else None, t)
            self._update_cover(p, st)
            if st.frame.reload and st.ammo < self.cfg.max_ammo:
                st.ammo = self.cfg.max_ammo
            if st.frame.shot is not None and self.round_over_t is None:
                self._fire(p.id, st, st.frame.shot, t)

        self._resolve_bullets(t)
        self._update_aims(players)
        self.effects = [e for e in self.effects if t - e.t < 0.8]

    def reset_round(self) -> None:
        for st in self.states.values():
            st.hp, st.ammo = self.cfg.max_hp, self.cfg.max_ammo
        self.bullets.clear()
        self.winner = None
        self.round_over_t = None

    def _new_state(self) -> PlayerState:
        return PlayerState(PlayerActions(self.cfg), self.cfg.max_hp, self.cfg.max_ammo)

    # ---------------------------------------------------------------- shooting
    def _fire(self, pid: int, st: PlayerState, shot: Shot, t: float) -> None:
        if st.ammo <= 0:
            return
        st.ammo -= 1
        length, _ = self._cast(pid, shot.origin, shot.direction)
        self.bullets.append(Bullet(pid, shot.origin, shot.direction, length, t, t + self.cfg.bullet_travel_s))
        self.effects.append(Effect("muzzle", shot.origin, t, pid))

    def _resolve_bullets(self, t: float) -> None:
        remaining = []
        for b in self.bullets:
            if t < b.t_arrive:
                remaining.append(b)
                continue
            dist, what = self._cast(b.shooter, b.origin, b.direction)
            pos = b.origin + b.direction * dist
            kind, target = what
            if kind in ("head", "body") and self.round_over_t is None:
                self._damage(b.shooter, target, kind, pos, t)
            elif kind == "cover":
                self.effects.append(Effect("blocked", pos, t, b.shooter))
            elif kind == "bottle":
                target.alive = False
                target.respawn_t = t + 1.2
                self.scores[b.shooter] = self.scores.get(b.shooter, 0) + 1
                self.effects.append(Effect("shatter", pos, t, b.shooter))
        self.bullets = remaining

    def _damage(self, shooter: int, target_id: int, kind: str, pos: np.ndarray, t: float) -> None:
        st = self.states[target_id]
        st.hp = max(0, st.hp - (self.cfg.head_damage if kind == "head" else self.cfg.body_damage))
        st.last_hit_t = t
        self.effects.append(Effect("headshot" if kind == "head" else "hit", pos, t, shooter))
        if st.hp == 0:
            self.winner = shooter
            self.round_over_t = t
            self.scores[shooter] = self.scores.get(shooter, 0) + 1

    def _cast(self, shooter: int, origin: np.ndarray, direction: np.ndarray):
        """First thing a ray from `shooter` hits. Returns (distance, (kind, target))."""
        best = ray_exit_distance(origin, direction, self.width, self.height)
        hit: tuple[str, object] = ("miss", None)

        def consider(dist: float | None, what: tuple[str, object]) -> None:
            nonlocal best, hit
            if dist is not None and dist < best:
                best, hit = dist, what

        for pid, p in self._players.items():
            if pid == shooter or not p.visible or p.smoothed is None:
                continue
            st = self.states[pid]
            if st.cover is not None:
                consider(ray_polygon(origin, direction, st.cover.polygon()), ("cover", pid))
            head, torso = self._hitboxes(p)
            if st.frame.ducked:
                # Behind cover: anything that would have hit the player is stopped by the crate.
                dists = (ray_circle(origin, direction, *head), ray_polygon(origin, direction, torso))
                body_hit = min((d for d in dists if d is not None), default=None)
                consider(body_hit, ("cover", pid))
            else:
                consider(ray_circle(origin, direction, *head), ("head", pid))
                consider(ray_polygon(origin, direction, torso), ("body", pid))
        for target in self.targets:
            if target.alive:
                consider(ray_circle(origin, direction, target.center, target.radius), ("bottle", target))
        return best, hit

    def _hitboxes(self, p: Player) -> tuple[tuple[np.ndarray, float], np.ndarray]:
        kp = p.smoothed
        torso_len = p.signals.torso_len if p.signals else 100.0
        if p.face is not None:
            x0, y0, x1, y1 = p.face.bbox
            head = (np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32), 0.6 * max(x1 - x0, y1 - y0))
        else:
            head = (kp[NOSE].copy(), 0.35 * torso_len)
        # Torso quad, widened a little so shots at the arms' base still count.
        quad = kp[[L_SHOULDER, R_SHOULDER, R_HIP, L_HIP]].astype(np.float32)
        center = quad.mean(axis=0)
        torso = center + (quad - center) * 1.15
        return head, torso

    # ---------------------------------------------------------------- cover, aims, targets
    def _update_cover(self, p: Player, st: PlayerState) -> None:
        acts = st.actions
        if not p.visible or p.smoothed is None or acts.ref_shoulder is None:
            return
        kp = p.smoothed
        hip_x = float((kp[L_HIP][0] + kp[R_HIP][0]) / 2)
        half_w = 0.9 * acts.ref_torso
        top = acts.ref_shoulder + self.cfg.cover_top * acts.ref_torso
        if st.cover is None:
            st.cover = Cover(hip_x - half_w, top, hip_x + half_w, self.height)
            return
        cx = (st.cover.x0 + st.cover.x1) / 2
        cx += COVER_FOLLOW * (hip_x - cx)
        st.cover = Cover(cx - half_w, top, cx + half_w, self.height)

    def _update_aims(self, players: list[Player]) -> None:
        self.aims = {}
        for p in players:
            st = self.states.get(p.id)
            if st is None or not (p.visible and st.frame.aiming and p.signals is not None):
                continue
            o, d = p.signals.aim_origin, p.signals.aim_dir
            dist, (kind, _) = self._cast(p.id, o, d)
            self.aims[p.id] = Aim(o, o + d * dist, kind in ("head", "body", "bottle"))

    def _update_targets(self, player: Player, t: float) -> None:
        if not self.targets:
            self.targets = [
                Target(np.zeros(2, np.float32), 0.0, alive=False) for _ in range(self.cfg.practice_targets)
            ]
        for target in self.targets:
            if not target.alive and t >= target.respawn_t:
                self._spawn_target(target, player)

    def _spawn_target(self, target: Target, player: Player) -> None:
        """Place a bottle where it is reachable but not on the player or another bottle."""
        radius = 0.04 * self.height
        kp = player.smoothed if player.smoothed is not None else None
        if kp is not None:
            body_x0 = float(min(kp[L_SHOULDER][0], kp[R_SHOULDER][0]))
            body_x1 = float(max(kp[L_SHOULDER][0], kp[R_SHOULDER][0]))
            margin = 0.5 * (player.signals.torso_len if player.signals else 100.0) + radius
        else:
            body_x0 = body_x1 = self.width / 2
            margin = 0.0
        others = [t.center for t in self.targets if t.alive and t is not target]
        top = 0.18 * self.height  # keep clear of the banner
        center = None
        for _ in range(30):  # rejection sampling; the free area is large, so this rarely loops
            c = np.array(
                [self._rng.uniform(0.06, 0.94) * self.width, self._rng.uniform(top, 0.6 * self.height)],
                np.float32,
            )
            if body_x0 - margin < c[0] < body_x1 + margin:
                continue
            if any(np.linalg.norm(c - o) < 4 * radius for o in others):
                continue
            center = c
            break
        if center is None:
            return  # no free spot this frame; try again next frame
        target.center = center
        target.radius = radius
        target.alive = True
