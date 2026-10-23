"""T4: game rules. Turns recognised actions into game state: shots, hits, HP, rounds.

Modes (chosen from how many players T3 currently sees):
  idle      fewer than two players in frame: waiting for a duel
  duel      two players: shoot each other; first to 0 HP loses the round

A shot is resolved when its bullet arrives, `bullet_travel_s` after firing, against the
target's geometry *at arrival time*. That delay is what makes ducking a real defence.

Powerups (duel only): every `powerup_interval_s` one is tossed up from below the screen and
falls back out, like fruit in Fruit Ninja. Its path is a closed-form parabola, so its position
at any time (including a bullet's arrival time) is exact. Whoever shoots it gets the effect:
  heal    HP back to max_hp
  double  this player's hits do double damage for double_damage_s
  rocket  this player's next shot is a rocket on a ballistic arc: it explodes on impact and
          the blast reaches players behind the wall. Each frame the rocket's path since the last
          frame is swept as a line segment, so it cannot tunnel through a thin target.
  grenade this player's next shot lobs a grenade at the opponent. It homes in on them and
          lands after grenade_flight_s; if they are not ducked at that moment, they die.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from highnoon.config import GameConfig
from highnoon.contracts import Player
from highnoon.game.actions import ActionFrame, PlayerActions, Shot
from highnoon.game.geometry import ray_circle, ray_exit_distance, ray_polygon, rect_polygon

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, NOSE = 11, 12, 23, 24, 0
POWERUP_KINDS = ("heal", "double", "rocket", "grenade")


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
    last_hit_t: float = -math.inf
    double_until: float = -math.inf
    rockets: int = 0
    grenades: int = 0


@dataclass
class Bullet:
    shooter: int
    origin: np.ndarray
    direction: np.ndarray
    length: float  # px to where it was heading at fire time (for drawing the tracer)
    t_fire: float
    t_arrive: float


@dataclass
class Grenade:
    thrower: int
    target: int
    origin: np.ndarray
    t_throw: float
    t_land: float
    pos: np.ndarray  # current position (updated by the game, read by the renderer)
    landing: np.ndarray  # where it is heading (the target's head, followed every frame)

    def progress(self, t: float) -> float:
        return min(1.0, (t - self.t_throw) / (self.t_land - self.t_throw))


@dataclass
class Rocket:
    shooter: int
    origin: np.ndarray
    velocity: np.ndarray  # px/s at launch
    gravity: float  # px/s^2, downward
    t_fire: float
    pos: np.ndarray  # where it was at the last update

    def at(self, t: float) -> np.ndarray:
        dt = t - self.t_fire
        return self.origin + self.velocity * dt + np.array([0.0, 0.5 * self.gravity * dt * dt], np.float32)

    def heading(self, t: float) -> np.ndarray:
        return self.velocity + np.array([0.0, self.gravity * (t - self.t_fire)], np.float32)


@dataclass
class Powerup:
    kind: str  # one of POWERUP_KINDS
    origin: np.ndarray  # position at t_spawn
    velocity: np.ndarray  # px/s at t_spawn
    gravity: float  # px/s^2, downward
    radius: float
    t_spawn: float
    t_end: float  # back below the screen

    def center(self, t: float) -> np.ndarray:
        dt = t - self.t_spawn
        return self.origin + self.velocity * dt + np.array([0.0, 0.5 * self.gravity * dt * dt], np.float32)


@dataclass
class Effect:
    kind: str  # "hit", "headshot", "blocked", "muzzle", "pickup", "explosion", "grenade", "safe"
    pos: np.ndarray
    t: float
    player: int  # shooter
    amount: int = 0  # damage dealt ("hit", "headshot")
    detail: str = ""  # powerup kind ("pickup")


@dataclass
class Aim:
    """What the laser sight of a player currently points at (for rendering)."""

    origin: np.ndarray
    end: np.ndarray
    on_target: bool
    path: np.ndarray | None = None  # predicted rocket arc (N, 2); None for a straight bullet


class Game:
    def __init__(self, cfg: GameConfig, frame_size: tuple[int, int], seed: int = 0):
        self.cfg = cfg
        self.width, self.height = frame_size
        self.states: dict[int, PlayerState] = {}
        self.bullets: list[Bullet] = []
        self.rockets: list[Rocket] = []
        self.grenades: list[Grenade] = []
        self.effects: list[Effect] = []
        self.powerup: Powerup | None = None
        self._next_powerup_t: float | None = None
        self._t = 0.0
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
        self._t = t
        for p in players:
            if p.id in reassigned or (p.visible and p.id not in self.states):
                self.states[p.id] = self._new_state()

        self.mode = "duel" if sum(p.visible for p in players) >= 2 else "idle"

        if self.round_over_t is not None and t - self.round_over_t >= self.cfg.round_end_s:
            self.reset_round()
        self._update_powerup(t)

        for p in players:
            st = self.states.get(p.id)
            if st is None:  # never seen yet
                continue
            st.frame = st.actions.update(p.signals if p.visible else None, t, self.height)
            if st.frame.reload and st.ammo < self.cfg.max_ammo:  # ducking reloads
                st.ammo = self.cfg.max_ammo
            if st.frame.shot is not None and self.round_over_t is None:
                self._fire(p.id, st, st.frame.shot, t)

        self._resolve_bullets(t)
        self._update_rockets(t)
        self._update_grenades(t)
        self._update_aims(players)
        self.effects = [e for e in self.effects if t - e.t < 0.8]

    def double_active(self, pid: int) -> bool:
        st = self.states.get(pid)
        return st is not None and self._t < st.double_until

    @property
    def cover(self) -> Cover:
        """The static wall along the bottom of the screen."""
        return Cover(0.0, (1.0 - self.cfg.cover_height) * self.height, float(self.width), float(self.height))

    def reset_round(self) -> None:
        for st in self.states.values():
            st.hp, st.ammo = self.cfg.max_hp, self.cfg.max_ammo
            st.double_until, st.rockets, st.grenades = -math.inf, 0, 0
        self.bullets.clear()
        self.rockets.clear()
        self.grenades.clear()
        self.powerup = None
        self._next_powerup_t = None  # the timer restarts with the round
        self.winner = None
        self.round_over_t = None

    def _new_state(self) -> PlayerState:
        return PlayerState(PlayerActions(self.cfg), self.cfg.max_hp, self.cfg.max_ammo)

    # ---------------------------------------------------------------- shooting
    def _fire(self, pid: int, st: PlayerState, shot: Shot, t: float) -> None:
        target = next((q for q in self._players.values() if q.id != pid and q.visible), None)
        if st.grenades > 0 and target is not None:  # aim does not matter: it homes in
            st.grenades -= 1
            land = self._grenade_landing(target)
            flight = self.cfg.grenade_flight_s
            self.grenades.append(
                Grenade(pid, target.id, shot.origin, t, t + flight, shot.origin.copy(), land)
            )
            self.effects.append(Effect("grenade", land, t, pid))
        elif st.rockets > 0:  # a rocket does not need bullets in the magazine
            st.rockets -= 1
            velocity, gravity = self._rocket_launch(shot.direction)
            self.rockets.append(Rocket(pid, shot.origin, velocity, gravity, t, shot.origin.copy()))
        elif st.ammo > 0:
            st.ammo -= 1
            length, _ = self._cast(pid, shot.origin, shot.direction)
            travel = self.cfg.bullet_travel_s
            self.bullets.append(Bullet(pid, shot.origin, shot.direction, length, t, t + travel))
        else:
            return
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
                amount = self.cfg.head_damage if kind == "head" else self.cfg.body_damage
                self._damage(b.shooter, target, amount, "headshot" if kind == "head" else "hit", pos, t)
            elif kind == "cover":
                self.effects.append(Effect("blocked", pos, t, b.shooter))
            elif kind == "powerup":
                self._collect(b.shooter, target, pos, t)
        self.bullets = remaining

    # ---------------------------------------------------------------- rockets
    def _rocket_launch(self, direction: np.ndarray) -> tuple[np.ndarray, float]:
        h = self.height
        return (direction * self.cfg.rocket_speed * h).astype(np.float32), self.cfg.rocket_gravity * h

    def _update_rockets(self, t: float) -> None:
        remaining = []
        for r in self.rockets:
            new = r.at(t)
            hit = self._sweep(r.shooter, r.pos, new)
            if hit is not None:
                pos, (kind, target) = hit
                if kind == "powerup":
                    self._collect(r.shooter, target, pos, t)
                self._explode(r.shooter, pos, t)
            elif -0.1 * self.width < new[0] < 1.1 * self.width and t - r.t_fire < 4.0:
                r.pos = new
                remaining.append(r)
        self.rockets = remaining

    def _sweep(self, shooter: int, a: np.ndarray, b: np.ndarray):
        """First thing on the segment a -> b, as (point, (kind, target)), or None."""
        if a[1] < 0 and b[1] < 0:
            return None  # above the screen: nothing to hit up there
        if a[1] < 0:  # coming back down into view: start at the top edge
            a = a + (b - a) * (-a[1] / (b[1] - a[1]))
        seg = b - a
        n = float(np.linalg.norm(seg))
        if n < 1e-6:
            return None
        d = seg / n
        dist, what = self._cast(shooter, a, d)
        if what[0] == "miss" or dist > n:
            return None
        return a + d * dist, what

    def _predict_arc(self, shooter: int, origin: np.ndarray, direction: np.ndarray):
        """Points along the arc a rocket fired now would fly, up to impact (for the aim preview)."""
        velocity, gravity = self._rocket_launch(direction)
        r = Rocket(shooter, origin, velocity, gravity, 0.0, origin)
        points, prev, what = [origin], origin, ("miss", None)
        for k in range(1, 61):
            new = r.at(k * 0.04)
            hit = self._sweep(shooter, prev, new)
            if hit is not None:
                points.append(hit[0])
                what = hit[1]
                break
            points.append(new)
            if not (0 <= new[0] <= self.width) or new[1] > self.height:
                break
            prev = new
        return np.array(points, np.float32), what

    def _explode(self, shooter: int, pos: np.ndarray, t: float) -> None:
        """Blast damage to every other player within the radius, wall or not."""
        self.effects.append(Effect("explosion", pos, t, shooter))
        if self.round_over_t is not None:
            return
        cfg = self.cfg
        radius = cfg.rocket_blast_radius * self.height
        for pid, p in self._players.items():
            if pid == shooter or not p.visible or p.smoothed is None:
                continue
            (head_c, head_r), torso = self._hitboxes(p)
            # Distance from the impact to the nearest point of the player (0 when inside).
            to_head = max(0.0, float(np.linalg.norm(pos - head_c)) - head_r)
            to_torso = max(0.0, -cv2.pointPolygonTest(torso, (float(pos[0]), float(pos[1])), True))
            dist = min(to_head, to_torso)
            if dist > radius:
                continue
            amount = round(cfg.rocket_damage - (cfg.rocket_damage - cfg.rocket_min_damage) * dist / radius)
            self._damage(shooter, pid, amount, "hit", torso.mean(axis=0), t)
            if self.round_over_t is not None:
                return

    # ---------------------------------------------------------------- grenades
    def _grenade_landing(self, p: Player) -> np.ndarray:
        head, _ = self._hitboxes(p)
        return head[0].copy()

    def _update_grenades(self, t: float) -> None:
        remaining = []
        for g in self.grenades:
            target = self._players.get(g.target)
            if target is not None and target.visible and target.smoothed is not None:
                g.landing = self._grenade_landing(target)
            # Straight line to the (moving) target plus a parabolic lob on top: 4 * h * u * (1 - u).
            u = g.progress(t)
            lob = 4 * self.cfg.grenade_arc * self.height * u * (1 - u)
            g.pos = g.origin + (g.landing - g.origin) * u - np.array([0.0, lob], np.float32)
            if u < 1.0:
                remaining.append(g)
                continue
            self.effects.append(Effect("explosion", g.pos.copy(), t, g.thrower, detail="grenade"))
            st = self.states.get(g.target)
            if st is None or self.round_over_t is not None:
                continue
            if st.frame.ducked:
                self.effects.append(Effect("safe", g.pos.copy(), t, g.thrower))
            else:
                self._damage(g.thrower, g.target, st.hp, "hit", g.pos.copy(), t)  # all remaining HP
        self.grenades = remaining

    def _damage(
        self, shooter: int, target_id: int, amount: int, effect: str, pos: np.ndarray, t: float
    ) -> None:
        st = self.states[target_id]
        if t < self.states[shooter].double_until:
            amount *= 2
        st.hp = max(0, st.hp - amount)
        st.last_hit_t = t
        self.effects.append(Effect(effect, pos, t, shooter, amount))
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

        # The wall blocks every shot that reaches it, including shots fired from behind it.
        consider(ray_polygon(origin, direction, self.cover.polygon()), ("cover", None))

        for pid, p in self._players.items():
            if pid == shooter or not p.visible or p.smoothed is None:
                continue
            st = self.states[pid]
            head, torso = self._hitboxes(p)
            if st.frame.ducked:
                # Behind cover: anything that would have hit the player is stopped by the crate.
                dists = (ray_circle(origin, direction, *head), ray_polygon(origin, direction, torso))
                body_hit = min((d for d in dists if d is not None), default=None)
                consider(body_hit, ("cover", pid))
            else:
                consider(ray_circle(origin, direction, *head), ("head", pid))
                consider(ray_polygon(origin, direction, torso), ("body", pid))
        if self.powerup is not None:
            pu = self.powerup
            consider(ray_circle(origin, direction, pu.center(self._t), pu.radius), ("powerup", pu))
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

    # ---------------------------------------------------------------- aims
    def _update_aims(self, players: list[Player]) -> None:
        self.aims = {}
        for p in players:
            st = self.states.get(p.id)
            if st is None or not (p.visible and st.frame.aiming and p.signals is not None):
                continue
            o, d = p.signals.aim_origin, p.signals.aim_dir
            if st.rockets > 0:
                path, (kind, target) = self._predict_arc(p.id, o, d)
                # A rocket landing on the wall in front of a ducked player still hits them.
                on = kind in ("head", "body", "powerup") or (kind == "cover" and target is not None)
                self.aims[p.id] = Aim(o, path[-1], on, path)
                continue
            dist, (kind, _) = self._cast(p.id, o, d)
            self.aims[p.id] = Aim(o, o + d * dist, kind in ("head", "body", "powerup"))

    # ---------------------------------------------------------------- powerups
    def _update_powerup(self, t: float) -> None:
        if self.powerup is not None and t >= self.powerup.t_end:
            self.powerup = None
        # Only duels spawn powerups. Leaving the duel for a frame (a player briefly not detected)
        # does not reset the timer; only a new round does.
        if self.mode != "duel" or self.round_over_t is not None:
            return
        if self._next_powerup_t is None:
            self._next_powerup_t = t + self.cfg.powerup_interval_s
        if t >= self._next_powerup_t and self.powerup is None:
            self.powerup = self._spawn_powerup(t)
            self._next_powerup_t = t + self.cfg.powerup_interval_s

    def _spawn_powerup(self, t: float) -> Powerup:
        """Toss a powerup up from just below the screen so it peaks above the wall and falls back.

        With rise time T to a peak d px above the start: g = 2d / T^2 and v_y = -2d / T.
        """
        h, w, rng = self.height, self.width, self._rng
        radius = self.cfg.powerup_radius * h
        start = np.array([rng.uniform(0.3, 0.7) * w, h + radius], np.float32)
        peak_y = rng.uniform(0.22, 0.4) * h
        rise = self.cfg.powerup_rise_s
        d = float(start[1] - peak_y)
        land_x = rng.uniform(0.2, 0.8) * w
        velocity = np.array([(land_x - start[0]) / (2 * rise), -2 * d / rise], np.float32)
        kind = POWERUP_KINDS[int(rng.integers(len(POWERUP_KINDS)))]
        return Powerup(kind, start, velocity, 2 * d / rise**2, radius, t, t + 2 * rise)

    def _collect(self, pid: int, pu: Powerup, pos: np.ndarray, t: float) -> None:
        self.powerup = None  # a second bullet in the same frame now finds nothing
        st = self.states[pid]
        if pu.kind == "heal":
            st.hp = self.cfg.max_hp
        elif pu.kind == "double":
            st.double_until = t + self.cfg.double_damage_s
        elif pu.kind == "rocket":
            st.rockets = 1
        else:
            st.grenades = 1
        self.effects.append(Effect("pickup", pos, t, pid, detail=pu.kind))
