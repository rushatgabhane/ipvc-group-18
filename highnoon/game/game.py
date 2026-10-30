from __future__ import annotations

import cv2
import numpy as np

from highnoon.config import GameConfig
from highnoon.contracts import Player
from highnoon.game.actions import PlayerActions, Shot
from highnoon.game.entities import (
    Aim,
    Bottle,
    Bullet,
    Effect,
    Grenade,
    Hitbox,
    PlayerState,
    Powerup,
    Rocket,
    falling_path,
)
from highnoon.game.geometry import ray_exit_distance, rect_polygon, segment_circle_hits, segment_polygon_hits

NOSE = 0
TORSO_CORNERS = [11, 12, 24, 23]
PRACTICE_POWERUP_KINDS = ("rocket", "bomb")
GRENADE_EXPLOSION_SCALE = 1.4
EFFECT_LIFETIME_S = 0.8
HEAD_RADIUS_PER_FACE = 0.6
HEAD_RADIUS_PER_TORSO = 0.35
TORSO_HITBOX_SCALE = 1.15
DEFAULT_TORSO_PX = 100.0
ROCKET_MAX_FLIGHT_S = 4.0
ARC_PREVIEW_STEP_S = 0.04
LOB_PREVIEW_STEPS = np.linspace(0.0, 1.0, 25, dtype=np.float32)
ARC_PREVIEW_STEPS = 60


class Game:
    def __init__(self, cfg: GameConfig, frame_size: tuple[int, int], seed: int = 0):
        self.cfg = cfg
        self.width, self.height = frame_size
        self.states: dict[int, PlayerState] = {}
        self.bullets: list[Bullet] = []
        self.rockets: list[Rocket] = []
        self.grenades: list[Grenade] = []
        self.powerup: Powerup | None = None
        self.bottles: list[Bottle] = []
        self.effects: list[Effect] = []
        self.aims: dict[int, Aim] = {}
        self.mode = "idle"
        self.winner: int | None = None
        self.forfeited: int | None = None
        self.round_over_t: float | None = None
        self.round = 0
        self.round_start_t = 0.0
        self.scores: dict[int, int] = {}
        self._next_powerup_t: float | None = None
        self._next_bottle_t = 0.0
        self.countdown_end_t: float | None = None
        self._rng = np.random.default_rng(seed)
        self._players: dict[int, Player] = {}
        self._hitboxes: dict[int, Hitbox] = {}
        self._wall = self.wall
        self._t = 0.0

    def update(self, players: list[Player], t: float, reassigned: set[int]) -> None:
        self._t = t
        self._players = {p.id: p for p in players}
        self._hitboxes = {p.id: hitbox(p) for p in players if p.visible and p.smoothed is not None}
        self._wall = self.wall
        self._create_states(players, reassigned)
        self._update_mode(sum(p.visible for p in players) >= (1 if self.cfg.practice else 2), t)
        self._check_forfeit(t)
        if self.forfeited is None and self.round_over_t is not None and t - self.round_over_t >= self.cfg.round_end_s:
            self.reset_round()
            self.mode, self.countdown_end_t = "countdown", t + self.cfg.countdown_s
        self._update_powerup(t)
        self._update_bottles(t)
        for player in players:
            self._update_player(player, t)
        self._resolve_bullets(t)
        self._update_rockets(t)
        self._update_grenades(t)
        self._update_aims(players)
        self.effects = [e for e in self.effects if t - e.t < EFFECT_LIFETIME_S]

    @property
    def wall_top(self) -> float:
        return (1.0 - self.cfg.cover_height) * self.height

    @property
    def wall(self) -> np.ndarray:
        return rect_polygon(0.0, self.wall_top, float(self.width), float(self.height))

    @property
    def now(self) -> float:
        return self._t

    @property
    def round_live(self) -> bool:
        return self.mode == "duel" and self.round_over_t is None

    def _absent_for(self, player: Player) -> float:
        return self._t - player.last_seen if not player.visible and player.id in self.states else 0.0

    @property
    def missing_players(self) -> list[int]:
        if self.mode != "duel" or self.cfg.practice:
            return []
        return [p.id for p in self._players.values() if self._absent_for(p) > self.cfg.come_back_after_s]

    def leave_countdown(self, player_id: int) -> float:
        """Seconds until a missing player forfeits the game."""
        absent = self._absent_for(self._players[player_id])
        return max(0.0, self.cfg.come_back_after_s + self.cfg.leave_timeout_s - absent)

    def _check_forfeit(self, t: float) -> None:
        if self.forfeited is not None:
            return
        for player_id in self.missing_players:
            if self.leave_countdown(player_id) <= 0:
                self.forfeited = player_id
                self.mode = "forfeit"
                self.bullets.clear()
                self.rockets.clear()
                self.grenades.clear()
                return

    def countdown_left(self, t: float) -> float:
        return max(0.0, self.countdown_end_t - t) if self.mode == "countdown" else 0.0

    def _update_mode(self, both_in_view: bool, t: float) -> None:
        if self.mode == "idle" and both_in_view:
            self.mode, self.countdown_end_t = "countdown", t + self.cfg.countdown_s
        elif self.mode == "countdown" and not both_in_view:
            self.mode = "idle"
        elif self.mode == "countdown" and t >= self.countdown_end_t:
            self.reset_round()
            self.mode = "duel"

    def blast_radius(self, grenade: bool = False) -> float:
        return (self.cfg.grenade_blast_radius if grenade else self.cfg.rocket_blast_radius) * self.height

    def explosion_radius(self, grenade: bool = False) -> float:
        """The size the fireball and scorch marks are drawn at, which is smaller than a grenade's reach."""
        return self.cfg.rocket_blast_radius * self.height * (GRENADE_EXPLOSION_SCALE if grenade else 1.0)

    def reset_round(self) -> None:
        for state in self.states.values():
            state.hp, state.ammo = self.cfg.max_hp, self.cfg.max_ammo
            state.rockets = state.grenades = 0
        self.bullets.clear()
        self.rockets.clear()
        self.grenades.clear()
        self.powerup = None
        self._next_powerup_t = None
        self.bottles.clear()
        self._next_bottle_t = 0.0
        self.winner = None
        self.round_over_t = None
        self.round += 1
        self.round_start_t = self._t

    def _create_states(self, players: list[Player], reassigned: set[int]) -> None:
        for player in players:
            if player.id in reassigned or (player.visible and player.id not in self.states):
                self.states[player.id] = PlayerState(
                    PlayerActions(self.cfg), self.cfg.max_hp, self.cfg.max_ammo
                )

    def _update_player(self, player: Player, t: float) -> None:
        state = self.states.get(player.id)
        if state is None:
            return
        state.frame = state.actions.update(player.signals if player.visible else None, t, self.height)
        if state.frame.reload:
            self._reload(player.id, state, t)
        self._drain_if_camping(player.id, state, t)
        if state.frame.ignored_fist is not None:
            state.last_ignored_fist = (state.frame.ignored_fist, t)
        if state.frame.shot is not None and self.round_live:
            self._fire(player.id, state, state.frame.shot, t)

    def _reload(self, player_id: int, state: PlayerState, t: float) -> None:
        if state.ammo < self.cfg.max_ammo and self.round_live:
            box = self._hitboxes.get(player_id)
            pos = box.head_center.copy() if box else np.array([self.width / 2, self.wall_top], np.float32)
            self.effects.append(Effect("reload", pos, t, player_id))
        state.ammo = self.cfg.max_ammo

    def _drain_if_camping(self, player_id: int, state: PlayerState, t: float) -> None:
        camping = state.frame.ducked and state.frame.ducked_for >= self.cfg.camp_grace_s
        opponent = self._opponent_of(player_id)
        if not (camping and self.round_live and opponent is not None):
            state.next_camp_drain_t = None
            return
        if state.next_camp_drain_t is None:
            state.next_camp_drain_t = t
        if t >= state.next_camp_drain_t:
            state.next_camp_drain_t += self.cfg.camp_drain_interval_s
            drain = max(1, round(self.cfg.max_hp * self.cfg.camp_drain_fraction))
            box = self._hitboxes.get(player_id)
            pos = box.head_center.copy() if box else np.array([self.width / 2, self.wall_top], np.float32)
            self._damage(opponent.id, player_id, drain, "camping", pos, t, "camping")

    def _opponent_of(self, player_id: int) -> Player | None:
        if self.cfg.practice:
            return None
        return next((p for p in self._players.values() if p.id != player_id and p.id in self._hitboxes), None)

    def _fire(self, shooter: int, state: PlayerState, shot: Shot, t: float) -> None:
        opponent = self._opponent_of(shooter)
        weapon = self.weapon_of(state, opponent)
        origin = self.muzzle(shot.origin, shot.direction, weapon)
        if weapon == "gun" and state.ammo == 0:
            self.effects.append(Effect("dry_fire", origin, t, shooter))
            state.last_ignored_fist = ("no ammo", t)
            return
        if weapon == "grenade":
            state.grenades -= 1
            self._throw_grenade(shooter, opponent, origin, shot.direction, t)
        elif weapon == "rocket":
            state.rockets -= 1
            velocity, gravity = self._rocket_launch(shot.direction)
            self.rockets.append(Rocket(shooter, origin, velocity, gravity, t, origin.copy()))
        else:
            state.ammo -= 1
            length, _ = self._cast(shooter, origin, shot.direction)
            arrival = t + self.cfg.bullet_travel_s
            self.bullets.append(Bullet(shooter, origin, shot.direction, length, t, arrival))
        self.effects.append(Effect("muzzle", origin, t, shooter, detail=weapon))

    @staticmethod
    def weapon_of(state: PlayerState, opponent: Player | None) -> str:
        if state.grenades > 0 and opponent is not None:
            return "grenade"
        return "rocket" if state.rockets > 0 else "gun"

    def reach(self, weapon: str) -> float:
        cfg = self.cfg
        return {"gun": cfg.gun_reach, "rocket": cfg.rocket_reach, "grenade": cfg.grenade_reach}[weapon]

    def _resolve_bullets(self, t: float) -> None:
        in_flight = []
        for bullet in self.bullets:
            if t < bullet.t_arrive:
                in_flight.append(bullet)
                continue
            distance, (kind, target) = self._cast(bullet.shooter, bullet.origin, bullet.direction)
            impact = bullet.origin + bullet.direction * distance
            if kind in ("head", "body") and self.round_live:
                damage = self.cfg.head_damage if kind == "head" else self.cfg.body_damage
                effect = "headshot" if kind == "head" else "hit"
                self._damage(bullet.shooter, target, damage, effect, impact, t, "bullet")
            elif kind == "cover":
                self.effects.append(Effect("blocked", impact, t, bullet.shooter))
            elif kind == "powerup":
                self._collect(bullet.shooter, target, impact, t)
            elif kind == "bottle":
                self._shatter(bullet.shooter, target, t)
        self.bullets = in_flight

    def _rocket_launch(self, direction: np.ndarray) -> tuple[np.ndarray, float]:
        speed = self.cfg.rocket_speed * self.height
        return (direction * speed).astype(np.float32), self.cfg.rocket_gravity * self.height

    def _update_rockets(self, t: float) -> None:
        in_flight = []
        for rocket in self.rockets:
            next_pos = rocket.at(t)
            hit = self._first_hit_along(rocket.shooter, np.stack([rocket.pos, next_pos]))
            if hit is not None:
                _, impact, (kind, target) = hit
                if kind == "powerup":
                    self._collect(rocket.shooter, target, impact, t)
                self._explode(rocket.shooter, impact, t)
            elif self._rocket_in_play(rocket, next_pos, t):
                rocket.pos = next_pos
                in_flight.append(rocket)
        self.rockets = in_flight

    def _rocket_in_play(self, rocket: Rocket, pos: np.ndarray, t: float) -> bool:
        on_screen_width = -0.1 * self.width < pos[0] < 1.1 * self.width
        return on_screen_width and t - rocket.t_fire < ROCKET_MAX_FLIGHT_S

    def _predict_arc(self, shooter: int, origin: np.ndarray, direction: np.ndarray):
        velocity, gravity = self._rocket_launch(direction)
        times = np.arange(ARC_PREVIEW_STEPS + 1, dtype=np.float32) * ARC_PREVIEW_STEP_S
        path = falling_path(origin, velocity, gravity, times)
        off_screen = (path[:, 0] < 0) | (path[:, 0] > self.width) | (path[:, 1] > self.height)
        if off_screen.any():
            path = path[: int(np.argmax(off_screen)) + 1]
        hit = self._first_hit_along(shooter, path)
        if hit is None:
            return path, ("miss", None)
        segment, impact, what = hit
        return np.vstack([path[: segment + 1], impact]), what

    def _grenade_landing(self, thrower: int, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Where the aim line first meets the wall or another player, or else leaves the screen."""
        reach = ray_exit_distance(origin, direction, self.width, self.height)
        start, end = origin[None, :], (origin + direction * reach)[None, :]
        fractions = [1.0, float(segment_polygon_hits(start, end, self._wall)[0])]
        for player_id, box in self._hitboxes.items():
            if player_id != thrower:
                fractions.append(float(segment_circle_hits(start, end, box.head_center, box.head_radius)[0]))
                fractions.append(float(segment_polygon_hits(start, end, box.torso)[0]))
        return (origin + direction * reach * min(fractions)).astype(np.float32)

    def _throw_grenade(
        self, thrower: int, target: Player, origin: np.ndarray, direction: np.ndarray, t: float
    ) -> None:
        landing = self._grenade_landing(thrower, origin, direction)
        flight_end = t + self.cfg.grenade_flight_s
        self.grenades.append(Grenade(thrower, target.id, origin, t, flight_end, origin.copy(), landing))
        self.effects.append(Effect("grenade", landing, t, thrower))

    def _update_grenades(self, t: float) -> None:
        in_flight = []
        for grenade in self.grenades:
            progress = grenade.progress(t)
            grenade.pos = self.lob_path(grenade.origin, grenade.landing, np.array([progress], np.float32))[0]
            if progress < 1.0:
                in_flight.append(grenade)
            else:
                self._detonate_grenade(grenade, t)
        self.grenades = in_flight

    def _detonate_grenade(self, grenade: Grenade, t: float) -> None:
        self.effects.append(Effect("explosion", grenade.pos.copy(), t, grenade.thrower, detail="grenade"))
        cfg, radius = self.cfg, self.blast_radius(grenade=True)
        # The target goes first, so a blast that catches both gives the round to the thrower. Splash on
        # the thrower counts as damage from the opponent, so blowing yourself up loses the round.
        for player_id, credited in ((grenade.target, grenade.thrower), (grenade.thrower, grenade.target)):
            box = self._hitboxes.get(player_id)
            if box is None or not self.round_live:
                continue
            distance = distance_to_hitbox(grenade.pos, box)
            if distance > radius:
                continue
            if self.states[player_id].frame.ducked:
                if player_id == grenade.target:
                    self.effects.append(Effect("safe", grenade.pos.copy(), t, grenade.thrower))
                continue
            damage = splash_damage(distance, radius, cfg.grenade_damage, cfg.grenade_min_damage)
            self._damage(credited, player_id, damage, "hit", box.torso.mean(axis=0), t, "grenade")

    def _explode(self, shooter: int, center: np.ndarray, t: float) -> None:
        self.effects.append(Effect("explosion", center, t, shooter))
        radius = self.blast_radius()
        for bottle in list(self.bottles):
            if np.linalg.norm(bottle.center - center) <= radius + bottle.radius:
                self._shatter(shooter, bottle, t)
        for player_id, box in self._hitboxes.items():
            if player_id == shooter or not self.round_live or self.cfg.practice:
                continue
            distance = distance_to_hitbox(center, box)
            if distance <= radius:
                damage = splash_damage(distance, radius, self.cfg.rocket_damage, self.cfg.rocket_min_damage)
                self._damage(shooter, player_id, damage, "hit", box.torso.mean(axis=0), t, "rocket")

    def _damage(
        self, shooter: int, target: int, amount: int, effect: str, pos: np.ndarray, t: float, source: str
    ) -> None:
        state = self.states[target]
        state.hp = max(0, state.hp - amount)
        state.last_hit_t = t
        self.effects.append(Effect(effect, pos, t, shooter, amount, detail=source))
        if state.hp == 0:
            self.round_over_t = t
            if shooter != target:
                self.winner = shooter
                self.scores[shooter] = self.scores.get(shooter, 0) + 1

    def muzzle(self, hand: np.ndarray, direction: np.ndarray, weapon: str = "gun") -> np.ndarray:
        return (hand + direction * self.reach(weapon) * self.height).astype(np.float32)

    def lob_path(self, origin: np.ndarray, landing: np.ndarray, progress: np.ndarray) -> np.ndarray:
        path = origin + np.outer(progress, landing - origin)
        path[:, 1] -= 4 * self.cfg.grenade_arc * self.height * progress * (1 - progress)
        return path.astype(np.float32)

    def _cast(self, shooter: int, origin: np.ndarray, direction: np.ndarray):
        reach = ray_exit_distance(origin, direction, self.width, self.height)
        hit = self._first_hit(shooter, origin[None, :], (origin + direction * reach)[None, :])
        if hit is None:
            return reach, ("miss", None)
        _, fraction, what = hit
        return fraction * reach, what

    def _first_hit_along(self, shooter: int, path: np.ndarray):
        hit = self._first_hit(shooter, path[:-1], path[1:])
        if hit is None:
            return None
        segment, fraction, what = hit
        impact = path[segment] + (path[segment + 1] - path[segment]) * fraction
        return segment, impact.astype(np.float32), what

    def _first_hit(self, shooter: int, starts: np.ndarray, ends: np.ndarray):
        fractions, targets = self._hit_fractions(shooter, starts, ends)
        hit_segments = np.flatnonzero(np.isfinite(fractions).any(axis=0))
        if not len(hit_segments):
            return None
        segment = int(hit_segments[0])
        target = int(np.argmin(fractions[:, segment]))
        return segment, float(fractions[target, segment]), targets[target]

    def _hit_fractions(self, shooter: int, starts: np.ndarray, ends: np.ndarray):
        rows = [segment_polygon_hits(starts, ends, self._wall)]
        targets: list[tuple[str, object]] = [("cover", None)]
        for player_id, box in self._hitboxes.items():
            if player_id == shooter or player_id not in self.states or self.cfg.practice:
                continue
            head = segment_circle_hits(starts, ends, box.head_center, box.head_radius)
            body = segment_polygon_hits(starts, ends, box.torso)
            if self.states[player_id].frame.ducked:
                rows.append(np.minimum(head, body))
                targets.append(("cover", player_id))
            else:
                rows += [head, body]
                targets += [("head", player_id), ("body", player_id)]
        if self.powerup is not None:
            center = self.powerup.center(self._t)
            hit_radius = self.powerup.radius * self.cfg.powerup_hit_scale
            rows.append(segment_circle_hits(starts, ends, center, hit_radius))
            targets.append(("powerup", self.powerup))
        for bottle in self.bottles:
            rows.append(segment_circle_hits(starts, ends, bottle.center, bottle.radius))
            targets.append(("bottle", bottle))
        return np.stack(rows), targets

    def _update_aims(self, players: list[Player]) -> None:
        self.aims = {}
        for player in players:
            state = self.states.get(player.id)
            if state is None or not (player.visible and state.frame.aiming and player.signals is not None):
                continue
            origin, direction = player.signals.aim_origin, player.signals.aim_dir
            opponent = self._opponent_of(player.id)
            weapon = self.weapon_of(state, opponent)
            muzzle = self.muzzle(origin, direction, weapon)
            spread = state.frame.spread_deg
            if weapon == "grenade":
                landing = self._grenade_landing(player.id, muzzle, direction)
                path = self.lob_path(muzzle, landing, LOB_PREVIEW_STEPS)
                in_blast = distance_to_hitbox(landing, self._hitboxes[opponent.id]) <= self.blast_radius(True)
                self.aims[player.id] = Aim(origin, landing, in_blast, path, spread, direction, weapon)
            elif weapon == "rocket":
                path, (kind, target) = self._predict_arc(player.id, muzzle, direction)
                lands_on_player = kind in ("head", "body", "powerup", "bottle") or (
                    kind == "cover" and target is not None
                )
                self.aims[player.id] = Aim(origin, path[-1], lands_on_player, path, spread, direction, weapon)
            else:
                distance, (kind, _) = self._cast(player.id, muzzle, direction)
                end = muzzle + direction * distance
                on_target = kind in ("head", "body", "powerup", "bottle")
                self.aims[player.id] = Aim(origin, end, on_target, None, spread, direction, weapon)

    def _update_powerup(self, t: float) -> None:
        if self.powerup is not None and t >= self.powerup.t_end:
            self.powerup = None
        if not self.round_live:
            return
        cfg = self.cfg
        interval = cfg.practice_powerup_interval_s if cfg.practice else cfg.powerup_interval_s
        if self._next_powerup_t is None:
            self._next_powerup_t = t + interval
        if t >= self._next_powerup_t and self.powerup is None:
            self.powerup = self._toss_powerup(t)
            self._next_powerup_t = t + interval

    def _toss_powerup(self, t: float) -> Powerup:
        rng, fall_s = self._rng, self.cfg.powerup_fall_s
        radius = self.cfg.powerup_radius * self.height
        spread = self.cfg.powerup_center_spread
        start = np.array([rng.uniform(0.5 - spread, 0.5 + spread) * self.width, -radius], np.float32)
        velocity = np.array([0.0, (self.wall_top + radius) / fall_s], np.float32)
        if self.cfg.practice:
            kind = PRACTICE_POWERUP_KINDS[int(rng.integers(len(PRACTICE_POWERUP_KINDS)))]
        else:
            kind = "grenade" if rng.random() < self.cfg.grenade_drop_chance else "rocket"
        return Powerup(kind, start, velocity, 0.0, radius, t, t + fall_s)

    def _update_bottles(self, t: float) -> None:
        if not (self.cfg.practice and self.round_live):
            return
        while len(self.bottles) < self.cfg.bottle_count and t >= self._next_bottle_t:
            self.bottles.append(self._place_bottle(t))

    def _place_bottle(self, t: float) -> Bottle:
        rng, radius = self._rng, self.cfg.bottle_radius * self.height
        for _ in range(20):
            center = np.array(
                [rng.uniform(0.08, 0.92) * self.width, rng.uniform(0.12, 0.9) * self.wall_top], np.float32
            )
            if all(np.linalg.norm(center - b.center) > 3 * radius for b in self.bottles):
                break
        return Bottle(center, radius, t)

    def _shatter(self, player_id: int, bottle: Bottle, t: float) -> None:
        if bottle not in self.bottles:
            return
        self.bottles.remove(bottle)
        self._next_bottle_t = t + self.cfg.bottle_respawn_s
        self.scores[player_id] = self.scores.get(player_id, 0) + 1
        self.effects.append(Effect("bottle", bottle.center.copy(), t, player_id))

    def _collect(self, player_id: int, powerup: Powerup, pos: np.ndarray, t: float) -> None:
        self.powerup = None
        state = self.states[player_id]
        if powerup.kind == "bomb":
            self._set_off_bomb(player_id, pos, t)
            return
        if powerup.kind == "rocket":
            state.rockets = 1
        else:
            state.grenades = 1
        self.effects.append(Effect("pickup", pos, t, player_id, detail=powerup.kind))

    def _set_off_bomb(self, player_id: int, pos: np.ndarray, t: float) -> None:
        self.effects.append(Effect("explosion", pos, t, player_id, detail="bomb"))
        if not self.round_live:
            return
        box = self._hitboxes.get(player_id)
        hurt_at = box.torso.mean(axis=0) if box else pos
        self._damage(player_id, player_id, self.cfg.bomb_damage, "hit", hurt_at, t, "bomb")


def hitbox(player: Player) -> Hitbox:
    keypoints = player.smoothed
    torso_len = player.signals.torso_len if player.signals else DEFAULT_TORSO_PX
    if player.face is not None:
        x0, y0, x1, y1 = player.face.bbox
        head_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        head_radius = HEAD_RADIUS_PER_FACE * float(max(x1 - x0, y1 - y0))
    else:
        head_center = keypoints[NOSE].copy()
        head_radius = HEAD_RADIUS_PER_TORSO * torso_len
    corners = keypoints[TORSO_CORNERS].astype(np.float32)
    middle = corners.mean(axis=0)
    torso = middle + (corners - middle) * TORSO_HITBOX_SCALE
    return Hitbox(head_center, head_radius, torso)


def splash_damage(distance: float, radius: float, center_damage: int, edge_damage: int) -> int:
    return round(center_damage - (center_damage - edge_damage) * distance / radius)


def distance_to_hitbox(point: np.ndarray, box: Hitbox) -> float:
    to_head = max(0.0, float(np.linalg.norm(point - box.head_center)) - box.head_radius)
    signed = cv2.pointPolygonTest(box.torso, (float(point[0]), float(point[1])), True)
    return min(to_head, max(0.0, -signed))
