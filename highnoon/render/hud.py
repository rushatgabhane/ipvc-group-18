from __future__ import annotations

import math

import cv2
import numpy as np

from highnoon.contracts import Player
from highnoon.game.game import Game
from highnoon.render import assets
from highnoon.render import fx as effects
from highnoon.render.assets import HUD_FONT, TITLE_FONT, TextCache, blit, blit_center, panel, rounded_rect
from highnoon.render.fx import FX
from highnoon.render.palette import BLACK, CREAM, GOLD, RED, SAFE_GREEN, WHITE, hp_color

TAG_FOLLOW = 0.35
HP_EASE_PER_S = 6.0
HP_TRAIL_PER_S = 40.0
HIT_FLASH_S = 0.25
DUEL_HINT = "CLOSE YOUR HAND TO SHOOT  |  DUCK TO RELOAD  |  SHOOT POWERUPS  |  FIRST TO ZERO LOSES"
BANNERS = {
    "idle": ("HIGH NOON", "TWO PLAYERS: STEP INTO VIEW"),
    "countdown": ("GET READY", "STAY IN VIEW"),
    "duel": ("DUEL", DUEL_HINT),
}
RELOAD_WARNING = "DUCK TO RELOAD!"
RELOAD_WARNING_SIZE = 40
GET_UP_WARNING = "GET UP!"
CAMPER_WARNING = "GET OUT CAMPER!"
COUNTDOWN_SIZE = 220
WAITING_LINES = (("TWO PLAYERS", 110), ("STEP INTO VIEW", 80))
COUNTDOWN_COLOR = GOLD
ZOOM_FROM = 0.3
ZOOM_OVERSHOOT = 1.15
SHRINK_TO = 0.4


class Hud:
    def __init__(self, colors: dict[int, tuple[int, int, int]], text: TextCache, fx: FX):
        self.colors = colors
        self.text = text
        self.fx = fx
        self._tag_pos: dict[int, np.ndarray] = {}
        self._hp_shown: dict[int, float] = {}
        self._hp_trail: dict[int, float] = {}
        self._winner_reveal_t: float | None = None
        self._knockout_shown = False

    def prewarm(self, height: int) -> None:
        scale = height / 720.0
        announcements = [effects.BOOM, effects.GRENADE_BOOM, effects.GRENADE_WARNING, effects.KNOCKOUT]
        for player_id in self.colors:
            color = self.colors[player_id]
            announcements.append(effects.headshot_announcement(color))
            announcements += [
                effects.pickup_announcement(player_id, kind, color) for kind in assets.POWERUP_LABELS
            ]
            self._winner_name(player_id, scale)
        for text, color, size in announcements:
            self._announcement_sprite(text, size, color, scale)
        for label, color in (("DUCK!", RED), ("SAFE", SAFE_GREEN)):
            self.text.get(label, int(46 * scale), color, TITLE_FONT, stroke=3)
        for player_id in self.colors:
            self._reload_warning_sprite(player_id, scale)
        for warning in (GET_UP_WARNING, CAMPER_WARNING):
            self._camping_sprite(warning, scale)
        self.text.get("RELOADING", int(26 * scale), GOLD, HUD_FONT, stroke=2)
        self._waiting_sprites(scale)
        for player_id in self.colors:
            self._come_back_sprite(player_id, scale)
        for mode in BANNERS:
            self._banner_sprites(mode, scale)
        for number in range(1, 4):
            self._countdown_sprite(number, scale)
        for seconds_left in range(5):
            self._round_over_texts(seconds_left, scale)

    def _round_over_texts(self, seconds_left: int, scale: float):
        subtitle = self.text.get("WINS THE ROUND", int(48 * scale), CREAM, HUD_FONT, stroke=2)
        countdown = self.text.get(f"NEXT ROUND IN {seconds_left}", int(28 * scale), GOLD, HUD_FONT)
        return subtitle, countdown

    def _announcement_sprite(self, text: str, size: int, color, scale: float):
        return self.text.get(
            text, int(size * scale), color, TITLE_FONT, stroke=4, stroke_color=(20, 25, 40), shadow=5
        )

    def _banner_sprites(self, mode: str, scale: float):
        title, hint = BANNERS[mode]
        title_sprite = self.text.get(
            title, int(30 * scale), GOLD, TITLE_FONT, stroke=2, stroke_color=(20, 30, 50)
        )
        return title_sprite, self.text.get(hint, int(19 * scale), CREAM, HUD_FONT)

    def _countdown_sprite(self, number: int, scale: float):
        return self._announcement_sprite(str(number), COUNTDOWN_SIZE, COUNTDOWN_COLOR, scale)

    def _winner_name(self, player_id: int, scale: float):
        color = self.colors.get(player_id, WHITE)
        return self.text.get(f"PLAYER {player_id}", int(110 * scale), color, TITLE_FONT, stroke=4, shadow=6)

    def draw(self, image: np.ndarray, game: Game, players: list[Player], t: float, dt: float) -> None:
        scale = image.shape[0] / 720.0
        visible = [p for p in players if p.visible and p.id in game.states]
        for player in visible:
            self._ease_hp(player.id, game.states[player.id].hp, dt)
            self._draw_name_tag(image, game, player, t, scale)
        self._draw_float_texts(image, t, scale)
        for player in visible:
            self._draw_card(image, game, player, t, scale)
        self._draw_reload_warnings(image, game, visible, t, scale)
        self._draw_banner(image, game, scale)
        if self._winner_revealed(game, t):
            self._draw_winner(image, game, t, scale)
        if game.mode == "countdown":
            self._draw_countdown(image, game.countdown_left(t), scale)
        if game.mode == "idle":
            self._draw_waiting(image, t, scale)
        self._draw_come_back(image, game.missing_players, t, scale)
        self._draw_announcements(image, t, scale)

    def _waiting_sprites(self, scale: float):
        return [
            self._announcement_sprite(line, size, GOLD if i == 0 else CREAM, scale)
            for i, (line, size) in enumerate(WAITING_LINES)
        ]

    def _draw_waiting(self, image, t: float, scale: float) -> None:
        height, width = image.shape[:2]
        pulse = 1.0 + 0.04 * math.sin(t * 3)
        top, bottom = self._waiting_sprites(scale)
        blit_center(image, assets.scaled(top, pulse), width / 2, height * 0.36)
        blit_center(image, assets.scaled(bottom, pulse), width / 2, height * 0.36 + top.height * 0.9)

    def _come_back_sprite(self, player_id: int, scale: float):
        color = self.colors.get(player_id, WHITE)
        return self._announcement_sprite(f"P{player_id}: COME BACK ON SCREEN", 64, color, scale)

    def _draw_come_back(self, image, missing: list[int], t: float, scale: float) -> None:
        height, width = image.shape[:2]
        alpha = 1.0 if int(t * 3) % 2 == 0 else 0.7
        for row, player_id in enumerate(missing):
            sprite = self._come_back_sprite(player_id, scale)
            blit_center(image, sprite, width / 2, height * 0.3 + row * sprite.height, alpha)

    def _draw_countdown(self, image, seconds_left: float, scale: float) -> None:
        if seconds_left <= 0:
            return
        number = math.ceil(seconds_left)
        into_second = number - seconds_left
        pop = 1.0 + 0.5 * max(0.0, 1 - into_second * 4)
        alpha = 1.0 if into_second < 0.75 else max(0.0, 1 - (into_second - 0.75) / 0.25)
        height, width = image.shape[:2]
        sprite = assets.scaled(self._countdown_sprite(number, scale), pop)
        blit_center(image, sprite, width / 2, height * 0.42, alpha)

    def _ease_hp(self, player_id: int, hp: int, dt: float) -> None:
        shown = self._hp_shown.get(player_id, float(hp))
        shown = shown + (hp - shown) * min(1.0, HP_EASE_PER_S * dt) if hp < shown else float(hp)
        trail = self._hp_trail.get(player_id, shown)
        trail = max(shown, trail - HP_TRAIL_PER_S * dt)
        self._hp_shown[player_id], self._hp_trail[player_id] = shown, trail

    def _tag_anchor(self, player: Player, scale: float) -> np.ndarray | None:
        if player.face is not None:
            x0, y0, x1, _ = player.face.bbox
            return np.array([(x0 + x1) / 2, y0 - 26 * scale], np.float32)
        if player.smoothed is not None:
            torso = player.signals.torso_len if player.signals else 150.0
            nose = player.smoothed[0]
            return np.array([nose[0], nose[1] - 0.6 * torso], np.float32)
        return None

    def _draw_name_tag(self, image, game: Game, player: Player, t: float, scale: float) -> None:
        target = self._tag_anchor(player, scale)
        if target is None:
            return
        previous = self._tag_pos.get(player.id)
        pos = target if previous is None else previous + TAG_FOLLOW * (target - previous)
        self._tag_pos[player.id] = pos
        height, width = image.shape[:2]
        cx = float(np.clip(pos[0], 60 * scale, width - 60 * scale))
        cy = float(np.clip(pos[1], 40 * scale, height - 40 * scale))

        label = self.text.get(f"P{player.id}", int(30 * scale), WHITE, HUD_FONT)
        tag_w, tag_h = label.width + int(22 * scale), label.height + int(4 * scale)
        x0, y0 = int(cx - tag_w / 2), int(cy - tag_h - 6 * scale)
        rounded_rect(image, x0 - 2, y0 - 2, x0 + tag_w + 2, y0 + tag_h + 2, tag_h // 2 + 2, BLACK)
        rounded_rect(image, x0, y0, x0 + tag_w, y0 + tag_h, tag_h // 2, self.colors.get(player.id, WHITE))
        blit_center(image, label, cx, y0 + tag_h / 2)

        bar_w, bar_h = int(84 * scale), max(5, int(8 * scale))
        bar_x, bar_y = int(cx - bar_w / 2), int(cy + 2 * scale)
        rounded_rect(image, bar_x - 2, bar_y - 2, bar_x + bar_w + 2, bar_y + bar_h + 2, bar_h, BLACK)
        fraction = self._hp_shown[player.id] / game.cfg.max_hp
        if fraction > 0.01:
            fill_x = bar_x + max(bar_h, int(bar_w * fraction))
            rounded_rect(image, bar_x, bar_y, fill_x, bar_y + bar_h, bar_h // 2, hp_color(fraction))

        badges = player_badges(game, player.id)
        if badges:
            badge = self.text.get("  ".join(badges), int(22 * scale), CREAM, HUD_FONT, stroke=2)
            blit_center(image, badge, cx, bar_y + bar_h + 16 * scale)

    def _draw_status(
        self, image, game: Game, player_id: int, cx: float, card_top: int, t: float, scale: float
    ):
        state = game.states[player_id]
        reload_y = card_top - 34 * scale
        if state.frame.ducked and state.ammo < game.cfg.max_ammo and game.mode == "duel":
            self._draw_reload_bar(image, cx, reload_y, state.frame.reload_progress, scale)
        ducked_for = state.frame.ducked_for if state.frame.ducked else 0.0
        if game.mode == "duel" and ducked_for >= game.cfg.camp_warning_s:
            warning = CAMPER_WARNING if ducked_for >= game.cfg.camp_grace_s else GET_UP_WARNING
            self._draw_camping_warning(image, warning, cx, reload_y - 52 * scale, t, scale)

    def _camping_sprite(self, warning: str, scale: float):
        return self.text.get(warning, int(RELOAD_WARNING_SIZE * scale), RED, TITLE_FONT, stroke=3, shadow=3)

    def _draw_camping_warning(self, image, warning: str, x: float, y: float, t: float, scale: float) -> None:
        pulse = 1.0 + 0.15 * abs(math.sin(t * 10))
        alpha = 1.0 if int(t * 6) % 2 == 0 else 0.5
        blit_center(image, assets.scaled(self._camping_sprite(warning, scale), pulse), x, y, alpha)

    def _draw_reload_bar(self, image, x: float, y: float, progress: float, scale: float) -> None:
        label = self.text.get("RELOADING", int(26 * scale), GOLD, HUD_FONT, stroke=2)
        blit_center(image, label, x, y - 12 * scale)
        bar_w, bar_h = int(120 * scale), max(6, int(10 * scale))
        left, top = int(x - bar_w / 2), int(y + 6 * scale)
        rounded_rect(image, left - 2, top - 2, left + bar_w + 2, top + bar_h + 2, bar_h, BLACK)
        filled = max(bar_h, int(bar_w * progress))
        rounded_rect(image, left, top, left + filled, top + bar_h, bar_h // 2, GOLD)

    def _reload_warning_sprite(self, player_id: int, scale: float):
        text = f"P{player_id}: {RELOAD_WARNING}"
        color = self.colors.get(player_id, WHITE)
        return self.text.get(text, int(RELOAD_WARNING_SIZE * scale), color, TITLE_FONT, stroke=3, shadow=3)

    def _draw_reload_warnings(self, image, game: Game, players: list[Player], t: float, scale: float) -> None:
        height, width = image.shape[:2]
        empty = [p.id for p in players if needs_reload(game, p.id) and not game.states[p.id].frame.ducked]
        pulse = 1.0 + 0.12 * abs(math.sin(t * 8))
        alpha = 1.0 if int(t * 4) % 2 == 0 else 0.55
        for player_id in empty:
            sprite = assets.scaled(self._reload_warning_sprite(player_id, scale), pulse)
            blit_center(image, sprite, width * player_side(player_id), height * 0.5, alpha)

    def _draw_float_texts(self, image, t: float, scale: float) -> None:
        for text in self.fx.texts:
            age = (t - text.t0) / text.duration
            pop = 1.0 + 0.4 * max(0.0, 1 - age * 6)
            alpha = 1.0 if age < 0.6 else max(0.0, 1 - (age - 0.6) / 0.4)
            sprite = self.text.get(
                text.text, int(text.size * scale), text.color, HUD_FONT, stroke=max(2, int(3 * scale))
            )
            rise = 70 * scale * age + 30 * scale
            blit_center(image, assets.scaled(sprite, pop), text.pos[0], text.pos[1] - rise, alpha)

    def _draw_card(self, image, game: Game, player: Player, t: float, scale: float) -> None:
        height, width = image.shape[:2]
        state = game.states[player.id]
        color = self.colors.get(player.id, WHITE)
        card_w, card_h, margin = int(310 * scale), int(96 * scale), int(16 * scale)
        x0 = margin if player.id == 1 else width - margin - card_w
        y0 = height - margin - card_h
        border = RED if t - state.last_hit_t < HIT_FLASH_S else color
        panel(image, x0, y0, x0 + card_w, y0 + card_h, r=int(14 * scale), opacity=0.68)
        rounded_rect(image, x0, y0, x0 + card_w, y0 + card_h, int(14 * scale), border, 2)
        accent = ((x0 + int(10 * scale), y0 + int(12 * scale)), (x0 + int(15 * scale), y0 + int(40 * scale)))
        cv2.rectangle(image, *accent, color, -1)

        name = self.text.get(f"PLAYER {player.id}", int(30 * scale), color, HUD_FONT)
        blit(image, name, x0 + int(22 * scale), y0 + int(6 * scale))
        score = self.text.get(f"{game.scores.get(player.id, 0)} WINS", int(24 * scale), CREAM, HUD_FONT)
        blit(image, score, x0 + card_w - score.width - int(12 * scale), y0 + int(10 * scale))

        self._draw_hp_bar(image, game, player.id, x0, y0, card_w, scale)
        self._draw_ammo_row(image, game, player.id, x0, y0, card_w, t, scale)
        self._draw_status(image, game, player.id, x0 + card_w / 2, y0, t, scale)

    def _draw_hp_bar(
        self, image, game: Game, player_id: int, x0: int, y0: int, card_w: int, scale: float
    ) -> None:
        left, top = x0 + int(12 * scale), y0 + int(44 * scale)
        right, bottom = x0 + card_w - int(12 * scale), y0 + int(60 * scale)
        radius = int(8 * scale)
        rounded_rect(image, left, top, right, bottom, radius, (45, 40, 38))
        shown, trail = self._hp_shown[player_id], self._hp_trail[player_id]
        trail_x = left + int((right - left) * trail / game.cfg.max_hp)
        shown_x = left + int((right - left) * shown / game.cfg.max_hp)
        if trail_x > left + 2 * radius:
            rounded_rect(image, left, top, trail_x, bottom, radius, (200, 220, 235))
        if shown_x > left + 2 * radius:
            rounded_rect(image, left, top, shown_x, bottom, radius, hp_color(shown / game.cfg.max_hp))
            cv2.line(image, (left + radius, top + 3), (shown_x - radius, top + 3), WHITE, 1, cv2.LINE_AA)
        number = self.text.get(f"{int(round(shown))}", int(18 * scale), WHITE, HUD_FONT, stroke=2)
        blit(image, number, right - number.width - int(4 * scale), top + (bottom - top - number.height) // 2)

    def _draw_ammo_row(
        self, image, game: Game, player_id: int, x0, y0, card_w, t: float, scale: float
    ) -> None:
        state = game.states[player_id]
        x = x0 + int(14 * scale)
        for i in range(game.cfg.max_ammo):
            icon = assets.bullet_icon(int(22 * scale), i < state.ammo)
            blit(image, icon, x, y0 + int(66 * scale))
            x += int(icon.width + 5 * scale)
        x += int(10 * scale)
        for kind, held in (("rocket", state.rockets), ("grenade", state.grenades)):
            if held:
                sprite = self.text.get(
                    kind.upper(), int(20 * scale), self.colors.get(player_id, WHITE), HUD_FONT
                )
                blit(image, sprite, x, y0 + int(68 * scale))
                x += sprite.width + int(10 * scale)
        if state.ammo == 0 and int(t * 3) % 2 == 0:
            hint = self.text.get("DUCK TO RELOAD", int(20 * scale), GOLD, HUD_FONT)
            blit(image, hint, x0 + card_w - hint.width - int(12 * scale), y0 + int(68 * scale))

    def _draw_banner(self, image, game: Game, scale: float) -> None:
        width = image.shape[1]
        title_sprite, hint_sprite = self._banner_sprites(game.mode, scale)
        panel_w = max(title_sprite.width, hint_sprite.width) + int(40 * scale)
        panel_h = title_sprite.height + hint_sprite.height + int(10 * scale)
        x0, y0 = width // 2 - panel_w // 2, int(10 * scale)
        panel(image, x0, y0, x0 + panel_w, y0 + panel_h, r=int(16 * scale), opacity=0.55)
        title_y = y0 + int(4 * scale) + title_sprite.height / 2
        blit_center(image, title_sprite, width / 2, title_y)
        blit_center(image, hint_sprite, width / 2, title_y + title_sprite.height / 2 + hint_sprite.height / 2)

    def _winner_revealed(self, game: Game, t: float) -> bool:
        if game.winner is None:
            self._winner_reveal_t, self._knockout_shown = None, False
            return False
        if self._winner_reveal_t is None:
            if not self._knockout_shown:
                text, color, size = effects.KNOCKOUT
                self.fx.interrupt(text, t, color, size, effects.KNOCKOUT_S)
                self._knockout_shown = True
            if self.fx.announcing():
                return False
            self._winner_reveal_t = t
        return True

    def _draw_winner(self, image, game: Game, t: float, scale: float) -> None:
        height, width = image.shape[:2]
        cv2.convertScaleAbs(image, dst=image, alpha=0.45)
        age = t - self._winner_reveal_t
        name = self._winner_name(game.winner, scale)
        left = game.cfg.round_end_s - (t - (game.round_over_t or t))
        subtitle, countdown = self._round_over_texts(max(0, math.ceil(left)), scale)
        zoom, alpha = zoom_in(age)
        y = height * 0.42
        blit_center(image, assets.scaled(name, zoom), width / 2, y, alpha)
        blit_center(image, subtitle, width / 2, y + name.height * 0.75, alpha)
        blit_center(image, countdown, width / 2, y + name.height * 0.75 + subtitle.height * 1.2, alpha)

    def _draw_announcements(self, image, t: float, scale: float) -> None:
        announcement = self.fx.announcement
        if announcement is None:
            return
        height, width = image.shape[:2]
        zoom, alpha = zoom_in_out(t - announcement.t0, announcement.duration)
        sprite = self._announcement_sprite(announcement.text, announcement.size, announcement.color, scale)
        blit_center(image, assets.scaled(sprite, zoom), width / 2, height * 0.36, alpha)


def zoom_in(age: float) -> tuple[float, float]:
    grow = min(1.0, age / effects.ANNOUNCE_IN_S)
    settle = min(1.0, max(0.0, age - effects.ANNOUNCE_IN_S) / effects.ANNOUNCE_IN_S)
    if grow < 1.0:
        return ZOOM_FROM + (ZOOM_OVERSHOOT - ZOOM_FROM) * (1 - (1 - grow) ** 3), grow
    return ZOOM_OVERSHOOT + (1.0 - ZOOM_OVERSHOOT) * settle, 1.0


def zoom_in_out(age: float, duration: float) -> tuple[float, float]:
    left = duration - age
    if left >= effects.ANNOUNCE_OUT_S:
        return zoom_in(age)
    shrink = 1 - max(0.0, left) / effects.ANNOUNCE_OUT_S
    return 1.0 + (SHRINK_TO - 1.0) * shrink**2, 1 - shrink


def player_side(player_id: int) -> float:
    return 0.25 if player_id == 1 else 0.75


def needs_reload(game: Game, player_id: int) -> bool:
    state = game.states[player_id]
    return game.mode == "duel" and state.ammo == 0 and not state.rockets and not state.grenades


def player_badges(game: Game, player_id: int) -> list[str]:
    state = game.states[player_id]
    flags = (("DUCKED", state.frame.ducked), ("ROCKET", state.rockets > 0), ("GRENADE", state.grenades > 0))
    return [name for name, on in flags if on]


def draw_signal_debug(image: np.ndarray, game: Game, players: list[Player], colors: dict) -> None:
    for player in players:
        state = game.states.get(player.id)
        if not player.visible or player.signals is None or state is None or player.smoothed is None:
            continue
        lines = debug_lines(game, player)
        panel_w, line_h = 400, 20
        panel_h = line_h * len(lines) + 12
        x, y = debug_panel_origin(image, player, panel_w, panel_h)
        roi = image[y : y + panel_h, x : x + panel_w]
        cv2.convertScaleAbs(roi, dst=roi, alpha=0.3)
        for i, text in enumerate(lines):
            origin = (x + 8, y + 18 + i * line_h)
            color = colors.get(player.id, WHITE)
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def debug_lines(game: Game, player: Player) -> list[str]:
    signals, frame, cfg = player.signals, game.states[player.id].frame, game.cfg
    wall_top = 1 - cfg.cover_height
    hand = (
        "hand: not tracked"
        if signals.hand_openness is None
        else f"hand {signals.hand_openness:4.2f} {'FIST' if frame.hand_closed else 'open'}"
        f" (fist < {cfg.fist_closed:.2f}, open > {cfg.fist_open:.2f})"
    )
    return [
        f"arm {signals.extension:4.2f}  elev {np.degrees(signals.elevation):+4.0f} (aim -50..80)",
        f"aim {'ON' if frame.aiming else 'off'}  head {frame.head_pos:.2f} (duck > {wall_top:.2f})",
        f"ammo {game.states[player.id].ammo}",
        hand,
        ignored_fist_line(game.states[player.id], game.now),
    ]


def ignored_fist_line(state, t: float) -> str:
    if state.last_ignored_fist is None:
        return "last fist: ok"
    reason, when = state.last_ignored_fist
    return f"last ignored fist: {reason} ({t - when:.1f}s ago)"


def debug_panel_origin(image, player: Player, panel_w: int, panel_h: int) -> tuple[int, int]:
    height, width = image.shape[:2]
    if player.face is not None:
        face_x0, face_y0, face_x1, _ = player.face.bbox.astype(int)
    else:
        nose = player.smoothed[0].astype(int)
        face_x0, face_y0, face_x1 = nose[0] - 60, nose[1] - 60, nose[0] + 60
    x = face_x1 + 15 if width - face_x1 > face_x0 else face_x0 - 15 - panel_w
    x = min(max(5, x), width - panel_w - 5)
    y = min(max(60, face_y0), height - panel_h - 5)
    return x, y
