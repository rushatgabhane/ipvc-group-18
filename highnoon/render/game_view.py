"""T5: draws the game layer on top of the composited camera image.

Layer order, back to front:
  1. players (already composited onto the backdrop by the renderer)
  2. the wall, which occludes the lower body, so a ducking player disappears behind it
  3. bottles, particles
  4. additive glow layer: laser sights, tracers, muzzle flashes
  5. reticles, name tags above heads, floating numbers
  6. HUD cards, banner, announcements

Glow is drawn into one black layer and added with cv2.add (one full-frame op). Text comes from
the sprite cache in assets.py. Everything else uses OpenCV primitives on small regions.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from highnoon.contracts import Player
from highnoon.game.game import Game
from highnoon.render import assets
from highnoon.render.assets import HUD_FONT, TITLE_FONT, TextCache, blit, blit_center, panel, rounded_rect
from highnoon.render.fx import FX

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GOLD, RED, CREAM = (70, 200, 255), (60, 60, 235), (215, 240, 250)
TAG_FOLLOW = 0.35  # name tags lerp toward the head each frame (removes jitter)
HP_DRAIN = 6.0  # displayed HP catches up with real HP at this rate (1/s)


def _pt(p) -> tuple[int, int]:
    return int(p[0]), int(p[1])


def _hp_color(frac: float) -> tuple[int, int, int]:
    if frac > 0.5:
        return (90, 210, 110)
    if frac > 0.25:
        return (60, 200, 245)
    return (60, 70, 235)


class GameView:
    def __init__(self, colors: dict[int, tuple[int, int, int]]):
        self.colors = colors
        self.text = TextCache()
        self.fx = FX()
        self._tags: dict[int, np.ndarray] = {}
        self._hp_shown: dict[int, float] = {}
        self._hp_trail: dict[int, float] = {}
        self._t_last: float | None = None
        self._mode: str | None = None
        self._winner: int | None = None
        self._round = 1
        self._glow: np.ndarray | None = None

    # ------------------------------------------------------------------ main entry
    def draw(self, image: np.ndarray, game: Game, players: list[Player], t: float) -> None:
        h, w = image.shape[:2]
        s = h / 720.0  # UI scale
        dt = 0.0 if self._t_last is None else min(t - self._t_last, 0.1)
        self._t_last = t
        self._update_announcements(game, t)
        self.fx.update(t, game.effects, self.colors)
        visible = [p for p in players if p.visible and game.states.get(p.id) is not None]

        self._draw_bottles(image, game, t, s)
        self._draw_particles(image)
        self._draw_glow(image, game, t, s)
        for pid, aim in game.aims.items():
            self._draw_reticle(image, aim, pid, t, s)
        for p in visible:
            self._draw_name_tag(image, game, p, dt, s)
        self._draw_float_texts(image, t, s)
        self._draw_cards(image, game, visible, dt, t, s)
        self._draw_banner(image, game, t, s)
        if game.winner is not None:
            self._draw_winner(image, game, t, s)
        self._draw_announcements(image, t, s)

    def shake(self, t: float) -> tuple[int, int]:
        return self.fx.shake_offset(t)

    # ------------------------------------------------------------------ world elements
    def _draw_bottles(self, image, game: Game, t: float, s: float) -> None:
        for i, target in enumerate(game.targets):
            if not target.alive:
                continue
            age = t - target.spawned_t
            pop = min(1.0, 0.4 + age / 0.18) if age < 0.18 else 1.0 + 0.08 * math.exp(-(age - 0.18) * 12)
            sprite = assets.scaled(assets.bottle_sprite(int(target.radius * 2.6)), pop)
            bob = math.sin(t * 2.2 + i * 1.7) * 3 * s
            # Soft glow behind the bottle so it reads against the sky.
            blit_center(image, assets.glow_sprite(int(target.radius * 3.2), (200, 240, 255)),
                        target.center[0], target.center[1] + bob, alpha=0.35)  # fmt: skip
            blit_center(image, sprite, target.center[0], target.center[1] + bob)

    def _draw_particles(self, image) -> None:
        p = self.fx.particles
        if not len(p.life):
            return
        fade = p.fade()
        for (x, y), col, size, f in zip(p.pos, p.color, p.size, fade, strict=True):
            r = max(1, int(size * (0.4 + 0.6 * f)))
            cv2.circle(image, (int(x), int(y)), r, tuple(int(c) for c in col), -1, cv2.LINE_AA)

    def _draw_glow(self, image, game: Game, t: float, s: float) -> None:
        muzzles = [e for e in game.effects if e.kind == "muzzle" and t - e.t < 0.09]
        if not (game.aims or game.bullets or muzzles):
            return
        if self._glow is None or self._glow.shape != image.shape:
            self._glow = np.zeros_like(image)
        glow = self._glow
        glow[:] = 0
        for pid, aim in game.aims.items():
            col = np.array(RED if aim.on_target else self.colors.get(pid, WHITE), np.float32)
            pulse = 0.85 + 0.15 * math.sin(t * 9)
            outer = tuple((col * 0.30 * pulse).tolist())
            cv2.line(glow, _pt(aim.origin), _pt(aim.end), outer, max(2, int(7 * s)), cv2.LINE_AA)
            cv2.line(glow, _pt(aim.origin), _pt(aim.end), tuple((col * 0.85).tolist()), max(1, int(2 * s)),
                     cv2.LINE_AA)  # fmt: skip
        for b in game.bullets:
            progress = min(1.0, (t - b.t_fire) / max(b.t_arrive - b.t_fire, 1e-6))
            head = b.origin + b.direction * b.length * progress
            tail = b.origin + b.direction * b.length * max(0.0, progress - 0.35)
            cv2.line(glow, _pt(tail), _pt(head), (30, 110, 160), max(3, int(12 * s)), cv2.LINE_AA)
            cv2.line(glow, _pt(tail), _pt(head), (90, 210, 255), max(2, int(4 * s)), cv2.LINE_AA)
            cv2.line(glow, _pt(tail), _pt(head), (255, 255, 255), 1, cv2.LINE_AA)
        for e in muzzles:
            k = 1 - (t - e.t) / 0.09
            for r, inten in ((46, 0.25), (30, 0.5), (16, 1.0)):
                color = (int(120 * inten * k), int(220 * inten * k), int(255 * inten * k))
                cv2.circle(glow, _pt(e.pos), int(r * s * (0.6 + 0.4 * k)), color, -1, cv2.LINE_AA)
        cv2.add(image, glow, dst=image)

    def _draw_reticle(self, image, aim, pid: int, t: float, s: float) -> None:
        color = RED if aim.on_target else self.colors.get(pid, WHITE)
        c = _pt(aim.end)
        r = int((20 if aim.on_target else 14) * s * (1 + (0.08 * math.sin(t * 12) if aim.on_target else 0)))
        cv2.circle(image, c, r + 2, BLACK, 3, cv2.LINE_AA)
        cv2.circle(image, c, r, color, 2, cv2.LINE_AA)
        spin = t * (3.0 if aim.on_target else 1.0)
        for k in range(4):
            a = spin + k * math.pi / 2
            p0 = (int(c[0] + math.cos(a) * r * 0.55), int(c[1] + math.sin(a) * r * 0.55))
            p1 = (int(c[0] + math.cos(a) * r * 1.45), int(c[1] + math.sin(a) * r * 1.45))
            cv2.line(image, p0, p1, color, 2, cv2.LINE_AA)
        cv2.circle(image, c, 2, color, -1, cv2.LINE_AA)

    def _draw_name_tag(self, image, game: Game, p: Player, dt: float, s: float) -> None:
        """Name tag + mini HP bar floating above the head (T1's extra visual)."""
        st = game.states[p.id]
        color = self.colors.get(p.id, WHITE)
        if p.face is not None:
            x0, y0, x1, _ = p.face.bbox
            target = np.array([(x0 + x1) / 2, y0 - 26 * s], np.float32)
        elif p.smoothed is not None:
            torso = p.signals.torso_len if p.signals else 150.0
            target = np.array([p.smoothed[0][0], p.smoothed[0][1] - 0.6 * torso], np.float32)
        else:
            return
        prev = self._tags.get(p.id)
        pos = target if prev is None else prev + TAG_FOLLOW * (target - prev)
        self._tags[p.id] = pos
        h, w = image.shape[:2]
        cx = float(np.clip(pos[0], 60 * s, w - 60 * s))
        cy = float(np.clip(pos[1], 40 * s, h - 40 * s))

        label = self.text.get(f"P{p.id}", int(30 * s), WHITE, HUD_FONT)
        lw, lh = label.shape[1] + int(22 * s), label.shape[0] + int(4 * s)
        x0, y0 = int(cx - lw / 2), int(cy - lh - 6 * s)
        rounded_rect(image, x0 - 2, y0 - 2, x0 + lw + 2, y0 + lh + 2, int(lh / 2) + 2, BLACK)
        rounded_rect(image, x0, y0, x0 + lw, y0 + lh, int(lh / 2), color)
        blit_center(image, label, cx, y0 + lh / 2)
        # Mini HP bar under the tag.
        bw, bh = int(84 * s), max(5, int(8 * s))
        bx, by = int(cx - bw / 2), int(cy + 2 * s)
        rounded_rect(image, bx - 2, by - 2, bx + bw + 2, by + bh + 2, bh, BLACK)
        frac = self._hp_shown.get(p.id, st.hp) / game.cfg.max_hp
        if frac > 0.01:
            rounded_rect(image, bx, by, bx + max(bh, int(bw * frac)), by + bh, bh // 2, _hp_color(frac))
        if st.frame.ducked:
            badge = self.text.get("DUCKED", int(22 * s), CREAM, HUD_FONT, stroke=2)
            blit_center(image, badge, cx, by + bh + 16 * s)

    def _draw_float_texts(self, image, t: float, s: float) -> None:
        for ft in self.fx.texts:
            age = (t - ft.t0) / ft.duration
            pop = 1.0 + 0.4 * max(0.0, 1 - age * 6)
            sprite = self.text.get(ft.text, int(ft.size * s), ft.color, HUD_FONT, stroke=max(2, int(3 * s)))
            alpha = 1.0 if age < 0.6 else max(0.0, 1 - (age - 0.6) / 0.4)
            blit_center(
                image, assets.scaled(sprite, pop), ft.pos[0], ft.pos[1] - 70 * s * age - 30 * s, alpha
            )

    # ------------------------------------------------------------------ HUD
    def _draw_cards(self, image, game: Game, visible: list[Player], dt: float, t: float, s: float) -> None:
        h, w = image.shape[:2]
        cfg = game.cfg
        cw, ch, m = int(310 * s), int(96 * s), int(16 * s)
        for p in visible:
            st = game.states[p.id]
            color = self.colors.get(p.id, WHITE)
            # Animated HP: the bar eases down, and a light "damage trail" follows more slowly.
            shown = self._hp_shown.get(p.id, float(st.hp))
            shown += (st.hp - shown) * min(1.0, HP_DRAIN * dt) if st.hp < shown else st.hp - shown
            trail = self._hp_trail.get(p.id, shown)
            trail = max(shown, trail - 40 * dt) if trail > shown else shown
            self._hp_shown[p.id], self._hp_trail[p.id] = shown, trail

            x0 = m if p.id == 1 else w - m - cw
            y0 = h - m - ch
            hit_flash = t - st.last_hit_t < 0.25
            panel(image, x0, y0, x0 + cw, y0 + ch, r=int(14 * s), opacity=0.68)
            rounded_rect(image, x0, y0, x0 + cw, y0 + ch, int(14 * s), RED if hit_flash else color, 2)
            cv2.rectangle(
                image, (x0 + int(10 * s), y0 + int(12 * s)), (x0 + int(15 * s), y0 + int(40 * s)), color, -1
            )

            name = self.text.get(f"PLAYER {p.id}", int(30 * s), color, HUD_FONT)
            blit(image, name, x0 + int(22 * s), y0 + int(6 * s))
            label = "BOTTLES" if game.mode == "practice" else "WINS"
            score = self.text.get(f"{game.scores.get(p.id, 0)} {label}", int(24 * s), CREAM, HUD_FONT)
            blit(image, score, x0 + cw - score.shape[1] - int(12 * s), y0 + int(10 * s))

            bx0, by0 = x0 + int(12 * s), y0 + int(44 * s)
            bx1, by1 = x0 + cw - int(12 * s), y0 + int(60 * s)
            rounded_rect(image, bx0, by0, bx1, by1, int(8 * s), (45, 40, 38))
            span = bx1 - bx0
            r = int(8 * s)
            trail_x = bx0 + int(span * trail / cfg.max_hp)
            shown_x = bx0 + int(span * shown / cfg.max_hp)
            if trail_x > bx0 + 2 * r:
                rounded_rect(image, bx0, by0, trail_x, by1, r, (200, 220, 235))
            if shown_x > bx0 + 2 * r:
                frac = shown / cfg.max_hp
                rounded_rect(image, bx0, by0, shown_x, by1, r, _hp_color(frac))
                cv2.line(image, (bx0 + r, by0 + 3), (shown_x - r, by0 + 3), (255, 255, 255), 1, cv2.LINE_AA)
            hp_text = self.text.get(f"{int(round(shown))}", int(18 * s), WHITE, HUD_FONT, stroke=2)
            blit(
                image, hp_text, bx1 - hp_text.shape[1] - int(4 * s), by0 + (by1 - by0 - hp_text.shape[0]) // 2
            )

            icon_h = int(22 * s)
            for i in range(cfg.max_ammo):
                icon = assets.bullet_icon(icon_h, i < st.ammo)
                blit(image, icon, x0 + int(14 * s) + i * int(icon.shape[1] + 5 * s), y0 + int(66 * s))
            if st.ammo == 0 and int(t * 3) % 2 == 0:
                hint = self.text.get("RELOAD: LOWER BOTH ARMS", int(20 * s), GOLD, HUD_FONT)
                blit(image, hint, x0 + cw - hint.shape[1] - int(12 * s), y0 + int(68 * s))

    def _draw_banner(self, image, game: Game, t: float, s: float) -> None:
        w = image.shape[1]
        trigger = "CLOSE YOUR HAND TO SHOOT" if game.cfg.trigger == "fist" else "FLICK UP TO SHOOT"
        title, hint = {
            "idle": ("HIGH NOON", "STEP INTO VIEW"),
            "practice": ("PRACTICE", f"POINT YOUR ARM  |  {trigger}  |  DUCK BEHIND THE WALL"),
            "duel": ("DUEL", f"{trigger}  |  DUCK BEHIND THE WALL  |  FIRST TO ZERO LOSES"),
        }[game.mode]
        title_s = self.text.get(title, int(30 * s), GOLD, TITLE_FONT, stroke=2, stroke_color=(20, 30, 50))
        hint_s = self.text.get(hint, int(19 * s), CREAM, HUD_FONT)
        pw = max(title_s.shape[1], hint_s.shape[1]) + int(40 * s)
        ph = title_s.shape[0] + hint_s.shape[0] + int(10 * s)
        x0, y0 = w // 2 - pw // 2, int(10 * s)
        panel(image, x0, y0, x0 + pw, y0 + ph, r=int(16 * s), opacity=0.55)
        blit_center(image, title_s, w / 2, y0 + int(4 * s) + title_s.shape[0] / 2)
        blit_center(image, hint_s, w / 2, y0 + int(4 * s) + title_s.shape[0] + hint_s.shape[0] / 2)

    def _draw_winner(self, image, game: Game, t: float, s: float) -> None:
        h, w = image.shape[:2]
        cv2.convertScaleAbs(image, dst=image, alpha=0.45)  # dim the scene behind the result
        color = self.colors.get(game.winner, WHITE)
        age = t - (game.round_over_t or t)
        pop = 1.0 + 0.25 * math.exp(-age * 6)
        big = self.text.get(f"PLAYER {game.winner}", int(110 * s), color, TITLE_FONT, stroke=4, shadow=6)
        blit_center(image, assets.scaled(big, pop), w / 2, h * 0.42)
        sub = self.text.get("WINS THE ROUND", int(48 * s), CREAM, HUD_FONT, stroke=2)
        blit_center(image, sub, w / 2, h * 0.42 + big.shape[0] * 0.75)
        left = max(0, math.ceil(game.cfg.round_end_s - age))
        nxt = self.text.get(f"NEXT ROUND IN {left}", int(28 * s), GOLD, HUD_FONT)
        blit_center(image, nxt, w / 2, h * 0.42 + big.shape[0] * 0.75 + sub.shape[0] * 1.2)

    # ------------------------------------------------------------------ announcements
    def _update_announcements(self, game: Game, t: float) -> None:
        if game.mode != self._mode:
            if game.mode == "duel":
                self.fx.announce("DRAW!", t, GOLD)
            elif game.mode == "practice" and self._mode == "duel":
                self.fx.announce("PRACTICE", t, CREAM, size=72)
            self._mode = game.mode
        if self._winner is not None and game.winner is None:
            self._round += 1
            self.fx.announce(f"ROUND {self._round}", t, GOLD)
        self._winner = game.winner

    def _draw_announcements(self, image, t: float, s: float) -> None:
        h, w = image.shape[:2]
        for a in self.fx.announcements:
            age = (t - a.t0) / a.duration
            pop = 1.0 + 0.6 * max(0.0, 1 - age * 5)
            alpha = 1.0 if age < 0.7 else max(0.0, 1 - (age - 0.7) / 0.3)
            sprite = self.text.get(
                a.text, int(a.size * s), a.color, TITLE_FONT, stroke=4, stroke_color=(20, 25, 40), shadow=5
            )
            blit_center(image, assets.scaled(sprite, pop), w / 2, h * 0.36, alpha)


def draw_signal_debug(image: np.ndarray, game: Game, players: list[Player], colors: dict) -> None:
    """Per-player readout of the values the action thresholds act on (for tuning live)."""
    cfg = game.cfg
    for p in players:
        st = game.states.get(p.id)
        if not p.visible or p.signals is None or st is None or p.smoothed is None:
            continue
        s, f = p.signals, st.frame
        lines = [
            f"arm {s.extension:4.2f}  elev {np.degrees(s.elevation):+4.0f} (aim -50..80)",
            f"aim {'ON' if f.aiming else 'off'}  head {f.head_pos:.2f} (duck > {1 - cfg.cover_height:.2f})",
            f"arms down {'YES' if s.arms_down else 'no'}  ammo {st.ammo}",
            "hand: not tracked"
            if s.hand_openness is None
            else f"hand {s.hand_openness:4.2f} {'FIST' if f.hand_closed else 'open'}"
            f" (fist < {cfg.fist_closed:.2f}, open > {cfg.fist_open:.2f})",
        ]
        # Beside the face (the side with more room), on a dark panel so it reads over anything.
        h, w = image.shape[:2]
        panel_w, line_h = 400, 20
        panel_h = line_h * len(lines) + 12
        if p.face is not None:
            fx0, fy0, fx1, _ = p.face.bbox.astype(int)
        else:
            nose = p.smoothed[0].astype(int)
            fx0, fy0, fx1 = nose[0] - 60, nose[1] - 60, nose[0] + 60
        x = fx1 + 15 if w - fx1 > fx0 else fx0 - 15 - panel_w
        x = min(max(5, x), w - panel_w - 5)
        y = min(max(60, fy0), h - panel_h - 5)
        roi = image[y : y + panel_h, x : x + panel_w]
        cv2.addWeighted(roi, 0.3, np.zeros_like(roi), 0.7, 0, dst=roi)
        for i, text in enumerate(lines):
            org = (x + 8, y + 18 + i * line_h)
            cv2.putText(
                image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(p.id, WHITE), 1, cv2.LINE_AA
            )
