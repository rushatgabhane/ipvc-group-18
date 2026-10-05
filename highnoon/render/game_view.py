"""T5: draws the game layer on top of the composited camera image.

Draw order matters: crates are drawn after the players, so they occlude the lower body.
A ducking player really disappears behind cover.
Everything uses OpenCV primitives on small regions, never full-frame per-pixel numpy ops.
"""

from __future__ import annotations

import cv2
import numpy as np

from highnoon.contracts import Player
from highnoon.game.game import Game

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, BLACK, RED, YELLOW, GREEN = (255, 255, 255), (0, 0, 0), (40, 40, 235), (60, 220, 255), (90, 220, 90)
WOOD, WOOD_DARK = (40, 95, 150), (25, 60, 100)


def draw_game(image: np.ndarray, game: Game, players: list[Player], t: float, colors: dict) -> None:
    for p in players:
        st = game.states.get(p.id)
        if p.visible and st is not None and st.cover is not None:
            _draw_crate(image, st.cover)
    for target in game.targets:
        if target.alive:
            _draw_bottle(image, target.center, target.radius)
    for pid, aim in game.aims.items():
        _draw_laser(image, aim, colors.get(pid, WHITE))
    for b in game.bullets:
        progress = min(1.0, (t - b.t_fire) / max(b.t_arrive - b.t_fire, 1e-6))
        head = b.origin + b.direction * b.length * progress
        tail = b.origin + b.direction * b.length * max(0.0, progress - 0.3)
        cv2.line(image, _pt(tail), _pt(head), YELLOW, 4, cv2.LINE_AA)
        cv2.line(image, _pt(tail), _pt(head), WHITE, 1, cv2.LINE_AA)
    for e in game.effects:
        _draw_effect(image, e, t)
    for p in players:
        st = game.states.get(p.id)
        if p.visible and st is not None and st.frame.ducked and st.cover is not None:
            cx = int((st.cover.x0 + st.cover.x1) / 2)
            _text_center(image, "DUCKED", (cx, int(st.cover.y0) - 10), 0.7, colors.get(p.id, WHITE), 2)
    _draw_hud(image, game, {p.id for p in players if p.visible}, colors, t)


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(p[0]), int(p[1])


def _text_center(image, text, center, scale, color, thickness=2) -> None:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    org = (int(center[0] - w / 2), int(center[1] + h / 2))
    cv2.putText(image, text, org, FONT, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_crate(image: np.ndarray, c) -> None:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = int(max(c.x0, 0)), int(max(c.y0, 0)), int(min(c.x1, w - 1)), int(min(c.y1, h - 1))
    if x1 <= x0 or y1 <= y0:
        return
    cv2.rectangle(image, (x0, y0), (x1, y1), WOOD, -1)
    plank = max(12, (y1 - y0) // 5)
    for y in range(y0 + plank, y1, plank):
        cv2.line(image, (x0, y), (x1, y), WOOD_DARK, 2)
    cv2.line(image, (x0, y0), (x1, y1), WOOD_DARK, 4)
    cv2.line(image, (x1, y0), (x0, y1), WOOD_DARK, 4)
    cv2.rectangle(image, (x0, y0), (x1, y1), WOOD_DARK, 5)


def _draw_bottle(image: np.ndarray, center: np.ndarray, r: float) -> None:
    cx, cy, r = int(center[0]), int(center[1]), int(r)
    cv2.rectangle(image, (cx - r // 2, cy - r), (cx + r // 2, cy + r), (60, 140, 60), -1)
    cv2.rectangle(image, (cx - r // 5, cy - 2 * r), (cx + r // 5, cy - r), (60, 140, 60), -1)
    cv2.rectangle(image, (cx - r // 2, cy - r), (cx + r // 2, cy + r), (30, 80, 30), 2)
    cv2.circle(image, (cx, cy), r + 6, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_laser(image: np.ndarray, aim, color) -> None:
    cv2.line(image, _pt(aim.origin), _pt(aim.end), color, 2, cv2.LINE_AA)
    end = _pt(aim.end)
    if aim.on_target:
        cv2.circle(image, end, 7, RED, -1, cv2.LINE_AA)
        cv2.circle(image, end, 18, RED, 2, cv2.LINE_AA)
    else:
        cv2.circle(image, end, 5, color, -1, cv2.LINE_AA)


def _draw_effect(image: np.ndarray, e, t: float) -> None:
    age = t - e.t
    pos = _pt(e.pos)
    if e.kind == "muzzle":
        if age < 0.12:
            cv2.circle(image, pos, int(22 * (1 - age / 0.12)) + 4, YELLOW, -1, cv2.LINE_AA)
        return
    grow = int(10 + 60 * age)
    if e.kind in ("hit", "headshot"):
        cv2.circle(image, pos, grow, RED, 3, cv2.LINE_AA)
        cv2.circle(image, pos, max(4, 14 - int(20 * age)), RED, -1, cv2.LINE_AA)
        label = "HEADSHOT!" if e.kind == "headshot" else "HIT"
        _text_center(
            image, label, (pos[0], pos[1] - 40 - int(40 * age)), 1.0 if e.kind == "headshot" else 0.8, RED
        )
    elif e.kind == "blocked":
        for k in range(6):
            ang = k * np.pi / 3 + age * 3
            tip = (int(pos[0] + grow * np.cos(ang)), int(pos[1] + grow * np.sin(ang)))
            cv2.line(image, pos, tip, YELLOW, 2, cv2.LINE_AA)
        _text_center(image, "BLOCKED", (pos[0], pos[1] - 30 - int(30 * age)), 0.7, YELLOW)
    elif e.kind == "shatter":
        for k in range(8):
            ang = k * np.pi / 4
            p = (int(pos[0] + grow * np.cos(ang)), int(pos[1] + grow * np.sin(ang) + 80 * age * age))
            cv2.circle(image, p, 4, GREEN, -1, cv2.LINE_AA)
        _text_center(image, "+1", (pos[0], pos[1] - 30 - int(30 * age)), 0.9, GREEN)


def _draw_hud(image: np.ndarray, game: Game, visible: set[int], colors: dict, t: float) -> None:
    h, w = image.shape[:2]
    cfg = game.cfg
    for pid, st in sorted(game.states.items()):
        if pid not in visible:
            continue
        color = colors.get(pid, WHITE)
        left = pid == 1
        x = 20 if left else w - 300
        y = h - 70
        cv2.rectangle(image, (x - 8, y - 32), (x + 288, y + 52), BLACK, -1)
        flash = t - st.last_hit_t < 0.25
        cv2.rectangle(image, (x - 8, y - 32), (x + 288, y + 52), RED if flash else color, 2)
        score = game.scores.get(pid, 0)
        cv2.putText(image, f"P{pid}   score {score}", (x, y - 8), FONT, 0.7, color, 2, cv2.LINE_AA)
        frac = st.hp / cfg.max_hp
        cv2.rectangle(image, (x, y + 2), (x + 280, y + 20), (60, 60, 60), -1)
        bar = GREEN if frac > 0.5 else YELLOW if frac > 0.25 else RED
        cv2.rectangle(image, (x, y + 2), (x + int(280 * frac), y + 20), bar, -1)
        for i in range(cfg.max_ammo):
            cx = x + 10 + i * 22
            fill = YELLOW if i < st.ammo else (70, 70, 70)
            cv2.rectangle(image, (cx - 4, y + 28), (cx + 4, y + 46), fill, -1)
        if st.ammo == 0:
            cv2.putText(image, "lower arms to reload", (x + 140, y + 44), FONT, 0.5, YELLOW, 1, cv2.LINE_AA)

    banner = {
        "idle": "Step into view",
        "practice": "PRACTICE - point your arm, flick it up to shoot the bottles",
        "duel": "DUEL - duck behind your crate to dodge",
    }[game.mode]
    _text_center(image, banner, (w // 2, 30), 0.75, WHITE, 2)
    if game.winner is not None:
        _text_center(
            image, f"PLAYER {game.winner} WINS!", (w // 2, h // 2), 2.2, colors.get(game.winner, WHITE), 5
        )


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
            f"aim {'ON' if f.aiming else 'off'}  drop {f.duck_drop:+.2f} (duck > {cfg.duck_enter:.2f})",
            f"arms down {'YES' if s.arms_down else 'no'}  ammo {st.ammo}",
            "hand: not tracked"
            if s.hand_openness is None
            else f"hand {s.hand_openness:4.2f} {'FIST' if f.hand_closed else 'open'}"
            f" (fist < {cfg.fist_closed:.2f}, open > {cfg.fist_open:.2f})",
        ]
        # Under the face, where it stays on screen even when the shoulders are out of frame.
        anchor = p.face.bbox[[0, 3]] if p.face is not None else p.smoothed[0]
        x = int(anchor[0])
        y = int(anchor[1]) + 25
        for i, text in enumerate(lines):
            org = (min(max(5, x), image.shape[1] - 330), min(image.shape[0] - 160, y) + i * 20)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 4, cv2.LINE_AA)
            cv2.putText(
                image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(p.id, WHITE), 1, cv2.LINE_AA
            )
