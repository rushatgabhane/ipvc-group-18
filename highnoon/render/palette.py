from __future__ import annotations

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (70, 200, 255)
RED = (60, 60, 235)
CREAM = (215, 240, 250)
ORANGE = (30, 140, 255)
SAFE_GREEN = (140, 240, 160)
PLAYER_COLORS = {1: (70, 70, 235), 2: (235, 170, 50)}


def point(xy) -> tuple[int, int]:
    return int(xy[0]), int(xy[1])


def hp_color(fraction: float) -> tuple[int, int, int]:
    if fraction > 0.5:
        return (90, 210, 110)
    if fraction > 0.25:
        return (60, 200, 245)
    return (60, 70, 235)
