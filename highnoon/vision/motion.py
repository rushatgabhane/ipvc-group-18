"""T2: turns raw keypoints into smooth, stable per-player signals.

Filters are keyed by player id, so they need T3's identities. When a slot gets a new
identity, its filter is reset so a newcomer's pose is not blended with the previous one.
Keypoints below `min_visibility` are frozen at their last filtered value instead of
following noisy guesses.
"""

from __future__ import annotations

from highnoon.config import FilterConfig
from highnoon.contracts import Player
from highnoon.core.filters import OneEuroFilter


class MotionFilter:
    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        self._filters: dict[int, OneEuroFilter] = {}

    def _filter(self, player_id: int) -> OneEuroFilter:
        f = self._filters.get(player_id)
        if f is None:
            f = OneEuroFilter(self.cfg.min_cutoff, self.cfg.beta, self.cfg.d_cutoff)
            self._filters[player_id] = f
        return f

    def update(self, players: list[Player], t: float, reassigned: set[int]) -> None:
        for p in players:
            f = self._filter(p.id)
            if p.id in reassigned:
                f.reset()
            if p.pose is None:
                continue
            reliable = p.pose.visibility >= self.cfg.min_visibility
            p.smoothed = f(p.pose.keypoints, t, mask=reliable)
