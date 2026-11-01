from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = [
    "frame",
    "t",
    "player",
    "visible",
    "new_id",
    "locked",
    "x_px",
    "y_px",
    "x_m",
    "z_m",
    "has_face",
    "owner_similarity",
    "people_in_view",
    "people_without_id",
]


class IdentityLog:
    """One row per player slot per frame: who was seen where, and how sure the face match was.

    `tools/evaluate_clip.py` turns this into identity switches, recovery times and consistency.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)

    def record(self, frame_id: int, t: float, players, tracker, perception) -> None:
        for row in identity_rows(frame_id, t, players, tracker, perception):
            self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()


def identity_rows(frame_id: int, t: float, players, tracker, perception) -> list[list]:
    people, without_id = len(perception.poses), tracker.unassigned_people(perception)
    rows = []
    for player in players:
        center = player.pose.center() if player.visible and player.pose is not None else None
        position = player.position_m
        similarity = tracker.owner_similarity(player) if player.visible else None
        rows.append(
            [
                frame_id,
                f"{t:.3f}",
                player.id,
                int(player.visible),
                int(player.id in tracker.reassigned),
                int(tracker.locked),
                "" if center is None else f"{center[0]:.1f}",
                "" if center is None else f"{center[1]:.1f}",
                "" if position is None else f"{position[0]:.3f}",
                "" if position is None else f"{position[2]:.3f}",
                int(player.face is not None),
                "" if similarity is None else f"{similarity:.3f}",
                people,
                without_id,
            ]
        )
    return rows
