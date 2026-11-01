"""Replay a recorded clip through the full tracking pipeline and report the numbers the report needs.

    .venv/bin/python tools/evaluate_clip.py clips/session.mp4            # -> results/session_eval.md

Per task (see docs/PERFORMANCE.md, "Report evaluation checklist"):
  T1  face detection rate, hat jitter raw vs smoothed (position, tilt, size) and the lag smoothing adds
  T2  shoulder and wrist jitter raw vs smoothed, filter lag, share of unreliable keypoints
  T3  identity consistency and switches (checked against an offline face reference), gaps and how long
      re-identification took, new ids handed out after enrolment, one-frame position jumps, the face match
      score distributions behind the thresholds, positions in metres
  T5  per-module time (pose, face, face id, hands, tracking, filtering, face effects)

The face reference: each id's faces from its first few seconds (while the two players stand apart) form a
reference for that person. Every later face is matched against all references offline, and the frame
"agrees" when the best match is the id the tracker gave. This uses the same face model as the tracker, so
it is a consistency check rather than independent ground truth; scripted clips (docs/PERFORMANCE.md,
"Recording protocol") supply the expected events to compare against.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from highnoon.config import Config
from highnoon.contracts import FaceObservation
from highnoon.core import evaluation as ev
from highnoon.core.capture import open_source
from highnoon.core.identity_log import COLUMNS as IDENTITY_COLUMNS
from highnoon.core.identity_log import identity_rows
from highnoon.game.game import Game
from highnoon.render.face_damage import draw_face_damage
from highnoon.render.face_fx import FaceEffectRenderer
from highnoon.tracking.identity import IdentityTracker
from highnoon.vision.head import measure_head
from highnoon.vision.motion import L_SHOULDER, R_SHOULDER, R_WRIST, MotionFilter
from highnoon.vision.perception import Perceiver

TIMED = ("pose", "face", "face_id", "hand", "inference", "postprocess", "tracking", "filter", "face_fx")
JUMP_PER_WIDTH = 0.15  # one frame; nobody crosses 15% of the picture in 33 ms
SWITCH_MIN_S = 0.3  # shorter disagreements are single bad face crops, not a swapped id
APART_PER_WIDTH = 0.2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--enrol-s", type=float, default=4.0, help="seconds per id used as its face reference"
    )
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    cfg = Config()
    cfg.capture.source = str(args.clip)
    cfg.capture.realtime = False
    cfg.filters.camera_hfov_deg = args.hfov
    data = replay(cfg, args.max_frames)
    args.out.mkdir(parents=True, exist_ok=True)
    write_identity_csv(args.out / f"{args.clip.stem}_identity.csv", data["identity_rows"])
    report = build_report(args.clip, data, cfg, args.enrol_s)
    path = args.out / f"{args.clip.stem}_eval.md"
    path.write_text(report)
    print(report)
    print(f"\nWrote {path}")


def replay(cfg: Config, max_frames: int | None) -> dict:
    source = open_source(cfg.capture)
    source.start()
    perceiver = Perceiver(cfg.perception)
    tracker = IdentityTracker(cfg.tracking, cfg.perception.max_players)
    motion = MotionFilter(cfg.filters)
    face_fx = FaceEffectRenderer()
    game = None
    n = cfg.perception.max_players
    frames: list[dict] = []
    rows: list[list] = []
    last_id = 0
    try:
        while (frame := source.read(last_id)) is not None:
            last_id = frame.id
            game = game or Game(cfg.game, frame.size)
            t = frame.timestamp_ms / 1000.0
            frame.t_capture = t
            perception = perceiver(frame)
            timings = dict(perception.timings)
            started = time.perf_counter()
            players = tracker.update(perception)
            timings["tracking"] = ms_since(started)
            started = time.perf_counter()
            motion.update(players, t, tracker.reassigned, frame.size)
            timings["filter"] = ms_since(started)
            game.update(players, t, tracker.reassigned)
            if game.mode == "duel":
                tracker.lock()
            timings["face_fx"] = time_face_effects(face_fx, frame.image.copy(), players)
            rows += identity_rows(frame.id, t, players, tracker, perception)
            frames.append(
                {
                    "t": t,
                    "size": frame.size,
                    "timings": timings,
                    "poses": len(perception.poses),
                    "faces": len(perception.faces),
                    "players": [snapshot(p, tracker) for p in players[:n]],
                }
            )
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        perceiver.close()
        source.stop()
    return {"frames": frames, "identity_rows": rows, "locked": tracker.locked}


def ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def time_face_effects(face_fx: FaceEffectRenderer, image: np.ndarray, players) -> float:
    started = time.perf_counter()
    for player in players:
        if not player.visible or player.face is None:
            continue
        face = player.face
        if player.face_smoothed is not None:
            points = player.face_smoothed
            face = FaceObservation(points, np.concatenate([points.min(axis=0), points.max(axis=0)]))
        draw_face_damage(image, face, player.id, 0.5, math.inf)
        if player.head is not None:
            face_fx.draw(image, player.head)
    return ms_since(started)


def snapshot(player, tracker) -> dict:
    nan2, nan3 = np.full(2, np.nan), np.full(4, np.nan)
    if not player.visible or player.pose is None:
        return {
            "visible": False,
            "x": np.nan,
            "raw_sh": nan2,
            "smooth_sh": nan2,
            "raw_wr": nan2,
            "smooth_wr": nan2,
            "raw_hat": nan3,
            "smooth_hat": nan3,
            "embedding": None,
            "pos": None,
            "new_id": player.id in tracker.reassigned,
            "face": False,
            "reliable": np.nan,
            "sh_ok": False,
            "wr_ok": False,
        }
    raw = player.pose.keypoints
    smooth = player.smoothed if player.smoothed is not None else raw
    raw_head = measure_head(player.face) if player.face is not None else None
    return {
        "visible": True,
        "x": float(player.pose.center()[0]),
        "raw_sh": (raw[L_SHOULDER] + raw[R_SHOULDER]) / 2,
        "smooth_sh": (smooth[L_SHOULDER] + smooth[R_SHOULDER]) / 2,
        "raw_wr": raw[R_WRIST].copy(),
        "smooth_wr": smooth[R_WRIST].copy(),
        "raw_hat": head_vector(raw_head),
        "smooth_hat": head_vector(player.head),
        "embedding": None if player.face is None else player.face.embedding,
        "pos": None if player.position_m is None else player.position_m.copy(),
        "new_id": player.id in tracker.reassigned,
        "face": player.face is not None,
        "reliable": float((player.pose.visibility >= 0.5).mean()),
        # The filter holds an unreliable keypoint still, so jitter is only measured where it is reliable.
        "sh_ok": bool((player.pose.visibility[[L_SHOULDER, R_SHOULDER]] >= 0.5).all()),
        "wr_ok": bool(player.pose.visibility[R_WRIST] >= 0.5),
    }


def head_vector(head) -> np.ndarray:
    if head is None:
        return np.full(4, np.nan)
    return np.array([head.center[0], head.center[1], math.degrees(head.angle), head.width])


def write_identity_csv(path: Path, rows: list[list]) -> None:
    import csv

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(IDENTITY_COLUMNS)
        writer.writerows(rows)


def series(frames: list[dict], slot: int, key: str) -> np.ndarray:
    return np.array([f["players"][slot][key] for f in frames], dtype=np.float64)


def build_report(clip: Path, data: dict, cfg: Config, enrol_s: float) -> str:
    frames = data["frames"]
    if not frames:
        return f"# {clip.name}: no frames"
    t = np.array([f["t"] for f in frames])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    width = frames[0]["size"][0]
    slots = range(len(frames[0]["players"]))
    out = [
        f"# Evaluation of `{clip.name}`",
        "",
        f"{len(frames)} frames, {t[-1] - t[0]:.1f} s, {1 / dt:.0f} fps source, replayed frame by frame "
        f"(no frames dropped, so these are module costs, not live latency).",
        "",
    ]

    out += [
        "## Module timings (T5, and each task's real-time cost)",
        "",
        "| Module | mean ms | p95 ms | frames |",
        "|---|---|---|---|",
    ]
    for name in TIMED:
        values = [f["timings"][name] for f in frames if name in f["timings"]]
        if values:
            out.append(
                f"| {name} | {np.mean(values):.2f} | {np.percentile(values, 95):.2f} | {len(values)} |"
            )
    out.append("")

    visible = {s: series(frames, s, "visible").astype(bool) for s in slots}
    has_face = {s: series(frames, s, "face").astype(bool) for s in slots}
    out += ["## Detection (T1, T2, T3)", "", "| | P1 | P2 |", "|---|---|---|"]
    out.append(
        "| frames with the id visible | " + " | ".join(f"{visible[s].mean():.1%}" for s in slots) + " |"
    )
    out.append(
        "| face found while visible | "
        + " | ".join(f"{has_face[s][visible[s]].mean():.1%}" if visible[s].any() else "-" for s in slots)
        + " |"
    )
    reliable = {s: series(frames, s, "reliable") for s in slots}
    out.append(
        "| keypoints above confidence 0.5 | "
        + " | ".join(f"{np.nanmean(reliable[s]):.1%}" if visible[s].any() else "-" for s in slots)
        + " |"
    )
    both = visible[0] & visible[1] if len(slots) > 1 else visible[0]
    out += [
        "",
        f"Both players visible in {both.mean():.1%} of frames; poses detected per frame: "
        f"{np.mean([f['poses'] for f in frames]):.2f}, faces: {np.mean([f['faces'] for f in frames]):.2f}.",
        "",
    ]

    out += [
        "## Smoothing: jitter and lag (T1 hat, T2 keypoints)",
        "",
        "Jitter is the per-frame noise left after removing smooth motion (second-difference estimate). "
        "Lag is the delay that best aligns smoothed with raw velocity.",
        "",
        "| Player | Signal | raw jitter | smoothed jitter | reduction | lag ms |",
        "|---|---|---|---|---|---|",
    ]
    for s in slots:
        v = visible[s]
        for label, raw_key, smooth_key, column, unit, gate in (
            ("shoulder centre x", "raw_sh", "smooth_sh", 0, "px", "sh_ok"),
            ("shoulder centre y", "raw_sh", "smooth_sh", 1, "px", "sh_ok"),
            ("right wrist x", "raw_wr", "smooth_wr", 0, "px", "wr_ok"),
            ("right wrist y", "raw_wr", "smooth_wr", 1, "px", "wr_ok"),
            ("hat centre x", "raw_hat", "smooth_hat", 0, "px", None),
            ("hat centre y", "raw_hat", "smooth_hat", 1, "px", None),
            ("hat tilt", "raw_hat", "smooth_hat", 2, "deg", None),
            ("hat width", "raw_hat", "smooth_hat", 3, "px", None),
        ):
            raw = np.array([f["players"][s][raw_key][column] for f in frames], np.float64)
            smooth = np.array([f["players"][s][smooth_key][column] for f in frames], np.float64)
            ok = v & np.isfinite(raw) & np.isfinite(smooth)
            if gate is not None:
                ok &= np.array([f["players"][s][gate] for f in frames], bool)
            if ok.sum() < 10:
                continue
            j_raw, j_smooth = ev.noise_level(raw, ok), ev.noise_level(smooth, ok)
            lag = ev.filter_lag_s(raw, smooth, ok, dt) * 1000
            reduction = 1 - j_smooth / j_raw if j_raw > 0 else float("nan")
            cells = [f"P{s + 1}", label, f"{j_raw:.2f} {unit}", f"{j_smooth:.2f} {unit}"]
            out.append(table_row(*cells, f"{reduction:.0%}", f"{lag:.0f}"))
    out.append("")

    out += identity_section(frames, t, slots, visible, width, cfg, enrol_s, data["locked"])
    out += position_section(frames, slots, visible)
    return "\n".join(out) + "\n"


def identity_section(frames, t, slots, visible, width, cfg, enrol_s, locked) -> list[str]:
    out = ["## Identity (T3)", ""]
    new_ids = [(f["t"], s + 1) for f in frames for s in slots if f["players"][s]["new_id"]]
    out.append(
        "New ids handed out: " + (", ".join(f"P{p} at {tt:.1f} s" for tt, p in new_ids) or "none") + "."
    )
    out.append(f"Enrolment locked (duel started): {'yes' if locked else 'no'}.")
    out.append("")

    waiting = np.array([f["poses"] - sum(p["visible"] for p in f["players"]) > 0 for f in frames])
    all_gaps = [g for s in slots for g in ev.gaps(t, visible[s], waiting, s + 1)]
    out += ["### Disappearance and re-entry", ""]
    if all_gaps:
        out += [
            "| Player | left at | gone for | came back | re-identification wait |",
            "|---|---|---|---|---|",
        ]
        for g in all_gaps:
            back = "yes" if g.recovered else "no"
            out.append(
                table_row(
                    f"P{g.player}",
                    f"{g.start_t:.1f} s",
                    f"{g.duration_s:.2f} s",
                    back,
                    f"{g.reid_wait_s:.2f} s",
                )
            )
        short = [g.duration_s for g in all_gaps if g.duration_s < 0.5]
        waits = ev.summary_stats([g.reid_wait_s for g in all_gaps if g.recovered])
        out += [
            "",
            f"{len(all_gaps)} gaps, {len(short)} under 0.5 s (detection dropouts rather than leaving). "
            f"Re-identification wait: {fmt_stats(waits)}.",
        ]
    else:
        out.append("No gaps: every id stayed visible once seen.")
    out.append("")

    jumps = [
        (tt, s + 1)
        for s in slots
        for tt in ev.position_jumps(t, series(frames, s, "x"), visible[s], JUMP_PER_WIDTH * width)
    ]
    out += [
        "### One-frame position jumps",
        "",
        "The body centre moved more than 15% of the frame width in one frame. That happens when part of the "
        "body is hidden (behind the wall, or by the other player); it is only a swap if the face check "
        "below disagrees at the same time.",
        "",
        ("none" if not jumps else ", ".join(f"P{p} at {tt:.2f} s" for tt, p in sorted(jumps))),
        "",
    ]

    out += ["### Consistency against the face reference", ""]
    references = {}
    for s in slots:
        first = np.flatnonzero(visible[s])
        if not len(first):
            continue
        start = t[first[0]]
        picks = []
        for f in frames:
            if not (start <= f["t"] <= start + enrol_s):
                continue
            p = f["players"][s]
            others = [o["x"] for i, o in enumerate(f["players"]) if i != s and o["visible"]]
            apart = all(abs(p["x"] - x) > APART_PER_WIDTH * width for x in others)
            if p["visible"] and p["embedding"] is not None and apart:
                picks.append(p["embedding"])
        if len(picks) >= 3:
            references[s + 1] = np.array(picks)
    if len(references) < 2:
        out += ["Needs faces of two players in the first seconds; not enough here.", ""]
        return out

    out += [
        "| Player | frames checked | agree | switches (>= 0.3 s) | longest switch |",
        "|---|---|---|---|---|",
    ]
    same, other = [], []
    switch_list = []
    for s in slots:
        if s + 1 not in references:
            continue
        idx = [
            i
            for i, f in enumerate(frames)
            if f["players"][s]["visible"] and f["players"][s]["embedding"] is not None
        ]
        if not idx:
            continue
        emb = np.array([frames[i]["players"][s]["embedding"] for i in idx])
        best, scores = ev.reference_identity(emb, references)
        ids = sorted(references)
        own = ids.index(s + 1)
        same += scores[:, own].tolist()
        other += np.delete(scores, own, axis=1).max(axis=1).tolist()
        agree = np.zeros(len(frames), bool)
        checked = np.zeros(len(frames), bool)
        agree[idx], checked[idx] = best == s + 1, True
        found = ev.disagreements(t, agree, checked, s + 1, SWITCH_MIN_S)
        switch_list += found
        longest = max((d.duration_s for d in found), default=0.0)
        out.append(
            f"| P{s + 1} | {len(idx)} | {np.mean(best == s + 1):.1%} | {len(found)} | {longest:.2f} s |"
        )
    if switch_list:
        out += [
            "",
            "Switches: "
            + ", ".join(f"P{d.player} at {d.start_t:.1f} s for {d.duration_s:.2f} s" for d in switch_list),
        ]
    accept, reject = cfg.tracking.face_accept, cfg.tracking.face_reject
    same, other = np.array(same), np.array(other)
    out += [
        "",
        "### Face match scores (behind the thresholds)",
        "",
        "| | mean | 5th pct | 95th pct |",
        "|---|---|---|---|",
        score_row("same person", same),
        score_row("best other person", other),
        "",
        f"Same-person faces below accept ({accept}): {(same < accept).mean():.1%}. "
        f"Other-person faces at or above accept: {(other >= accept).mean():.1%}; above reject ({reject}): "
        f"{(other >= reject).mean():.1%}.",
        "",
    ]
    return out


def position_section(frames, slots, visible) -> list[str]:
    out = [
        "## Position in metres (T3 spatial output)",
        "",
        "From the torso length (tracking/spatial.py); x is left/right of the camera axis, z the distance.",
        "",
        "| Player | median distance | distance range (5-95%) | x range (5-95%) | distance jitter |",
        "|---|---|---|---|---|",
    ]
    for s in slots:
        pos = [f["players"][s]["pos"] for f in frames]
        ok = np.array([p is not None for p in pos]) & visible[s]
        if ok.sum() < 10:
            continue
        z = np.array([p[2] if p is not None else np.nan for p in pos])
        x = np.array([p[0] if p is not None else np.nan for p in pos])
        z_lo, z_hi = np.nanpercentile(z[ok], [5, 95])
        x_lo, x_hi = np.nanpercentile(x[ok], [5, 95])
        jitter_cm = ev.noise_level(z, ok) * 100
        out.append(
            table_row(
                f"P{s + 1}",
                f"{np.nanmedian(z[ok]):.2f} m",
                f"{z_lo:.2f}-{z_hi:.2f} m",
                f"{x_lo:+.2f} to {x_hi:+.2f} m",
                f"{jitter_cm:.1f} cm",
            )
        )
    out += [
        "",
        "Check against a tape-measured distance (stand at a marked 2.0 m and 3.0 m) to report the error.",
        "",
    ]
    return out


def table_row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def score_row(label: str, scores: np.ndarray) -> str:
    return table_row(label, f"{scores.mean():.3f}", *(f"{v:.3f}" for v in np.percentile(scores, [5, 95])))


def fmt_stats(stats: dict) -> str:
    if not stats["n"]:
        return "n/a"
    return (
        f"median {stats['median']:.2f} s, p95 {stats['p95']:.2f} s, max {stats['max']:.2f} s (n={stats['n']})"
    )


if __name__ == "__main__":
    main()
