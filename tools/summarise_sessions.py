"""Aggregate the per-clip replays of the current code into the report's measured results (T1-T5).

usage (from the repo root): .venv/bin/python tools/summarise_sessions.py results/latest_e86b5f9 > summary.md
"""

# ruff: noqa: E501, E731, B023, B905  (report tables printed as long f-strings)

import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, ".")
import numpy as np

from highnoon.core import evaluation as ev

D = sys.argv[1]
E = f"{D}/exp"
SKIP = {"session_20261101_1148"}
NOT_TWO_PLAYER = {"session_20261101_1143"}  # aborted take, faces mostly out of view: shown, not in totals
stems = sorted(os.path.basename(p)[:-4] for p in glob.glob("clips/session_*.mp4"))
stems = [s for s in stems if s not in SKIP]
short = lambda s: s.replace("session_20261101_", "")
pct = lambda x: "-" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"
f1 = lambda x: "-" if x is None or not np.isfinite(x) else f"{x:.1f}"


def load_exp(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    by = {}
    for r in j["rows"]:
        by.setdefault(r["k"], {})[r["id"]] = r
    keep = [k for k in sorted(by) if 1 in by[k] and 2 in by[k]]
    return {"by": by, "keep": keep, "dups": j["dups"], "rows": j["rows"]}


def dt_of(x):
    ts = np.array([x["by"][k][1]["t"] for k in x["keep"]])
    return float(np.median(np.diff(ts))) if len(ts) > 2 else 1 / 30


def detection(x):
    by, keep = x["by"], x["keep"]
    vis = [by[k][i] for k in keep for i in (1, 2) if by[k][i]["vis"]]
    if not vis:
        return None
    both = [k for k in keep if by[k][1]["vis"] and by[k][2]["vis"]]
    raised = [r for r in vis if r.get("wr_ok")]
    sims = [r["sim"] for r in vis if r.get("sim") is not None]
    return dict(
        frames=len(keep),
        both_vis=len(both) / len(keep),
        face=np.mean([r["face"] for r in vis]),
        both_faces=np.mean([by[k][1].get("face", False) and by[k][2].get("face", False) for k in both])
        if both
        else np.nan,
        kp=np.mean([r["kp_ok"] for r in vis]),
        hand=np.mean([r["hand"] for r in raised]) if raised else np.nan,
        wrong=sum(s < 0.2 for s in sims),
        checks=len(sims),
        own=np.mean(sims) if sims else np.nan,
    )


def smoothing(x):
    """Jitter still/moving, filter lag, hat alignment, per player."""
    by, keep, dt = x["by"], x["keep"], dt_of(x)
    out = {}
    for pid in (1, 2):
        pr = [by[k][pid] for k in keep]

        def arr(key, n):
            return np.array([r[key] if (r["vis"] and key in r) else [np.nan] * n for r in pr], float)

        ok = np.array([r["vis"] and r.get("sh_ok", False) for r in pr])
        wr_ok = np.array([r["vis"] and r.get("wr_ok", False) for r in pr])
        raw_sh, sm_sh, raw_wr, sm_wr = arr("raw_sh", 2), arr("sm_sh", 2), arr("raw_wr", 2), arr("sm_wr", 2)
        raw_hat, sm_hat = arr("raw_hat", 4), arr("sm_hat", 4)
        hat_ok = np.isfinite(raw_hat[:, 0]) & np.isfinite(sm_hat[:, 0])
        speed = np.r_[np.nan, np.linalg.norm(np.diff(sm_sh, axis=0), axis=1)]
        still, moving = ok & (speed < 1.5), ok & (speed > 4)
        res = {"still_share": still.mean(), "moving_share": moving.mean()}
        for name, raw, sm, m0 in (
            ("shoulder", raw_sh, sm_sh, ok),
            ("wrist", raw_wr, sm_wr, wr_ok),
            ("hat", raw_hat[:, :2], sm_hat[:, :2], hat_ok),
            ("tilt", raw_hat[:, 2], sm_hat[:, 2], hat_ok),
        ):
            for label, m in (("still", still), ("moving", moving)):
                mm = m0 & m
                res[f"{name}_{label}"] = (
                    (ev.noise_level(raw, mm, 4), ev.noise_level(sm, mm, 4))
                    if mm.sum() > 20
                    else (np.nan, np.nan)
                )
            a = raw[:, 0] if raw.ndim > 1 else raw
            b = sm[:, 0] if sm.ndim > 1 else sm
            res[f"{name}_lag_ms"] = ev.filter_lag_s(a, b, m0, dt) * 1000 if m0.sum() > 50 else np.nan
        off = np.linalg.norm(raw_hat[:, :2] - sm_hat[:, :2], axis=1)
        fw = raw_hat[:, 3] / 1.5
        for label, m in (("still", still), ("moving", moving)):
            mm = hat_ok & m
            res[f"align_{label}"] = (
                (np.nanmean(off[mm]), np.nanpercentile(off[mm], 95), np.nanmean(off[mm] / fw[mm]))
                if mm.sum() > 10
                else (np.nan,) * 3
            )
        out[pid] = res
    return out


def speed_bins(x, acc):
    by, keep, dt = x["by"], x["keep"], dt_of(x)
    bins = [0.5, 1.0, 2.0, 3.0, 99]
    for pid in (1, 2):
        pr = [by[k][pid] for k in keep]
        for a, b in zip(pr, pr[1:]):
            if not (a["vis"] and b["vis"] and a.get("torso") and "sm_wr" in a and "sm_wr" in b):
                continue
            v = np.linalg.norm(np.subtract(b["sm_wr"], a["sm_wr"])) / dt * 0.5 / a["torso"]
            if v < bins[0] or v > 20:  # > 20 m/s is a one-frame tracking jump
                continue
            i = int(np.searchsorted(bins, v, side="right") - 1)
            c = acc.setdefault(i, [0, 0, 0])
            c[0] += 1
            c[1] += b["hand"]
            c[2] += bool(b.get("wr_ok"))
    return bins


def crossings(x, hold_s=1.0, window_s=1.5):
    """Side swaps: the left-right order of P1 and P2 flips and stays flipped >= hold_s."""
    by, keep = x["by"], x["keep"]
    seq = [
        (by[k][1]["t"], np.sign(by[k][1]["x"] - by[k][2]["x"]))
        for k in keep
        if by[k][1]["vis"] and by[k][2]["vis"] and "x" in by[k][1] and "x" in by[k][2]
    ]
    runs = []  # (sign, start, end)
    for t, s in seq:
        if runs and runs[-1][0] == s:
            runs[-1][2] = t
        else:
            runs.append([s, t, t])
    runs = [r for r in runs if r[2] - r[1] >= hold_s]
    merged = []
    for r in runs:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    events = []
    bad_t = [r["t"] for r in x["rows"] if r.get("sim") is not None and r["sim"] < 0.2]
    for a, b in zip(merged, merged[1:]):
        t = (a[2] + b[1]) / 2
        wrong = sum(abs(bt - t) <= window_s for bt in bad_t)
        events.append((t, wrong))
    return events


def eval_md(stem):
    p = f"{D}/{stem}_eval.md"
    if not os.path.exists(p):
        return None
    s = open(p).read()
    res = {"agree": [], "switches": 0, "longest": 0.0}
    for m in re.finditer(r"^\| P(\d) \| (\d+) \| ([\d.]+)% \| (\d+) \| ([\d.]+) s \|$", s, re.M):
        res["agree"].append((int(m[1]), int(m[2]), float(m[3])))
        res["switches"] += int(m[4])
        res["longest"] = max(res["longest"], float(m[5]))
    gaps = re.findall(r"^\| P\d \| ([\d.]+) s \| ([\d.]+) s \| (yes|no) \| ([\d.]+) s \|$", s, re.M)
    res["gaps"] = [(float(a), float(b), c == "yes", float(d)) for a, b, c, d in gaps]
    m = re.search(r"New ids handed out: (.*)\.", s)
    res["new_ids"] = m[1] if m else ""
    m = re.search(r"^(\d+) frames, ([\d.]+) s", s, re.M)
    res["frames"], res["dur"] = (int(m[1]), float(m[2])) if m else (0, 0)
    pos = re.findall(r"^\| P(\d) \| ([\d.]+) m \| ([\d.]+)-([\d.]+) m \| .* \| ([\d.]+) cm \|$", s, re.M)
    res["pos"] = [(int(a), float(b), float(c), float(d), float(e)) for a, b, c, d, e in pos]
    return res


def metrics(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))[30:]  # drop the warm-up second
    if len(rows) < 30:
        return None
    col = lambda k: np.array([float(r.get(k) or "nan") for r in rows])
    t = col("t")
    ft = np.diff(t)
    ft = ft[ft > 0]
    lat = col("latency")
    return dict(
        frames=len(rows),
        fps=1 / np.mean(ft),
        fps_p5=1 / np.percentile(ft, 95),
        fps_p1=1 / np.percentile(ft, 99),
        lat=np.nanmean(lat),
        lat95=np.nanpercentile(lat, 95),
        lat99=np.nanpercentile(lat, 99),
        dropped=np.nanmean(col("dropped")),
        worst_ms=1000 * ft.max(),
        **{
            k: np.nanmean(col(k))
            for k in (
                "pose",
                "face",
                "face_id",
                "hand",
                "inference",
                "tracking",
                "filter",
                "game",
                "render",
                "camera",
                "hand_lag",
            )
        },
    )


def ps_stats(path):
    if not os.path.exists(path):
        return None
    v = np.loadtxt(path, ndmin=2)
    if len(v) < 5:
        return None
    v = v[3:]
    return dict(cpu=v[:, 0].mean(), mem=v[:, 1].mean() / 1024, mem_peak=v[:, 1].max() / 1024)


print("# Measured results across all recordings, current code (e86b5f9)\n")
print(
    "Every raw camera recording of 1 Nov 2026 replayed through the current pipeline. "
    "`session_20261101_1148` (3.6 s) is left out. Repeated frames (written by the recorder to keep real "
    "time) are skipped for jitter and detection.\n"
)

# ---------------------------------------------------------------- per clip overview
DET, SM, CROSS, EVAL, FAST, PACED, PS = {}, {}, {}, {}, {}, {}, {}
SPEED = {}
for s in stems:
    x = load_exp(f"{E}/{s}_g1.0.json")
    if x:
        DET[s] = detection(x)
        SM[s] = smoothing(x)
        CROSS[s] = crossings(x)
        bins = speed_bins(x, SPEED)
    EVAL[s] = eval_md(s)
    FAST[s] = metrics(f"{D}/{s}_metrics.csv")
    PACED[s] = metrics(f"{E}/{s}_paced_metrics.csv")
    PS[s] = ps_stats(f"{E}/{s}_paced_ps.txt")


def wavg(key, src, weight="frames"):
    items = [
        (v[key], v[weight]) for s, v in src.items() if s not in NOT_TWO_PLAYER and v and np.isfinite(v[key])
    ]
    return sum(a * w for a, w in items) / sum(w for _, w in items) if items else np.nan


print("## T1 Face tracking and effects\n")
print(
    "| Clip | unique frames | both players visible | face found (visible player) | both faces (both visible) | wrong-face checks |"
)
print("|---|---|---|---|---|---|")
for s in stems:
    d = DET.get(s)
    if d:
        print(
            f"| {short(s)} | {d['frames']} | {pct(d['both_vis'])} | {pct(d['face'])} | {pct(d['both_faces'])} | {d['wrong']} / {d['checks']} |"
        )
print(
    f"| **all (without 1143)** | {sum(d['frames'] for s, d in DET.items() if d and s not in NOT_TWO_PLAYER)} | {pct(wavg('both_vis', DET))} | {pct(wavg('face', DET))} | "
    f"{pct(wavg('both_faces', DET))} | {sum(d['wrong'] for d in DET.values() if d)} / {sum(d['checks'] for d in DET.values() if d)} |\n"
)


def pooled(name):
    """median over clips and players of (raw, smoothed) jitter."""
    out = {}
    for label in ("still", "moving"):
        vals = [SM[s][p][f"{name}_{label}"] for s in SM for p in (1, 2)]
        vals = [v for v in vals if np.isfinite(v[0])]
        out[label] = (
            (np.median([v[0] for v in vals]), np.median([v[1] for v in vals]), len(vals))
            if vals
            else (np.nan, np.nan, 0)
        )
    lags = [SM[s][p][f"{name}_lag_ms"] for s in SM for p in (1, 2)]
    lags = [v for v in lags if np.isfinite(v)]
    out["lag"] = np.median(lags) if lags else np.nan
    return out


print(
    "Jitter (frame-to-frame noise after removing smooth motion, px or deg per frame), median over every "
    "clip and player; still = shoulders < 1.5 px/frame, moving = > 4 px/frame.\n"
)
print("| Signal | still raw → smoothed | reduction | moving raw → smoothed | reduction | filter lag |")
print("|---|---|---|---|---|---|")
for name, label in (
    ("hat", "T1 hat centre (px)"),
    ("tilt", "T1 hat tilt (deg)"),
    ("shoulder", "T2 shoulder centre (px)"),
    ("wrist", "T2 right wrist (px)"),
):
    p = pooled(name)
    cells = []
    for lab in ("still", "moving"):
        a, b, n = p[lab]
        cells += [f"{a:.2f} → {b:.2f} (n={n})", pct(1 - b / a) if a else "-"]
    print(f"| {label} | {' | '.join(cells)} | {p['lag']:.0f} ms |")
print()
print("Hat alignment error (smoothed hat vs raw face mesh in the same frame):\n")
print("| | mean px | p95 px | share of face width |")
print("|---|---|---|---|")
for lab in ("still", "moving"):
    vals = [SM[s][p][f"align_{lab}"] for s in SM for p in (1, 2)]
    vals = [v for v in vals if np.isfinite(v[0])]
    if vals:
        print(
            f"| {lab} | {np.median([v[0] for v in vals]):.1f} | {np.median([v[1] for v in vals]):.1f} | {np.median([v[2] for v in vals]):.1%} |"
        )
print()

print("## T2 Pose and motion\n")
print("| Clip | upper-body keypoints confident | hand found with wrist raised |")
print("|---|---|---|")
for s in stems:
    d = DET.get(s)
    if d:
        print(f"| {short(s)} | {pct(d['kp'])} | {pct(d['hand'])} |")
print(f"| **all (without 1143)** | {pct(wavg('kp', DET))} | {pct(wavg('hand', DET))} |\n")
print("Hand tracking against right-wrist speed (all clips, normal light):\n")
print("| Wrist speed | frames | pose wrist confident | hand landmarks found |")
print("|---|---|---|---|")
names = ["0.5-1 m/s", "1-2 m/s", "2-3 m/s", "> 3 m/s"]
for i in sorted(SPEED):
    n, h, w = SPEED[i]
    print(f"| {names[i]} | {n} | {w / n:.0%} | {h / n:.0%} |")
print()

print("## T3 Identity\n")
print(
    "| Clip | length | face-reference agreement P1 / P2 | switches ≥ 0.3 s | side swaps (wrong-face checks within 1.5 s) | gaps (≥ 0.5 s) | re-entries recovered | re-id wait median / p95 / max |"
)
print("|---|---|---|---|---|---|---|---|")
all_gaps, all_agree, all_sw, all_dur, all_cross = [], [], 0, 0.0, []
for s in stems:
    e = EVAL.get(s)
    if not e:
        continue
    ag = " / ".join(f"{a:.1f}% ({n})" for _, n, a in sorted(e["agree"]))
    all_agree += e["agree"]
    all_sw += e["switches"]
    all_dur += e["dur"]
    long_gaps = [g for g in e["gaps"] if g[1] >= 0.5]
    all_gaps += long_gaps
    back = [g for g in long_gaps if g[2]]
    waits = [g[3] for g in back]
    cr = CROSS.get(s, [])
    all_cross += cr
    crs = ", ".join(f"{t:.1f} s ({w})" for t, w in cr) or "none"
    wtxt = f"{np.median(waits):.2f} / {np.percentile(waits, 95):.2f} / {max(waits):.2f} s" if waits else "-"
    print(
        f"| {short(s)} | {e['dur']:.0f} s | {ag} | {e['switches']} | {crs} | {len(long_gaps)} | {len(back)} / {len(long_gaps)} | {wtxt} |"
    )
n_chk = sum(n for _, n, _ in all_agree)
agree = sum(n * a / 100 for _, n, a in all_agree) / n_chk if n_chk else np.nan
back = [g for g in all_gaps if g[2]]
waits = [g[3] for g in back]
ok_cross = sum(w == 0 for _, w in all_cross)
print(
    f"\n**All clips:** identity agreement {agree:.1%} of {n_chk} face-checked player-frames; {all_sw} switches "
    f"≥ 0.3 s in {all_dur / 60:.1f} min ({all_sw / (all_dur / 60):.2f} per minute); side swaps with both identities "
    f"preserved (no wrong-face check within 1.5 s): {ok_cross} / {len(all_cross)}; re-entries after ≥ 0.5 s out of view "
    f"recovered: {len(back)} / {len(all_gaps)}"
    + (
        f", re-identification median {np.median(waits):.2f} s, p95 {np.percentile(waits, 95):.2f} s, max {max(waits):.2f} s."
        if waits
        else "."
    )
)
pos = [p for e in EVAL.values() if e for p in e["pos"]]
if pos:
    print(
        f"Distance estimate: medians {min(p[1] for p in pos):.2f}-{max(p[1] for p in pos):.2f} m across clips, distance jitter "
        f"{np.median([p[4] for p in pos]):.1f} cm (median); no tape-measured reference.\n"
    )

print("## T4 Game control (live logs of every session: results/games.csv, results/*_signals.csv)\n")
g = list(csv.DictReader(open("results/games.csv"))) if os.path.exists("results/games.csv") else []
if g:
    tot = lambda k: sum(int(float(r[f"p1_{k}"] or 0)) + int(float(r[f"p2_{k}"] or 0)) for r in g)
    fists, cd = tot("fists"), tot("cooldown")
    shots, dry = tot("shots"), tot("dry_fires")
    na, du, se = tot("ignored_not_aiming"), tot("ignored_ducked"), tot("ignored_aim_settling")
    counted = fists - cd
    print(
        f"{len(g)} rounds ({sum(float(r['duration_s']) for r in g):.0f} s of duel). Fists {fists}: fired {shots + dry} "
        f"({shots} shots, {dry} empty-gun clicks), cooldown {cd}, not aiming {na}, ducked {du}, aim settling {se}. "
        f"Fired share of non-cooldown fists: {(shots + dry) / counted:.0%} ({shots + dry} / {counted}). "
        f"Bullet hits {tot('bullet_hits')} (headshots {tot('headshots')}), blocked {tot('blocked')}, ducks {tot('ducks')}, "
        f"reloads {tot('reloads')}, powerups {tot('powerups')}.\n"
    )
    print(
        "| Round | clip time | result | P1 fists / fired % | P2 fists / fired % | P1 hand tracked | P2 hand tracked |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in g:
        print(
            f"| {r['session_round']} | {r['ended_at']} | {r['result']} ({r['duration_s']} s) | {r['p1_fists']} / {r['p1_fists_fired_pct']}% | "
            f"{r['p2_fists']} / {r['p2_fists_fired_pct']}% | {r['p1_hand_tracked_pct']}% | {r['p2_hand_tracked_pct']}% |"
        )
    print()
sim = 0
both = 0
for s in stems:
    p = f"results/{s}_signals.csv"
    if not os.path.exists(p):
        continue
    shots = {1: [], 2: []}
    for r in csv.DictReader(open(p)):
        if "SHOT" in (r["event"] or ""):
            shots[int(r["player"])].append(float(r["t"]))
    for t in shots[1]:
        if any(abs(t - u) <= 0.3 for u in shots[2]):
            sim += 1
print(f"Simultaneous shots (both players fired within 0.3 s, both registered): {sim}.\n")

print("## T5 Integration\n")
print("Real-time replay (paced like the live camera, headless), and maximum throughput (unpaced):\n")
print(
    "| Clip | real-time FPS mean / p5 / p1 | latency mean / p95 / p99 ms | dropped per frame | longest frame | CPU | memory mean / peak | max throughput fps | pose / face / face ID / hand ms |"
)
print("|---|---|---|---|---|---|---|---|---|")
for s in stems:
    p, f, q = PACED.get(s), FAST.get(s), PS.get(s)
    if not (p or f):
        continue
    a = (
        f"{p['fps']:.1f} / {p['fps_p5']:.1f} / {p['fps_p1']:.1f} | {p['lat']:.1f} / {p['lat95']:.1f} / {p['lat99']:.1f} | {p['dropped']:.2f} | {p['worst_ms']:.0f} ms"
        if p
        else "- | - | - | -"
    )
    b = f"{q['cpu']:.0f}% | {q['mem']:.0f} / {q['mem_peak']:.0f} MB" if q else "- | -"
    c = f"{f['fps']:.1f}" if f else "-"
    m = p or f
    print(
        f"| {short(s)} | {a} | {b} | {c} | {m['pose']:.1f} / {m['face']:.1f} / {m['face_id']:.1f} / {m['hand']:.1f} |"
    )
print(
    f"| **all (frame-weighted)** | {f1(wavg('fps', PACED))} | {f1(wavg('lat', PACED))} / {f1(wavg('lat95', PACED))} | {wavg('dropped', PACED):.2f} | | "
    f"{np.mean([q['cpu'] for q in PS.values() if q]):.0f}% | {np.mean([q['mem'] for q in PS.values() if q]):.0f} / {max([q['mem_peak'] for q in PS.values() if q], default=0):.0f} MB | {f1(wavg('fps', FAST))} | |\n"
)

print("## Lighting (current code)\n")
print(
    "| Clip | condition | both ids visible | face found | upper-body keypoints | hand with wrist raised | wrong-face checks | own-face score |"
)
print("|---|---|---|---|---|---|---|---|")
for s in ("session_20261101_1106", "session_20261101_1053"):
    for tag, label in (
        ("g1.0", "normal"),
        ("g0.5_n0", "50%"),
        ("g0.3_n0", "30%"),
        ("g0.3_n6", "30% + noise"),
        ("g0.15_n0", "15%"),
        ("g0.15_n6", "15% + noise"),
    ):
        x = load_exp(f"{E}/{s}_{tag}.json")
        d = detection(x) if x else None
        if d:
            print(
                f"| {short(s)} | {label} | {pct(d['both_vis'])} | {pct(d['face'])} | {pct(d['kp'])} | {pct(d['hand'])} | {d['wrong']} / {d['checks']} | {d['own']:.2f} |"
            )

# ---------------------------------------------------------------- impact factors
print("\n## Impact factor: repetition (spread across sessions)\n")
print(
    "Mean ± standard deviation over the sessions (each session one repetition; sessions shorter than 20 s left out).\n"
)
print("| Metric | mean ± SD | min - max | sessions |")
print("|---|---|---|---|")
long_enough = [s for s in stems if EVAL.get(s) and EVAL[s]["dur"] >= 20 and s not in NOT_TWO_PLAYER]


def rep(label, vals, fmt):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) >= 2:
        print(
            f"| {label} | {fmt(np.mean(vals))} ± {fmt(np.std(vals, ddof=1))} | {fmt(min(vals))} - {fmt(max(vals))} | {len(vals)} |"
        )


P = lambda v: f"{100 * v:.1f}%"
F = lambda v: f"{v:.1f}"
rep("T1 face found per visible player", [DET[s]["face"] for s in long_enough if s in DET], P)
rep("T1 both faces when both visible", [DET[s]["both_faces"] for s in long_enough if s in DET], P)
rep("T2 upper-body keypoints confident", [DET[s]["kp"] for s in long_enough if s in DET], P)
rep("T2 hand found with wrist raised", [DET[s]["hand"] for s in long_enough if s in DET], P)
rep(
    "T3 face-reference agreement",
    [
        sum(n * a for _, n, a in EVAL[s]["agree"]) / 100 / sum(n for _, n, _ in EVAL[s]["agree"])
        for s in long_enough
        if EVAL[s]["agree"]
    ],
    P,
)
rep("T5 real-time FPS", [PACED[s]["fps"] for s in long_enough if PACED.get(s)], F)
rep("T5 latency mean (ms)", [PACED[s]["lat"] for s in long_enough if PACED.get(s)], F)
rep("T5 max throughput (fps)", [FAST[s]["fps"] for s in long_enough if FAST.get(s)], F)

print("\n## Impact factor: distance from the camera\n")
print(
    "Distance from the torso length (0.5 m torso, 70° field of view, 1280 px wide); all sessions, normal light.\n"
)
FOCAL = 640 / np.tan(np.radians(35))
edges = [0, 1.25, 1.75, 2.25, 99]
names = ["< 1.25 m", "1.25-1.75 m", "1.75-2.25 m", "> 2.25 m"]
acc = {i: {"n": 0, "face": 0, "kp": 0.0, "raised": 0, "hand": 0, "chk": 0, "wrong": 0} for i in range(4)}
for s in stems:
    x = load_exp(f"{E}/{s}_g1.0.json")
    if not x:
        continue
    for r in x["rows"]:
        if not (r["vis"] and r.get("torso")):
            continue
        z = FOCAL * 0.5 / r["torso"]
        i = int(np.searchsorted(edges, z, side="right") - 1)
        a = acc[min(i, 3)]
        a["n"] += 1
        a["face"] += r["face"]
        a["kp"] += r["kp_ok"]
        if r.get("wr_ok"):
            a["raised"] += 1
            a["hand"] += r["hand"]
        if r.get("sim") is not None:
            a["chk"] += 1
            a["wrong"] += r["sim"] < 0.2
print(
    "| Distance | player-frames | face found | upper-body keypoints | hand with wrist raised | wrong-face checks |"
)
print("|---|---|---|---|---|---|")
for i, a in acc.items():
    if a["n"]:
        print(
            f"| {names[i]} | {a['n']} | {a['face'] / a['n']:.1%} | {a['kp'] / a['n']:.1%} | "
            f"{(a['hand'] / a['raised']) if a['raised'] else float('nan'):.1%} | {a['wrong']} / {a['chk']} |"
        )
