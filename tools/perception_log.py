"""Per-frame perception log of one clip under one lighting condition (current code).

usage: .venv/bin/python tools/perception_log.py CLIP GAIN NOISE OUT.json
Frames the recorder repeated to keep real time are skipped (they would fake zero jitter).
"""

# ruff: noqa: E501, E731, B023, B905  (report tables printed as long f-strings)

import json
import math
import sys

sys.path.insert(0, ".")
import cv2
import numpy as np

from highnoon.config import Config
from highnoon.core.capture import open_source
from highnoon.tracking.identity import IdentityTracker
from highnoon.vision.head import measure_head
from highnoon.vision.motion import R_WRIST, MotionFilter
from highnoon.vision.perception import Perceiver

clip, gain, noise, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
UPPER = [0, 11, 12, 13, 14, 15, 16]
RNG = np.random.default_rng(0)
cfg = Config()
cfg.capture.source = clip
cfg.capture.realtime = False
src = open_source(cfg.capture)
src.start()
per = Perceiver(cfg.perception)
tr = IdentityTracker(cfg.tracking, 2)
mf = MotionFilter(cfg.filters)
rows, last, k, dups, prev, t0 = [], 0, 0, 0, None, None
while (f := src.read(last)) is not None:
    last = f.id
    small = cv2.resize(f.image, (160, 90), interpolation=cv2.INTER_AREA)
    if prev is not None and np.abs(small.astype(np.int16) - prev).mean() < 0.5:
        dups += 1
        continue
    prev = small.astype(np.int16)
    k += 1
    t = f.timestamp_ms / 1000.0
    t0 = t if t0 is None else t0
    f.t_capture = t
    if gain != 1.0:
        f.image = cv2.convertScaleAbs(f.image, alpha=gain)
    if noise:
        f.image = np.clip(
            f.image.astype(np.int16) + RNG.normal(0, noise, f.image.shape).astype(np.int16), 0, 255
        ).astype(np.uint8)
    p = per(f)
    ps = tr.update(p)
    mf.update(ps, t, tr.reassigned, f.size)
    if t - t0 >= 12:  # ids handed out during the first countdown, then locked
        tr.lock()
    for pl in ps:
        r = {"k": k, "t": t - t0, "id": pl.id, "vis": pl.visible}
        if pl.visible and pl.pose is not None:
            raw = pl.pose.keypoints
            sm = pl.smoothed if pl.smoothed is not None else raw
            r["x"] = float(pl.pose.center()[0])
            r["kp_ok"] = float((pl.pose.visibility[UPPER] >= 0.5).mean())
            r["sh_ok"] = bool((pl.pose.visibility[[11, 12]] >= 0.5).all())
            r["wr_ok"] = bool(pl.pose.visibility[R_WRIST] >= 0.5)
            r["raw_sh"] = ((raw[11] + raw[12]) / 2).tolist()
            r["sm_sh"] = ((sm[11] + sm[12]) / 2).tolist()
            r["raw_wr"] = raw[R_WRIST].tolist()
            r["sm_wr"] = sm[R_WRIST].tolist()
            r["torso"] = pl.signals.torso_len if pl.signals else None
            r["hand"] = bool(pl.hands)
            r["face"] = pl.face is not None
            if pl.face is not None and pl.head is not None:
                rh = measure_head(pl.face)
                if rh is not None:
                    r["raw_hat"] = [*rh.center.tolist(), math.degrees(rh.angle), rh.width]
                    r["sm_hat"] = [*pl.head.center.tolist(), math.degrees(pl.head.angle), pl.head.width]
            r["sim"] = tr.owner_similarity(pl)
        rows.append(r)
per.close()
src.stop()
json.dump({"clip": clip, "gain": gain, "noise": noise, "dups": dups, "rows": rows}, open(out, "w"))
print(f"{clip} gain {gain} noise {noise}: {k} unique frames, {dups} repeated frames skipped")
