import csv
import time

from highnoon.core.profiler import Profiler


def test_stage_timing_and_csv(tmp_path):
    path = tmp_path / "m.csv"
    prof = Profiler(csv_path=path)
    for i in range(5):
        t_capture = time.perf_counter()
        with prof.stage("work"):
            time.sleep(0.002)
        prof.add({"pose": 10.0})
        prof.end_frame(i + 1, t_capture)
    prof.close()

    stats = prof.stats()
    assert stats["work"][0] >= 2.0
    assert stats["pose"] == (10.0, 10.0)
    assert stats["latency"][0] >= stats["work"][0]

    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 5
    assert set(rows[0]) >= {"frame_id", "fps", "work", "pose", "latency"}
