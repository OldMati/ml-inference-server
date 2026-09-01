# metrics.py
import csv
import threading

import numpy as np

# IMPORTANT: these are SERVER-SIDE numbers, for DECOMPOSITION only (queue-wait vs
# inference-time vs batch-size). They are NOT your headline latency — that comes
# from the CO-correct loadgen CSV. Don't report these as the project's p99.


class MetricsCollector:
    def __init__(self):
        self._rows = []                 # (arrival, batch_start, done, batch_size)
        self._lock = threading.Lock()   # worker appends; snapshot/dump read

    def record(self, arrival_ts, batch_start_ts, done_ts, batch_size):
        with self._lock:
            self._rows.append((arrival_ts, batch_start_ts, done_ts, batch_size))

    def _derive(self):
        # Returns columns in ms. Row order is completion order, but each row
        # carries its own arrival_ts so t_rel is correct regardless of ordering.
        with self._lock:
            rows = list(self._rows)     # copy under lock, compute outside
        if not rows:
            return None
        t0 = min(r[0] for r in rows)    # first arrival = time origin
        out = []
        for arrival, batch_start, done, bs in rows:
            out.append({
                "t_rel_s":       arrival - t0,
                "queue_wait_ms": (batch_start - arrival) * 1000.0,
                "inference_ms":  (done - batch_start) * 1000.0,
                "total_ms":      (done - arrival) * 1000.0,
                "batch_size":    bs,
            })
        return out

    def snapshot(self):
        rows = self._derive()
        if not rows:
            return {"count": 0}

        def pct(key):
            vals = np.array([r[key] for r in rows])
            return {"p50": float(np.percentile(vals, 50)),
                    "p99": float(np.percentile(vals, 99))}

        return {
            "count":          len(rows),
            "queue_wait_ms":  pct("queue_wait_ms"),
            "inference_ms":   pct("inference_ms"),
            "total_ms":       pct("total_ms"),
            "avg_batch_size": float(np.mean([r["batch_size"] for r in rows])),
        }

    def dump_csv(self, path):
        rows = self._derive()
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)