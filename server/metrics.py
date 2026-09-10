# metrics.py
import csv
import threading

import numpy as np


def _grow(a, new_cap):
    b = np.empty(new_cap, dtype=a.dtype)
    b[:a.size] = a
    return b


class MetricsCollector:

    def __init__(self, capacity=200_000):
        self._lock = threading.Lock()

        self._n, self._cap = 0, capacity
        self._arrival     = np.empty(capacity, dtype=np.float64)
        self._body        = np.empty(capacity, dtype=np.float64)
        self._batch_start = np.empty(capacity, dtype=np.float64)
        self._done        = np.empty(capacity, dtype=np.float64)
        self._batch_size  = np.empty(capacity, dtype=np.int32)
        self._failed      = np.zeros(capacity, dtype=bool)

        self._n_rej, self._rej_cap = 0, capacity
        self._rej_arrival = np.empty(capacity, dtype=np.float64)

        # one entry per BATCH
        self._n_batch, self._batch_cap = 0, capacity
        self._b_start = np.empty(capacity, dtype=np.float64)
        self._b_infer = np.empty(capacity, dtype=np.float64)
        self._b_size  = np.empty(capacity, dtype=np.int32)

    # ---------- writers ----------

    def record_batch(self, arrival_ts, body_ts, batch_start_ts, done_ts,
                     batch_size, failed=False):
        """One call per batch, from the WORKER thread.

        arrival_ts / body_ts are sequences, one entry per request in the batch.
        """
        k = len(arrival_ts)
        with self._lock:
            if self._n + k > self._cap:
                self._cap = max(self._cap * 2, self._n + k)
                for name in ("_arrival", "_body", "_batch_start",
                             "_done", "_batch_size", "_failed"):
                    setattr(self, name, _grow(getattr(self, name), self._cap))

            s = slice(self._n, self._n + k)
            self._arrival[s]     = arrival_ts
            self._body[s]        = body_ts
            self._batch_start[s] = batch_start_ts
            self._done[s]        = done_ts
            self._batch_size[s]  = batch_size
            self._failed[s]      = failed
            self._n += k

            if self._n_batch >= self._batch_cap:
                self._batch_cap *= 2
                for name in ("_b_start", "_b_infer", "_b_size"):
                    setattr(self, name, _grow(getattr(self, name), self._batch_cap))
            i = self._n_batch
            self._b_start[i] = batch_start_ts
            self._b_infer[i] = (done_ts - batch_start_ts) * 1000.0
            self._b_size[i]  = batch_size
            self._n_batch += 1

    def record_rejection(self, arrival_ts):
        with self._lock:
            if self._n_rej >= self._rej_cap:
                self._rej_cap *= 2
                self._rej_arrival = _grow(self._rej_arrival, self._rej_cap)
            self._rej_arrival[self._n_rej] = arrival_ts
            self._n_rej += 1


    def _views(self):
        with self._lock:
            n, r = self._n, self._n_rej
            return (self._arrival[:n].copy(), self._body[:n].copy(),
                    self._batch_start[:n].copy(), self._done[:n].copy(),
                    self._batch_size[:n].copy(), self._failed[:n].copy(),
                    self._rej_arrival[:r].copy())

    def snapshot(self):
        arrival, body, bstart, done, bsize, failed, rej = self._views()
        if arrival.size == 0 and rej.size == 0:
            return {"count": 0}

        ok = ~failed
        n_served, n_rej, n_fail = int(ok.sum()), int(rej.size), int(failed.sum())
        if n_served == 0:
            return {"count": 0, "rejected": n_rej, "failed": n_fail}

        def pct(v):
            return {"p50":   float(np.percentile(v, 50)),
                    "p99":   float(np.percentile(v, 99)),
                    "p99.9": float(np.percentile(v, 99.9))}

        return {
            "count":              n_served,
            "deserialization_ms": pct((body[ok]   - arrival[ok]) * 1000.0),
            "queue_wait_ms":      pct((bstart[ok] - body[ok])    * 1000.0),
            "inference_ms":       pct((done[ok]   - bstart[ok])  * 1000.0),
            "total_ms":           pct((done[ok]   - arrival[ok]) * 1000.0),
            "avg_batch_size":     float(bsize[ok].mean()),
            "failed":             n_fail,
            "rejected":           n_rej,
            "rejection_rate":     n_rej / (n_served + n_fail + n_rej),
        }

    def batch_period_stats(self):

        with self._lock:
            m = self._n_batch
            starts = self._b_start[:m].copy()
            infer  = self._b_infer[:m].copy()
            sizes  = self._b_size[:m].copy()
        if m < 2:
            return None

        period = np.diff(starts) * 1000.0
        return {
            "batches": int(m),
            "period_ms": {"p50": float(np.percentile(period, 50)),
                          "p99": float(np.percentile(period, 99)),
                          "mean": float(period.mean())},
            "infer_ms":  {"p50": float(np.percentile(infer, 50)),
                          "p99": float(np.percentile(infer, 99)),
                          "mean": float(infer.mean())},
            "batch_size_mean":  float(sizes.mean()),
            "throughput_check": float(sizes.mean() / (period.mean() / 1000.0)),
        }

    def dump_csv(self, path):
        arrival, body, bstart, done, bsize, failed, rej = self._views()
        if arrival.size == 0 and rej.size == 0:
            return

        t0 = min(arrival.min() if arrival.size else np.inf,
                 rej.min()     if rej.size     else np.inf)

        FIELDS = ["t_rel_s", "outcome", "deserialization_ms", "queue_wait_ms",
                  "inference_ms", "total_ms", "batch_size"]

        rows = [(arrival[i] - t0,
                 "failed" if failed[i] else "served",
                 (body[i]   - arrival[i]) * 1000.0,
                 (bstart[i] - body[i])    * 1000.0,
                 (done[i]   - bstart[i])  * 1000.0,
                 (done[i]   - arrival[i]) * 1000.0,
                 int(bsize[i]))
                for i in range(arrival.size)]
        rows += [(rej[i] - t0, "rejected", "", "", "", "", "")
                 for i in range(rej.size)]
        rows.sort(key=lambda r: r[0])

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerows(rows)