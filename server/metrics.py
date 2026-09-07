# metrics.py
import csv
import threading

import numpy as np
import matplotlib.pyplot as plt

# IMPORTANT: these are SERVER-SIDE numbers, for DECOMPOSITION only (queue-wait vs
# inference-time vs batch-size). They are NOT your headline latency — that comes
# from the CO-correct loadgen CSV. Don't report these as the project's p99.


class MetricsCollector:
    def __init__(self):
        self._rows = []                 # (arrival, batch_start, done, batch_size)
        self._rejections = []           # arrival_ts only
        self._lock = threading.Lock()   # worker appends; snapshot/dump read
        self._inference_time = []
        self.batch_start_time = []
        self.qsize_at_infer = []
        self.qsizes_bef_aft = []
        self.admit_check_latency = []

    def record_rejection(self, arrival_ts):
        # Called from the EVENT LOOP thread (submit), not the worker. Same lock:
        # contention is negligible and correctness matters more.
        with self._lock:
            self._rejections.append(arrival_ts)

    def record(self, arrival_ts, batch_start_ts, done_ts, batch_size):
        with self._lock:
            self._rows.append((arrival_ts, batch_start_ts, done_ts, batch_size))

    def _derive(self):
        with self._lock:
            rows = list(self._rows)
            rejections = list(self._rejections)
        if not rows and not rejections:
            return None

        # Time origin must span BOTH lists, or rejected rows get negative t_rel.
        arrivals = [r[0] for r in rows] + rejections
        t0 = min(arrivals)

        out = []
        for arrival, batch_start, done, bs in rows:
            out.append({
                "t_rel_s":       arrival - t0,
                "outcome":       "served",
                "queue_wait_ms": (batch_start - arrival) * 1000.0,
                "inference_ms":  (done - batch_start) * 1000.0,
                "total_ms":      (done - arrival) * 1000.0,
                "batch_size":    bs,
            })
        for arrival in rejections:
            out.append({
                "t_rel_s":       arrival - t0,
                "outcome":       "rejected",
                "queue_wait_ms": "",     # empty -> NaN in pandas, not 0
                "inference_ms":  "",
                "total_ms":      "",
                "batch_size":    "",
            })
        out.sort(key=lambda r: r["t_rel_s"])
        return out

    def snapshot(self):
        rows = self._derive()
        if not rows:
            return {"count": 0}
        served = [r for r in rows if r["outcome"] == "served"]
        n_rej = len(rows) - len(served)
        if not served:
            return {"count": 0, "rejected": n_rej}

        def pct(key):
            vals = np.array([r[key] for r in served], dtype=float)
            return {"p50": float(np.percentile(vals, 50)),
                    "p99": float(np.percentile(vals, 99)),
		    "p99.9": float(np.percentile(vals, 99.9))}
        return {
            "count":          len(served),
            "queue_wait_ms":  pct("queue_wait_ms"),
            "inference_ms":   pct("inference_ms"),
            "total_ms":       pct("total_ms"),
            "avg_batch_size": float(np.mean([r["batch_size"] for r in served])),
            "rejected":       n_rej,
            "rejection_rate": n_rej / len(rows),
        }

    def dump_csv(self, path):
        rows = self._derive()
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def plot_inference_latencies(self):
        if self._inference_time:
            self._inference_time.sort()
            n = len(self._inference_time)
            print(f""""Inference Latency: \n
                p50: {self._inference_time[(n-1) // 2]}\n
                p99: {self._inference_time[n * 99//100]}""")
            plt.plot(self._inference_time)
            plt.savefig('ínference_latencies_2.png')

        if self.batch_start_time:
            batch_latency = []
            for i in range(1, len(self.batch_start_time)):
                batch_latency.append((self.batch_start_time[i] - self.batch_start_time[i - 1]) * 1000)
            n = len(batch_latency)
            batch_latency.sort()
            print(f"""Batch latencies:
                p50:    {batch_latency[(n - 1) // 2]} ms
                p99:    {batch_latency[n * 99 // 100]} ms
                p99.9:  {batch_latency[n * 999//1000]} ms
                min: {batch_latency[0]} ms
                max: {batch_latency[-1]} ms
            """)

            q = sorted(self.qsize_at_infer)
            n = len(q)
            print(f"""Queue sizes:
                p50:    {q[(n - 1) // 2]} ms
                p99:    {q[n * 99 // 100]} ms
                p99.9:  {q[n * 999//1000]} ms
                min: {q[0]} ms
                max: {q[-1]} ms
            """)
            plt.clf()
            # self.qsizes_bef_aft.sort()
            qsize_bef = [x[0] for x in self.qsizes_bef_aft]
            qsize_aft = [x[1] for x in self.qsizes_bef_aft]
            plt.plot(qsize_bef)
            plt.plot(qsize_aft)
            plt.savefig('qsizes.png')
            print('Qsizes:')
            print(self.qsizes_bef_aft[:10])

            print("Admit check latency:")
            arr = self.admit_check_latency
            plt.clf()
            plt.plot(arr)
            plt.savefig('admit_check_latency.png')
            arr.sort()
            plt.clf()
            plt.plot(arr)
            plt.savefig('admit_check_latency_sorted.png')
            n = len(arr)
            print(f"""
            p50: {arr[(n+1)//2] * 1000}
            p99: {arr[n*99//100] * 1000}
            """)

