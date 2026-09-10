import queue
import threading
import time
from dataclasses import dataclass
from math import ceil

import numpy as np
_SHUTDOWN = object()

@dataclass
class Request:
    input: np.ndarray                 # ONE 3x224x224 float32 tensor, no batch dim
    arrival_ts: float                 # time.monotonic(), stamped in the handler
    body_ts: float                    # time.monotonic(), stamped in the handler
    future: "asyncio.Future"          # created via loop.create_future() in handler
    loop: "asyncio.AbstractEventLoop" # the loop that owns `future`


class NaiveScheduler:
    def __init__(self, backend, max_batch_size, max_wait_s, metrics=None):
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_s
        self.metrics = metrics
        self._buf = np.zeros((max_batch_size, 3, 224, 224), dtype=np.float32)
        self.queue: "queue.Queue" = queue.Queue()
        self._thread = None 

    def start(self):
        self._thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.queue.put(_SHUTDOWN)     # wakes the worker if it's blocked in get()
        if self._thread is not None:
            self._thread.join()

    def submit(self, req: Request):
        self.queue.put(req)

    def _collect_batch(self):
        first = self.queue.get()
        if first is _SHUTDOWN:
            return _SHUTDOWN

        batch = [first]
        
        deadline = time.monotonic() + self.max_wait_s

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break                              # timeout: ship what we have
            try:
                item = self.queue.get(timeout=remaining)
            except queue.Empty:
                break                              # not enough traffic in time —
                                                   # normal, not an error
            if item is _SHUTDOWN:
                # Shutdown landed mid-collect. Put it back for the next
                # _collect_batch to see, and ship the in-flight batch cleanly.
                self.queue.put(_SHUTDOWN)
                break
            batch.append(item)

        return batch


    def _batch_loop(self):
        while True:
            batch = self._collect_batch()
            if batch is _SHUTDOWN:
                break

            batch_start_ts = time.monotonic()

            try:
                n = len(batch)
                for i, r in enumerate(batch):
                    np.copyto(self._buf[i], r.input)
                outputs = self.backend.infer(self._buf)[:n]   # always shape (4,...)
                err = None
            except Exception as e:
                outputs, err = None, e

            done_ts = time.monotonic()

            # Build ONE callback for the whole batch
            if err is None:
                payload = [(r.future, outputs[i]) for i, r in enumerate(batch)]

                def _fulfil(payload=payload):
                    for fut, out in payload:
                        if not fut.done():
                            fut.set_result(out)
            else:
                futures = [r.future for r in batch]

                def _fulfil(futures=futures, err=err):
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(err)

            batch[0].loop.call_soon_threadsafe(_fulfil)

            if self.metrics is not None:
                self.metrics.record_batch(
                    arrival_ts=[r.arrival_ts for r in batch],
                    body_ts=[r.body_ts for r in batch],
                    batch_start_ts=batch_start_ts,
                    done_ts=done_ts,
                    batch_size=len(batch),
                    failed=err is not None,
                )

class DynamicScheduler(NaiveScheduler):
    
    def _collect_batch(self):
        first = self.queue.get()
        if first is _SHUTDOWN:
            return _SHUTDOWN

        batch = [first]

        qsize = self.queue.qsize()

        if qsize == 0:  # if no more requests, return instantly
            return batch
        elif qsize >= self.max_batch_size - 1: # fill the batch instantly
            for _ in range(self.max_batch_size - 1):
                item = self.queue.get()
                if item is _SHUTDOWN:
                    self.queue.put(_SHUTDOWN)
                    break
                batch.append(item)
            return batch

        # dynamically adjusted deadline
        deadline = time.monotonic() + self.max_wait_s * (1 - qsize / self.max_batch_size)

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break                              # timeout: ship what we have
            try:
                item = self.queue.get(timeout=remaining)
            except queue.Empty:
                break                              # not enough traffic in time
            if item is _SHUTDOWN:
                # Shutdown landed mid-collect. Put it back for the next
                # _collect_batch to see, and ship the in-flight batch cleanly.
                self.queue.put(_SHUTDOWN)
                break
            batch.append(item)

        return batch


class AdmissionScheduler(DynamicScheduler):
    def __init__(self, backend, max_batch_size, max_wait_s,
                 metrics=None, slo_ms=50.0, per_batch_ms=7.6):
        super().__init__(backend, max_batch_size, max_wait_s, metrics=metrics)
        self.slo_ms = slo_ms
        self.per_batch_ms = per_batch_ms
        self.rejected = 0

    def should_admit(self, arrival_ts) -> bool:
        qsize = self.queue.qsize()
        projected_ms = ceil(qsize / self.max_batch_size + 1) * self.per_batch_ms + self.per_batch_ms + (time.monotonic() - arrival_ts) * 1000
        if projected_ms > self.slo_ms:
            self.rejected += 1
            self.metrics.record_rejection(arrival_ts=arrival_ts)
            return False

        return True
