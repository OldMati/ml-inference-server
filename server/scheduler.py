# scheduler.py
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
# NOTE: we annotate asyncio types as strings below so this file doesn't need
# to import asyncio — the worker thread never touches asyncio directly, it only
# calls loop.call_soon_threadsafe on the loop object handed to it.

# Poison pill: a unique sentinel pushed onto the queue at shutdown to wake the
# worker out of a blocking get(). Identity (is) is all that matters.
_SHUTDOWN = object()


@dataclass
class Request:
    input: np.ndarray                 # ONE 3x224x224 float32 tensor, no batch dim
    arrival_ts: float                 # time.monotonic(), stamped in the handler
    future: "asyncio.Future"          # created via loop.create_future() in handler
    loop: "asyncio.AbstractEventLoop" # the loop that owns `future`


class NaiveScheduler:
    def __init__(self, backend, max_batch_size, max_wait_s, metrics=None):
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_s
        self.metrics = metrics
        self.queue: "queue.Queue" = queue.Queue()   # unbounded ON PURPOSE:
        self._thread = None                          # accept-until-collapse IS
                                                     # the naive behavior.

    def start(self):
        self._thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.queue.put(_SHUTDOWN)     # wakes the worker if it's blocked in get()
        if self._thread is not None:
            self._thread.join()

    def submit(self, req: Request):
        # Called from the event-loop thread. Queue is thread-safe, so no lock.
        # The ABSENCE of a capacity check here is the naive policy. This is one
        # of the two methods that changes when you add admission control (Wk 5).
        self.queue.put(req)

    def _collect_batch(self):
        # Block for the FIRST request — no spinning while idle.
        first = self.queue.get()
        if first is _SHUTDOWN:
            return _SHUTDOWN

        batch = [first]
        # Deadline is anchored to the first request and NEVER reset as more
        # arrive. Resetting it would let a steady trickle extend the batch and
        # make the first request wait far longer than max_wait_s.
        deadline = time.monotonic() + self.max_wait_s

        # Check length BEFORE each get so we never overfill by one.
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

            # Single boundary stamp for the whole batch (see metrics reasoning).
            batch_start_ts = time.monotonic()

            try:
                batched = np.stack([r.input for r in batch])   # order preserved
                outputs = self.backend.infer(batched)
            except Exception as e:
                # Fulfill EVERY future with the error, then keep the thread
                # alive. Skipping this makes those requests await forever and
                # vanish from the latency distribution — invisible, worse than a
                # crash for a measurement project.
                for r in batch:
                    r.loop.call_soon_threadsafe(r.future.set_exception, e)
                continue

            done_ts = time.monotonic()

            # Scatter results. output row i belongs to batch[i] — the stack
            # preserved order, so never reorder `batch`. Future is NOT
            # thread-safe, so we go back through the loop, not set_result direct.
            for i, r in enumerate(batch):
                r.loop.call_soon_threadsafe(r.future.set_result, outputs[i])
                if self.metrics is not None:
                    self.metrics.record(
                        arrival_ts=r.arrival_ts,
                        batch_start_ts=batch_start_ts,
                        done_ts=done_ts,
                        batch_size=len(batch),
                    )