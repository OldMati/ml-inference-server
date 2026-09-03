# Function glossary — `api.py` + `scheduler.py`

Every function/method from outside your own code, grouped by package, with
what it does and exactly where/why it shows up in these two files.

Two genuinely different kinds of package here: **fastapi** and **numpy** are
third-party (`pip install`); **asyncio**, **queue**, **threading**, **time**,
**dataclasses**, **contextlib**, **os**, **pathlib** are all standard library
— no install needed, always available. Worth knowing which is which, since
the stdlib ones are stable API surface you'll reuse in every future Python
project, not something specific to this server.

---

## fastapi (third-party)

### `FastAPI(lifespan=...)`
Constructs the application object. `lifespan` takes an async context manager
(see `@asynccontextmanager` below) that FastAPI calls automatically: enters
it once on server startup, exits it once on shutdown. This is the modern
replacement for the older `@app.on_event("startup")` decorators.

**Here:** `app = FastAPI(lifespan=lifespan)` — wires your `lifespan()`
function in as the startup/shutdown hook that builds the backend, scheduler,
and metrics collector.

### `@app.post(path)` / `@app.get(path)`
Route decorators. Registers the decorated function as the handler for
`METHOD path`, and tells FastAPI to introspect its signature (type hints,
parameter names) to auto-generate request validation and the OpenAPI schema.

**Here:** `@app.post("/predict")` and `@app.get("/healthz")`.

### `Request` (from `fastapi`, re-exported from Starlette)
Represents the incoming HTTP request. Gives you access to headers, the raw
body, query params, and — critically for this codebase — `request.app`,
which is how a route handler reaches back to objects stashed on the
application instance.

- **`await request.body()`** — asynchronously reads the full raw request
  body as `bytes`. The `await` means the coroutine suspends while the
  socket delivers data, instead of blocking the thread.
- **`request.app`** — reference back to the owning `FastAPI` instance, used
  here as `request.app.state.scheduler` to reach the scheduler from inside
  the handler without it being a global.

**Note:** this is a *different* `Request` from the `Request` dataclass in
`scheduler.py` — see the aliasing note in `api.py`'s imports. Same name,
unrelated types.

### `HTTPException(status_code, detail)`
Raising this inside a route handler short-circuits normal execution and
makes FastAPI return an HTTP error response with that status code and a
JSON body containing `detail`. It's exception-based control flow — you
`raise` it like any Python exception, but FastAPI intercepts it specifically
(rather than it propagating as an unhandled 500) and turns it into a
well-formed response.

**Here:** used for a `400` on a malformed payload size, and a `500` if
inference itself raises (e.g. a CUDA OOM under overload).

### `app.state`
A plain namespace object FastAPI attaches to every app instance for
stashing arbitrary objects across the app's lifetime. Nothing magic — it's
just attribute storage (`app.state.scheduler = x` then later
`app.state.scheduler`), used because the scheduler and metrics collector are
built once in `lifespan()` but need to be reachable from every later request
handler call.

---

## numpy (third-party)

### `np.frombuffer(buffer, dtype)`
Constructs an `ndarray` that's a *view* over existing bytes — no copy. The
array shares memory with the original `bytes`/`bytearray` object, which is
why it comes back **read-only** (you can't safely let two objects think they
own the same mutable memory).

**Here:** `np.frombuffer(raw, dtype="<f4")` interprets the raw POST body as
little-endian 32-bit floats. `dtype="<f4"` is exact and load-bearing — `<`
is byte order, `f4` is 4-byte float; get this wrong (or mismatched with
whatever the C++ load generator writes) and you get silently wrong numbers,
not an error.

### `ndarray.reshape(shape)`
Returns a new view over the same underlying data, reinterpreted with a
different shape, as long as the total element count matches. No data is
moved or copied — it's just a different stride/shape descriptor over the
same bytes.

**Here:** `.reshape(3, 224, 224)` gives the flat float buffer CHW structure
(channels, height, width), matching what the ONNX model expects per image.

### `np.stack(arrays)`
Takes a list of arrays with identical shape and joins them along a **new**
leading axis, producing one array with shape `(N, *original_shape)`. Unlike
`reshape`, this genuinely copies data — it has to, since it's assembling
scattered arrays (each backed by a different request's `bytes` object) into
one contiguous block.

**Here:** `np.stack([r.input for r in batch])` turns a Python list of N
separate `(3,224,224)` arrays into one `(N,3,224,224)` batch, in the same
order as `batch` — order preservation here is exactly why the code scatters
results back with `enumerate(batch)` afterward rather than any other
matching scheme.

### `np.argmax(array)`
Returns the index of the largest element. With no `axis` argument on a 1-D
array, it's just "which class had the highest logit."

**Here:** `np.argmax(result)` turns a raw output vector for one image into
a single predicted class ID.

---

## asyncio (standard library)

### `asyncio.get_running_loop()`
Returns a reference to the event loop that is *currently* executing the
calling code. Only valid from inside a coroutine actually being driven by a
loop — call it from a plain thread with no loop of its own and it raises
`RuntimeError`.

**Here:** captured once per request in `predict()`, then carried inside the
`Request` dataclass so the scheduler's worker thread — which has no loop of
its own — has a way to reach back into the correct loop later.

### `loop.create_future()`
Creates a new `asyncio.Future`: a placeholder object representing a result
that doesn't exist yet. Must be created via a loop method (rather than
`asyncio.Future()` directly) so it's correctly bound to that loop.

**Here:** one `Future` per request, created fresh in `predict()`, later
resolved from the worker thread once that request's batch finishes.

### `Future` — `await`, `set_result(value)`, `set_exception(exc)`
A `Future` starts pending. `await future` suspends the calling coroutine
until it's resolved. `future.set_result(value)` resolves it successfully —
every coroutine awaiting it wakes up and `await future` evaluates to
`value`. `future.set_exception(exc)` resolves it with an error instead —
`await future` re-raises `exc` at the await point.

**Important constraint:** none of this is thread-safe. `Future` assumes only
its owning loop's thread ever touches it directly.

**Here:** `predict()` does `result = await future`, suspending until the
scheduler's worker thread (on a different OS thread entirely) resolves it —
via `call_soon_threadsafe`, never directly, exactly because of that
thread-safety constraint.

### `loop.call_soon_threadsafe(callback, *args)`
The one sanctioned way to schedule a callback onto a loop **from a different
thread**. Internally it appends to the loop's ready queue under a lock, and
wakes the loop out of its blocking wait (it's parked in an OS-level
`epoll`/`select` call) so the callback actually runs promptly instead of
waiting for some unrelated socket event.

**Here:** the worker thread's only two ways of touching a request's
`Future` — `r.loop.call_soon_threadsafe(r.future.set_result, outputs[i])`
on success, or `r.loop.call_soon_threadsafe(r.future.set_exception, e)` on
failure. Both route the actual `set_result`/`set_exception` call through the
owning loop's thread instead of calling it directly from the worker thread.

### `asyncio.AbstractEventLoop` (type)
The base class/interface for event loop objects. Used here purely as a type
annotation (`loop: "asyncio.AbstractEventLoop"`) — not instantiated
directly, since `asyncio.get_running_loop()` hands you a concrete
implementation already.

**Note on the quoting:** in `scheduler.py`, both `"asyncio.Future"` and
`"asyncio.AbstractEventLoop"` are written as string literals rather than
real type annotations. Because they're strings, Python never actually
evaluates them at class-definition time — which is exactly why this file
gets away with never importing `asyncio` at all, per its own top comment.
This is a manual version of what `from __future__ import annotations` does
automatically for every annotation in a file.

---

## contextlib (standard library)

### `@asynccontextmanager`
Decorator that turns a generator function containing exactly one `yield`
into an async context manager. Code before `yield` runs on `__aenter__`
(entry); code after `yield` runs on `__aexit__` (exit) — including on
exceptions, similar to `try`/`finally`.

**Here:** decorates `lifespan()`. Startup code (build backend, start
scheduler) runs before `yield`; shutdown code (stop scheduler, dump
metrics) runs after. FastAPI drives this itself — you never call it by hand.

---

## queue (standard library)

### `queue.Queue()`
A thread-safe FIFO queue, built on a `deque` internally with a `Lock` plus
condition variables for blocking `get`/`put`. Safe to have multiple threads
`put()`/`get()` on concurrently with no extra locking on your part — this is
precisely why it's the hand-off point between the event loop thread and the
worker thread here, instead of a plain list.

**Here:** `self.queue = queue.Queue()` — deliberately **unbounded**. The
code comment is explicit that this is the naive policy: no backpressure, no
capacity check, accept-until-collapse. A bounded queue (or a capacity check
in `submit()`) is one of the two things admission control changes later.

### `Queue.put(item)`
Pushes an item onto the queue and wakes up one thread blocked in `get()`, if
any. Never blocks itself (queue is unbounded here, so there's no "queue
full" backpressure case to worry about).

**Here:** `submit()` calls `self.queue.put(req)` from the event loop thread.
`stop()` calls `self.queue.put(_SHUTDOWN)` to wake a worker that might be
blocked waiting for the next request.

### `Queue.get()` / `Queue.get(timeout=remaining)`
Pops the oldest item. With no arguments, blocks indefinitely until
something is available — used for the *first* item of a batch, since
there's no point spinning while genuinely idle. With `timeout=`, blocks at
most that many seconds, then raises `queue.Empty` if nothing arrived —
used for every item *after* the first, so a batch ships once its deadline
passes even with an empty queue.

**Here:** `self.queue.get()` (no timeout) blocks for the first request of a
batch; `self.queue.get(timeout=remaining)` inside the collection loop
enforces the `max_wait_s` deadline for subsequent requests in the same
batch.

### `queue.Empty`
The exception `Queue.get(timeout=...)` raises when the timeout expires
before an item arrives. Caught explicitly here — it's the *expected*,
normal way a batch ships early due to low traffic, not an error condition.

---

## threading (standard library)

### `threading.Thread(target=fn, daemon=True)`
Constructs (but does not yet start) a new OS thread that will run `fn` when
started. `daemon=True` means this thread won't keep the process alive on
its own — if the main thread exits, daemon threads are killed automatically
rather than blocking process exit.

**Here:** `threading.Thread(target=self._batch_loop, daemon=True)` — the
scheduler's background worker. `daemon=True` is a safety net (so a bug that
skips clean shutdown doesn't hang the whole process), not a substitute for
`stop()`'s proper `join()`-based shutdown.

### `Thread.start()`
Actually spawns the OS thread and begins executing `target` concurrently
with the calling thread. Returns immediately — does not wait for `target`
to finish.

**Here:** `self._thread.start()` in `NaiveScheduler.start()`, kicking off
`_batch_loop()` on its own thread while the main process goes on to run
uvicorn's event loop.

### `Thread.join()`
Blocks the calling thread until the target thread has actually finished
running. Used for clean shutdown — you want to be sure the worker has
genuinely exited before the process moves on.

**Here:** `self._thread.join()` in `stop()`, called after pushing
`_SHUTDOWN` onto the queue — waits for `_batch_loop` to see the sentinel and
return before considering shutdown complete.

---

## time (standard library)

### `time.monotonic()`
Returns a float number of seconds from some fixed, arbitrary starting point
— **not** wall-clock time, and guaranteed never to jump backward (unlike
`time.time()`, which can shift if the system clock is adjusted, e.g. via
NTP sync). The absolute value is meaningless; only differences between two
calls are meaningful.

**Here:** every timestamp in both files — `arrival_ts`, `batch_start_ts`,
`done_ts`, and the batch-collection `deadline` — uses this, specifically
because your metrics pipeline needs elapsed-time differences to be
trustworthy even if something adjusts the system clock mid-experiment.

---

## dataclasses (standard library)

### `@dataclass`
Class decorator that auto-generates `__init__`, `__repr__`, and `__eq__`
from the class body's type-annotated fields, so you don't hand-write
boilerplate like `def __init__(self, input, arrival_ts, future, loop): self.input = input; ...`.

**Here:** decorates `Request` in `scheduler.py`. Its four fields
(`input`, `arrival_ts`, `future`, `loop`) become constructor parameters
automatically — that's what makes
`InferenceRequest(input=arr, arrival_ts=arrival_ts, future=future, loop=loop)`
in `api.py` work with no `__init__` ever written by hand.

---

## os (standard library)

### `os.environ.get(key, default)`
Reads an environment variable by name, returning `default` if it isn't set
— never raises, unlike `os.environ[key]`.

**Here:** the pattern behind every piece of runtime config —
`MAX_BATCH_SIZE`, `MAX_WAIT_S`, `MODEL_PATH`, `METRICS_CSV` — all have a
hardcoded sane default but can be overridden per-run without touching code,
which matters once you're scripting sweeps across many configurations.

---

## pathlib (standard library)

### `Path(__file__)`
`__file__` is a string: this script's own path. Wrapping it in `Path` gives
you an object with path-manipulation methods instead of raw string
handling (`os.path.join` etc.).

### `Path.resolve()`
Converts a path to an absolute, symlink-resolved form. Turns a possibly
relative `__file__` into something unambiguous regardless of the current
working directory the process was launched from.

### `Path.parent`
The containing directory of a path. Chaining it (`.parent.parent`) walks up
multiple directory levels.

**Here:** `Path(__file__).resolve().parent.parent` — from
`.../inference-server/server/api.py`, walks up to `.../inference-server/`,
giving a stable default `MODEL_PATH` regardless of where you `python -m
uvicorn` from.
