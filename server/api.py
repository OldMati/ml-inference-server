# api.py
import asyncio
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from backend import InferenceBackend
from metrics import MetricsCollector
# NAME CLASH TRAP: the scheduler's dataclass is also called Request, which is
# FastAPI's request type. Alias it so the two never collide.
from scheduler import NaiveScheduler, Request as InferenceRequest

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models" / "resnet18.onnx"))

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
MAX_WAIT_S     = float(os.environ.get("MAX_WAIT_S", "0.005"))   # 5 ms floor
METRICS_CSV    = os.environ.get("METRICS_CSV", "")              # empty = no dump

# Wire contract (must match the loadgen exactly — swap points #1 and #2):
# raw little-endian float32, one CHW image = (3,224,224) in C order, no batch dim.
EXPECTED_BYTES = 3 * 224 * 224 * 4                              # 602112


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = InferenceBackend(MODEL_PATH)
    backend.warmup(range(1, MAX_BATCH_SIZE + 1))   # warm EVERY size, not just 1
    metrics = MetricsCollector()
    scheduler = NaiveScheduler(backend, MAX_BATCH_SIZE, MAX_WAIT_S, metrics=metrics)
    scheduler.start()
    app.state.scheduler = scheduler
    app.state.metrics = metrics
    yield
    scheduler.stop()
    if METRICS_CSV:
        metrics.dump_csv(METRICS_CSV)


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: Request):
    arrival_ts = time.monotonic()          # STAMP FIRST, before deserialization
    raw = await request.body()
    if len(raw) != EXPECTED_BYTES:         # clear error beats a cryptic reshape
        raise HTTPException(status_code=400,
                            detail=f"expected {EXPECTED_BYTES} bytes, got {len(raw)}")

    # frombuffer view is read-only, but np.stack in the worker copies anyway.
    # NOTE: no batch dim here — the scheduler stacks these into (B,3,224,224).
    arr = np.frombuffer(raw, dtype="<f4").reshape(3, 224, 224)

    loop = asyncio.get_running_loop()
    future = loop.create_future()          # create ON the loop, never in worker
    req = InferenceRequest(input=arr, arrival_ts=arrival_ts, future=future, loop=loop)
    request.app.state.scheduler.submit(req)

    try:
        result = await future              # worker fulfills this cross-thread
    except Exception as e:                 # e.g. OOM under overload — fail visibly
        raise HTTPException(status_code=500, detail=str(e))

    return {"class_id": int(np.argmax(result))}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}