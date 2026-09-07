# api.py
import asyncio
import os
import time
from contextlib import asynccontextmanager
import sys

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from backend import InferenceBackend
from metrics import MetricsCollector
# NAME CLASH TRAP: the scheduler's dataclass is also called Request, which is
# FastAPI's request type. Alias it so the two never collide.
from scheduler import NaiveScheduler, DynamicScheduler, AdmissionScheduler, Request as InferenceRequest

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models" / "resnet18.onnx"))

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))
MAX_WAIT_S     = float(os.environ.get("MAX_WAIT_S", "0.005"))   # 5 ms floor
METRICS_CSV    = os.environ.get("METRICS_CSV", "")              # empty = no dump
SCHEDULER = os.environ.get("SCHEDULER", "naive")

# Wire contract (must match the loadgen exactly — swap points #1 and #2):
# raw little-endian float32, one CHW image = (3,224,224) in C order, no batch dim.
EXPECTED_BYTES = 3 * 224 * 224 * 4                              # 602112

sys.setswitchinterval(0.001)

@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = InferenceBackend(MODEL_PATH)
    backend.warmup(range(1, MAX_BATCH_SIZE + 1))   # warm EVERY size, not just 1
    metrics = MetricsCollector()
    if SCHEDULER == "naive":
        scheduler = NaiveScheduler(backend, MAX_BATCH_SIZE, MAX_WAIT_S, metrics=metrics)
    elif SCHEDULER == "dynamic":
        scheduler = DynamicScheduler(backend, MAX_BATCH_SIZE, MAX_WAIT_S, metrics=metrics)
    else:
        scheduler = AdmissionScheduler(backend, MAX_BATCH_SIZE, MAX_WAIT_S, metrics=metrics)
    scheduler.start()
    app.state.scheduler = scheduler
    app.state.metrics = metrics
    yield
    scheduler.stop()
    metrics.plot_inference_latencies()
    if METRICS_CSV:
        metrics.dump_csv(METRICS_CSV)
    print('\n')
    print(metrics.snapshot())
    


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: Request):
    
    arrival_ts = time.monotonic()          # STAMP FIRST, before deserialization

    raw = await request.body()
    if len(raw) != EXPECTED_BYTES:         # clear error beats a cryptic reshape
        raise HTTPException(status_code=400,
                            detail=f"expected {EXPECTED_BYTES} bytes, got {len(raw)}")

    if SCHEDULER == 'admission':
        request.app.state.scheduler.metrics.admit_check_latency.append(arrival_ts)

        if not request.app.state.scheduler.should_admit(arrival_ts):
            raise HTTPException(status_code=503, detail="SLO cannot be met")


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

@app.post("/echo")
async def echo(request: Request):
    t0 = time.monotonic()
    raw = await request.body()
    return {"read_ms": (time.monotonic() - t0) * 1000, "n": len(raw)}
