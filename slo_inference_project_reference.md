# SLO-Aware ML Inference Server — Project Reference

*A self-contained reference for building this project from scratch. Written to be
usable on its own, in a new Claude conversation, or as a personal reference while
coding — no prior context assumed.*

---

## 1. Overview

An ML inference server that accepts prediction requests and processes them
intelligently rather than naively: instead of using a fixed batch size and
accepting every request unconditionally, it **dynamically batches** requests and
**decides whether to admit them at all**, based on a target latency promise (an
SLO — Service Level Objective), so that p99 latency stays bounded even under
bursty, unpredictable load.

Target model: a CNN (e.g. ResNet-18), served via ONNX Runtime on GPU (CUDA),
optionally INT8-quantized. The novel engineering is in the scheduling and
admission logic sitting in front of that model, not the model itself.

**Why build this:** it's a systems/performance-engineering project, not an ML
modeling project — the goal is CV- and UROP-worthy proof that you can design,
build, and rigorously evaluate a real latency-sensitive system. It's aimed at
demonstrating fit with Imperial's LSDS group (specifically Marios Kogias's
work on microsecond-scale tail-latency systems) while also standing on its
own for general SWE / performance-engineering roles.

---

## 2. The core tension (why this problem exists at all)

Two facts, in conflict:

1. **GPUs like batches.** A GPU processing 32 images in one forward pass takes
   barely longer than processing 1 — the hardware is built for massive
   parallelism. Bigger batches → dramatically better *throughput* (total
   requests handled per second, across everyone).
2. **Requests hate waiting.** If you're the first request to arrive and the
   server waits to collect 31 more before running the batch, you paid a
   latency cost for zero benefit to yourself. Bigger batches → worse
   *latency* for whoever arrives early.

Every real inference-serving system has to navigate this trade-off. A naive
server picks one fixed strategy and hopes it works across all load
conditions. This project's server picks the strategy dynamically.

---

## 3. Prerequisite reading

Read these two before writing code — everything else can be learned by doing.

1. **["Serving DNNs like Clockwork: Performance Predictability from the
   Bottom Up"](https://www.usenix.org/system/files/osdi20-gujarati.pdf)**
   — Gujarati et al., OSDI 2020. The closest existing system to what you're
   building. Focus on the motivation section (why naive serving fails) and
   the evaluation methodology (how they measure and present results) — that's
   what you'll be mimicking in your own report. ([presentation
   page](https://www.usenix.org/conference/osdi20/presentation/gujarati))

2. **["How NOT to Measure Latency"](https://www.youtube.com/watch?v=lJ8ydIuPFeU)**
   — Gil Tene (talk), or [this written summary](https://bravenewgeek.com/everything-you-know-about-latency-is-wrong/)
   if you'd rather read than watch. Explains **coordinated omission** — the
   single most common bug in latency benchmarking, and the reason your load
   generator needs to be built carefully rather than thrown together. Read
   this before writing the load generator specifically.

---

## 4. Key concepts glossary

**Inference vs. training** — training teaches a model; inference *uses* an
already-trained, frozen model to make one prediction. This project is
entirely about inference.

**Latency vs. throughput** — latency is how long *one* request takes,
start to finish. Throughput is how many requests the *whole system* finishes
per second. You can often trade one for the other — that trade-off is the
subject of this project.

**SLO (Service Level Objective)** — a latency promise, phrased as a
percentile: "99% of requests complete in under 50ms." Not "on average" —
averages are easy to hit and easy to hide problems behind.

**p50 / p99 latency** — p50 (median) is the latency half of requests beat.
p99 is the latency 99% of requests beat, i.e. the boundary for your worst 1%.
At scale, "rare" bad outcomes happen to real people constantly — 1% of a
million daily requests is 10,000 bad experiences, every day.

**Dynamic batching** — batch size and wait time adjust to current
conditions: bigger/slower batches when load is high and there's slack in the
latency budget; smaller/faster batches when load is low or the SLO is at
risk.

**Admission control** — the server's ability to say "no" (or "not yet") to
an incoming request *before* processing it, if accepting it would blow the
SLO for requests already queued. Most naive servers skip this entirely and
will accept requests until the whole system collapses under load.

**INT8 quantization** — model weights are normally 32-bit floats;
quantization rounds them to 8-bit integers — a 4x memory reduction and a
real speed boost on GPU, at the cost of a small, usually-acceptable accuracy
hit. Used here mainly to buy extra throughput headroom for the scheduling
policies to work with, not as the project's main contribution.

**ONNX Runtime** — a model-execution engine that runs a model (exported in
the ONNX format) across multiple backends (CPU, CUDA, TensorRT) through one
API. This is what the batching/admission logic calls to actually run
inference.

**Execution provider (CUDA / TensorRT)** — ONNX Runtime's term for "which
hardware backend runs the math." CUDA EP uses the GPU directly; TensorRT EP
additionally compiles/optimizes the model graph for extra speed.

**Open-loop load generation** — a load generator that sends requests on a
fixed schedule regardless of whether previous requests have finished
(mimics real, independent users). The alternative — closed-loop, waiting for
a response before sending the next request — silently hides tail latency:
if the server is struggling, a closed-loop generator automatically "backs
off" and never shows you how bad things got.

**Coordinated omission** — the specific bug that happens when a closed-loop
(or badly-implemented) load generator under-samples exactly the slow
periods it should be measuring, because it's blocked waiting rather than
sending. Can skew reported percentiles by orders of magnitude. See §3.

---

## 5. System architecture

### Process architecture

Load generator and server run as **separate OS processes**, communicating
over real HTTP on localhost — not one script simulating both sides
in-memory. This matters: running in one process hides the network stack
overhead, OS scheduling jitter, and serialization cost that real tail
latency comes from.

### Language and libraries

**Python for the server, C++ for the load generator.** This split is
deliberate, not arbitrary:

- The server's actual compute-heavy work (`session.run()` in ONNX Runtime)
  is a C++/CUDA call that releases Python's GIL while running — so Python
  isn't the bottleneck there. The interesting engineering (scheduling,
  admission decisions) is fundamentally about timing/queueing logic, not
  raw compute, and Python is fine for that at this scale.
- The load generator is the one place where implementation language
  *actually affects data validity*: its whole job is precise timing, and
  Python's GIL/asyncio event loop introduce their own scheduling jitter —
  exactly the kind of noise that shouldn't contaminate latency
  measurements. A C++ generator using real threads and tighter timer
  control is a genuine accuracy improvement, not just resume-padding.

Suggested libraries: FastAPI + Uvicorn (server, async), `onnxruntime`
(inference), `pandas` + `matplotlib` (analysis). Load generator: plain C++
with real threads (or a small async library) — no framework needed.

### The one architectural trap to know about in advance

The server's event loop handles incoming requests *and* has to trigger GPU
inference calls. If `session.run()` is called directly inside an async
handler, it **blocks the entire event loop** for the duration of inference —
meaning no new requests can be admitted or timestamped while a batch is
running, which quietly corrupts admission-control timing. Fix: run
inference in a background thread (e.g. `run_in_executor()`), so the event
loop stays free to keep accepting/queueing requests while the GPU works
through the current batch. This is the kind of bug that's invisible until
p99 numbers look inexplicably wrong.

### Data flow

```
                    ┌───────────────────┐
  requests -------->│  Load Generator    │  (C++, open-loop, configurable
                    │                    │   rate, bursty traffic patterns)
                    └─────────┬──────────┘
                              │  HTTP, localhost
                              ▼
                    ┌───────────────────┐
                    │  Admission         │  accept / reject / defer,
                    │  Controller        │  based on projected latency
                    └─────────┬──────────┘
                              │ (accepted requests)
                              ▼
                    ┌───────────────────┐
                    │  Batching          │  groups requests into batches
                    │  Scheduler         │  dynamically (size + timeout)
                    └─────────┬──────────┘
                              │ (a batch)
                              ▼
                    ┌───────────────────┐
                    │  Inference         │  ONNX Runtime + CUDA EP,
                    │  Backend           │  optionally INT8 quantized
                    └─────────┬──────────┘
                              │ (results)
                              ▼
                    ┌───────────────────┐
                    │  Metrics           │  per-request completion time
                    │  Collector         │  → p50/p99, throughput, histograms
                    └───────────────────┘
```

### Suggested module breakdown

```
inference-server/
  server/
    api.py          # FastAPI endpoint — receives request, timestamps arrival
    admission.py      # admission controller — accept/reject logic
    scheduler.py        # batch-formation policies (naive / dynamic / SLO-aware)
    backend.py            # ONNX Runtime wrapper, batched inference
    metrics.py              # per-request timestamps → percentiles
  loadgen/                     # C++
    generator.cpp        # open-loop client, Poisson arrivals
    workloads.hpp          # traffic patterns (constant, bursty, step)
  scripts/
    export_model.py    # model → ONNX, dynamic batch axis, optional INT8
    run_experiment.py    # sweeps (policy × load level), orchestrates runs
    plot_results.py        # throughput/latency plots, histograms
  tests/
    test_admission.py
    test_scheduler.py
```

Rough scope: ~1,400 lines total. The admission controller and scheduler
combined are only ~350 lines but are where most of the actual thinking
lives; the load generator and plotting scripts are mechanically simpler but
still take real time to get right (fiddly timing bugs vs. genuine design
decisions).

---

## 6. Policies to implement (in order of increasing complexity)

1. **Naive baseline** — fixed batch size, no admission control. Everything
   is accepted; batches form on a fixed timer or fixed count. The "what
   everyone does by default" comparison point.
2. **Dynamic batching only** — batch size/timeout adapt to current queue
   depth, but still no admission control (degrades under overload, but
   doesn't protect the SLO).
3. **SLO-aware admission control** — the centerpiece: the controller
   estimates whether an incoming request can still complete within its SLO
   given current queue state, and rejects/sheds load if not, protecting
   requests already admitted.
4. *(Stretch)* **Multi-replica routing** — if there's time, run multiple
   backend instances and route with a "join bounded shortest queue"
   (JBSQ)-style policy rather than round-robin.

---

## 7. Evaluation plan

- **Core plot:** throughput (x-axis) vs. p50/p99 latency (y-axis) per
  policy, across increasing request rates. Look for the "knee" — the load
  level where tail latency explodes — and show admission control pushes it
  further out, or at least fails gracefully instead of catastrophically.
- **Latency histograms** per policy at a fixed load level, to show the
  distribution shape, not just summary percentiles.
- **Flamegraphs** of the inference backend, to show where time is actually
  spent — useful as both a sanity check and an extra "I understand the
  system" artifact.
- *(Optional)* Fairness across request types, if multiple priorities/models
  are introduced.

---

## 8. Suggested build order (6–8 weeks)

| Week | Goal |
|---|---|
| 1 | ONNX Runtime + CUDA EP running a single synchronous inference end-to-end. Confirm INT8 quantization works; measure the speed/accuracy trade-off. |
| 2 | Build the open-loop (C++) load generator. Worth doing carefully — every later measurement depends on it being correct. |
| 3 | Naive baseline server (fixed batching, no admission control); first throughput-vs-latency plot. |
| 4 | Dynamic batching. Compare against baseline. |
| 5 | SLO-aware admission control — budget real time here. |
| 6 | Full benchmark harness: automated sweeps across load levels, all metrics, all plots. |
| 7 | Polish: flamegraphs, README, write-up contrasting all policies. |
| 8 (stretch) | Multi-replica routing, or a second model/workload type. |

Weeks 1–6 alone give a complete, presentable project even without the
stretch goals.

---

## 9. Known gotchas (found during setup testing)

- **PyTorch's ONNX exporter:** recent PyTorch versions (2.9+) default to a
  new "dynamo"-based exporter. In testing, it silently produced a corrupt,
  incomplete `.onnx` file for a plain ResNet-18 (a fraction of the expected
  file size — missing most of the weight data — while still printing
  "success"). Force the older, stable exporter explicitly:
  `torch.onnx.export(..., dynamo=False)`. Worth checking output file size
  against a rough expectation (params × 4 bytes for FP32) as a sanity check
  after any export.
- **Dynamic batch axis:** export with `dynamic_axes` (or equivalent) so the
  ONNX model accepts variable batch sizes at inference time — required
  since the entire project is about varying batch size at runtime. Exporting
  with a fixed batch size silently locks you into that one size later.
- **Silent CPU fallback:** `onnxruntime` (CPU-only package) and
  `onnxruntime-gpu` are different pip packages. Requesting the CUDA
  execution provider when only the CPU package is installed does not
  error — it silently falls back to CPU. Always check
  `session.get_providers()[0]` after creating a session and confirm it says
  `CUDAExecutionProvider`, not just that the code ran.
- **CPU batching gives little/no benefit:** confirmed via quick test —
  batch=1 vs. batch=4 latency scaled almost perfectly linearly on CPU
  (no batching benefit at all), which is expected and is exactly the
  contrast that motivates using GPU: on GPU, batch=4 should take barely
  longer than batch=1. Worth verifying this gap actually shows up on your
  own RTX 4060 before building any scheduling logic on top of it — it's the
  effect the entire project depends on.

---

## 10. Deliverables (what "done" looks like)

- Working server implementing at least: naive baseline, dynamic batching,
  and SLO-aware admission control
- Correct, open-loop C++ load generator with configurable, bursty traffic
  patterns
- Throughput-vs-p50/p99-latency plots comparing all policies
- Latency histograms and at least one flamegraph
- A short written report explaining the mechanism, results, and trade-offs
  of each policy
- A clean README

---

## 11. How this connects to Marios Kogias's own work (for outreach, not for the CV)

Kogias's own research (with Edouard Bugnion, his PhD advisor, mostly pre-Imperial)
is the direct intellectual ancestor of this project:

- **[R2P2: Making RPCs first-class datacenter citizens](https://www.usenix.org/system/files/atc19-kogias-r2p2_0.pdf)**
  (USENIX ATC 2019) — his RPC transport layer. Includes **SVEN**, an
  SLO-aware RPC admission control mechanism — the direct conceptual
  ancestor of this project's admission controller, just generalized from
  generic RPCs to ML inference specifically.
- **[ZygOS: Achieving Low Tail Latency for Microsecond-scale Networked Tasks](https://www.semanticscholar.org/paper/ZygOS:-Achieving-Low-Tail-Latency-for-Networked-Prekas-Kogias/e2635ec7a07be23e193622eb3681c0421a2c10ae)**
  (SOSP 2017) — work-conserving scheduling to avoid head-of-line blocking.
  Relevant if the multi-replica/JBSQ stretch goal is attempted.
- **[HovercRaft: Scalability and Fault-tolerance for micro-second scale
  Datacenter Services](https://dl.acm.org/doi/10.1145/3342195.3387545)**
  (EuroSys 2020) — integrates Raft replication directly into R2P2's
  transport layer rather than bolting it on separately. Relevant only if
  extending this project with a fault-tolerance component later.
- **Lancet: A self-correcting Latency Measuring Tool** (USENIX ATC 2019,
  Kogias, Mallon, Bugnion) — his own tooling for avoiding exactly the
  coordinated-omission problem covered in §3. Worth explicitly mentioning
  in the load generator's design rationale.

Live, current research: [marioskogias.github.io](https://marioskogias.github.io/).
Imperial's authenticated project portal (for currently-listed student
project topics, requires an Imperial login): [project-portal.doc.ic.ac.uk](https://project-portal.doc.ic.ac.uk/).

**Framing note:** don't pitch this as solving an open research problem — the
ML-inference-serving space is active and crowded (Clockwork, Shepherd,
Clipper, and others). Pitch it as a well-scoped exploration of SVEN's idea
applied to a new domain, which is exactly the right scope for a UROP
application and doesn't require novelty to be a strong result.
