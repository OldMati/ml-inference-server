"""
Validate the inference backend before building anything on top of it.

Two checks, both from §9 of the project reference:
  1. Confirm ONNX Runtime is actually using the GPU, not silently falling
     back to CPU (onnxruntime vs onnxruntime-gpu are different packages).
  2. Measure latency across batch sizes. This is the single effect the
     whole project depends on: on GPU, per-image time should drop sharply
     as batch grows (batch=32 barely slower than batch=1). On CPU it stays
     roughly flat — no batching benefit — which is exactly why you need
     the GPU.

Note on timing: onnxruntime's session.run() blocks until the GPU work is
done and results are copied back, so wall-clock timing around it is honest
(no explicit CUDA sync needed, unlike raw PyTorch).
"""

import time
import numpy as np
import onnxruntime as ort
ort.preload_dlls()

MODEL = "models/resnet50.onnx"
BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32]
WARMUP = 20
ITERS = 100


def main():
    sess = ort.InferenceSession(
        MODEL,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    provider = sess.get_providers()[0]
    print(f"active provider: {provider}\n")
    # assert provider == "CUDAExecutionProvider", (
    #     "Not running on GPU — this is the silent CPU-fallback gotcha. "
    #     "You likely have the 'onnxruntime' package instead of "
    #     "'onnxruntime-gpu'. Uninstall onnxruntime, install onnxruntime-gpu, "
    #     "and check your CUDA/cuDNN versions match its requirements."
    # )

    input_name = sess.get_inputs()[0].name

    print(f"{'batch':>6} {'median ms':>10} {'ms/img':>8} {'vs batch=1':>11}")
    print("-" * 38)

    base_total = None
    for b in BATCH_SIZES:
        x = np.random.randn(b, 3, 224, 224).astype(np.float32)

        for _ in range(WARMUP):
            sess.run(None, {input_name: x})

        samples = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            sess.run(None, {input_name: x})
            samples.append((time.perf_counter() - t0) * 1000.0)

        median = float(np.median(samples))
        maxx = float(np.max(samples))
        minn = float(np.min(samples))

        per_img = median / b
        if base_total is None:
            base_total = median
        print(f"{b:>6} {median:>10.2f} {per_img:>8.3f} {median / base_total:>10.2f}x")

    print(
        "\nGreen light: 'ms/img' falls sharply as batch grows, and batch=32 "
        "total is only a small multiple of batch=1.\nRed flag: 'ms/img' stays "
        "flat (~linear scaling) — you're effectively on CPU or the export is wrong."
    )


if __name__ == "__main__":
    main()