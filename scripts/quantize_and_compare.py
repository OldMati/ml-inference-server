"""
INT8 static quantization + FP32-vs-INT8 speed/accuracy comparison.

Completes Week 1: quantize ResNet-18, measure the throughput gain AND the
accuracy cost, so the trade-off is a measured number rather than a guess.

DATA REQUIRED (this is the one piece of legwork):
  A few hundred labelled ImageNet-val images in ImageFolder layout:
      data/imagenet_val/<synset_id>/<image>.JPEG
  e.g. data/imagenet_val/n01440764/ILSVRC2012_val_00000293.JPEG
  Folders named by synset id (n01440764, ...) make ImageFolder assign the
  exact class indices the pretrained model uses (sorted-synset order), so
  labels line up with no manual mapping. A subset is fine.

Run order: export_model.py first (makes models/resnet18.onnx), then this.
"""

import os
import time
import random

import numpy as np
import onnxruntime as ort

ort.preload_dlls()  # same CUDA-DLL fix as validate_backend.py — must run before any session

import torchvision
from torchvision.datasets import ImageFolder
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

FP32_MODEL = "models/resnet18.onnx"
PREP_MODEL = "models/resnet18.prep.onnx"   # shape-inferred/optimised, fed to the quantiser
INT8_MODEL = "models/resnet18.int8.onnx"

VAL_DIR = "imagenette2-320/val"
CALIB_SIZE = 200      # images used to calibrate activation ranges
EVAL_SIZE = 500       # DISJOINT images used to score accuracy
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
WARMUP, ITERS = 20, 100
SEED = 0

# Exact preprocessing (256 resize -> 224 crop -> ImageNet normalise) for these weights.
_weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
_preprocess = _weights.transforms()


def _to_chw(pil_img):
    """PIL image -> (3, 224, 224) float32, preprocessed for ResNet-18."""
    return _preprocess(pil_img).numpy().astype(np.float32)


# True ImageNet-1k indices for the 10 Imagenette classes (folder name == wnid).
# ImageFolder numbers folders alphabetically over ONLY those present, so a 10-class
# subset gets labels 0-9 -- which are NOT the 1000-class indices the pretrained model
# outputs. We remap through this table. To use a different subset, add its wnids here.
IMAGENETTE_WNID_TO_IDX = {
    "n01440764": 0,   "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}


def build_label_map(ds):
    """Map each ImageFolder label -> true ImageNet-1k index."""
    wnids = ds.classes  # folder names, in ImageFolder index order
    if len(wnids) == 1000:
        return {i: i for i in range(1000)}  # full val: folder order already == true order
    missing = [w for w in wnids if w not in IMAGENETTE_WNID_TO_IDX]
    if missing:
        raise SystemExit(f"No true-index mapping for {missing}. Add them to IMAGENETTE_WNID_TO_IDX.")
    return {i: IMAGENETTE_WNID_TO_IDX[w] for i, w in enumerate(wnids)}


def load_split():
    ds = ImageFolder(VAL_DIR)  # no transform: items are (PIL, label); we preprocess ourselves
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    calib_idx = idx[:CALIB_SIZE]
    eval_idx = idx[CALIB_SIZE:CALIB_SIZE + EVAL_SIZE]  # disjoint from calibration
    return ds, calib_idx, eval_idx


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds calibration images to the quantiser one at a time (low memory)."""

    def __init__(self, ds, indices, input_name):
        self.ds, self.indices, self.input_name = ds, indices, input_name
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.indices):
            return None
        img = self.ds[self.indices[self.pos]][0]
        self.pos += 1
        return {self.input_name: _to_chw(img)[None, ...]}  # (1, 3, 224, 224)

    def rewind(self):
        self.pos = 0


def input_name_of(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.get_inputs()[0].name


def export_int8(ds, calib_idx):
    name = input_name_of(FP32_MODEL)

    # ORT recommends this pre-pass (shape inference + graph opt) before static
    # quantisation; skipping it is a common cause of quantise failures.
    quant_pre_process(FP32_MODEL, PREP_MODEL)

    quantize_static(
        PREP_MODEL,
        INT8_MODEL,
        calibration_data_reader=ImageCalibrationReader(ds, calib_idx, name),
        quant_format=QuantFormat.QDQ,          # modern default; keeps a portable graph
        per_channel=True,                       # better accuracy for conv weights
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,        # symmetric int8 — TensorRT-friendly
        # calibrate_method defaults to MinMax; Percentile/Entropy can recover accuracy
    )

    fp32_mb = os.path.getsize(FP32_MODEL) / 1e6
    int8_mb = os.path.getsize(INT8_MODEL) / 1e6
    print(f"size: FP32 {fp32_mb:.1f} MB -> INT8 {int8_mb:.1f} MB "
          f"(ratio {int8_mb / fp32_mb:.2f}; expect ~0.3-0.4 for QDQ, not a clean 0.25)")


def make_session(path):
    sess = ort.InferenceSession(
        path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    return sess, sess.get_providers()[0], sess.get_inputs()[0].name


def benchmark(path, label):
    sess, provider, name = make_session(path)
    print(f"[{label}] provider: {provider}")
    out = {}
    for b in BATCH_SIZES:
        x = np.random.randn(b, 3, 224, 224).astype(np.float32)
        for _ in range(WARMUP):
            sess.run(None, {name: x})
        samples = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            sess.run(None, {name: x})
            samples.append((time.perf_counter() - t0) * 1000.0)
        out[b] = float(np.median(samples))
    return out


def accuracy(path, label, ds, eval_idx, label_map, batch=32):
    sess, _, name = make_session(path)
    top1 = top5 = total = 0
    for start in range(0, len(eval_idx), batch):
        chunk = eval_idx[start:start + batch]
        xs = np.stack([_to_chw(ds[i][0]) for i in chunk])
        ys = np.array([label_map[ds[i][1]] for i in chunk])  # remap to true ImageNet index
        logits = sess.run(None, {name: xs})[0]
        top5_pred = np.argsort(-logits, axis=1)[:, :5]
        top1 += int((top5_pred[:, 0] == ys).sum())
        top5 += int(sum(ys[k] in top5_pred[k] for k in range(len(ys))))
        total += len(ys)
    t1, t5 = 100 * top1 / total, 100 * top5 / total
    print(f"[{label}] top-1 {t1:.2f}%   top-5 {t5:.2f}%   (n={total})")
    return t1, t5


def main():
    if not os.path.exists(FP32_MODEL):
        print(f"Missing {FP32_MODEL} — run export_model.py first.")
        return
    if not os.path.isdir(VAL_DIR):
        print(f"Missing validation data at {VAL_DIR}. Download Imagenette:\n"
              f"  curl.exe -L -o imagenette2-320.tgz "
              f"https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz\n"
              f"  tar -xzf imagenette2-320.tgz\n"
              f"then set VAL_DIR to the val/ folder (e.g. imagenette2-320/val).")
        return

    ds, calib_idx, eval_idx = load_split()
    label_map = build_label_map(ds)
    print(f"dataset: {len(ds)} images ({len(ds.classes)} classes) | "
          f"calibrate on {len(calib_idx)} | evaluate on {len(eval_idx)}\n")

    print("=== 1. INT8 static quantization ===")
    export_int8(ds, calib_idx)

    print("\n=== 2. Latency: FP32 vs INT8 (synthetic input) ===")
    fp32 = benchmark(FP32_MODEL, "FP32")
    int8 = benchmark(INT8_MODEL, "INT8")
    print(f"\n{'batch':>6} {'FP32 ms':>9} {'INT8 ms':>9} {'speedup':>8}")
    print("-" * 34)
    for b in BATCH_SIZES:
        print(f"{b:>6} {fp32[b]:>9.2f} {int8[b]:>9.2f} {fp32[b] / int8[b]:>7.2f}x")

    print("\n=== 3. Accuracy: FP32 vs INT8 (same held-out images) ===")
    a1f, a5f = accuracy(FP32_MODEL, "FP32", ds, eval_idx, label_map)
    a1q, a5q = accuracy(INT8_MODEL, "INT8", ds, eval_idx, label_map)
    print(f"\ntrade-off: top-1 drop {a1f - a1q:+.2f} pts | "
          f"top-5 drop {a5f - a5q:+.2f} pts")


if __name__ == "__main__":
    main()