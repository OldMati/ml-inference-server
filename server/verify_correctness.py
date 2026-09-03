#!/usr/bin/env python3
"""
Week 3 correctness check: does the CHW byte layout actually round-trip correctly?

A 200 OK from the server proves the plumbing works, not that the payload was
interpreted correctly -- random data returns a valid class_id at any layout.
This script runs one *real* preprocessed Imagenette image through the model
locally, and optionally through the server, and compares top-1 against known
ground truth (the synset in the folder name).

Local-only mode needs no network and no CUDA, so it runs fine on a MacBook:

    python server/verify_correctness.py \
        --image imagenette2-320/val/n03394916/ILSVRC2012_val_00033682.JPEG \
        --dump fixtures/french_horn_566.bin

Server comparison (needs the GPU box + a running server):

    python server/verify_correctness.py --image <same> --server http://127.0.0.1:8000/predict

Interpreting the result:
    local top-5 coherent + match True  -> layout is correct, move on
    local correct but match False      -> wire/reshape bug (CHW vs HWC)
    local garbage                      -> preprocessing bug (resize / normalize)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# --- constants -------------------------------------------------------------

# Imagenette's 10 classes mapped to their TRUE ImageNet-1k indices. Imagenette's
# own labels are 0-9 in folder-sort order, but a stock pretrained ResNet-18 has
# the full 1000-way head -- comparing against 0-9 produces a spurious mismatch
# and sends you hunting for a layout bug that isn't there.
IMAGENETTE_SYNSET_TO_IMAGENET_IDX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

# Standard ImageNet preprocessing -- must match whatever the ONNX export assumed.
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EXPECTED_BYTES = 3 * CROP_SIZE * CROP_SIZE * 4  # 602112

# Resolve the model relative to THIS file, not the launch directory -- a bare
# relative path resolves from cwd and breaks depending on where you run from.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "resnet18.onnx"


# --- labels ----------------------------------------------------------------

def load_categories():
    """Human-readable ImageNet class names, or None if unavailable.

    torchvision keeps the category list in the weights' .meta dict, which is a
    plain Python structure -- reading it does not download the weights file.
    """
    try:
        from torchvision.models import ResNet18_Weights
        return ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    except Exception:
        return None


def label(categories, idx):
    if categories is None:
        return f"class {idx}"
    return f"{idx} ({categories[idx]})"


# --- preprocessing ---------------------------------------------------------

def preprocess(image_path: Path) -> np.ndarray:
    """JPEG -> normalized float32 CHW array, C-contiguous, shape (3, 224, 224).

    No batch axis: the server's wire format is a single image and the scheduler
    adds the batch dimension via np.stack.
    """
    img = Image.open(image_path).convert("RGB")

    # Resize shortest side to 256, preserving aspect ratio.
    w, h = img.size
    if w < h:
        new_w, new_h = RESIZE_SHORT_SIDE, round(h * RESIZE_SHORT_SIDE / w)
    else:
        new_w, new_h = round(w * RESIZE_SHORT_SIDE / h), RESIZE_SHORT_SIDE
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to 224x224.
    left = (new_w - CROP_SIZE) // 2
    top = (new_h - CROP_SIZE) // 2
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    # HWC uint8 [0,255] -> HWC float32 [0,1] -> normalized -> CHW.
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))

    # ascontiguousarray matters: transpose only changes strides, and .tobytes()
    # on a non-contiguous array would silently serialize in a different order
    # than the server expects to read.
    return np.ascontiguousarray(arr, dtype=np.float32)


# --- inference -------------------------------------------------------------

def run_local(model_path: Path, chw: np.ndarray):
    """Run the ONNX model in-process on CPU. Returns the (1000,) logit vector."""
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    batch = chw[np.newaxis, ...]  # (1, 3, 224, 224)
    logits = session.run(None, {input_name: batch})[0]
    return logits[0]


def run_server(url: str, payload: bytes):
    """POST the raw bytes to the server. Returns (class_id, raw_json)."""
    import requests  # imported lazily so local-only mode needs no network deps

    resp = requests.post(
        url,
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    class_id = body.get("class_id", body.get("prediction"))
    return class_id, body


def top_k(logits: np.ndarray, k: int = 5):
    idx = np.argsort(logits)[::-1][:k]
    return [(int(i), float(logits[i])) for i in idx]


# --- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path,
                        help="path to an Imagenette val JPEG")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--expected", type=int, default=None,
                        help="true ImageNet index; inferred from the synset folder if omitted")
    parser.add_argument("--server", type=str, default=None,
                        help="server /predict URL; local-only if omitted")
    parser.add_argument("--dump", type=Path, default=None,
                        help="write the preprocessed tensor here as raw little-endian f32")
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"image not found: {args.image}")
    if not args.model.exists():
        sys.exit(f"model not found: {args.model}")

    # Ground truth comes from the containing folder's synset.
    synset = args.image.parent.name
    expected = args.expected
    if expected is None:
        expected = IMAGENETTE_SYNSET_TO_IMAGENET_IDX.get(synset)
        if expected is None:
            sys.exit(f"unknown synset {synset!r}; pass --expected explicitly")

    categories = load_categories()

    print(f"image    : {args.image}")
    print(f"synset   : {synset}")
    print(f"expected : {label(categories, expected)}")
    print()

    # --- preprocess
    chw = preprocess(args.image)
    payload = chw.astype("<f4").tobytes()
    print(f"tensor   : shape={chw.shape} dtype={chw.dtype} "
          f"contiguous={chw.flags['C_CONTIGUOUS']}")
    print(f"bytes    : {len(payload)} (expected {EXPECTED_BYTES})")
    if len(payload) != EXPECTED_BYTES:
        sys.exit("FAIL: payload size mismatch -- wrong shape or dtype upstream")
    print()

    # --- local inference
    logits = run_local(args.model, chw)
    local_top1 = int(np.argmax(logits))
    print("local top-5:")
    for i, (idx, score) in enumerate(top_k(logits), 1):
        marker = "  <-- expected" if idx == expected else ""
        print(f"  {i}. {label(categories, idx):<40} {score:8.3f}{marker}")
    local_ok = local_top1 == expected
    print(f"\nlocal top-1 == expected : {local_ok}")
    if not local_ok:
        print("  -> if the top-5 is also incoherent, the bug is in PREPROCESSING")
        print("     (resize / crop / normalize), not in the wire format.")
    print()

    # --- dump fixture
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_bytes(payload)
        print(f"wrote {len(payload)} bytes -> {args.dump}")
        print(f"  point the loadgen at this file; every response should be "
              f"class {expected}.")
        print()

    # --- server comparison
    if args.server:
        server_class, body = run_server(args.server, payload)
        print(f"server response : {body}")
        print(f"server class_id : {label(categories, server_class)}")
        match = server_class == local_top1
        print(f"match (server == local) : {match}")
        if not match:
            print("  -> local was correct but the server disagrees: the payload is")
            print("     being reinterpreted. Suspect CHW vs HWC, or endianness.")
        sys.exit(0 if (local_ok and match) else 1)

    sys.exit(0 if local_ok else 1)


if __name__ == "__main__":
    main()