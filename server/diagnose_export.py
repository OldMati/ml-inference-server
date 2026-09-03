#!/usr/bin/env python3
"""
Diagnose a wrong top-1 from verify_correctness.py.

Two independent checks:

  1. CROSS-CHECK -- run the *same* preprocessed tensor through torchvision's
     PyTorch ResNet-18 and through the exported .onnx file. Identical input,
     two engines. If they disagree, the ONNX export is broken and preprocessing
     is exonerated. If they agree (even on a wrong answer), the export is fine
     and the image or the preprocessing is at fault.

  2. SWEEP -- run N images from every Imagenette class and report accuracy.
     Establishes a base rate. One hard image proves nothing; 0/50 does.

Usage:
    python server/diagnose_export.py --data imagenette2-320/val
    python server/diagnose_export.py --data imagenette2-320/val --per-class 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_correctness import (  # noqa: E402
    IMAGENETTE_SYNSET_TO_IMAGENET_IDX,
    DEFAULT_MODEL_PATH,
    preprocess,
    load_categories,
    label,
    top_k,
)


def onnx_logits(session, chw):
    name = session.get_inputs()[0].name
    return session.run(None, {name: chw[np.newaxis, ...]})[0][0]


def torch_logits(model, chw):
    import torch
    with torch.no_grad():
        out = model(torch.from_numpy(chw).unsqueeze(0))
    return out[0].numpy()


def cross_check(session, image_path, expected, categories):
    """Same tensor, two engines. Isolates export bugs from preprocessing bugs."""
    print("=" * 68)
    print("CROSS-CHECK: PyTorch vs ONNX on an identical input tensor")
    print("=" * 68)

    try:
        import torch  # noqa: F401
        from torchvision.models import resnet18, ResNet18_Weights
    except Exception as e:
        print(f"skipped -- torch/torchvision unavailable ({e})")
        return None

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    chw = preprocess(image_path)
    t_logits = torch_logits(model, chw)
    o_logits = onnx_logits(session, chw)

    t_top1 = int(np.argmax(t_logits))
    o_top1 = int(np.argmax(o_logits))

    print(f"image        : {image_path.name}")
    print(f"expected     : {label(categories, expected)}\n")

    print("PyTorch top-5:")
    for i, (idx, s) in enumerate(top_k(t_logits), 1):
        print(f"  {i}. {label(categories, idx):<40} {s:8.3f}")
    print("\nONNX top-5:")
    for i, (idx, s) in enumerate(top_k(o_logits), 1):
        print(f"  {i}. {label(categories, idx):<40} {s:8.3f}")

    # Max absolute difference across all 1000 logits. FP32 CPU vs FP32 CPU on
    # the same graph should agree to ~1e-4; anything larger means the exported
    # graph is not the same function as the source model.
    max_diff = float(np.max(np.abs(t_logits - o_logits)))
    print(f"\nmax |logit difference| : {max_diff:.6f}")
    print(f"PyTorch top-1 : {label(categories, t_top1)}")
    print(f"ONNX    top-1 : {label(categories, o_top1)}")

    if t_top1 != o_top1 or max_diff > 1e-2:
        print("\n>>> VERDICT: the two engines disagree. The ONNX EXPORT IS BROKEN.")
        print("    Re-export with torch.onnx.export(..., dynamo=False) and")
        print("    check the output file is ~45 MB before trusting it.")
        return "export"
    if t_top1 != expected:
        print("\n>>> VERDICT: both engines agree, and both are wrong. The export")
        print("    is FINE. Either this image is genuinely hard, or the")
        print("    preprocessing recipe is off. See the sweep below.")
        return "image_or_preproc"
    print("\n>>> VERDICT: both engines agree and are correct.")
    return "ok"


def sweep(session, data_dir, per_class, categories):
    """Accuracy across all 10 Imagenette classes. Establishes the base rate."""
    print()
    print("=" * 68)
    print(f"SWEEP: up to {per_class} image(s) per class")
    print("=" * 68)

    total = correct = 0
    for synset, expected in sorted(IMAGENETTE_SYNSET_TO_IMAGENET_IDX.items(),
                                   key=lambda kv: kv[1]):
        folder = data_dir / synset
        if not folder.is_dir():
            print(f"{synset}: folder missing, skipped")
            continue

        # Prefer real ILSVRC val images over the train-split files that
        # Imagenette also ships in these folders.
        images = sorted(folder.glob("ILSVRC2012_val_*.JPEG")) or sorted(folder.glob("*.JPEG"))
        images = images[:per_class]

        hits = 0
        for img in images:
            try:
                pred = int(np.argmax(onnx_logits(session, preprocess(img))))
            except Exception as e:
                print(f"  {img.name}: error {e}")
                continue
            hits += (pred == expected)
            total += 1
        correct += hits

        name = categories[expected] if categories else str(expected)
        bar = "#" * hits + "." * (len(images) - hits)
        print(f"  {name:<20} {hits}/{len(images)}  {bar}")

    if total:
        acc = correct / total
        print(f"\noverall: {correct}/{total} = {acc:.1%}")
        print("\ninterpretation:")
        print("  >70%  -- pipeline is correct; the failing image is just hard.")
        print("  20-60% -- systematically degraded input. Suspect the")
        print("            normalize/resize recipe, or a mismatch between the")
        print("            export's assumed input range and what you feed it.")
        print("  <10%  -- broken model or scrambled layout. Trust the")
        print("            cross-check verdict above.")
    return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("imagenette2-320/val"))
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--image", type=Path, default=None,
                   help="image for the cross-check; defaults to the first French horn")
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--skip-sweep", action="store_true")
    args = p.parse_args()

    if not args.model.exists():
        sys.exit(f"model not found: {args.model}")

    size_mb = args.model.stat().st_size / 1e6
    print(f"model: {args.model}  ({size_mb:.1f} MB)")
    if size_mb < 40:
        print("  !! ResNet-18 FP32 should be ~45 MB (11.7M params x 4 bytes).")
        print("     A short file means a truncated/corrupt export -- fix that first.")
    print()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    categories = load_categories()

    image = args.image
    if image is None:
        candidates = sorted((args.data / "n03394916").glob("*.JPEG"))
        if not candidates:
            sys.exit(f"no images found under {args.data / 'n03394916'}")
        image = candidates[0]

    expected = IMAGENETTE_SYNSET_TO_IMAGENET_IDX.get(image.parent.name)
    cross_check(session, image, expected, categories)

    if not args.skip_sweep:
        sweep(session, args.data, args.per_class, categories)


if __name__ == "__main__":
    main()