"""
Export ResNet-18 to ONNX with a dynamic batch axis.

Week 1 deliverable (FP32 only — add INT8 quantization once this pipeline
is proven end-to-end). Guards against the two export gotchas from the
project reference:
  - the dynamo exporter can silently write a corrupt/incomplete .onnx file
  - a fixed batch axis silently locks you out of runtime batch-size changes
"""

import os
import torch
import torchvision

OUT_DIR = "models"
OUT_PATH = os.path.join(OUT_DIR, "resnet18.onnx")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        OUT_PATH,
        input_names=["input"],
        output_names=["output"],
        # Batch dimension (axis 0) is dynamic — the whole project varies it.
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # gotcha: dynamo exporter silently produced corrupt files
    )

    # Sanity-check file size against a rough FP32 weight-size expectation.
    # A corrupt export shows up here as a file far smaller than expected.
    n_params = sum(p.numel() for p in model.parameters())
    expected_bytes = n_params * 4  # 4 bytes per FP32 param
    actual_bytes = os.path.getsize(OUT_PATH)
    ratio = actual_bytes / expected_bytes

    print(f"params:        {n_params:,}")
    print(f"expected size: ~{expected_bytes / 1e6:.1f} MB (FP32 weights)")
    print(f"actual size:    {actual_bytes / 1e6:.1f} MB  (ratio {ratio:.2f})")

    assert ratio > 0.8, (
        "ONNX file is much smaller than expected — export is probably "
        "corrupt. Confirm dynamo=False took effect and re-run."
    )
    print(f"OK — wrote {OUT_PATH}")


if __name__ == "__main__":
    main()