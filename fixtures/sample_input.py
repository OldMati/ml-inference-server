import numpy as np
from PIL import Image

IMG_PATH = "imagenette2-320/val/n01440764/ILSVRC2012_val_00009111.JPEG"  # any real file
OUT_PATH = "sample_input.bin"

img = Image.open(IMG_PATH).convert("RGB")

# resize shorter side to 256, then center-crop to 224x224 -- standard
# ImageNet preprocessing recipe, matches what the pretrained weights expect
w, h = img.size
if w < h:
    new_w, new_h = 256, round(256 * h / w)
else:
    new_w, new_h = round(256 * w / h), 256
img = img.resize((new_w, new_h), Image.BILINEAR)

left, top = (new_w - 224) // 2, (new_h - 224) // 2
img = img.crop((left, top, left + 224, top + 224))

arr = np.asarray(img, dtype=np.float32) / 255.0        # HWC, uint8[0,255] -> float32[0,1]

# ImageNet normalization -- SKIP this block if export_model.py already
# bakes a normalization layer into the ONNX graph. Check that first.
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
arr = (arr - mean) / std

arr = arr.transpose(2, 0, 1)                            # HWC -> CHW
arr = np.ascontiguousarray(arr, dtype="<f4")             # force layout + byte order

assert arr.shape == (3, 224, 224), arr.shape
assert arr.nbytes == 602112, arr.nbytes
assert arr.dtype == np.dtype("<f4")

arr.tofile(OUT_PATH)
print(f"wrote {OUT_PATH}: {arr.nbytes} bytes")