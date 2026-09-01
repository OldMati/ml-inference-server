import numpy as np
from backend import InferenceBackend


b = InferenceBackend("/home/mateu/ml-inference-server/models/resnet18.onnx")   # will raise if not on CUDA EP
b.warmup([1])
out = b.infer(np.zeros((1, 3, 224, 224), dtype=np.float32))
print(out.shape) 