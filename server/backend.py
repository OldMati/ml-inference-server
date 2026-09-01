import numpy as np
import onnxruntime as ort


model_path = "models/resnet18.onnx"

class InferenceBackend:
    
    def __init__(self, model_path, providers=("CUDAExecutionProvider",)):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=[("CUDAExecutionProvider", {"device_id": 0})])

        active = self.session.get_providers()
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"CUDA EP not active; providers = {active}")

        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

    def warmup(self, batch_sizes, iters=15):
        for b in batch_sizes:
            dummy = np.zeros((b, 3, 224, 224), dtype=np.float32)
            for _ in range(iters):
                self.infer(dummy)

    def infer(self, batch: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.out_name], {self.in_name: batch})
        return outputs[0]