import numpy as np, requests
x = np.random.rand(3, 224, 224).astype("<f4")
r = requests.post("http://127.0.0.1:8000/predict", data=x.tobytes())
print(r.status_code, r.json())