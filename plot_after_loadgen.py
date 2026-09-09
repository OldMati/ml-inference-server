import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
df[df.phase == "measure"].plot.scatter("intended_ms", "latency_ms", s=1)
m = df[df.phase == "measure"]
for s, g in m.groupby("status"):
    plt.scatter(g.intended_ms, g.latency_ms, s=1, label=f"status {s}")
plt.legend()
plt.savefig("scatter.png", dpi=300)