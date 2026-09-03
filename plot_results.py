import glob, re
import pandas as pd
import matplotlib.pyplot as plt

rows = []
for f in sorted(glob.glob("results/rate_*.csv")):
    rate = int(re.search(r"rate_(\d+)", f).group(1))
    df = pd.read_csv(f)
    df = df[(df.phase == "measure") & (df.status == 200)]
    if df.empty:
        continue
    duration_s = (df.intended_ms.max() - df.intended_ms.min()) / 1000
    rows.append({
        "offered_rate": rate,
        "achieved_tput": len(df) / duration_s,
        "p50": df.latency_ms.quantile(0.50),
        "p99": df.latency_ms.quantile(0.99),
    })

s = pd.DataFrame(rows).sort_values("offered_rate")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(s.achieved_tput, s.p50, "o-", label="p50")
ax.plot(s.achieved_tput, s.p99, "s-", label="p99")
ax.set_xlabel("Achieved throughput (req/s)")
ax.set_ylabel("Latency (ms)")
ax.set_title("Naive baseline: throughput vs latency")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig("baseline_curve.png", dpi=150, bbox_inches="tight")
print(s.to_string(index=False))
