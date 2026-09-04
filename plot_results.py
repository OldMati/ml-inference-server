import glob, re
import pandas as pd
import matplotlib.pyplot as plt

<<<<<<< HEAD
rows = []
for f in sorted(glob.glob("results/rate_*.csv")):
    rate = int(re.search(r"rate_(\d+)", f).group(1))
    df = pd.read_csv(f)
    df = df[(df.phase == "measure") & (df.status == 200)]
    if df.empty:
        continue
    df["completion_ms"] = df.intended_ms + df.latency_ms
    duration_s = (df.completion_ms.max() - df.completion_ms.min()) / 1000
    rows.append({
        "offered_rate": rate,
        "achieved_tput": len(df) / duration_s,
        "p50": df.latency_ms.quantile(0.50),
        "p99": df.latency_ms.quantile(0.99),
    })

s = pd.DataFrame(rows).sort_values("offered_rate")
=======
def load_sweep(policy):
    rows = []
    for f in sorted(glob.glob(f"results/{policy}/rate_*.csv")):
        rate = int(re.search(r"rate_(\d+)", f).group(1))
        df = pd.read_csv(f)
        df = df[(df.phase == "measure") & (df.status == 200)]
        if df.empty:
            continue
        df = df.assign(completion_ms=df.intended_ms + df.latency_ms)
        dur = (df.completion_ms.max() - df.completion_ms.min()) / 1000
        rows.append({
            "offered_rate": rate,
            "achieved_tput": len(df) / dur,
            "p50": df.latency_ms.quantile(0.50),
            "p99": df.latency_ms.quantile(0.99),
        })
    return pd.DataFrame(rows).sort_values("offered_rate")
>>>>>>> 83e2bc0531d78d6afd365a1dd51be101eccb2978

fig, ax = plt.subplots(figsize=(7, 5))
for policy, colour in [("naive", "tab:blue"), ("dynamic", "tab:orange")]:
    s = load_sweep(policy)
    ax.plot(s.achieved_tput, s.p50, "o-", color=colour, label=f"{policy} p50")
    ax.plot(s.achieved_tput, s.p99, "s--", color=colour, label=f"{policy} p99")

ax.set_xlabel("Achieved throughput (req/s)")
ax.set_ylabel("Latency (ms)")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig("policy_comparison.png", dpi=150, bbox_inches="tight")