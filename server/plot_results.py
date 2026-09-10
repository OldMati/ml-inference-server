import glob, re
import pandas as pd
import matplotlib.pyplot as plt

SLO   = 50.0
CLAMP = 3 * SLO                      # display only — state it in the caption
POLICIES = [("naive", "tab:blue"), ("dynamic", "tab:green"), ("admission", "tab:red")]


COLUMNS = ["offered_rate", "throughput", "goodput", "rejection_rate",
           "failure_rate", "slo_attainment", "p50", "p99"]


def load_sweep(policy):
    files = sorted(glob.glob(f"results/{policy}/rate_*.csv"))
    if not files:
        print(f"  [warn] no files matched results/{policy}/rate_*.csv — skipping")
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for f in files:
        rate = int(re.search(r"rate_(\d+)", f).group(1))
        df = pd.read_csv(f)
        m = df[df.phase == "measure"]        # keep ALL statuses
        if m.empty:
            print(f"  [warn] {f} has no measure-phase rows — skipping")
            continue

        dur = (m.intended_ms.max() - m.intended_ms.min()) / 1000.0

        ok   = m[m.status == 200]
        good = ok[ok.latency_ms <= SLO]

        rows.append({
            "offered_rate":   rate,
            "throughput":     len(ok)   / dur,
            "goodput":        len(good) / dur,
            "rejection_rate": (m.status == 503).mean(),
            "failure_rate":   (m.status == -1).mean(),
            "slo_attainment": len(good) / len(m),
            "p50": ok.latency_ms.quantile(0.50) if len(ok) else float("nan"),
            "p99": ok.latency_ms.quantile(0.99) if len(ok) else float("nan"),
        })

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows).sort_values("offered_rate")


fig, ax = plt.subplots(3, 1, figsize=(7, 11), sharex=True)

for policy, c in POLICIES:
    s = load_sweep(policy)
    if s.empty:
        continue
    ax[0].plot(s.offered_rate, s.goodput, "o-", color=c, label=policy)
    ax[1].plot(s.offered_rate, s.p50.clip(upper=CLAMP), "o-",  color=c, label=f"{policy} p50")
    ax[1].plot(s.offered_rate, s.p99.clip(upper=CLAMP), "s--", color=c, label=f"{policy} p99")
    ax[2].plot(s.offered_rate, s.rejection_rate * 100, "o-",   color=c, label=policy)

lim = ax[0].get_xlim()
ax[0].plot(lim, lim, ":", color="grey", lw=1, label="offered (ideal)")
ax[0].set_xlim(lim)

ax[0].set_ylabel("Goodput (req/s)\nstatus 200 AND ≤ SLO")
ax[1].axhline(SLO, color="k", ls=":", lw=1)
ax[1].set_ylabel(f"Latency (ms)\n[clamped at {CLAMP:.0f}]")
ax[2].set_ylabel("Rejected (%)")
ax[2].set_xlabel("Offered rate (req/s)")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

fig.tight_layout()
fig.savefig("figures/policy_comparison.png", dpi=150, bbox_inches="tight")