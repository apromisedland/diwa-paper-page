"""Regenerate measured result plots while preserving the author-supplied method image."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" if (ROOT / "paper").is_dir() else ROOT
OUT = PAPER / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA = json.loads((ROOT / "evidence/results_ledger.json").read_text())
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
TEAL = "#117F82"
INK = "#263744"
MUTED = "#657581"

# architecture.jpg is the original author-supplied asset; do not regenerate it.

labels=list(DATA["latency_ms"])
latency=list(DATA["latency_ms"].values())
fig, ax=plt.subplots(figsize=(5.5,2.1))
fig.subplots_adjust(left=0.25,right=0.93,bottom=0.23,top=0.93)
colors=[TEAL,"#8B9AA6","#B6C1C9","#D5DCE1"]
bars=ax.barh(range(4),latency,height=0.57,color=colors,edgecolor="white",linewidth=0.7)
ax.invert_yaxis()
ax.set_yticks(range(4),labels)
ax.set_xlim(0,355)
ax.set_xticks([0,100,200,300])
ax.set_xlabel("Mean end-to-end latency (ms) ↓",fontsize=9)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#AAB6BF")
ax.tick_params(axis="y",length=0,pad=7)
ax.tick_params(axis="x",color="#AAB6BF",labelsize=8)
ax.xaxis.grid(True,color="#E5E9EC",linewidth=0.7)
ax.set_axisbelow(True)
for bar,val in zip(bars,latency):
    ax.text(val+6,bar.get_y()+bar.get_height()/2,str(val),va="center",fontsize=9,
            fontweight="bold" if val==89 else "normal",color=INK)
fig.savefig(OUT/"latency.pdf",metadata={"Title":"Measured inference latency"})
fig.savefig(OUT/"latency.png",dpi=220)
plt.close(fig)

# Fixed counts and the adaptive ceiling are distinct measured settings.
fixed = list(reversed(DATA["fixed_budget"]))
adaptive = DATA["adaptive_budget"]
fig, ax = plt.subplots(figsize=(5.5, 2.45))
fig.subplots_adjust(left=.10, right=.97, bottom=.22, top=.91)
ax.plot([r["latency_ms"] for r in fixed], [r["standard_success_pct"] for r in fixed],
        color="#7E8B95", marker="o", markersize=5, linewidth=1.2,
        label="Fixed query count", zorder=2)
for r, offset in zip(fixed, [(8,-9), (7,-12), (7,8), (0,9), (-7,-13)]):
    ax.annotate(f"{r['selected_queries']} queries", (r["latency_ms"], r["standard_success_pct"]),
                xytext=offset, textcoords="offset points", fontsize=8, color=INK,
                ha="right" if r["selected_queries"] == 48 else "left")
ax.scatter([adaptive["latency_ms"]], [adaptive["standard_success_pct"]],
           s=100, marker="*", color=TEAL, edgecolors="white", linewidths=.4,
           label="Adaptive, ceiling 25%", zorder=4)
ax.annotate("Adaptive: 73.5%, 89 ms", (89,73.5), xytext=(113,69.8),
            fontsize=8, color=TEAL,
            arrowprops=dict(arrowstyle="-", color=TEAL, linewidth=.8))
ax.set(xlim=(48,230), ylim=(60.5,77), xlabel="Mean end-to-end latency (ms)", ylabel="Task success (%)")
ax.set_xticks([60,90,120,150,180,210])
ax.set_yticks([62,66,70,74])
ax.grid(True, color="#E5E9EC", linewidth=.6)
ax.set_axisbelow(True)
ax.tick_params(labelsize=8)
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.savefig(OUT/"budget_frontier.pdf", metadata={"Title":"Measured query-budget tradeoff"})
fig.savefig(OUT/"budget_frontier.png", dpi=220)
plt.close(fig)
print("Generated 2 vector result plots; preserved the author-supplied architecture image.")
