"""NLST figure: S4 - S3 differences (bootstrap 95 % CI over patients) by vendor x kernel class, from
p2_nlst/strata.csv (scripts/p2_nlst_analysis.py). Left: candidates per scan; right: median candidate score.
Output: p2_nlst/fig_nlst_forest.{pdf,png}"""
from pathlib import Path
import sys
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C

OUT = C.WORK / "wp0" / "analysis" / "p2_nlst"
INK, MUTED, GRID = "#1f2328", "#8a8f98", "#d8dbe0"

d = pd.read_csv(OUT / "strata.csv"); d["round"] = d["round"].astype(str)
nl = d[d.dataset == "NLST"]
rows = [("All NLST", nl[(nl.vendor == "all") & (nl.kernel_class == "all") & (nl["round"] == "all")].iloc[0], INK)]
for v in ("GE", "PHILIPS", "SIEMENS", "TOSHIBA"):
    for k in ("soft", "sharp"):
        r = nl[(nl.vendor == v) & (nl.kernel_class == k) & (nl["round"] == "all")].iloc[0]
        rows.append((f"{v.title() if v != 'GE' else 'GE'} {k}", r, INK))
rows.append(("LUNA16 test (reference)", d[d.dataset != "NLST"].iloc[0], MUTED))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
panels = [("cands_per_scan", "Δ candidates per scan (S4 − S3)"), ("score_q50", "Δ median candidate score (S4 − S3)")]
y = list(range(len(rows)))[::-1]
for ax, (stat, xlabel) in zip(axes, panels):
    for yi, (lab, r, col) in zip(y, rows):
        ax.plot([r[f"{stat}_diff_lo"], r[f"{stat}_diff_hi"]], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(r[f"{stat}_diff"], yi, "o", ms=6, color=col, mec="white", mew=1.5, zorder=3)
    ax.axvline(0, color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_xlabel(xlabel, fontsize=9); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0, labelsize=8.5); ax.tick_params(axis="x", labelsize=8)
    ax.axhline(y[0] - 0.5, color=GRID, lw=0.8); ax.axhline(y[-1] + 0.5, color=GRID, lw=0.8)
axes[0].set_yticks(y); axes[0].set_yticklabels([f"{lab}  (n = {int(r.n_series)})" for lab, r, _ in rows])
for ax, t in zip(axes, "ab"): ax.set_title(t, loc="left", fontweight="bold", fontsize=10)
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_nlst_forest.{ext}", dpi=300, bbox_inches="tight")
print("wrote", OUT / "fig_nlst_forest.pdf")
