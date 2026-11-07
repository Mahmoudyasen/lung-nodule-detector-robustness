"""Generate the manuscript figures (paper/figures/fig*.pdf + .png) from the project's result files and images.
Matplotlib only; no generated imagery. Run: $PY paper/make_figures.py
  fig1  one LUNA16 test nodule under each perturbation + noise-only images (incl. the real TCIA 10 % - full noise)
  fig2  CPM vs perturbation strength, S3 vs S4, pooled over the four test folds
  fig3  paired Delta CPM (S4 - S3) for all 21 conditions, primary families marked
  fig4  TCIA paired full vs 10 % dose
  fig5  NLST real-kernel pairs (same exam, soft and sharp kernel): agreement S3 vs S4, by vendor, calibrated blur
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from lungcad import config as C
from lungcad.preprocess import world_to_voxel, load_volume
from lungcad.stage1.dataset import load_full_volume

AN = C.WORK / "wp0" / "analysis"; POOL = AN / "e1_dose_pooled"; OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#6b7079", "#d8dbe0"
COL = {"S3": "#2a78d6", "S4": "#eb6834"}; NAME = {"S3": "S3 (no augmentation)", "S4": "S4 (noise + blur/sharpening augmentation)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "pdf.fonttype": 42,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
spec = importlib.util.spec_from_file_location("e1", ROOT / "scripts" / "e1_dose_robustness.py")
e1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e1)
MM = 1 / 25.4


def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", OUT / f"{name}.pdf")


def clean_axes(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)


# --------------------------------------------------------------------------------------------- figure 1
def fig1():
    nod = pd.read_csv(AN / "e5_misses" / "all_nodules_hit_status.csv")
    n = nod[(nod.fold == 0) & nod.hit & nod.diameter_mm.between(7, 10)].sort_values("diameter_mm").iloc[0]
    img, meta = load_full_volume(n.seriesuid, int(n.subset))
    pn, pmeta = load_volume(C.THESIS_DATA / "luna16" / "projnoise" / "a0.1" / f"subset{int(n.subset)}" / n.seriesuid)
    assert tuple(pn.shape) == tuple(img.shape[2:]) and pmeta["crop_offset_zyx"] == meta["crop_offset_zyx"]
    z, y, x = np.round(world_to_voxel(meta, [n.coordX, n.coordY, n.coordZ])[0]).astype(int)
    sp = np.array(meta["spacing_zyx"]); h = int(30 / sp[1])                      # 60 mm field of view
    sl = (slice(None), slice(None), slice(max(z - 8, 0), z + 9), slice(y - h - 8, y + h + 8), slice(x - h - 8, x + h + 8))
    sub = img[sl]; zc = min(z, 8)
    to_hu = lambda t: t[0, 0, zc, 8:-8, 8:-8].numpy() * 1400 - 1000
    clean = to_hu(sub)
    allc = {**e1.CONDITIONS, **e1.CONDITIONS_EXTRA}
    def pert(k):
        c = allc[k]; return to_hu(e1.perturb(sub.clone(), c[0], c[1], c[2], seed=1234, corr=c[3] if len(c) > 3 else 0.0))
    proj = pn[max(z - 8, 0):z + 9, y - h - 8:y + h + 8, x - h - 8:x + h + 8][zc, 8:-8, 8:-8].astype(np.float32)
    tx_, txm = load_volume(C.THESIS_DATA / "luna16" / "texture" / "tex_nps" / f"subset{int(n.subset)}" / n.seriesuid)
    assert tuple(tx_.shape) == tuple(img.shape[2:]) and txm["crop_offset_zyx"] == meta["crop_offset_zyx"]
    tex = tx_[max(z - 8, 0):z + 9, y - h - 8:y + h + 8, x - h - 8:x + h + 8][zc, 8:-8, 8:-8].astype(np.float32)
    top = [("clean", clean), ("white noise\n80 HU", pert("noise80")), ("correlated\nnoise 80 HU,\n1.4 mm grain", pert("cnoise80_c2")),
           ("projection-\ndomain noise,\n10 % dose", proj), (f"3D Gaussian blur\nσ {1.5 * C.LUNA_SPACING[1]:.2f} mm in-plane", pert("soft1.5")),
           ("unsharp\nmask 1.5", pert("sharp1.5"))]
    # real reference noise: TCIA 10 % - full dose of one patient (same detector grid and window), lesion C012 L1
    T = C.THESIS_DATA / "tcia_ldct" / "preprocessed"
    tf, tm = load_volume(T / "C012_full"); tl, _ = load_volume(T / "C012_low")
    les = pd.read_csv(C.THESIS_DATA / "tcia_ldct" / "eval" / "lesions_checked.csv").query("case == 'C012' and lesion == 1").iloc[0]
    tz, ty, tx = np.round(world_to_voxel(tm, [les.coordX, les.coordY, les.coordZ])[0]).astype(int)
    tdiff = (tl[tz, ty - h:ty + h, tx - h:tx + h].astype(np.float32) - tf[tz, ty - h:ty + h, tx - h:tx + h])
    bottom = [("white\n80 HU", top[1][1] - clean), ("correlated\n80 HU, 1.4 mm", top[2][1] - clean),
              ("projection-domain,\n10 % dose", proj - clean), ("stationary noise,\nreal TCIA NPS", tex - clean), ("TCIA real pair:\n10 % − full dose", tdiff)]
    fig = plt.figure(figsize=(183 * MM, 82 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.38, wspace=0.06)
    r = n.diameter_mm / 2 / sp[2]
    for i, (lab, im) in enumerate(top):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(im, cmap="gray", vmin=-1000, vmax=200, interpolation="nearest")
        ax.add_patch(plt.Circle((h, h), r * 1.7, fill=False, color="#3d9a3a", lw=0.9)); ax.set_title(lab, fontsize=7); ax.axis("off")
        if i == 0: ax.text(-0.08, 1.25, "a", transform=ax.transAxes, fontweight="bold", fontsize=9)
    for i, (lab, im) in enumerate(bottom):
        ax = fig.add_subplot(gs[1, i + 1]); ax.imshow(im, cmap="gray", vmin=-300, vmax=300, interpolation="nearest")
        ax.set_title(lab, fontsize=7); ax.axis("off")
        if i == 0: ax.text(-0.08, 1.25, "b", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    ax.text(0.0, 0.5, "noise only\n(perturbed − clean)\nwindow ±300 HU\n\n60 mm field of view,\nlung window\n[−1000, 200] HU (a)",
            fontsize=6.5, color=MUTED, va="center")
    save(fig, "fig1_perturbations"); return n


# --------------------------------------------------------------------------------------------- figure 2
RV = AN / "p2_review"; XY = C.LUNA_SPACING[1]                       # in-plane voxel size (mm); E1 blur sigma is in voxels


def level_panel(ax):
    """e: noise texture x level. CPM loss from clean (95 % CI) for three stationary textures at 0.5, 1 and 2 times the per-scan
    level of projection noise at 10 % dose, inserted on the same path; projection noise itself at x1 for reference."""
    t = pd.read_csv(RV / "levels_contrasts.csv"); t = t[t.kind.str.startswith("drop ")]
    p = pd.read_csv(AN / "p2_texture" / "texture_contrasts.csv"); p = p[(p.subset == "all") & (p.kind == "drop proj0.1")]
    sty = {"tex_nps": ("-", "o", "real NPS"), "tex_white": ("--", "s", "white"), "tex_gauss": (":", "^", "Gaussian grain")}
    for j, m in enumerate(("S3", "S4")):
        off = 1 + (j - 0.5) * 0.06
        for c, (ls, mk, lab) in sty.items():
            r = [t[(t.kind == f"drop {c}_x{f:g}") & (t.model == m)].iloc[0] for f in (0.5, 1, 2)]
            x = np.array([0.5, 1, 2]) * off; y = np.array([-q.estimate for q in r])
            ax.plot(x, y, ls=ls, color=COL[m], lw=1.3)
            ax.errorbar(x, y, yerr=[y - np.array([-q.ci_hi for q in r]), np.array([-q.ci_lo for q in r]) - y], fmt=mk, color=COL[m],
                        ms=3.6, mec="white", mew=0.6, lw=0.9, capsize=0)
        q = p[p.model == m].iloc[0]
        ax.errorbar([off], [-q.estimate], yerr=[[-q.estimate + q.ci_hi], [-q.ci_lo + q.estimate]], fmt="*", color=COL[m], ms=6, mec=INK, mew=0.4, lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.01, linscale=0.6); ax.minorticks_off()   # linear below 0.01: keeps near-zero losses and negative CI ends
    ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["×0.5", "×1", "×2"]); ax.set_xlim(0.4, 2.5)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
    ax.set_ylim(-0.02, 1.2); ax.set_yticks([0, 0.01, 0.1, 1]); ax.set_yticklabels(["0", "0.01", "0.1", "1"])
    ax.set_xlabel("noise level, relative to projection noise at 10 % dose (per scan)"); ax.set_ylabel("CPM loss from clean\n(log above 0.01)")
    clean_axes(ax, grid="y")
    for c, (ls, mk, lab) in sty.items():
        ax.plot([], [], ls=ls, marker=mk, color=MUTED, lw=1.2, ms=3.6, label=lab)
    ax.plot([], [], "*", color=MUTED, ms=6, label="projection noise, 10 % dose"); ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    ax.set_title("e  Noise texture and level, same insertion path", loc="left", fontweight="bold")


def fig2():
    cp = pd.read_csv(POOL / "final_cpm.csv", index_col=0)
    def g(m, c): r = cp.loc[f"{m}|{c}"]; return r.cpm, r.ci_lo, r.ci_hi
    fig = plt.figure(figsize=(183 * MM, 105 * MM))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1, 1, 1], height_ratios=[1.35, 0.85], hspace=0.62, wspace=0.12)
    axs = [fig.add_subplot(gs[0, 0])]
    axs += [fig.add_subplot(gs[0, k], sharey=axs[0]) for k in (1, 2, 3)]
    axs.append(fig.add_subplot(gs[1, 0:3]))
    def line(ax, xs, conds, m, ls="-", marker="o", label=None, alpha_band=0.12):
        v = np.array([g(m, c) for c in conds])
        ax.fill_between(xs, v[:, 1], v[:, 2], color=COL[m], alpha=alpha_band, lw=0)
        ax.plot(xs, v[:, 0], ls=ls, color=COL[m], lw=1.6, marker=marker, ms=4, mec="white", mew=0.8, label=label)
    for m in ("S3", "S4"):
        line(axs[0], [0, 20, 40, 60, 80, 100], ["clean", "noise20", "noise40", "noise60", "noise80", "noise100"], m, "-", "o", alpha_band=0.10)
        line(axs[0], [0, 40, 80], ["clean", "cnoise40_c1", "cnoise80_c1"], m, "--", "s", alpha_band=0.0)
        line(axs[0], [0, 40, 80], ["clean", "cnoise40_c2", "cnoise80_c2"], m, ":", "^", alpha_band=0.10)
        line(axs[1], [100, 50, 25, 10], ["clean", "proj0.5", "proj0.25", "proj0.1"], m)
        line(axs[2], [0, 0.5 * XY, 1.0 * XY, 1.5 * XY], ["clean", "soft0.5", "soft1.0", "soft1.5"], m)
        ij = json.load(open(RV / "inplane_results.json")); cal = ij["xycal_mm"]
        axs[2].plot([0, cal, XY, 1.5 * XY], [g(m, "clean")[0]] + [ij["cpm"][f"{m}|{c}"] for c in ("xycal", "xy1.0", "xy1.5")],
                    ls="--", color=COL[m], lw=1.3, marker="D", ms=3.2, mec="white", mew=0.6)
        line(axs[3], [0, 0.5, 1.0, 1.5], ["clean", "sharp0.5", "sharp1.0", "sharp1.5"], m)
    axs[0].text(101, g("S3", "noise100")[0] + 0.012, "white", fontsize=6.5, color=MUTED, ha="right")
    axs[0].text(84, g("S4", "cnoise80_c1")[0], "corr.\n0.7 mm", fontsize=6.5, color=MUTED, ha="left", va="center")
    axs[0].text(84, g("S3", "cnoise80_c2")[0] + 0.02, "corr.\n1.4 mm", fontsize=6.5, color=MUTED, ha="left", va="center")
    axs[1].set_xscale("log"); axs[1].minorticks_off(); axs[1].set_xticks([100, 50, 25, 10]); axs[1].set_xticklabels(["100", "50", "25", "10"]); axs[1].invert_xaxis()
    titles = ["a  Image-domain noise", "b  Projection noise", "c  Gaussian blur", "d  Unsharp masking"]
    xl = ["nominal added noise SD (HU)", "dose (% of reference)", "blur σ in-plane (mm)", "unsharp-mask strength"]
    for ax in axs[1:4]: ax.tick_params(labelleft=False)
    level_panel(axs[4])
    axs[2].axvline(cal, color=MUTED, lw=0.8, ls="-."); axs[2].text(cal, 0.57, " NLST\n soft vs\n sharp", fontsize=6, color=MUTED, ha="left", va="bottom")
    axs[2].text(1.5 * XY, g("S3", "soft1.5")[0] - 0.035, "3D", fontsize=6.5, color=MUTED, ha="right")
    axs[2].text(0.86, 0.935, "in-plane\nonly (- -)", fontsize=6.5, color=MUTED, ha="center")
    for ax, t, x_ in zip(axs[:4], titles, xl):
        ax.set_title(t, loc="left", fontweight="bold"); ax.set_xlabel(x_); clean_axes(ax)
    axs[2].axvspan(0.3 * XY, 1.0 * XY, color=GRID, alpha=0.45, lw=0); axs[3].axvspan(0.3, 1.0, color=GRID, alpha=0.45, lw=0)
    axs[0].axvspan(10, 60, color=GRID, alpha=0.45, lw=0)
    axs[0].set_ylabel("CPM (356 test scans)"); axs[0].set_ylim(0.55, 0.96)
    h_, l_ = [plt.Line2D([], [], color=COL[m], lw=1.6, marker="o", ms=4, mec="white") for m in ("S3", "S4")], [NAME["S3"], NAME["S4"]]
    h_.append(plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.45)); l_.append("S4 training range")
    fig.legend(h_, l_, loc="center left", ncol=1, frameon=False, bbox_to_anchor=(0.765, 0.24))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.11)
    pe = axs[4].get_position(); axs[4].set_position([0.20, pe.y0, 0.52, pe.height]); save(fig, "fig2_cpm_curves")


# --------------------------------------------------------------------------------------------- figure 3
LAB = {"clean": "clean", "noise20": "white 20 HU", "noise40": "white 40 HU", "noise60": "white 60 HU", "noise80": "white 80 HU",
       "noise100": "white 100 HU", **{f"soft{v}": f"3D blur σ {float(v) * C.LUNA_SPACING[1]:.2f} mm" for v in ("0.5", "1.0", "1.5")},
       "sharp0.5": "unsharp 0.5", "sharp1.0": "unsharp 1.0", "sharp1.5": "unsharp 1.5", "sharp1.0+noise60": "unsharp 1.0 + white 60 HU",
       "soft1.0+noise60": f"3D blur σ {C.LUNA_SPACING[1]:.2f} mm + white 60 HU", "cnoise40_c1": "correlated 40 HU, 0.7 mm", "cnoise80_c1": "correlated 80 HU, 0.7 mm",
       "cnoise40_c2": "correlated 40 HU, 1.4 mm", "cnoise80_c2": "correlated 80 HU, 1.4 mm †", "proj0.5": "projection 50 % dose",
       "proj0.25": "projection 25 % dose", "proj0.1": "projection 10 % dose"}


def fig3():
    d = pd.read_csv(POOL / "final_delta.csv").set_index("condition")
    groups = [("F1 blur (primary)", ["soft1.0", "soft1.5", "soft1.0+noise60"]),
              ("F2 realistic noise (primary)", ["cnoise40_c1", "cnoise80_c1", "cnoise40_c2", "cnoise80_c2", "proj0.5", "proj0.25", "proj0.1"]),
              ("Secondary", ["clean", "noise20", "noise40", "noise60", "noise80", "noise100", "soft0.5", "sharp0.5", "sharp1.0", "sharp1.5", "sharp1.0+noise60"])]
    rows = []; ylab = []; y = 0; heads = []
    for gname, cs in groups:
        heads.append((y, gname)); y -= 1
        for c in cs: rows.append((y, c)); y -= 1
        y -= 0.4
    fig, ax = plt.subplots(figsize=(120 * MM, 120 * MM))
    for yy, c in rows:
        r = d.loc[c]; prim = not pd.isna(r.p_holm); sig = prim and r.p_holm < 0.05
        col = INK if sig else MUTED
        ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(r.estimate, yy, "o" if sig else ("o" if prim else "D"), ms=4.5 if prim else 3.5, color=col if sig else "white",
                mec=col, mew=1.1, zorder=3)
        if prim:
            pt = f"P$_{{Holm}}$ = {r.p_holm:.4f}" if r.p_holm < 0.01 else f"P$_{{Holm}}$ = {r.p_holm:.2f}"
            ax.text(0.128, yy, pt, va="center", fontsize=6.5, color=col)
    for yy, gname in heads: ax.text(-0.058, yy, gname, fontweight="bold", fontsize=7.5, va="center", color=INK)
    ax.set_yticks([yy for yy, _ in rows]); ax.set_yticklabels([LAB[c] for _, c in rows])
    ax.axvline(0, color=MUTED, lw=0.8, ls="--"); ax.axvspan(-0.02, 0.02, color=GRID, alpha=0.35, lw=0)
    ax.set_xlim(-0.06, 0.17); ax.set_xlabel("ΔCPM, S4 − S3 (paired bootstrap 95 % CI)")
    clean_axes(ax, "x"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "fig3_delta_forest")


# --------------------------------------------------------------------------------------------- figure 4
def fig4():
    tc = pd.read_csv(AN / "p2_tcia" / "tcia_cpm.csv", index_col=0).cpm
    con = pd.read_csv(AN / "p2_tcia" / "tcia_contrasts.csv"); cons = pd.read_csv(AN / "p2_tcia" / "tcia_contrasts_solid.csv")
    fig, axs = plt.subplots(1, 2, figsize=(160 * MM, 62 * MM), gridspec_kw={"width_ratios": [0.8, 1.2]})
    ax = axs[0]
    for m, off in (("S3", -0.06), ("S4", 0.06)):
        for k in range(4):
            a, b = tc[f"{m}_f{k}_full"], tc[f"{m}_f{k}_low"]
            ax.plot([0 + off, 1 + off], [a, b], color=COL[m], lw=1.0, alpha=0.75, marker="o", ms=3.5, mec="white", mew=0.6)
        mf = np.mean([tc[f"{m}_f{k}_full"] for k in range(4)]); ml = np.mean([tc[f"{m}_f{k}_low"] for k in range(4)])
        ax.plot([0 + off, 1 + off], [mf, ml], color=COL[m], lw=2.6, label=NAME[m].split(" (")[0] + " (mean of 4 folds)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["full dose", "10 % dose"]); ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(f"CPM (49 patients, 34 lesions)"); ax.set_title("a  Each fold model, full vs 10 % dose", loc="left", fontweight="bold")
    clean_axes(ax); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.8)
    ax = axs[1]
    items = [("ΔΔ = S4 change − S3 change, primary", "DD (S4 loses less than S3, mean of 4 pairs)"),
             ("ΔCPM S4 − S3 at 10 %, primary", "D_low S4-S3 at 10 %"), ("ΔCPM S4 − S3 at full dose", "D_full S4-S3 at full dose"),
             ("S3 change, full → 10 %", "S3 drop full->10 % (mean of 4)"), ("S4 change, full → 10 %", "S4 drop full->10 % (mean of 4)")]
    for i, (lab, key) in enumerate(items):
        for df, off, mk, col in ((con, 0.13, "o", INK), (cons, -0.13, "s", MUTED)):
            r = df[df.contrast == key].iloc[0]; yy = -i + off
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=1.5); ax.plot(r.estimate, yy, mk, color=col, ms=4, mec="white", mew=0.7)
    ax.set_yticks([-i for i in range(len(items))]); ax.set_yticklabels([l for l, _ in items])
    ax.axvline(0, color=MUTED, lw=0.8, ls="--"); ax.set_xlabel("CPM difference (bootstrap over patients, 95 % CI)")
    ax.set_title("b  Paired contrasts (mean of 4 fold pairs)", loc="left", fontweight="bold")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    clean_axes(ax, "x"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend([plt.Line2D([], [], color=INK, marker="o", lw=1.5), plt.Line2D([], [], color=MUTED, marker="s", lw=1.5)],
              ["all 34 lesions", "22 solid lesions"], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.8)
    fig.tight_layout(); save(fig, "fig4_tcia")


# --------------------------------------------------------------------------------------------- figure 5
def fig5():
    """NLST real-kernel pairs: a agreement of confident detections per fold model; b S4 - S3 by vendor; c calibrated blur."""
    res = pd.read_csv(RV / "nlst_pairs_results.csv").set_index("quantity"); nv = pd.read_csv(RV / "nlst_pairs_vendor.csv")
    cal = pd.read_csv(RV / "nlst_pairs" / "kernel_calibration.csv", dtype={"pid": str})
    ok = pd.read_csv(RV / "nlst_pairs" / "geometry_check.csv", dtype={"pid": str}); cal = cal[cal.pid.isin(set(ok[ok.geometry_ok].pid))]
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 64 * MM), gridspec_kw={"width_ratios": [0.9, 1.15, 1.1]})
    ax = axs[0]
    for k in range(4):
        a = [res.loc[f"agreement {m}_f{k}", "estimate"] for m in ("S3", "S4")]
        ax.plot([0, 1], a, color=MUTED, lw=0.8, alpha=0.8); ax.text(1.06, a[1], f"fold {k}", fontsize=6, color=MUTED, va="center")
    for i, m in enumerate(("S3", "S4")):
        r = res.loc[f"agreement {m} (mean)"]
        ax.errorbar([i], [r.estimate], yerr=[[r.estimate - r.ci_lo], [r.ci_hi - r.estimate]], fmt="o", color=COL[m], ms=5, mec="white", mew=0.8, lw=1.6, capsize=0, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["S3", "S4"]); ax.set_xlim(-0.4, 1.5)
    ax.set_ylabel("agreement, soft vs sharp kernel"); clean_axes(ax)
    ax.set_title("a  Same exam, two real kernels", loc="left", fontweight="bold")
    ax = axs[1]; rows = nv.copy(); y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows.itertuples()):
        col = INK if r.vendor == "all" else MUTED
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round"); ax.plot(r.delta, yi, "o", color=col, ms=4.5, mec="white", mew=1.0, zorder=3)
        ax.plot([getattr(r, f"delta_f{k}") for k in range(4)], [yi] * 4, "|", color=COL["S4"], ms=6, mew=1.0, alpha=0.8)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--"); ax.set_yticks(y)
    ax.set_yticklabels([("All exams" if r.vendor == "all" else ("GE" if r.vendor == "GE" else r.vendor.title())) + f" (n = {r.n_exams})" for r in rows.itertuples()])
    ax.set_xlabel("Δ agreement (S4 − S3)"); clean_axes(ax, "x"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("b  Effect of augmentation", loc="left", fontweight="bold")
    ax = axs[2]; bins = np.arange(-0.025, 1.226, 0.05)
    ax.hist(cal.sigma_mm, bins=bins, color=GRID, edgecolor=MUTED, lw=0.5)
    ax.axvspan(0.3 * XY, 1.0 * XY, color=COL["S4"], alpha=0.10, lw=0)
    top = ax.get_ylim()[1]
    for x_, lab, yy in ((cal.sigma_mm.median(), "median", 0.97), (1.0 * XY, "tested\nin-plane", 0.72), (1.5 * XY, "tested\nin-plane", 0.97)):
        ax.axvline(x_, color=INK if lab == "median" else MUTED, lw=0.9, ls="-" if lab == "median" else "--")
        ax.text(x_ + 0.015, top * yy, f"{lab}\n{x_:.2f} mm", fontsize=6, color=INK if lab == "median" else MUTED, va="top")
    ax.text(0.3 * XY + 0.01, top * 0.45, "S4\ntraining\nrange", fontsize=6, color=COL["S4"], va="top")
    ax.set_xlabel("in-plane blur σ that turns sharp into soft (mm)"); ax.set_ylabel("exams"); ax.set_xlim(0, 1.2); clean_axes(ax)
    ax.set_title("c  Real kernel difference as blur", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=2.0); save(fig, "fig5_nlst_kernels")


if __name__ == "__main__":
    with torch.no_grad():
        n = fig1()
    print("fig1 nodule:", n.seriesuid, f"{n.diameter_mm:.1f} mm")
    fig2(); fig3(); fig4(); fig5()
