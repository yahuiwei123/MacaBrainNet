#!/usr/bin/env python3
"""
Generate publication-quality figure for OOD evaluation results.
Panel A: Per-class Dice (bars + per-sample dots + error bars)
Panel B: Per-class Hausdorff95 (bars + per-sample dots + error bars)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches

# ── Load results ──
with open("ood_results.json", "r") as f:
    data = json.load(f)

fg_names = data["label_names"][1:]  # skip Background
fg_meta = [data["per_class"][n] for n in fg_names]

dice_mean = np.array([m["dice_mean"] for m in fg_meta])
dice_std  = np.array([m["dice_std"]  for m in fg_meta])
hd95_mean = np.array([m["hd95_mean_mm"] for m in fg_meta])
hd95_std  = np.array([m["hd95_std_mm"]  for m in fg_meta])

# ── Per-sample data for scatter overlay ──
per_sample = data["per_sample"]
dice_per_sample = np.array([
    [s["dice_per_class"][n] for n in fg_names]
    for s in per_sample
])  # [N_samples, N_classes]

hd95_per_sample = np.array([
    [s["hd95_per_class"][n] if s["hd95_per_class"][n] is not None else np.nan
     for n in fg_names]
    for s in per_sample
])

# ── Sort by Dice descending ──
order = np.argsort(dice_mean)[::-1]
names_sorted      = [fg_names[i] for i in order]
dice_mean_sorted  = dice_mean[order]
dice_std_sorted   = dice_std[order]
hd95_mean_sorted  = hd95_mean[order]
hd95_std_sorted   = hd95_std[order]
dice_ps_sorted    = dice_per_sample[:, order]    # [N, C]
hd95_ps_sorted    = hd95_per_sample[:, order]

n = len(names_sorted)
N_samples = dice_ps_sorted.shape[0]
y_pos = np.arange(n)

# ── Anatomical groupings (color coding) ──
group_def = {
    "Cortical":          ["Cerebral WM", "Cerebral Cortex"],
    "Subcortical":       ["Thalamus Proper", "Caudate", "Putamen",
                          "Pallidum", "Hippocampus", "Amygdala",
                          "Accumbens Area", "Substantia Nigra",
                          "Ventral Diencephalon", "Claustrum"],
    "Cerebellum":        ["Cerebellum WM", "Cerebellum Cortex"],
    "Brainstem / Other": ["Brain Stem", "Lateral Ventricle", "CSF", "Cornea"],
}
group_colors = {
    "Cortical":          "#2166AC",
    "Subcortical":       "#B2182B",
    "Cerebellum":        "#4DAF4A",
    "Brainstem / Other": "#E67E00",
}
class_group = {}
for grp, members in group_def.items():
    for m in members:
        class_group[m] = grp
bar_colors = [group_colors[class_group[n]] for n in names_sorted]

# ── Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 400,
})

fig = plt.figure(figsize=(13, 5.8))

# Use GridSpec for manual layout control — left=0.13 ensures y-tick labels are not clipped
gs = fig.add_gridspec(1, 2, left=0.13, right=0.97, top=0.89, bottom=0.12,
                       wspace=0.42)

ax_dice = fig.add_subplot(gs[0, 0])
ax_hd   = fig.add_subplot(gs[0, 1])

bar_height = 0.6
jitter_scale = 0.08  # vertical jitter for dots
dot_size = 4.5
dot_alpha = 0.45

# ═══════════════════════════════════════════
# Panel A: Dice Score
# ═══════════════════════════════════════════
# Per-sample scatter (light dots behind bars)
rng = np.random.default_rng(42)  # deterministic jitter
for i in range(n):
    vals = dice_ps_sorted[:, i]
    vals = vals[~np.isnan(vals)]
    jitter = rng.uniform(-jitter_scale, jitter_scale, size=len(vals))
    ax_dice.scatter(vals, y_pos[i] + jitter,
                    s=dot_size, alpha=dot_alpha, facecolors="0.3",
                    edgecolors="none", zorder=2, linewidths=0)

# Bars with error
ax_dice.barh(y_pos, dice_mean_sorted, xerr=dice_std_sorted,
             color=bar_colors, edgecolor="white", linewidth=0.5,
             error_kw={"ecolor": "0.15", "capsize": 2.5, "linewidth": 1.0,
                       "capthick": 0.8},
             height=bar_height, zorder=3, alpha=0.92)

# Overall mean
overall_dice = data["overall_dice_mean"]
ax_dice.axvline(overall_dice, color="0.15", linestyle=(0, (4, 2.5)), linewidth=1.0, alpha=0.8)
ax_dice.text(overall_dice + 0.003, n - 0.25, f"Mean = {overall_dice:.3f}",
             fontsize=7.5, color="0.15", va="bottom", style="italic",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

# Value labels
for i in range(n):
    ax_dice.text(dice_mean_sorted[i] + dice_std_sorted[i] + 0.006, y_pos[i],
                 f"{dice_mean_sorted[i]:.3f}",
                 va="center", fontsize=5.8, color="0.2", zorder=5)

ax_dice.set_yticks(y_pos)
ax_dice.set_yticklabels(names_sorted, fontsize=8)
ax_dice.set_xlabel("Dice Score", fontsize=9)
ax_dice.set_title("A   Per-class Dice Score", loc="left", fontsize=10, fontweight="bold")
ax_dice.set_xlim(0.70, 0.975)
ax_dice.xaxis.set_major_locator(MultipleLocator(0.05))
ax_dice.xaxis.set_minor_locator(MultipleLocator(0.01))
ax_dice.tick_params(axis="x", labelsize=7.5)
ax_dice.grid(axis="x", alpha=0.2, linewidth=0.4)
ax_dice.set_axisbelow(True)
ax_dice.invert_yaxis()
ax_dice.set_ylim(n - 0.55, -0.35)

# Legend
legend_patches = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
leg = ax_dice.legend(handles=legend_patches, loc="lower right", framealpha=0.85,
                     edgecolor="0.7", fontsize=6.5, ncol=2,
                     title="Anatomical group", title_fontsize=6.5)
leg.set_zorder(10)

# ═══════════════════════════════════════════
# Panel B: Hausdorff95 Distance
# ═══════════════════════════════════════════
# Per-sample scatter
rng2 = np.random.default_rng(123)  # independent seed
for i in range(n):
    vals = hd95_ps_sorted[:, i]
    vals = vals[~np.isnan(vals)]
    jitter = rng2.uniform(-jitter_scale, jitter_scale, size=len(vals))
    ax_hd.scatter(vals, y_pos[i] + jitter,
                  s=dot_size, alpha=dot_alpha, facecolors="0.3",
                  edgecolors="none", zorder=2, linewidths=0)

ax_hd.barh(y_pos, hd95_mean_sorted, xerr=hd95_std_sorted,
           color=bar_colors, edgecolor="white", linewidth=0.5,
           error_kw={"ecolor": "0.15", "capsize": 2.5, "linewidth": 1.0,
                     "capthick": 0.8},
           height=bar_height, zorder=3, alpha=0.92)

overall_hd = data["overall_hd95_mean"]
ax_hd.axvline(overall_hd, color="0.15", linestyle=(0, (4, 2.5)), linewidth=1.0, alpha=0.8)
ax_hd.text(overall_hd + 0.015, n - 0.25, f"Mean = {overall_hd:.2f} mm",
           fontsize=7.5, color="0.15", va="bottom", style="italic",
           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

for i in range(n):
    ax_hd.text(hd95_mean_sorted[i] + hd95_std_sorted[i] + 0.03, y_pos[i],
               f"{hd95_mean_sorted[i]:.2f}",
               va="center", fontsize=5.8, color="0.2", zorder=5)

ax_hd.set_yticks(y_pos)
ax_hd.set_yticklabels(names_sorted, fontsize=8)
ax_hd.set_xlabel("Hausdorff95 Distance (mm)", fontsize=9)
ax_hd.set_title("B   Per-class Hausdorff95 Distance", loc="left", fontsize=10, fontweight="bold")
ax_hd.set_xlim(0.0, max(hd95_mean_sorted) * 1.28)
ax_hd.xaxis.set_major_locator(MultipleLocator(0.2))
ax_hd.xaxis.set_minor_locator(MultipleLocator(0.05))
ax_hd.tick_params(axis="x", labelsize=7.5)
ax_hd.grid(axis="x", alpha=0.2, linewidth=0.4)
ax_hd.set_axisbelow(True)
ax_hd.invert_yaxis()
ax_hd.set_ylim(n - 0.55, -0.35)

# ── Global title ──
fig.suptitle("OOD Generalization Performance — 5-fold SwinUNETR Ensemble (32 scans, 4 held-out sites)",
             fontsize=11, fontweight="bold", y=0.975)

# ── Footer ──
fig.text(0.5, 0.02,
         "OOD sites: ds001875 (n=9), ds003989 (n=13), ds004620 (n=8), ds005521 (n=2)  |  "
         "0.4 mm isotropic  |  T1w brain-extracted  |  "
         "Per-sample dots shown behind bars (± jitter)",
         ha="center", fontsize=6.5, color="0.4", style="italic")

fig.savefig("ood_results_figure.png", dpi=400, facecolor="white", edgecolor="none")
fig.savefig("ood_results_figure.pdf", facecolor="white", edgecolor="none")
print("Saved: ood_results_figure.png / ood_results_figure.pdf")
plt.close()
