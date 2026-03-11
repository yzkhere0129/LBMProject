#!/usr/bin/env python3
"""Fair comparison: TVD-MC vs PLIC, both with sharp init, same Zalesak benchmark."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

D = Path("/home/yzk/LBMProject/scripts/viz")
NX, NY = 128, 128

def load(name):
    with (D / name).open() as f:
        return np.array([list(map(float, l.split(","))) for l in f if l.strip()], dtype=np.float32)

cmap = mcolors.LinearSegmentedColormap.from_list("d", [
    (0, "#f5f5f5"), (0.45, "#a8d8ea"), (0.55, "#3a7fc1"), (1, "#1a3a6c")])

init = load("cmp_tvd_deg000.csv")
tvd  = load("cmp_tvd_deg360.csv")
plic = load("cmp_plic_deg360.csv")

m0 = init.sum()
tvd_err  = (m0 - tvd.sum())  / m0 * 100
plic_err = (m0 - plic.sum()) / m0 * 100

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="white",
                          gridspec_kw={"wspace": 0.12})
fig.suptitle("Zalesak Slotted-Disk  —  TVD-MC vs PLIC-VOF  (128×128, sharp init, 800 steps)",
             fontsize=14, fontweight="bold", y=0.98)

xs, ys = np.linspace(0, NX, NX), np.linspace(0, NY, NY)

for ax, data, title in [(axes[0], init, "0° initial (sharp)"),
                          (axes[1], tvd,  "360° TVD-MC"),
                          (axes[2], plic, "360° PLIC")]:
    ax.set_facecolor("#f5f5f5")
    ax.imshow(data, origin="lower", cmap=cmap, vmin=0, vmax=1,
              extent=[0, NX, 0, NY], interpolation="none")
    ax.contour(xs, ys, data, levels=[0.5], colors=["white"], linewidths=1.4)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("x [cells]", fontsize=10)
    ax.set_ylabel("y [cells]", fontsize=10)
    ax.set_aspect("equal")
    ax.plot(64, 64, "+", color="#ff4444", ms=10, mew=2, zorder=5)

# Overlay initial contour (orange dashed) on result panels
for ax in [axes[1], axes[2]]:
    ax.contour(xs, ys, init, levels=[0.5], colors=["#ff8800"],
               linewidths=1.2, linestyles="--", alpha=0.85)

axes[1].text(0.97, 0.03, f"mass loss: {tvd_err:.4f}%",
             transform=axes[1].transAxes, ha="right", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#ccc"))
axes[2].text(0.97, 0.03, f"mass loss: {plic_err:.4f}%",
             transform=axes[2].transAxes, ha="right", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#ccc"))

axes[2].legend(handles=[
    mpatches.Patch(fc="none", ec="#ff8800", linestyle="--", lw=1.5, label="initial (0°)"),
    mpatches.Patch(fc="none", ec="white", lw=1.5, label="final (360°)"),
], loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#ccc")

fig.savefig(D / "zalesak_plic_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {D / 'zalesak_plic_comparison.png'}")
print(f"  TVD  mass loss: {tvd_err:.4f}%")
print(f"  PLIC mass loss: {plic_err:.4f}%")
