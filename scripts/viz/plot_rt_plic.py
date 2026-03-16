#!/usr/bin/env python3
"""
Plot Rayleigh-Taylor instability: PLIC-VOF + TRT + Guo forcing.

Reads CSV snapshots from rt_plic_data/ and renders a multi-panel figure.
No post-processing smoothing — raw fill level data is shown as-is
to display PLIC's geometric sharpness.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

DIR = Path(__file__).parent
DATA_DIR = DIR / "rt_plic_data"
OUT_PATH = DIR / "rt_plic.png"

# Must match viz_rt_plic.cu
NX, NY = 128, 512
DX = 1e-4  # m/cell
TAU = 0.6
NU_LB = (TAU - 0.5) / 3.0
NU_PHYS = 5e-6
DT = NU_LB * DX**2 / NU_PHYS
AT = 0.333

SNAP_INTERVAL = 2000
STEPS = [0, 2000, 4000, 6000, 8000]

# ---- Colormap: sharp blue/red with narrow white interface ----
_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rt_sharp",
    [(0.0,  "#0a2463"),   # deep blue  (light fluid, f=0)
     (0.40, "#3e92cc"),
     (0.49, "#e8e8e8"),
     (0.51, "#e8e8e8"),
     (0.60, "#d63230"),
     (1.0,  "#6b0504")],  # dark red   (heavy fluid, f=1)
)


def load_csv(path):
    """Load 2D CSV → float array. Row j = lattice y=j."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split(",")])
    return np.array(rows, dtype=np.float32)


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} not found. Run viz_rt_plic first.")
        sys.exit(1)

    panels = []
    for step in STEPS:
        fname = DATA_DIR / f"rt_step{step:04d}.csv"
        if fname.exists():
            panels.append((step, load_csv(fname)))
        else:
            print(f"WARNING: {fname} not found, skipping")

    if not panels:
        print("No data found!")
        sys.exit(1)

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 10),
                             gridspec_kw={"wspace": 0.05})
    if n_panels == 1:
        axes = [axes]

    # Crop to interface region (with generous margin)
    all_data = np.concatenate([d for _, d in panels], axis=0)
    row_range = np.ptp(all_data, axis=1)
    mixed = np.where(row_range > 0.02)[0]
    if len(mixed) > 0:
        y0 = max(0, mixed.min() - 20)
        y1 = min(NY, mixed.max() + 20)
    else:
        y0, y1 = 0, NY

    extent_x = [0, NX * DX * 100]  # cm
    extent_y = [y0 * DX * 100, y1 * DX * 100]  # cm

    for ax, (step, data) in zip(axes, panels):
        view = data[y0:y1, :]
        t_phys = step * DT

        im = ax.imshow(
            view, origin="lower", aspect="equal",
            cmap=_CMAP, vmin=0.0, vmax=1.0,
            extent=[extent_x[0], extent_x[1], extent_y[0], extent_y[1]],
            interpolation="nearest",  # NO smoothing — show raw PLIC cells
        )

        # f=0.5 contour
        y_coords = np.linspace(extent_y[0], extent_y[1], view.shape[0])
        x_coords = np.linspace(extent_x[0], extent_x[1], view.shape[1])
        ax.contour(x_coords, y_coords, view, levels=[0.5],
                   colors=["white"], linewidths=0.6, alpha=0.8)

        ax.set_title(f"t = {t_phys:.2f} s", fontsize=10, pad=4)
        ax.set_xlabel("x [cm]", fontsize=8)
        ax.tick_params(labelsize=7)

        if ax is not axes[0]:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("y [cm]", fontsize=8)

    fig.suptitle(
        f"Rayleigh-Taylor Instability  (At = {AT:.2f},  PLIC-VOF + TRT + Guo)",
        fontsize=12, fontweight="bold", y=0.98)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Fill level f", fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=7)

    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
