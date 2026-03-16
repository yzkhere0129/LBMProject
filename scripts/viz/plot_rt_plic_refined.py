#!/usr/bin/env python3
"""
Plot Rayleigh-Taylor instability: PLIC-VOF + TRT + Guo + CSF.

Textbook-quality figure:
  - Equal aspect ratio (no distortion)
  - Select best frames: early growth → mushroom → just before wall impact
  - Sharp colormap, f=0.5 contour
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

DIR = Path(__file__).parent
DATA_DIR = DIR / "rt_plic_data"
OUT_PATH = DIR / "rt_plic_refined.png"

NX, NY = 128, 512

# Colormap: sharp blue/red with narrow white interface
_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rt_sharp",
    [(0.0,  "#0a2463"),
     (0.40, "#3e92cc"),
     (0.49, "#e8e8e8"),
     (0.51, "#e8e8e8"),
     (0.60, "#d63230"),
     (1.0,  "#6b0504")],
)


def load_csv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split(",")])
    return np.array(rows, dtype=np.float32)


def find_spike_tip(data, threshold=0.5):
    """Find lowest y where fill > threshold (spike tip of heavy fluid)."""
    for j in range(data.shape[0]):
        if np.any(data[j, :] > threshold):
            return j
    return 0


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} not found. Run viz_rt_plic first.")
        sys.exit(1)

    # Find all CSV snapshots
    all_files = sorted(DATA_DIR.glob("rt_step*.csv"))
    if not all_files:
        print("No data found!")
        sys.exit(1)

    # Load all snapshots
    snapshots = []
    for f in all_files:
        step = int(f.stem.replace("rt_step", ""))
        data = load_csv(f)
        spike_y = find_spike_tip(data)
        snapshots.append((step, data, spike_y))
        print(f"  step {step:5d}: spike at y={spike_y}")

    # Select 5 panels: initial + 3 intermediate + best "just before bottom"
    # Find the latest frame where spike tip is above y = 0.08*NY (8% from bottom)
    min_y_margin = int(0.08 * NY)  # don't show wall impact
    valid = [(s, d, y) for s, d, y in snapshots if y >= min_y_margin or s == 0]

    if len(valid) <= 5:
        selected = valid
    else:
        # Always include first and last valid
        selected = [valid[0]]
        # Pick 3 evenly spaced from middle
        n = len(valid)
        for i in [n // 4, n // 2, 3 * n // 4]:
            selected.append(valid[i])
        selected.append(valid[-1])

    n_panels = len(selected)
    print(f"\nSelected {n_panels} panels for figure")

    # Compute y-range: crop to region with activity + generous margin
    all_data = np.concatenate([d for _, d, _ in selected], axis=0)
    row_range = np.ptp(all_data, axis=1)
    mixed = np.where(row_range > 0.02)[0]
    if len(mixed) > 0:
        y0 = max(0, mixed.min() - 15)
        y1 = min(NY, mixed.max() + 15)
    else:
        y0, y1 = 0, NY

    # Physical extent (for axis labels, DX_PHYS=1e-4 m → cm)
    DX_PHYS = 1e-4
    extent_x = [0, NX * DX_PHYS * 100]
    extent_y = [y0 * DX_PHYS * 100, y1 * DX_PHYS * 100]

    # Create figure with equal aspect ratio
    panel_height = (y1 - y0) / NX  # aspect ratio of each panel
    fig_w = 2.8 * n_panels + 0.8  # extra for colorbar
    fig_h = 2.8 * panel_height + 1.2

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.05})
    if n_panels == 1:
        axes = [axes]

    for ax, (step, data, spy) in zip(axes, selected):
        view = data[y0:y1, :]

        im = ax.imshow(
            view, origin="lower", aspect="equal",
            cmap=_CMAP, vmin=0.0, vmax=1.0,
            extent=[extent_x[0], extent_x[1], extent_y[0], extent_y[1]],
            interpolation="nearest",
        )

        # f=0.5 contour (white, thin)
        y_coords = np.linspace(extent_y[0], extent_y[1], view.shape[0])
        x_coords = np.linspace(extent_x[0], extent_x[1], view.shape[1])
        ax.contour(x_coords, y_coords, view, levels=[0.5],
                   colors=["white"], linewidths=0.5, alpha=0.7)

        ax.set_title(f"step {step}", fontsize=9, pad=4)
        ax.set_xlabel("x [cm]", fontsize=8)
        ax.tick_params(labelsize=7)

        if ax is not axes[0]:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("y [cm]", fontsize=8)

    fig.suptitle(
        f"Rayleigh-Taylor Instability  (At={AT:.1f}, Re={int(round(Re))}, "
        f"PLIC-VOF + TRT + CSF)",
        fontsize=11, fontweight="bold", y=0.98)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Fill level f", fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=7)

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUT_PATH}")


# Parameters from viz_rt_plic.cu (for title)
RHO_HEAVY, RHO_LIGHT = 3.0, 1.0
AT = (RHO_HEAVY - RHO_LIGHT) / (RHO_HEAVY + RHO_LIGHT)
TAU = 0.6
NU_LB = (TAU - 0.5) / 3.0
G_LB = 5e-5
NX_SIM = 128
U_char = (AT * G_LB * NX_SIM) ** 0.5
Re = U_char * NX_SIM / NU_LB

if __name__ == "__main__":
    main()
