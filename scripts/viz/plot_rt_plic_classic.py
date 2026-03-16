#!/usr/bin/env python3
"""
Classic Re=256, At=0.5, sigma=0 Rayleigh-Taylor instability.

FULL DOMAIN, NO CROPPING. Axes in lattice units.
Strict: xlim(0,128), ylim(0,512), aspect='equal'.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

DIR = Path(__file__).parent
DATA_DIR = DIR / "rt_plic_data"
OUT_PATH = DIR / "rt_plic_classic_1to1.png"

NX, NY = 128, 512

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
    """Lowest y where fill > threshold (heavy fluid spike)."""
    for j in range(data.shape[0]):
        if np.any(data[j, :] > threshold):
            return j
    return 0


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} not found.")
        sys.exit(1)

    all_files = sorted(DATA_DIR.glob("rt_step*.csv"))
    if not all_files:
        print("No data found!")
        sys.exit(1)

    snapshots = []
    for f in all_files:
        step = int(f.stem.replace("rt_step", ""))
        data = load_csv(f)
        spike_y = find_spike_tip(data)
        snapshots.append((step, data, spike_y))
        print(f"  step {step:5d}: spike at y={spike_y}")

    # Select frames showing spike descent from y=384 to y~100
    # Prefer evenly spaced in SPIKE POSITION (not time) for visual clarity
    valid = [s for s in snapshots if s[2] >= 80 or s[0] == 0]
    if not valid:
        valid = snapshots

    if len(valid) <= 7:
        selected = valid
    else:
        # Evenly space by spike y-position
        y_start = valid[0][2]
        y_end = valid[-1][2]
        n_target = 7
        target_ys = [y_start - i * (y_start - y_end) / (n_target - 1)
                     for i in range(n_target)]

        selected = []
        for ty in target_ys:
            best = min(valid, key=lambda s: abs(s[2] - ty))
            if best not in selected:
                selected.append(best)
        # Always include first and last
        if valid[0] not in selected:
            selected.insert(0, valid[0])
        if valid[-1] not in selected:
            selected.append(valid[-1])

    n_panels = len(selected)
    print(f"\nSelected {n_panels} panels")

    # Figure: each panel is 128x512 (1:4 aspect)
    pw = 2.2
    ph = pw * (NY / NX)
    fig_w = pw * n_panels + 1.2
    fig_h = ph + 1.5

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.06})
    if n_panels == 1:
        axes = [axes]

    for ax, (step, data, spy) in zip(axes, selected):
        im = ax.imshow(
            data, origin="lower", aspect="equal",
            cmap=_CMAP, vmin=0.0, vmax=1.0,
            extent=[0, NX, 0, NY],
            interpolation="nearest",
        )

        # f=0.5 contour
        y_coords = np.linspace(0, NY, data.shape[0])
        x_coords = np.linspace(0, NX, data.shape[1])
        ax.contour(x_coords, y_coords, data, levels=[0.5],
                   colors=["white"], linewidths=0.4, alpha=0.6)

        # STRICT: no cropping
        ax.set_xlim(0, NX)
        ax.set_ylim(0, NY)
        ax.set_aspect("equal")

        ax.set_title(f"step {step}", fontsize=9, pad=4)
        ax.set_xlabel("x", fontsize=8)
        ax.tick_params(labelsize=6)

        if ax is not axes[0]:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("y", fontsize=8)

    fig.suptitle(
        "Rayleigh–Taylor Instability  (Re = 256, At = 0.5, σ = 0, PLIC-VOF + TRT)",
        fontsize=11, fontweight="bold", y=0.99)

    cbar_ax = fig.add_axes([0.93, 0.10, 0.010, 0.78])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Fill fraction f", fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=7)

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
