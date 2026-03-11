#!/usr/bin/env python3
"""
Rayleigh-Taylor instability visualization.

Reads VOF fill-level CSV snapshots produced by viz_rt.cu and renders a
4-panel figure showing the mushroom-cap formation sequence.

Layout: step 0 | step 2000 | step 4000 | step 6000
Each panel shows the f=0.5 contour on a blue/red colormap.
Physical orientation: y increases upward; heavy fluid (f=1, red) starts on
top and falls as spikes while light fluid (f=0, blue) rises as bubbles.

Output: /home/yzk/LBMProject/scripts/viz/rt_mushroom.png
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR  = Path("/home/yzk/LBMProject/scripts/viz")
OUT_PATH  = DATA_DIR / "rt_mushroom.png"

# Simulation parameters (must match viz_rt.cu)
NX, NY = 128, 512        # grid cells
DX_M   = 1e-4            # m / cell
TAU_F  = 0.6
NU_LBM = (TAU_F - 0.5) / 3.0
NU_PHY = 5e-6            # m²/s
DT     = min(NU_LBM * DX_M**2 / NU_PHY, 5e-5)   # s / step

PANELS = [
    ("step 0000", 0,    "t = 0 s"),
    ("step 2000", 2000, "t = {:.2f} s".format(2000 * DT)),
    ("step 4000", 4000, "t = {:.2f} s".format(4000 * DT)),
    ("step 6000", 6000, "t = {:.2f} s".format(6000 * DT)),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> np.ndarray:
    """Load a 2-D CSV (rows = y, cols = x) into a float32 array.

    CSV row j corresponds to lattice y=j (y=0 at domain bottom).
    imshow with origin='lower' renders row 0 at the bottom, which preserves
    the physical orientation: light fluid (f=0, blue) at bottom, heavy fluid
    (f=1, red) at top.
    """
    with path.open() as fh:
        rows = [list(map(float, line.split(","))) for line in fh if line.strip()]
    return np.asarray(rows, dtype=np.float32)


def crop_interface(data: np.ndarray, margin: int = 10) -> tuple[np.ndarray, int, int]:
    """
    Crop vertically to the interface region plus a fixed margin.
    Returns (cropped_array, y_start, y_end) in original row indices.
    Heavy fluid (f~1) is at top in data rows (low row index = low y).
    """
    ny = data.shape[0]
    # Rows that contain an interface (not all 0 or all 1)
    row_min = np.min(data, axis=1)
    row_max = np.max(data, axis=1)
    mixed = np.where((row_max - row_min) > 0.05)[0]
    if len(mixed) == 0:
        return data, 0, ny
    y0 = max(0,  mixed.min() - margin)
    y1 = min(ny, mixed.max() + margin + 1)
    return data[y0:y1, :], y0, y1


# ---------------------------------------------------------------------------
# Custom colormap: blue → white → red (fluid phases)
# ---------------------------------------------------------------------------
_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rt",
    [(0.0,  "#1a3a6c"),   # deep blue  (light fluid)
     (0.45, "#4b9cd3"),   # sky blue
     (0.5,  "#f8f8f8"),   # white at interface
     (0.55, "#e05c3a"),   # orange-red
     (1.0,  "#8b0000")],  # dark red   (heavy fluid)
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fig, axes = plt.subplots(
        1, len(PANELS),
        figsize=(12, 7),
        facecolor="#0d0d0d",
        gridspec_kw={"wspace": 0.04, "hspace": 0.0},
    )

    fig.suptitle(
        "Rayleigh-Taylor Instability   (At = 0.33, TVD-MC VOF)",
        color="white", fontsize=13, fontweight="bold", y=0.97,
    )

    im_ref = None

    for ax, (label, step_idx, time_label) in zip(axes, PANELS):
        fname = DATA_DIR / f"rt_step{step_idx:04d}.csv"
        data  = load_csv(fname)          # shape (ny, nx) = (512, 128)

        # Crop all panels to the interface-active region
        view, y0, y1 = crop_interface(data, margin=30)
        extent_y = (y0 * DX_M * 100, y1 * DX_M * 100)

        extent = [0, NX * DX_M * 100, extent_y[0], extent_y[1]]   # cm

        ax.set_facecolor("#0d0d0d")
        im = ax.imshow(
            view,
            origin="lower",
            aspect="auto",
            cmap=_CMAP,
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            interpolation="lanczos",
        )
        if im_ref is None:
            im_ref = im

        # f=0.5 contour
        y_cells = np.linspace(extent_y[0], extent_y[1], view.shape[0])
        x_cells = np.linspace(0, NX * DX_M * 100, view.shape[1])
        ax.contour(
            x_cells, y_cells, view,
            levels=[0.5],
            colors=["white"],
            linewidths=0.8,
            alpha=0.9,
        )

        ax.set_title(time_label, color="white", fontsize=10, pad=4)
        ax.set_xlabel("x  [cm]", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        # Only label y-axis on leftmost panel
        if ax is axes[0]:
            ax.set_ylabel("y  [cm]", color="#aaaaaa", fontsize=8)
        else:
            ax.set_yticklabels([])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.915, 0.12, 0.012, 0.72])
    cb = fig.colorbar(im_ref, cax=cbar_ax)
    cb.set_label("Fill level f", color="white", fontsize=9, labelpad=8)
    cb.ax.yaxis.set_tick_params(color="#888888", labelcolor="#aaaaaa", labelsize=7)
    cb.outline.set_edgecolor("#444444")

    # Annotation on the mushroom-cap panel (step 4000)
    mushroom_ax = axes[2]
    mushroom_ax.text(
        0.50, 0.04,
        "mushroom cap",
        transform=mushroom_ax.transAxes,
        color="#ffdd99", fontsize=8,
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.7, edgecolor="none"),
    )

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight",
                facecolor="#0d0d0d", edgecolor="none")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
