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

# Simulation parameters (must match viz_rt.cu — JAX air/helium config)
NX, NY = 128, 512                       # grid cells
LX_M   = 1.0                            # m
DX_M   = LX_M / NX                      # = 7.8125 mm / cell
TAU_F  = 0.55
NU_LBM = (TAU_F - 0.5) / 3.0
NU_PHY = 2.5551e-3                      # m²/s (matches air at heavy phase)
DT     = min(NU_LBM * DX_M**2 / NU_PHY, 5e-4)   # s / step

# Find closest available step file to each target time
TIMES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
import glob, re
_files = sorted(glob.glob(str(DATA_DIR / "rt_step*.csv")))
_avail = sorted(int(re.search(r"rt_step(\d+)", f).group(1)) for f in _files)

def _nearest(target_step):
    return min(_avail, key=lambda s: abs(s - target_step))

PANELS = []
for t in TIMES:
    target = int(round(t / DT))
    s = _nearest(target)
    PANELS.append((f"step {s:05d}", s, f"t = {t:.1f} s"))

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
        figsize=(15, 8),
        facecolor="#0d0d0d",
        gridspec_kw={"wspace": 0.02, "hspace": 0.0},
    )

    fig.suptitle(
        "Rayleigh-Taylor Instability — air/helium  (At = 0.758, g = 9.81 m/s²)",
        color="white", fontsize=14, fontweight="bold", y=0.96,
    )

    im_ref = None

    for ax, (label, step_idx, time_label) in zip(axes, PANELS):
        # files are zero-padded to 4 or 5 digits — try both
        fname = DATA_DIR / f"rt_step{step_idx:04d}.csv"
        if not fname.exists():
            fname = DATA_DIR / f"rt_step{step_idx:05d}.csv"
        data  = load_csv(fname)          # shape (ny, nx) = (512, 128)

        # FULL DOMAIN (y from 0 to 4 m) — no crop so the slim spike tail
        # and rollups are fully visible (matches JAX figure framing).
        view = data
        extent_y = (0.0, NY * DX_M)
        extent = [0, NX * DX_M, extent_y[0], extent_y[1]]   # m

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
        x_cells = np.linspace(0, NX * DX_M, view.shape[1])
        ax.contour(
            x_cells, y_cells, view,
            levels=[0.5],
            colors=["white"],
            linewidths=0.7,
            alpha=0.85,
        )

        ax.set_title(time_label, color="white", fontsize=11, pad=4)
        ax.set_xlabel("x  [m]", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        # Only label y-axis on leftmost panel
        if ax is axes[0]:
            ax.set_ylabel("y  [m]", color="#aaaaaa", fontsize=8)
        else:
            ax.set_yticklabels([])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.915, 0.12, 0.012, 0.72])
    cb = fig.colorbar(im_ref, cax=cbar_ax)
    cb.set_label("Fill level f", color="white", fontsize=9, labelpad=8)
    cb.ax.yaxis.set_tick_params(color="#888888", labelcolor="#aaaaaa", labelsize=7)
    cb.outline.set_edgecolor("#444444")

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight",
                facecolor="#0d0d0d", edgecolor="none")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
