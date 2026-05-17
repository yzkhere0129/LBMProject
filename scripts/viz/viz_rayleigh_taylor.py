#!/usr/bin/env python3
"""
Rayleigh-Taylor instability visualization for showcase gallery.

Reads VOF fill-level CSV snapshots produced by
/home/yzk/LBMProject/scripts/viz/viz_rt_openfoam_match.cu (iter 14 config:
variable-ν 2-phase, asym Bouss with sqrt(rho_H/rho_local) inertia
correction, patched TAU_MIN=0.505, free-slip BC fix). The simulation
replicates the OpenFOAM interFoam case at
/home/yzk/JAX-LaserAM/examples/RT_air_helium_nOuter3.

Layout: 6 panels at t = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 s
Each panel shows alpha (heavy=1, light=0) on blue/white/red colormap with
f=0.5 contour overlay.

Output: /home/yzk/LBMProject_showcase/gallery/rt_instability.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR  = Path("/home/yzk/LBMProject/scripts/viz/rt_of_data")
OUT_PATH  = Path("/home/yzk/LBMProject_showcase/gallery/rt_instability.png")

# Simulation parameters (must match viz_rt_openfoam_match.cu)
NX, NY = 128, 512
LX_M, LY_M = 1.0, 4.0
DX_M = LX_M / NX
DY_M = LY_M / NY

# Times to plot (correspond to rt_of_t00 ... rt_of_t10)
TIMES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_csv(path):
    """Load 2D CSV (rows=y, cols=x) as float32 array of shape (NY, NX)."""
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


# ---------------------------------------------------------------------------
# Blue → white → red colormap (light → interface → heavy)
# ---------------------------------------------------------------------------
_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rt",
    [(0.0,  "#1a3a6c"),
     (0.45, "#4b9cd3"),
     (0.5,  "#f8f8f8"),
     (0.55, "#e05c3a"),
     (1.0,  "#8b0000")],
)


def main():
    fig, axes = plt.subplots(
        1, len(TIMES),
        figsize=(15, 8),
        facecolor="#0d0d0d",
        gridspec_kw={"wspace": 0.02, "hspace": 0.0},
    )

    fig.suptitle(
        "Rayleigh-Taylor Instability — air/helium  "
        "(At = 0.758, g = 9.81 m/s²)",
        color="white", fontsize=14, fontweight="bold", y=0.96,
    )

    im_ref = None
    extent = [0.0, LX_M, 0.0, LY_M]

    for ax, t in zip(axes, TIMES):
        frame_idx = int(round(t * 10))
        fname = DATA_DIR / f"rt_of_t{frame_idx:02d}.csv"
        data = load_csv(fname)
        if data.shape != (NY, NX):
            raise RuntimeError(
                f"{fname}: shape {data.shape} != expected ({NY}, {NX})")

        ax.set_facecolor("#0d0d0d")
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=_CMAP,
            vmin=0.0, vmax=1.0,
            extent=extent,
            interpolation="lanczos",
        )
        if im_ref is None:
            im_ref = im

        # f=0.5 contour
        x_cells = np.linspace(DX_M / 2, LX_M - DX_M / 2, NX)
        y_cells = np.linspace(DY_M / 2, LY_M - DY_M / 2, NY)
        ax.contour(
            x_cells, y_cells, data,
            levels=[0.5],
            colors=["white"],
            linewidths=0.7,
            alpha=0.85,
        )

        ax.set_title(f"t = {t:.1f} s", color="white", fontsize=11, pad=4)
        ax.set_xlabel("x  [m]", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        if ax is axes[0]:
            ax.set_ylabel("y  [m]", color="#aaaaaa", fontsize=8)
        else:
            ax.set_yticklabels([])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.915, 0.12, 0.012, 0.72])
    cb = fig.colorbar(im_ref, cax=cbar_ax)
    cb.set_label("Fill level  f", color="white", fontsize=9, labelpad=8)
    cb.ax.yaxis.set_tick_params(color="#888888", labelcolor="#aaaaaa", labelsize=7)
    cb.outline.set_edgecolor("#444444")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight",
                facecolor="#0d0d0d", edgecolor="none")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
