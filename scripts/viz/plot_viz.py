"""
plot_viz.py
-----------
Generate two publication-quality figures:
  1. scripts/viz/rt_mushroom.png     -- RT instability mushroom cloud
  2. scripts/viz/zalesak_disk.png    -- Zalesak slotted-disk benchmark

Usage:
    python3 scripts/viz/plot_viz.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")           # no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.contour import QuadContourSet

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    return np.loadtxt(path, delimiter=",")


def interface_contour(ax, data, level=0.5, **kwargs):
    """Draw the 0.5-level contour (the interface line)."""
    ny, nx = data.shape
    x = np.arange(nx)
    y = np.arange(ny)
    ax.contour(x, y, data, levels=[level], **kwargs)


# ---------------------------------------------------------------------------
# Figure 1: Rayleigh-Taylor mushroom cloud
# ---------------------------------------------------------------------------

def plot_rt_mushroom():
    pattern = os.path.join(OUT_DIR, "rt_step*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("  No RT CSV files found - skipping RT plot.")
        return

    print(f"  Found {len(files)} RT snapshots.")

    # Pick up to 7 frames evenly spaced
    max_panels = 7
    if len(files) <= max_panels:
        selected = files
    else:
        idxs = np.linspace(0, len(files)-1, max_panels, dtype=int)
        selected = [files[i] for i in idxs]

    ncols = len(selected)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.2, 8),
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    fig.patch.set_facecolor("#111111")
    cmap = plt.get_cmap("coolwarm")

    for ax, fpath in zip(axes, selected):
        data = load_csv(fpath)
        ny, nx = data.shape

        # y-axis: flip so y=0 at bottom (physical orientation)
        img = ax.imshow(data, origin="lower", cmap=cmap,
                        vmin=0.0, vmax=1.0, aspect="auto",
                        interpolation="bilinear",
                        extent=[0, nx, 0, ny])

        # Interface contour
        interface_contour(ax, data, level=0.5,
                          colors=["white"], linewidths=[0.8])

        # Step number from filename
        basename = os.path.basename(fpath)
        step_str = basename.replace("rt_step", "").replace(".csv", "")
        try:
            step = int(step_str)
        except ValueError:
            step = 0
        ax.set_title(f"step {step}", color="white", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.6, pad=0.02)
    cbar.set_label("Fill level (f)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(colors="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle("Rayleigh-Taylor Instability  (At = 0.33,  TVD-MC VOF)",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    out = os.path.join(OUT_DIR, "rt_mushroom.png")
    fig.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Zalesak slotted disk
# ---------------------------------------------------------------------------

def plot_zalesak():
    deg_labels = [0, 90, 180, 270, 360]
    files = [os.path.join(OUT_DIR, f"zalesak_deg{d:03d}.csv") for d in deg_labels]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"  Missing Zalesak files: {missing}  -- skipping.")
        return

    print(f"  Found all {len(files)} Zalesak snapshots.")

    data_list = [load_csv(f) for f in files]
    ny, nx = data_list[0].shape
    x = np.arange(nx)
    y = np.arange(ny)

    # ---- Panel layout: 5 snapshots in a row + an overlay comparison panel
    fig = plt.figure(figsize=(15, 4.5), facecolor="#111111")
    fig.patch.set_facecolor("#111111")

    nrows, ncols = 1, 6
    gs = fig.add_gridspec(nrows, ncols, wspace=0.08, hspace=0.15,
                          left=0.02, right=0.92, top=0.88, bottom=0.05)

    cmap = plt.get_cmap("Blues")

    for col, (deg, data) in enumerate(zip(deg_labels, data_list)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#111111")

        ax.imshow(data, origin="lower", cmap=cmap,
                  vmin=0.0, vmax=1.0, aspect="equal",
                  interpolation="bilinear",
                  extent=[0, nx, 0, ny])
        interface_contour(ax, data, level=0.5,
                          colors=["#FFD700"], linewidths=[1.2])

        ax.set_title(f"{deg}°", color="white", fontsize=11, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")

    # ---- Overlay panel: initial vs final contours
    ax_ov = fig.add_subplot(gs[0, 5])
    ax_ov.set_facecolor("#111111")
    ax_ov.set_aspect("equal")
    ax_ov.set_title("Overlay: 0° vs 360°", color="white", fontsize=10, pad=3)

    # Draw filled region for initial disk (blue-ish tint)
    ax_ov.contourf(x, y, data_list[0], levels=[0.5, 1.0],
                   colors=["#3399FF"], alpha=0.25)
    # Initial interface (solid)
    ax_ov.contour(x, y, data_list[0], levels=[0.5],
                  colors=["#3399FF"], linewidths=[1.8],
                  linestyles=["-"])
    # Final interface (dashed)
    ax_ov.contour(x, y, data_list[-1], levels=[0.5],
                  colors=["#FF6633"], linewidths=[1.8],
                  linestyles=["--"])

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#3399FF", lw=2, label="Initial (0°)"),
        Line2D([0], [0], color="#FF6633", lw=2, ls="--", label="Final (360°)"),
    ]
    ax_ov.legend(handles=legend_elems, loc="lower right",
                 fontsize=7, facecolor="#222222", edgecolor="#555555",
                 labelcolor="white")
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])
    for sp in ax_ov.spines.values():
        sp.set_edgecolor("#444444")

    # Mass error annotation
    m0 = data_list[0].sum()
    m1 = data_list[-1].sum()
    merr = abs(m1 - m0) / m0 * 100.0
    fig.text(0.93, 0.5, f"Mass loss\n{merr:.1f}%",
             color="white", fontsize=9, va="center", ha="left",
             transform=fig.transFigure)

    fig.suptitle("Zalesak Slotted-Disk Rotation  (128×128, TVD-MC, 360°)",
                 color="white", fontsize=13, fontweight="bold")

    out = os.path.join(OUT_DIR, "zalesak_disk.png")
    fig.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating visualization figures...")
    print("\n[1/2] Rayleigh-Taylor mushroom:")
    plot_rt_mushroom()
    print("\n[2/2] Zalesak slotted disk:")
    plot_zalesak()
    print("\nDone.")
