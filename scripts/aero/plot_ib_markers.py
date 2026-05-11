#!/usr/bin/env python3
"""Phase A sanity figure: IB markers overlaid on lattice grid.

Plots three panels:
  1. NACA α=0   : markers + lattice grid + outward normals (sample of normals)
  2. NACA α=+8° : markers + lattice grid + outward normals
  3. Circle (R=0.05 m, dx=2.5 mm): markers + lattice + radial normals

Reads the CSVs written by build/test_ib_markers.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    rows = {"x": [], "y": [], "z": [], "ds": [], "nx": [], "ny": [], "nz": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in rows:
                rows[k].append(float(r[k]))
    return {k: np.asarray(v) for k, v in rows.items()}


def plot_with_grid(ax, markers, dx, title,
                   xlim=None, ylim=None, normal_stride=4, normal_len=None):
    if xlim is None:
        xlim = (markers["x"].min() - 0.05, markers["x"].max() + 0.05)
    if ylim is None:
        ylim = (markers["y"].min() - 0.10, markers["y"].max() + 0.10)
    if normal_len is None:
        normal_len = 4 * dx

    # Lattice grid (light grey)
    xs = np.arange(np.floor(xlim[0] / dx) * dx,
                   np.ceil(xlim[1] / dx) * dx + dx, dx)
    ys = np.arange(np.floor(ylim[0] / dx) * dx,
                   np.ceil(ylim[1] / dx) * dx + dx, dx)
    for x in xs:
        ax.axvline(x, color="lightgrey", lw=0.4, zorder=0)
    for y in ys:
        ax.axhline(y, color="lightgrey", lw=0.4, zorder=0)

    # Markers
    ax.scatter(markers["x"], markers["y"], s=18, c="C3", zorder=3,
               edgecolors="white", linewidths=0.5, label=f"markers (N={len(markers['x'])})")

    # Outward normals (sample)
    sel = slice(0, len(markers["x"]), normal_stride)
    ax.quiver(markers["x"][sel], markers["y"][sel],
              markers["nx"][sel] * normal_len, markers["ny"][sel] * normal_len,
              angles="xy", scale_units="xy", scale=1,
              width=0.003, color="C0", alpha=0.8, zorder=2,
              label=f"normals (every {normal_stride}th)")

    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(False)


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.normpath(os.path.join(here, "..", ".."))
    in_dir = os.path.join(root, "output_phaseA")
    out_path = os.path.join(root, "output_phaseA", "ib_markers_phaseA.png")

    paths = {
        "naca_a0": os.path.join(in_dir, "naca_a0_markers.csv"),
        "naca_a8": os.path.join(in_dir, "naca_a8_markers.csv"),
        "circle":  os.path.join(in_dir, "circle_markers.csv"),
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"Missing: {p}  (did you run build/test_ib_markers?)", file=sys.stderr)
            sys.exit(1)

    naca_a0 = load_csv(paths["naca_a0"])
    naca_a8 = load_csv(paths["naca_a8"])
    circle  = load_csv(paths["circle"])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # NACA α=0 — chord-aligned LE at (0,0), c=1, dx=0.025
    plot_with_grid(axs[0], naca_a0, dx=0.025,
                   title=f"NACA0012 α=0  (chord=1, dx=0.025, ds≈dx)\nN={len(naca_a0['x'])} markers — mirror-symmetric ✓",
                   xlim=(-0.10, 1.15), ylim=(-0.15, 0.15),
                   normal_stride=3)

    # NACA α=+8° — LE at (10,10), c=1, dx=0.025
    plot_with_grid(axs[1], naca_a8, dx=0.025,
                   title=f"NACA0012 α=+8°  (LE=(10,10), c=1)\nN={len(naca_a8['x'])} markers — surface-only point cloud",
                   xlim=(9.85, 11.10), ylim=(9.85, 10.30),
                   normal_stride=3)

    # Circle R=0.05, dx_circle=0.0025 (smaller for clarity in zoom)
    plot_with_grid(axs[2], circle, dx=0.005,
                   title=f"Circle (R=0.05 m, ds=0.0025 m)\nN={len(circle['x'])} markers — radial normals",
                   xlim=(-0.07, 0.07), ylim=(-0.07, 0.07),
                   normal_stride=8, normal_len=0.012)

    fig.suptitle("Phase A — IB marker generation sanity (Wu-Shu 2009 framework)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
