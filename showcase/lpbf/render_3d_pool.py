#!/usr/bin/env python3
"""Render the 3D melt-pool isosurface from the extracted .npz.

Single-panel showcase figure. VOF=0.5 isosurface, coloured by temperature,
in laser-relative coordinates.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main(npz_in, png_out):
    d = np.load(npz_in)
    v = d["verts_um"]
    T = d["T_K"]
    step = int(d["step"])
    dt = float(d["dt_s"])
    t_us = step * dt * 1e6

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(v[:, 0], v[:, 1], v[:, 2],
                    c=T, cmap="hot", s=1.5, vmin=300, vmax=4000,
                    rasterized=True)

    cbar = plt.colorbar(sc, ax=ax, label="Temperature [K]",
                        shrink=0.55, pad=0.08)
    cbar.set_label("Temperature [K]", fontsize=11)

    ax.set_xlabel("x' (scan direction)  [μm]", fontsize=11, labelpad=8)
    ax.set_ylabel("y  [μm]", fontsize=11, labelpad=8)
    ax.set_zlabel("z'  [μm]", fontsize=11, labelpad=8)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-130, 30)
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((4, 1.2, 1.6))

    ax.set_title("LBM-CUDA melt-pool simulation — 316L, 150 W laser, "
                 f"v_scan = 0.8 m/s\n"
                 f"VOF=0.5 isosurface, T-coloured  •  t = {t_us:.0f} μs  •  "
                 f"{len(v):,} surface vertices",
                 fontsize=12, pad=14)

    fig.tight_layout()
    fig.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{npz_in} → {png_out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
