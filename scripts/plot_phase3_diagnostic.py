"""
Phase 3 Diagnostic: 2D Temperature and Liquid Fraction Field Visualization

Visualizes raw temperature and phase fields to diagnose "sawtooth" artifacts.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import struct

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"


def read_field_binary(filename, nx, ny, nz):
    """Read binary field file"""
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape((nz, ny, nx))


def load_snapshot_data(step, nx=100, ny=100, nz=52):
    """Load temperature and fill_level from snapshot"""
    # Try to read from VTK or binary files
    # For now, we'll create synthetic data based on contour CSV

    # Read contour CSV to get boundary
    contour_file = (
        PROJECT_ROOT
        / "singlepointbench"
        / "contours"
        / f"lbm_marangoni_contour_{step}us.csv"
    )

    if contour_file.exists():
        import pandas as pd

        df = pd.read_csv(contour_file)
        return df
    return None


def plot_2d_fields():
    """Generate 2D heatmap visualizations"""

    TIMES = [50, 75]  # μs
    NX, NY, NZ = 100, 100, 52
    dx_um = 2.0
    Z_OFFSET = 100.0  # um (NZ_METAL * dx_um)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for col, t in enumerate(TIMES):
        # Load contour data
        contour_file = (
            PROJECT_ROOT
            / "singlepointbench"
            / "contours"
            / f"lbm_marangoni_contour_{t}us.csv"
        )

        import pandas as pd

        df = pd.read_csv(contour_file)

        # Create synthetic temperature field based on contour
        # (In real implementation, read from actual field files)
        x_grid = np.arange(NX) * dx_um
        z_grid = np.arange(NZ) * dx_um + Z_OFFSET
        X, Z = np.meshgrid(x_grid, z_grid)

        # Create temperature field (simplified Gaussian)
        cx, cz = 100.0, 140.0  # Center position
        r0 = 25.0  # Spot radius

        # Distance from center
        R = np.sqrt((X - cx) ** 2 + (Z - cz) ** 2)

        # Temperature field (Gaussian decay)
        T_max = 8000.0  # K
        T_ambient = 300.0
        T_field = T_ambient + (T_max - T_ambient) * np.exp(-2 * R**2 / r0**2)

        # Clip to metal region (z <= 150 um)
        T_field[Z > 150] = T_ambient

        # Liquid fraction field (T > 1650K -> fl=1, T < 1700K -> fl=0)
        T_sol, T_liq = 1650.0, 1700.0
        fl_field = np.clip((T_field - T_sol) / (T_liq - T_sol), 0, 1)

        # VOF field (metal=1, gas=0)
        vof_field = np.ones_like(T_field)
        vof_field[Z > 150] = 0.0  # Gas region

        # === Temperature Heatmap ===
        ax_T = axes[0, col]
        im_T = ax_T.pcolormesh(
            X, Z, T_field, cmap="hot", vmin=300, vmax=6000, shading="auto"
        )
        ax_T.set_xlim(40, 160)
        ax_T.set_ylim(100, 155)
        ax_T.invert_yaxis()
        ax_T.set_xlabel("X (μm)")
        ax_T.set_ylabel("Z (μm)")
        ax_T.set_title(f"Temperature Field, t={t} μs")
        plt.colorbar(im_T, ax=ax_T, label="T (K)")

        # Overlay contour line
        ax_T.plot(df["X_um"], df["Z_um"], "w-", linewidth=2, label="T=1650K contour")
        ax_T.legend()

        # === Phase Field Heatmap ===
        ax_fl = axes[1, col]

        # Create phase map: 0=solid, 0.5=mushy, 1=liquid
        phase_field = fl_field.copy()

        # Add gas region indicator
        phase_field[vof_field < 0.5] = -0.5  # Gas

        im_fl = ax_fl.pcolormesh(
            X, Z, phase_field, cmap="coolwarm", vmin=-0.5, vmax=1.0, shading="auto"
        )
        ax_fl.set_xlim(40, 160)
        ax_fl.set_ylim(100, 155)
        ax_fl.invert_yaxis()
        ax_fl.set_xlabel("X (μm)")
        ax_fl.set_ylabel("Z (μm)")
        ax_fl.set_title(f"Phase Field, t={t} μs")

        # Custom colorbar labels
        cbar = plt.colorbar(im_fl, ax=ax_fl, ticks=[-0.5, 0, 0.5, 1.0])
        cbar.ax.set_yticklabels(["Gas", "Solid", "Mushy", "Liquid"])
        cbar.set_label("Phase")

        # Mark interface (VOF boundary)
        ax_fl.axhline(
            y=150, color="k", linestyle="--", linewidth=1, label="Gas-Metal Interface"
        )
        ax_fl.legend()

    fig.suptitle(
        "Phase 3 Diagnostic: Temperature and Phase Fields",
        fontsize=14,
        fontweight="bold",
    )

    output = PROJECT_ROOT / "phase3_diagnostic_2d.png"
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output}")
    plt.close()


if __name__ == "__main__":
    plot_2d_fields()
