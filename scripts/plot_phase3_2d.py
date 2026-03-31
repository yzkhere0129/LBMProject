"""
Phase 3 Diagnostic: Generate 2D temperature slice visualization from LBM

Reads contour CSV and creates synthetic 2D field for visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path("/home/yzk/LBMProject")
LBM_DIR = PROJECT_ROOT / "singlepointbench" / "contours"
OF_DIR = (
    PROJECT_ROOT / "openfoam" / "spot_melting_marangoni" / "postProcessing" / "contours"
)

NX, NZ = 100, 52
dx_um = 2.0
Z_OFFSET = 104.0  # NZ * dx_um


def create_field_from_contour(contour_df, nx, nz, dx, z_offset, T_max=8000):
    """
    Create synthetic 2D temperature field from contour points.
    Assumes Gaussian temperature distribution centered at contour centroid.
    """
    # Find contour centroid
    cx = contour_df["X_um"].mean()
    cz_surface = 150.0  # Surface position

    # Estimate melt pool depth
    cz_bottom = contour_df["Z_um"].min()

    x = np.arange(nx) * dx
    z = np.arange(nz) * dx + z_offset
    X, Z = np.meshgrid(x, z)

    # Distance from center axis
    R = np.sqrt((X - cx) ** 2)

    # Temperature field: Gaussian radial + linear vertical
    r0 = 25.0  # Spot radius
    T_radial = np.exp(-2 * R**2 / r0**2)

    # Vertical decay from surface
    depth = np.maximum(0, cz_surface - Z)
    max_depth = cz_surface - cz_bottom
    if max_depth > 0:
        T_vertical = np.exp(-depth / max_depth)
    else:
        T_vertical = np.ones_like(Z)

    # Combined temperature
    T_ambient = 300.0
    T_field = T_ambient + (T_max - T_ambient) * T_radial * T_vertical

    # Set gas region to 300K (above surface)
    T_field[Z > cz_surface] = 300.0

    return X, Z, T_field


def plot_diagnostic():
    TIMES = [50, 75]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Custom colormap for phases
    phase_colors = [
        "#440154",
        "#31688e",
        "#35b779",
        "#fde725",
    ]  # Gas, Solid, Mushy, Liquid
    phase_cmap = LinearSegmentedColormap.from_list("phase", phase_colors, N=4)

    for col, t in enumerate(TIMES):
        # Load LBM contour
        lbm_file = LBM_DIR / f"lbm_marangoni_contour_{t}us.csv"
        of_file = OF_DIR / f"openfoam_contour_{t}us.csv"

        lbm = pd.read_csv(lbm_file)
        of = pd.read_csv(of_file)

        # Create temperature field
        X, Z, T_field = create_field_from_contour(lbm, NX, NZ, dx_um, Z_OFFSET)

        # === Temperature Heatmap ===
        ax_T = axes[0, col]
        im_T = ax_T.pcolormesh(
            X, Z, T_field, cmap="hot", vmin=300, vmax=6000, shading="auto"
        )
        ax_T.contour(X, Z, T_field, levels=[1650], colors="white", linewidths=2)
        ax_T.plot(
            lbm["X_um"], lbm["Z_um"], "c-", linewidth=1.5, label="LBM contour (CSV)"
        )
        ax_T.plot(of["X_um"], of["Z_um"], "w--", linewidth=2, label="OpenFOAM")
        ax_T.set_xlim(40, 160)
        ax_T.set_ylim(100, 155)
        ax_T.invert_yaxis()
        ax_T.set_xlabel("X (μm)")
        ax_T.set_ylabel("Z (μm)")
        ax_T.set_title(f"Temperature Field, t={t} μs")
        ax_T.legend(loc="lower left", fontsize=8)
        plt.colorbar(im_T, ax=ax_T, label="T (K)")

        # === Phase Field ===
        ax_fl = axes[1, col]

        # Create phase map
        T_sol, T_liq = 1650.0, 1700.0
        phase = np.zeros_like(T_field)
        phase[T_field >= T_liq] = 1.0  # Liquid
        phase[(T_field >= T_sol) & (T_field < T_liq)] = 0.5  # Mushy
        phase[Z > 150] = -0.5  # Gas

        im_fl = ax_fl.pcolormesh(
            X, Z, phase, cmap=phase_cmap, vmin=-0.5, vmax=1.0, shading="auto"
        )
        ax_fl.contour(X, Z, T_field, levels=[1650], colors="black", linewidths=1)
        ax_fl.set_xlim(40, 160)
        ax_fl.set_ylim(100, 155)
        ax_fl.invert_yaxis()
        ax_fl.set_xlabel("X (μm)")
        ax_fl.set_ylabel("Z (μm)")
        ax_fl.set_title(f"Phase Field, t={t} μs")

        # Add colorbar with labels
        cbar = plt.colorbar(im_fl, ax=ax_fl, ticks=[-0.5, 0, 0.5, 1.0])
        cbar.ax.set_yticklabels(["Gas", "Solid", "Mushy", "Liquid"])

        # Mark gas-metal interface
        ax_fl.axhline(y=150, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax_fl.text(45, 149, "Gas-Metal Interface", color="red", fontsize=8)

    fig.suptitle(
        "Phase 3 Diagnostic: 2D Temperature and Phase Fields\n(Reconstructed from contour data)",
        fontsize=14,
        fontweight="bold",
    )

    output = PROJECT_ROOT / "phase3_2d_diagnostic.png"
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output}")

    # Print diagnostic info
    print("\n=== Diagnostic Summary ===")
    for t in TIMES:
        lbm = pd.read_csv(LBM_DIR / f"lbm_marangoni_contour_{t}us.csv")
        of = pd.read_csv(OF_DIR / f"openfoam_contour_{t}us.csv")

        lbm_depth = 150 - lbm["Z_um"].min()
        of_depth = 150 - of["Z_um"].min()

        print(f"\nt={t} μs:")
        print(f"  LBM: {len(lbm)} points, depth={lbm_depth:.1f} μm")
        print(f"  OF:  {len(of)} points, depth={of_depth:.1f} μm")
        print(f"  Ratio: {lbm_depth / of_depth:.2f}")


if __name__ == "__main__":
    plot_diagnostic()
