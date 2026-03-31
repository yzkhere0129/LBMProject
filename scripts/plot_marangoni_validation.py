"""
Phase 3 Validation: LBM vs OpenFOAM comparison

Uses native matplotlib contour() for proper isotherm extraction.
Reads 2D temperature field from CSV file.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LBM_DIR = PROJECT_ROOT / "singlepointbench" / "contours"
OF_DIR = (
    PROJECT_ROOT / "openfoam" / "spot_melting_marangoni" / "postProcessing" / "contours"
)
OUTPUT = PROJECT_ROOT / "marangoni_V_shape_validation.png"

TIMES = [25, 50, 60, 75]  # μs

# === Figure setup ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axes = axes.flatten()

for idx, t in enumerate(TIMES):
    ax = axes[idx]

    # Load OpenFOAM contour data (pre-extracted points)
    of_file = OF_DIR / f"openfoam_contour_{t}us.csv"
    of = pd.read_csv(of_file)

    # --- Plot OpenFOAM contour (line only, no markers) ---
    ax.plot(
        of["X_um"],
        of["Z_um"],
        color="#4A4A4A",
        linewidth=3,
        linestyle="-",
        label="Results from OpenFOAM" if idx == 3 else None,
    )

    # --- Load LBM 2D field and use native contour() ---
    lbm_field_file = LBM_DIR / f"lbm_temperature_{t}us.csv"

    if lbm_field_file.exists():
        # Load 2D temperature field from CSV
        field_df = pd.read_csv(lbm_field_file)

        # Pivot to 2D grid for contour()
        pivot = field_df.pivot(index="Z_um", columns="X_um", values="T_K")
        X, Z = np.meshgrid(pivot.columns.values, pivot.index.values)
        T_field = pivot.values

        # Use native matplotlib contour for T=1650K isotherm
        cs = ax.contour(X, Z, T_field, levels=[1650], colors="#E6A125", linewidths=1.5)

        # Label for legend
        if idx == 3:
            ax.plot(
                [],
                [],
                color="#E6A125",
                linewidth=1.5,
                label="Results from my algorithm",
            )

        print(
            f"  t={t}us: Loaded {len(field_df)} points, T range [{T_field.min():.0f}, {T_field.max():.0f}] K"
        )
    else:
        print(f"  WARNING: Field file not found: {lbm_field_file}")
        print(f"  Run benchmark first to generate field data.")

    # Axes limits
    ax.set_xlim(40, 160)
    ax.set_ylim(80, 155)
    ax.invert_yaxis()

    # Labels
    ax.set_xlabel("X (μm)", fontsize=10)
    ax.set_ylabel("Z (μm)", fontsize=10)
    ax.set_title(f"t = {t} μs", fontsize=11, fontweight="bold")

    # Grid
    ax.grid(True, color="#D3D3D3", linewidth=0.5, linestyle="-")

    # Legend only on the last subplot
    if idx == 3:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

# Global title
fig.suptitle(
    "Validation results for melting process (Marangoni Phase)",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

# Save
plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT}")
plt.close()
