#!/usr/bin/env python3
"""
Rayleigh-Taylor instability interface visualization.

Reads fill_level from ASCII VTK StructuredPoints files at key timesteps,
extracts the center z-slice (k=2), and plots the 2D interface morphology
to verify mushroom-shaped cap formation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# === PARAMETERS ===
VTK_DIR   = "/home/yzk/LBMProject/build/output_rt_benchmark"
OUTPUT_DIR = VTK_DIR

# Snapshots to analyse (step numbers)
STEPS = [0, 8000, 16000, 24000, 32000, 36000]

# Grid dimensions as written in VTK header: NX x NY x NZ
NX, NY, NZ = 256, 1024, 4

# Center z-slice index (0-based)
K_SLICE = 2

# Interface contour level
INTERFACE_F = 0.5


# === VTK READER ===

def read_vtk_fill_level(filepath):
    """
    Parse ASCII VTK StructuredPoints file and return fill_level array
    shaped (NZ, NY, NX) in C order (matching VTK FORTRAN / column-major
    ordering: x varies fastest, then y, then z).
    """
    scalar_vals = []
    in_scalars  = False
    skip_next   = False   # skip LOOKUP_TABLE line

    with open(filepath, 'r') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("SCALARS fill_level"):
                in_scalars = True
                skip_next  = True   # next line is LOOKUP_TABLE
                continue
            if in_scalars and skip_next:
                skip_next = False   # skip the LOOKUP_TABLE line
                continue
            if in_scalars:
                # Stop when we hit the VECTORS block or another section header
                if line.startswith("VECTORS") or line.startswith("SCALARS"):
                    break
                if line:
                    scalar_vals.append(float(line))

    if len(scalar_vals) != NX * NY * NZ:
        raise ValueError(
            f"{filepath}: expected {NX*NY*NZ} scalars, got {len(scalar_vals)}"
        )

    # VTK stores data with x varying fastest, then y, then z
    arr = np.array(scalar_vals, dtype=np.float32)
    # Reshape to (NZ, NY, NX)
    arr = arr.reshape(NZ, NY, NX)
    return arr


# === MAIN ===

print("Reading VTK snapshots...")
slices = {}   # step -> 2D fill_level array (NY x NX)

for step in STEPS:
    fname = os.path.join(VTK_DIR, f"rt_benchmark_step{step:06d}.vtk")
    if not os.path.isfile(fname):
        print(f"  WARNING: {fname} not found, skipping.")
        continue
    data3d = read_vtk_fill_level(fname)
    # Extract center z-slice: shape (NY, NX)
    slices[step] = data3d[K_SLICE, :, :]
    f_min, f_max = slices[step].min(), slices[step].max()
    print(f"  step {step:6d}: fill_level range [{f_min:.4f}, {f_max:.4f}]")

# --- Individual PNG files ---
for step, f2d in slices.items():
    fig, ax = plt.subplots(figsize=(4, 12))

    # Colormap: blue=0 (light fluid), red=1 (heavy fluid)
    im = ax.imshow(
        f2d,
        origin='lower',
        cmap='RdBu_r',
        vmin=0.0, vmax=1.0,
        aspect='auto',
        interpolation='nearest'
    )
    # f=0.5 interface contour
    ax.contour(f2d, levels=[INTERFACE_F], colors='lime', linewidths=1.2)

    ax.set_title(f"RT instability  step={step}\n"
                 f"(blue=light, red=heavy, green=f=0.5 interface)", fontsize=9)
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="fill_level")

    out_path = os.path.join(OUTPUT_DIR, f"rt_snapshot_step{step:06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")

# --- Combined 3x2 figure ---
fig, axes = plt.subplots(2, 3, figsize=(15, 20))
axes_flat = axes.flatten()

for idx, step in enumerate(STEPS):
    ax = axes_flat[idx]
    if step not in slices:
        ax.set_visible(False)
        continue

    f2d = slices[step]
    im  = ax.imshow(
        f2d,
        origin='lower',
        cmap='RdBu_r',
        vmin=0.0, vmax=1.0,
        aspect='auto',
        interpolation='nearest'
    )
    ax.contour(f2d, levels=[INTERFACE_F], colors='lime', linewidths=1.0)
    ax.set_title(f"step = {step}", fontsize=11)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)

    # Add colorbar only on rightmost column
    if idx % 3 == 2:
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="fill_level")

fig.suptitle(
    "Rayleigh-Taylor Instability — fill_level evolution (z-slice k=2)\n"
    "Blue = light fluid (bottom), Red = heavy fluid (top), "
    "Green contour = f=0.5 interface",
    fontsize=13, y=1.01
)

combined_path = os.path.join(OUTPUT_DIR, "rt_evolution_combined.png")
fig.savefig(combined_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nCombined figure saved: {combined_path}")
