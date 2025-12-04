#!/usr/bin/env python3
"""
Analyze VOF (Volume of Fluid) field from VTK simulation output.

Extracts and analyzes fill level (phase fraction), interface location,
interface sharpness, and mass conservation from multiphysics simulations.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
import sys

# === PARAMETERS (modify these) ===
VTK_DIR = "/home/yzk/LBMProject/build/phase6_test2c_visualization"
VTK_PATTERN = "marangoni_flow_*.vtk"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/results"
SLICE_Z_INDEX = 16  # Middle of z-dimension for 2D slices
INTERFACE_THRESHOLD = 0.5  # VOF value defining interface (0.0 to 1.0)
LIQUID_THRESHOLD = 0.9  # VOF > this is considered liquid
GAS_THRESHOLD = 0.1  # VOF < this is considered gas

def load_vtk_file(filepath):
    """Load VTK file and return mesh with data."""
    try:
        mesh = pv.read(filepath)
        print(f"Loaded: {filepath}")
        print(f"  Dimensions: {mesh.dimensions}")
        print(f"  Points: {mesh.n_points}")
        return mesh
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_vof_field(mesh, output_prefix):
    """Extract and analyze VOF field statistics."""

    if 'FillLevel' not in mesh.array_names:
        print("ERROR: No 'FillLevel' field found in VTK file")
        print(f"Available fields: {mesh.array_names}")
        return None

    fill_level = mesh['FillLevel']
    print(f"\n=== VOF Fill Level Analysis ===")
    print(f"Shape: {fill_level.shape}")

    # Statistics
    print(f"\nFill Level Statistics:")
    print(f"  Min:  {fill_level.min():.6f}")
    print(f"  Max:  {fill_level.max():.6f}")
    print(f"  Mean: {fill_level.mean():.6f}")
    print(f"  Std:  {fill_level.std():.6f}")

    # Phase distribution
    liquid_cells = np.sum(fill_level > LIQUID_THRESHOLD)
    gas_cells = np.sum(fill_level < GAS_THRESHOLD)
    interface_cells = np.sum((fill_level >= GAS_THRESHOLD) & (fill_level <= LIQUID_THRESHOLD))
    total_cells = len(fill_level)

    print(f"\nPhase Distribution:")
    print(f"  Liquid cells (F > {LIQUID_THRESHOLD}):  {liquid_cells}/{total_cells} ({100*liquid_cells/total_cells:.2f}%)")
    print(f"  Gas cells (F < {GAS_THRESHOLD}):     {gas_cells}/{total_cells} ({100*gas_cells/total_cells:.2f}%)")
    print(f"  Interface cells ({GAS_THRESHOLD} ≤ F ≤ {LIQUID_THRESHOLD}): {interface_cells}/{total_cells} ({100*interface_cells/total_cells:.2f}%)")

    # Total liquid volume (sum of fill levels)
    total_liquid_volume = np.sum(fill_level)
    cell_volume = np.prod(mesh.spacing)
    liquid_volume_m3 = total_liquid_volume * cell_volume

    print(f"\nLiquid Volume:")
    print(f"  Total fill level sum: {total_liquid_volume:.2f}")
    print(f"  Cell volume: {cell_volume:.6e} m³")
    print(f"  Liquid volume: {liquid_volume_m3:.6e} m³")

    # Check for bound violations
    below_zero = np.sum(fill_level < 0.0)
    above_one = np.sum(fill_level > 1.0)
    if below_zero > 0 or above_one > 0:
        print(f"\nWARNING: Bound violations detected!")
        print(f"  Cells < 0.0: {below_zero}")
        print(f"  Cells > 1.0: {above_one}")

    # Check for NaN or Inf
    nan_count = np.sum(np.isnan(fill_level))
    inf_count = np.sum(np.isinf(fill_level))
    if nan_count > 0 or inf_count > 0:
        print(f"\nWARNING: Found {nan_count} NaN and {inf_count} Inf values!")

    return {
        'fill_level': fill_level,
        'total_volume': total_liquid_volume,
        'liquid_cells': liquid_cells,
        'gas_cells': gas_cells,
        'interface_cells': interface_cells,
        'liquid_volume_m3': liquid_volume_m3
    }

def compute_interface_sharpness(fill_level_3d):
    """
    Compute interface sharpness metric.

    Sharp interface: most cells are either 0 or 1
    Diffuse interface: many cells have intermediate values
    """
    # Compute how many cells are in intermediate range
    intermediate = np.sum((fill_level_3d > 0.1) & (fill_level_3d < 0.9))
    total = fill_level_3d.size

    # Sharpness: 1.0 = perfectly sharp (no intermediate), 0.0 = very diffuse
    sharpness = 1.0 - (intermediate / total)

    return sharpness, intermediate, total

def extract_interface_profile(mesh):
    """Extract interface position along x and y axes."""
    dims = mesh.dimensions
    spacing = mesh.spacing
    origin = mesh.origin

    fill_level = mesh['FillLevel']
    fill_3d = fill_level.reshape(dims, order='F')

    # Get centerline indices
    y_center = dims[1] // 2
    z_center = dims[2] // 2
    x_center = dims[0] // 2

    # Find interface position along x (at center y, z)
    x_coords = origin[0] + np.arange(dims[0]) * spacing[0]
    x_profile = fill_3d[:, y_center, z_center]

    # Find interface position along y (at center x, z)
    y_coords = origin[1] + np.arange(dims[1]) * spacing[1]
    y_profile = fill_3d[x_center, :, z_center]

    return {
        'x_coords': x_coords,
        'x_profile': x_profile,
        'y_coords': y_coords,
        'y_profile': y_profile
    }

def plot_vof_profiles(mesh, output_dir):
    """Plot VOF fill level profiles along x and y centerlines."""
    profile_data = extract_interface_profile(mesh)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # X-profile
    axes[0].plot(profile_data['x_coords'] * 1e6, profile_data['x_profile'], 'b-', linewidth=2)
    axes[0].axhline(y=INTERFACE_THRESHOLD, color='r', linestyle='--', label=f'Interface (F={INTERFACE_THRESHOLD})')
    axes[0].set_xlabel('X Position (μm)')
    axes[0].set_ylabel('Fill Level')
    axes[0].set_title('VOF Profile along X (centerline)')
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Y-profile
    axes[1].plot(profile_data['y_coords'] * 1e6, profile_data['y_profile'], 'b-', linewidth=2)
    axes[1].axhline(y=INTERFACE_THRESHOLD, color='r', linestyle='--', label=f'Interface (F={INTERFACE_THRESHOLD})')
    axes[1].set_xlabel('Y Position (μm)')
    axes[1].set_ylabel('Fill Level')
    axes[1].set_title('VOF Profile along Y (centerline)')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    output_file = Path(output_dir) / 'vof_profiles.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved VOF profiles to: {output_file}")
    plt.close()

def plot_vof_distribution(stats, output_dir):
    """Plot VOF fill level distribution histogram."""
    fill_level = stats['fill_level']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram with more bins to see distribution detail
    counts, bins, patches = ax.hist(fill_level, bins=100, color='blue', alpha=0.7, edgecolor='black')

    # Highlight regions
    for i, patch in enumerate(patches):
        if bins[i] < GAS_THRESHOLD:
            patch.set_facecolor('lightblue')  # Gas
        elif bins[i] > LIQUID_THRESHOLD:
            patch.set_facecolor('darkblue')  # Liquid
        else:
            patch.set_facecolor('orange')  # Interface

    ax.axvline(x=GAS_THRESHOLD, color='r', linestyle='--', linewidth=2, label=f'Gas threshold ({GAS_THRESHOLD})')
    ax.axvline(x=LIQUID_THRESHOLD, color='g', linestyle='--', linewidth=2, label=f'Liquid threshold ({LIQUID_THRESHOLD})')
    ax.axvline(x=INTERFACE_THRESHOLD, color='k', linestyle='--', linewidth=2, label=f'Interface ({INTERFACE_THRESHOLD})')

    ax.set_xlabel('Fill Level')
    ax.set_ylabel('Frequency')
    ax.set_title('VOF Distribution (Gas = light blue, Interface = orange, Liquid = dark blue)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'vof_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved VOF distribution to: {output_file}")
    plt.close()

def plot_2d_vof_slice(mesh, z_index, output_dir):
    """Plot 2D slice of VOF field."""
    dims = mesh.dimensions
    fill_level = mesh['FillLevel']
    fill_3d = fill_level.reshape(dims, order='F')

    if z_index >= dims[2]:
        z_index = dims[2] // 2

    # Extract slice
    vof_slice = fill_3d[:, :, z_index]

    # Create coordinate grids for plotting
    spacing = mesh.spacing
    origin = mesh.origin
    x = (origin[0] + np.arange(dims[0]) * spacing[0]) * 1e6  # μm
    y = (origin[1] + np.arange(dims[1]) * spacing[1]) * 1e6  # μm

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot as image
    im = ax.imshow(vof_slice.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')

    # Add contour for interface
    ax.contour(x, y, vof_slice.T, levels=[INTERFACE_THRESHOLD], colors='black', linewidths=2)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fill Level', fontsize=12)

    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title(f'VOF Field - 2D Slice at z_index={z_index}', fontsize=14)

    plt.tight_layout()
    output_file = Path(output_dir) / f'vof_slice_z{z_index}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 2D VOF slice to: {output_file}")
    plt.close()

def main():
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find VTK files
    vtk_files = sorted(Path(VTK_DIR).glob(VTK_PATTERN))

    if not vtk_files:
        print(f"ERROR: No VTK files found matching {VTK_DIR}/{VTK_PATTERN}")
        return 1

    print(f"Found {len(vtk_files)} VTK files")

    # Analyze first and last timesteps
    for i, idx in enumerate([0, -1]):
        vtk_file = vtk_files[idx]
        print(f"\n{'='*60}")
        print(f"Analyzing file {i+1}/2: {vtk_file.name}")
        print('='*60)

        mesh = load_vtk_file(vtk_file)
        if mesh is None:
            continue

        # Analyze VOF field
        stats = analyze_vof_field(mesh, vtk_file.stem)
        if stats is None:
            continue

        # Compute interface sharpness
        dims = mesh.dimensions
        fill_3d = stats['fill_level'].reshape(dims, order='F')
        sharpness, intermediate, total = compute_interface_sharpness(fill_3d)
        print(f"\nInterface Sharpness:")
        print(f"  Sharpness metric: {sharpness:.4f}")
        print(f"  Intermediate cells (0.1 < F < 0.9): {intermediate}/{total} ({100*intermediate/total:.2f}%)")

        # Plot profiles
        plot_vof_profiles(mesh, output_dir)

        # Plot distribution
        plot_vof_distribution(stats, output_dir)

        # Plot 2D slice
        plot_2d_vof_slice(mesh, SLICE_Z_INDEX, output_dir)

    # Time series analysis: mass conservation
    print(f"\n{'='*60}")
    print("Time Series Analysis: Mass Conservation")
    print('='*60)

    timesteps = []
    total_volumes = []
    liquid_volumes = []

    for vtk_file in vtk_files:
        mesh = pv.read(vtk_file)
        if 'FillLevel' not in mesh.array_names:
            continue

        fill_level = mesh['FillLevel']
        total_vol = np.sum(fill_level)

        # Extract timestep from filename
        timestep = int(vtk_file.stem.split('_')[-1])
        timesteps.append(timestep)
        total_volumes.append(total_vol)

    timesteps = np.array(timesteps)
    total_volumes = np.array(total_volumes)

    # Compute mass conservation error
    if len(total_volumes) > 0:
        initial_volume = total_volumes[0]
        mass_error = (total_volumes - initial_volume) / initial_volume * 100  # Percentage

        print(f"\nMass Conservation Analysis:")
        print(f"  Initial volume: {initial_volume:.2f}")
        print(f"  Final volume:   {total_volumes[-1]:.2f}")
        print(f"  Change:         {total_volumes[-1] - initial_volume:.2f} ({mass_error[-1]:.4f}%)")
        print(f"  Max error:      {np.abs(mass_error).max():.4f}%")
        print(f"  Mean error:     {np.abs(mass_error).mean():.4f}%")

        # Plot time series
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Total volume
        axes[0].plot(timesteps, total_volumes, 'b-o', linewidth=2)
        axes[0].axhline(y=initial_volume, color='r', linestyle='--', label='Initial volume')
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('Total Liquid Volume (sum of fill levels)')
        axes[0].set_title('Mass Conservation: Total Liquid Volume')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Mass error
        axes[1].plot(timesteps, mass_error, 'r-s', linewidth=2)
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=1)
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Mass Error (%)')
        axes[1].set_title('Mass Conservation Error Relative to Initial Volume')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / 'vof_mass_conservation.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved mass conservation plot to: {output_file}")
        plt.close()

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print('='*60)

    return 0

if __name__ == '__main__':
    sys.exit(main())
