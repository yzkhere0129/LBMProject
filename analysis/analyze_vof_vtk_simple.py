#!/usr/bin/env python3
"""
Analyze VOF field from VTK files using simple text parsing.

No external dependencies beyond numpy and matplotlib.
Extracts fill level data directly from ASCII VTK files.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# === PARAMETERS (modify these) ===
VTK_DIR = "/home/yzk/LBMProject/build/phase6_test2c_visualization"
VTK_PATTERN = "marangoni_flow_*.vtk"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/results"
INTERFACE_THRESHOLD = 0.5
LIQUID_THRESHOLD = 0.9
GAS_THRESHOLD = 0.1

def parse_vtk_header(lines):
    """Parse VTK header."""
    dims = None
    origin = None
    spacing = None

    for line in lines[:20]:
        if 'DIMENSIONS' in line:
            parts = line.split()
            dims = [int(parts[1]), int(parts[2]), int(parts[3])]
        elif 'ORIGIN' in line:
            parts = line.split()
            origin = [float(parts[1]), float(parts[2]), float(parts[3])]
        elif 'SPACING' in line:
            parts = line.split()
            spacing = [float(parts[1]), float(parts[2]), float(parts[3])]

    return dims, origin, spacing

def load_fill_level_from_vtk(filepath):
    """Load fill level from ASCII VTK file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        print(f"Loading: {filepath}")

        dims, origin, spacing = parse_vtk_header(lines)
        if dims is None:
            print("ERROR: Could not parse VTK dimensions")
            return None

        n_points = dims[0] * dims[1] * dims[2]
        print(f"  Dimensions: {dims}")
        print(f"  Points: {n_points}")

        # Find FillLevel data
        fill_level = []
        in_fill_section = False
        skip_next = False

        for i, line in enumerate(lines):
            if 'SCALARS FillLevel' in line:
                in_fill_section = True
                skip_next = True  # Skip LOOKUP_TABLE line
                print(f"  Found FillLevel at line {i}")
                continue

            if skip_next:
                skip_next = False
                continue

            if in_fill_section:
                if line.strip().startswith('SCALARS') or line.strip().startswith('VECTORS'):
                    break

                try:
                    value = float(line.strip())
                    fill_level.append(value)
                except ValueError:
                    continue

        fill_level = np.array(fill_level)

        if len(fill_level) != n_points:
            print(f"WARNING: Expected {n_points} values, got {len(fill_level)}")

        return {
            'fill_level': fill_level,
            'dims': dims,
            'origin': origin,
            'spacing': spacing
        }

    except Exception as e:
        print(f"Error loading VTK file: {e}")
        return None

def analyze_vof_stats(data):
    """Compute VOF statistics."""
    fill_level = data['fill_level']

    print(f"\n=== VOF Fill Level Analysis ===")
    print(f"Shape: {fill_level.shape}")

    print(f"\nFill Level Statistics:")
    print(f"  Min:  {fill_level.min():.6f}")
    print(f"  Max:  {fill_level.max():.6f}")
    print(f"  Mean: {fill_level.mean():.6f}")
    print(f"  Std:  {fill_level.std():.6f}")

    # Phase distribution
    liquid = np.sum(fill_level > LIQUID_THRESHOLD)
    gas = np.sum(fill_level < GAS_THRESHOLD)
    interface = np.sum((fill_level >= GAS_THRESHOLD) & (fill_level <= LIQUID_THRESHOLD))
    total = len(fill_level)

    print(f"\nPhase Distribution:")
    print(f"  Liquid (F > {LIQUID_THRESHOLD}):  {liquid}/{total} ({100*liquid/total:.2f}%)")
    print(f"  Gas (F < {GAS_THRESHOLD}):     {gas}/{total} ({100*gas/total:.2f}%)")
    print(f"  Interface:  {interface}/{total} ({100*interface/total:.2f}%)")

    # Total volume
    total_vol = np.sum(fill_level)
    cell_vol = np.prod(data['spacing'])
    liquid_vol = total_vol * cell_vol

    print(f"\nLiquid Volume:")
    print(f"  Total fill sum: {total_vol:.2f}")
    print(f"  Liquid volume: {liquid_vol:.6e} m³")

    # Check bounds
    below = np.sum(fill_level < 0.0)
    above = np.sum(fill_level > 1.0)
    if below > 0 or above > 0:
        print(f"\nWARNING: Bound violations!")
        print(f"  < 0: {below}, > 1: {above}")

    # Check NaN/Inf
    nan_count = np.sum(np.isnan(fill_level))
    inf_count = np.sum(np.isinf(fill_level))
    if nan_count > 0 or inf_count > 0:
        print(f"\nWARNING: Found {nan_count} NaN and {inf_count} Inf!")

    return {
        'total_volume': total_vol,
        'liquid_cells': liquid,
        'gas_cells': gas,
        'interface_cells': interface
    }

def plot_vof_profiles(data, output_dir):
    """Plot VOF profiles along centerlines."""
    dims = data['dims']
    spacing = data['spacing']
    origin = data['origin']
    fill_level = data['fill_level']

    # Reshape to 3D
    fill_3d = fill_level.reshape(dims[0], dims[1], dims[2], order='F')

    # Centerlines
    x_center, y_center, z_center = dims[0]//2, dims[1]//2, dims[2]//2

    x_coords = (origin[0] + np.arange(dims[0]) * spacing[0]) * 1e6  # μm
    y_coords = (origin[1] + np.arange(dims[1]) * spacing[1]) * 1e6  # μm

    x_profile = fill_3d[:, y_center, z_center]
    y_profile = fill_3d[x_center, :, z_center]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_coords, x_profile, 'b-', linewidth=2)
    axes[0].axhline(INTERFACE_THRESHOLD, color='r', linestyle='--', label=f'Interface (F={INTERFACE_THRESHOLD})')
    axes[0].set_xlabel('X (μm)')
    axes[0].set_ylabel('Fill Level')
    axes[0].set_title('VOF along X')
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(y_coords, y_profile, 'b-', linewidth=2)
    axes[1].axhline(INTERFACE_THRESHOLD, color='r', linestyle='--', label=f'Interface (F={INTERFACE_THRESHOLD})')
    axes[1].set_xlabel('Y (μm)')
    axes[1].set_ylabel('Fill Level')
    axes[1].set_title('VOF along Y')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'vof_profiles.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved VOF profiles to: {output_file}")
    plt.close()

def plot_vof_distribution(data, output_dir):
    """Plot VOF distribution."""
    fill_level = data['fill_level']

    fig, ax = plt.subplots(figsize=(10, 6))

    counts, bins, patches = ax.hist(fill_level, bins=100, color='blue', alpha=0.7, edgecolor='black')

    # Color regions
    for i, patch in enumerate(patches):
        if bins[i] < GAS_THRESHOLD:
            patch.set_facecolor('lightblue')
        elif bins[i] > LIQUID_THRESHOLD:
            patch.set_facecolor('darkblue')
        else:
            patch.set_facecolor('orange')

    ax.axvline(GAS_THRESHOLD, color='r', linestyle='--', linewidth=2, label=f'Gas ({GAS_THRESHOLD})')
    ax.axvline(LIQUID_THRESHOLD, color='g', linestyle='--', linewidth=2, label=f'Liquid ({LIQUID_THRESHOLD})')
    ax.axvline(INTERFACE_THRESHOLD, color='k', linestyle='--', linewidth=2, label=f'Interface ({INTERFACE_THRESHOLD})')

    ax.set_xlabel('Fill Level')
    ax.set_ylabel('Frequency')
    ax.set_title('VOF Distribution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'vof_distribution.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved distribution to: {output_file}")
    plt.close()

def plot_2d_slice(data, z_index, output_dir):
    """Plot 2D slice of VOF."""
    dims = data['dims']
    fill_level = data['fill_level']
    spacing = data['spacing']
    origin = data['origin']

    if z_index >= dims[2]:
        z_index = dims[2] // 2

    fill_3d = fill_level.reshape(dims[0], dims[1], dims[2], order='F')
    slice_2d = fill_3d[:, :, z_index]

    x = (origin[0] + np.arange(dims[0]) * spacing[0]) * 1e6
    y = (origin[1] + np.arange(dims[1]) * spacing[1]) * 1e6

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(slice_2d.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')

    ax.contour(x, y, slice_2d.T, levels=[INTERFACE_THRESHOLD], colors='black', linewidths=2)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fill Level')

    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title(f'VOF 2D Slice (z={z_index})')

    plt.tight_layout()
    output_file = Path(output_dir) / f'vof_slice_z{z_index}.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved 2D slice to: {output_file}")
    plt.close()

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    vtk_files = sorted(Path(VTK_DIR).glob(VTK_PATTERN))

    if not vtk_files:
        print(f"ERROR: No VTK files found in {VTK_DIR}")
        return 1

    print(f"Found {len(vtk_files)} VTK files\n")

    # Analyze first and last
    for idx in [0, -1]:
        vtk_file = vtk_files[idx]
        print(f"\n{'='*60}")
        print(f"Analyzing: {vtk_file.name}")
        print('='*60)

        data = load_fill_level_from_vtk(vtk_file)
        if data is None:
            continue

        stats = analyze_vof_stats(data)
        plot_vof_profiles(data, output_dir)
        plot_vof_distribution(data, output_dir)
        plot_2d_slice(data, data['dims'][2]//2, output_dir)

    # Time series - mass conservation
    print(f"\n{'='*60}")
    print("Time Series: Mass Conservation")
    print('='*60)

    timesteps, volumes = [], []

    for vtk_file in vtk_files[::5]:  # Sample every 5th
        data = load_fill_level_from_vtk(vtk_file)
        if data is None:
            continue

        vol = np.sum(data['fill_level'])
        timestep = int(vtk_file.stem.split('_')[-1])

        timesteps.append(timestep)
        volumes.append(vol)

    timesteps = np.array(timesteps)
    volumes = np.array(volumes)

    if len(volumes) > 0:
        initial = volumes[0]
        mass_error = (volumes - initial) / initial * 100

        print(f"\nMass Conservation:")
        print(f"  Initial: {initial:.2f}")
        print(f"  Final:   {volumes[-1]:.2f}")
        print(f"  Change:  {volumes[-1] - initial:.2f} ({mass_error[-1]:.4f}%)")
        print(f"  Max error: {np.abs(mass_error).max():.4f}%")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        axes[0].plot(timesteps, volumes, 'b-o', linewidth=2)
        axes[0].axhline(initial, color='r', linestyle='--', label='Initial')
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('Total Volume')
        axes[0].set_title('Mass Conservation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(timesteps, mass_error, 'r-s', linewidth=2)
        axes[1].axhline(0, color='k', linestyle='-')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Mass Error (%)')
        axes[1].set_title('Mass Error')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / 'vof_mass_conservation.png'
        plt.savefig(output_file, dpi=150)
        print(f"\nSaved mass conservation to: {output_file}")
        plt.close()

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results: {output_dir}")
    print('='*60)

    return 0

if __name__ == '__main__':
    sys.exit(main())
