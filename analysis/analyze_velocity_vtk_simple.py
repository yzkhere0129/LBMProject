#!/usr/bin/env python3
"""
Analyze velocity field from VTK files using simple text parsing.

No external dependencies beyond numpy and matplotlib.
Extracts velocity data directly from ASCII VTK files.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

# === PARAMETERS (modify these) ===
VTK_DIR = "/home/yzk/LBMProject/build/phase6_test2c_visualization"
VTK_PATTERN = "marangoni_flow_*.vtk"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/results"
VELOCITY_THRESHOLD = 1e-6  # m/s

def parse_vtk_header(lines):
    """Parse VTK header to extract grid dimensions and spacing."""
    dims = None
    origin = None
    spacing = None

    for line in lines[:20]:  # Header is in first few lines
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

def load_velocity_from_vtk(filepath):
    """Load velocity field from ASCII VTK file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        print(f"Loading: {filepath}")

        # Parse header
        dims, origin, spacing = parse_vtk_header(lines)
        if dims is None:
            print("ERROR: Could not parse VTK dimensions")
            return None

        n_points = dims[0] * dims[1] * dims[2]
        print(f"  Dimensions: {dims}")
        print(f"  Points: {n_points}")
        print(f"  Spacing: {spacing}")

        # Find velocity data
        velocity = []
        in_velocity_section = False

        for i, line in enumerate(lines):
            if 'VECTORS Velocity' in line:
                in_velocity_section = True
                print(f"  Found velocity at line {i}")
                continue

            if in_velocity_section:
                if line.strip().startswith('SCALARS') or line.strip().startswith('LOOKUP_TABLE'):
                    break  # End of velocity section

                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        vx, vy, vz = float(parts[0]), float(parts[1]), float(parts[2])
                        velocity.append([vx, vy, vz])
                    except ValueError:
                        continue

        velocity = np.array(velocity)

        if len(velocity) != n_points:
            print(f"WARNING: Expected {n_points} velocity vectors, got {len(velocity)}")

        return {
            'velocity': velocity,
            'dims': dims,
            'origin': origin,
            'spacing': spacing
        }

    except Exception as e:
        print(f"Error loading VTK file: {e}")
        return None

def analyze_velocity_stats(data):
    """Compute velocity statistics."""
    velocity = data['velocity']

    # Compute magnitude
    v_mag = np.linalg.norm(velocity, axis=1)
    v_x, v_y, v_z = velocity[:, 0], velocity[:, 1], velocity[:, 2]

    print(f"\n=== Velocity Field Analysis ===")
    print(f"Shape: {velocity.shape}")

    print(f"\nVelocity Magnitude (m/s):")
    print(f"  Min:  {v_mag.min():.6e}")
    print(f"  Max:  {v_mag.max():.6e}")
    print(f"  Mean: {v_mag.mean():.6e}")
    print(f"  Std:  {v_mag.std():.6e}")

    print(f"\nVelocity Components (m/s):")
    print(f"  Vx - Min: {v_x.min():.6e}, Max: {v_x.max():.6e}, Mean: {v_x.mean():.6e}")
    print(f"  Vy - Min: {v_y.min():.6e}, Max: {v_y.max():.6e}, Mean: {v_y.mean():.6e}")
    print(f"  Vz - Min: {v_z.min():.6e}, Max: {v_z.max():.6e}, Mean: {v_z.mean():.6e}")

    active_cells = np.sum(v_mag > VELOCITY_THRESHOLD)
    print(f"\nActive flow cells (v > {VELOCITY_THRESHOLD:.1e} m/s): {active_cells}/{len(v_mag)} ({100*active_cells/len(v_mag):.2f}%)")

    # Check for issues
    nan_count = np.sum(np.isnan(v_mag))
    inf_count = np.sum(np.isinf(v_mag))
    if nan_count > 0 or inf_count > 0:
        print(f"\nWARNING: Found {nan_count} NaN and {inf_count} Inf values!")

    return {
        'v_mag': v_mag,
        'v_x': v_x,
        'v_y': v_y,
        'v_z': v_z
    }

def plot_velocity_profiles(data, stats, output_dir):
    """Plot velocity profiles along centerlines."""
    dims = data['dims']
    spacing = data['spacing']
    origin = data['origin']

    v_mag = stats['v_mag']
    velocity = data['velocity']

    # Reshape to 3D
    v_mag_3d = v_mag.reshape(dims[0], dims[1], dims[2], order='F')
    vx_3d = velocity[:, 0].reshape(dims[0], dims[1], dims[2], order='F')
    vy_3d = velocity[:, 1].reshape(dims[0], dims[1], dims[2], order='F')

    # Centerline indices
    x_center, y_center, z_center = dims[0]//2, dims[1]//2, dims[2]//2

    # Coordinate arrays
    x_coords = (origin[0] + np.arange(dims[0]) * spacing[0]) * 1e6  # μm
    y_coords = (origin[1] + np.arange(dims[1]) * spacing[1]) * 1e6  # μm

    # Extract profiles
    x_profile = v_mag_3d[:, y_center, z_center] * 1000  # mm/s
    y_profile = v_mag_3d[x_center, :, z_center] * 1000  # mm/s

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_coords, x_profile, 'b-', linewidth=2)
    axes[0].set_xlabel('X Position (μm)')
    axes[0].set_ylabel('Velocity Magnitude (mm/s)')
    axes[0].set_title(f'Velocity along X (y={y_center}, z={z_center})')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(y_coords, y_profile, 'r-', linewidth=2)
    axes[1].set_xlabel('Y Position (μm)')
    axes[1].set_ylabel('Velocity Magnitude (mm/s)')
    axes[1].set_title(f'Velocity along Y (x={x_center}, z={z_center})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'velocity_profiles.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved velocity profiles to: {output_file}")
    plt.close()

def plot_velocity_distribution(stats, output_dir):
    """Plot velocity distribution histogram."""
    v_mag = stats['v_mag']
    active_v = v_mag[v_mag > VELOCITY_THRESHOLD]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(v_mag * 1000, bins=100, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Velocity (mm/s)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Velocity Distribution (All Cells)')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    if len(active_v) > 0:
        axes[1].hist(active_v * 1000, bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Velocity (mm/s)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Active Flow (v > {VELOCITY_THRESHOLD:.1e} m/s)')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'velocity_distribution.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved distribution to: {output_file}")
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

        data = load_velocity_from_vtk(vtk_file)
        if data is None:
            continue

        stats = analyze_velocity_stats(data)
        plot_velocity_profiles(data, stats, output_dir)
        plot_velocity_distribution(stats, output_dir)

    # Time series
    print(f"\n{'='*60}")
    print("Time Series: Maximum Velocity")
    print('='*60)

    timesteps, max_vels, mean_vels = [], [], []

    for vtk_file in vtk_files[::5]:  # Sample every 5th file
        data = load_velocity_from_vtk(vtk_file)
        if data is None:
            continue

        v_mag = np.linalg.norm(data['velocity'], axis=1)
        timestep = int(vtk_file.stem.split('_')[-1])

        timesteps.append(timestep)
        max_vels.append(v_mag.max())
        mean_vels.append(v_mag.mean())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timesteps, np.array(max_vels) * 1000, 'b-o', label='Max', linewidth=2)
    ax.plot(timesteps, np.array(mean_vels) * 1000, 'r-s', label='Mean', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_title('Velocity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = output_dir / 'velocity_time_series.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved time series to: {output_file}")
    plt.close()

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results: {output_dir}")
    print('='*60)

    return 0

if __name__ == '__main__':
    sys.exit(main())
