#!/usr/bin/env python3
"""
Analyze velocity field from VTK simulation output.

Extracts and analyzes velocity magnitude, components, streamlines,
and vorticity from multiphysics LBM-CUDA simulation data.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# === PARAMETERS (modify these) ===
VTK_DIR = "/home/yzk/LBMProject/build/phase6_test2c_visualization"
VTK_PATTERN = "marangoni_flow_*.vtk"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/results"
SLICE_Z_INDEX = 16  # Middle of z-dimension for 2D slices
VELOCITY_THRESHOLD = 1e-6  # m/s, minimum velocity to consider

def load_vtk_file(filepath):
    """Load VTK file and return mesh with data."""
    try:
        mesh = pv.read(filepath)
        print(f"Loaded: {filepath}")
        print(f"  Dimensions: {mesh.dimensions}")
        print(f"  Points: {mesh.n_points}")
        print(f"  Spacing: {mesh.spacing}")
        return mesh
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_velocity_field(mesh, output_prefix):
    """Extract and analyze velocity field statistics."""

    if 'Velocity' not in mesh.array_names:
        print("ERROR: No 'Velocity' field found in VTK file")
        print(f"Available fields: {mesh.array_names}")
        return None

    velocity = mesh['Velocity']
    print(f"\n=== Velocity Field Analysis ===")
    print(f"Shape: {velocity.shape}")

    # Compute velocity magnitude
    v_mag = np.linalg.norm(velocity, axis=1)
    v_x, v_y, v_z = velocity[:, 0], velocity[:, 1], velocity[:, 2]

    # Statistics
    print(f"\nVelocity Magnitude (m/s):")
    print(f"  Min:  {v_mag.min():.6e}")
    print(f"  Max:  {v_mag.max():.6e}")
    print(f"  Mean: {v_mag.mean():.6e}")
    print(f"  Std:  {v_mag.std():.6e}")

    print(f"\nVelocity Components (m/s):")
    print(f"  Vx - Min: {v_x.min():.6e}, Max: {v_x.max():.6e}, Mean: {v_x.mean():.6e}")
    print(f"  Vy - Min: {v_y.min():.6e}, Max: {v_y.max():.6e}, Mean: {v_y.mean():.6e}")
    print(f"  Vz - Min: {v_z.min():.6e}, Max: {v_z.max():.6e}, Mean: {v_z.mean():.6e}")

    # Check for active flow
    active_cells = np.sum(v_mag > VELOCITY_THRESHOLD)
    print(f"\nActive flow cells (v > {VELOCITY_THRESHOLD:.1e} m/s): {active_cells}/{len(v_mag)} ({100*active_cells/len(v_mag):.2f}%)")

    # Add magnitude to mesh for visualization
    mesh['VelocityMagnitude'] = v_mag

    # Check for NaN or Inf
    nan_count = np.sum(np.isnan(v_mag))
    inf_count = np.sum(np.isinf(v_mag))
    if nan_count > 0 or inf_count > 0:
        print(f"\nWARNING: Found {nan_count} NaN and {inf_count} Inf values!")

    return {
        'v_mag': v_mag,
        'v_x': v_x,
        'v_y': v_y,
        'v_z': v_z,
        'active_fraction': active_cells / len(v_mag)
    }

def extract_2d_slice(mesh, z_index):
    """Extract 2D slice at given z index."""
    dims = mesh.dimensions
    if z_index >= dims[2]:
        print(f"WARNING: z_index {z_index} >= dimensions[2] {dims[2]}, using middle")
        z_index = dims[2] // 2

    # Create slice plane
    z_coord = mesh.origin[2] + z_index * mesh.spacing[2]
    slice_plane = mesh.slice(normal='z', origin=(0, 0, z_coord))
    print(f"\nExtracted 2D slice at z={z_index} (z_coord={z_coord:.6e} m)")
    print(f"  Slice points: {slice_plane.n_points}")

    return slice_plane

def plot_velocity_profiles(mesh, output_dir):
    """Plot velocity profiles along x and y centerlines."""
    dims = mesh.dimensions
    spacing = mesh.spacing
    origin = mesh.origin

    # Get centerline indices
    x_center = dims[0] // 2
    y_center = dims[1] // 2
    z_center = dims[2] // 2

    velocity = mesh['Velocity']
    v_mag = np.linalg.norm(velocity, axis=1)

    # Reshape to 3D grid
    v_mag_3d = v_mag.reshape(dims, order='F')
    vx_3d = velocity[:, 0].reshape(dims, order='F')
    vy_3d = velocity[:, 1].reshape(dims, order='F')

    # Extract profiles
    x_coords = origin[0] + np.arange(dims[0]) * spacing[0]
    y_coords = origin[1] + np.arange(dims[1]) * spacing[1]

    # X-profile (along x at center y, z)
    x_profile = v_mag_3d[:, y_center, z_center]
    vx_profile = vx_3d[:, y_center, z_center]

    # Y-profile (along y at center x, z)
    y_profile = v_mag_3d[x_center, :, z_center]
    vy_profile = vy_3d[x_center, :, z_center]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # X-profile magnitude
    axes[0, 0].plot(x_coords * 1e6, x_profile * 1000, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('X Position (μm)')
    axes[0, 0].set_ylabel('Velocity Magnitude (mm/s)')
    axes[0, 0].set_title(f'Velocity Profile along X (y={y_center}, z={z_center})')
    axes[0, 0].grid(True, alpha=0.3)

    # X-component
    axes[0, 1].plot(x_coords * 1e6, vx_profile * 1000, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('X Position (μm)')
    axes[0, 1].set_ylabel('Vx (mm/s)')
    axes[0, 1].set_title('Vx Component along X')
    axes[0, 1].grid(True, alpha=0.3)

    # Y-profile magnitude
    axes[1, 0].plot(y_coords * 1e6, y_profile * 1000, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Y Position (μm)')
    axes[1, 0].set_ylabel('Velocity Magnitude (mm/s)')
    axes[1, 0].set_title(f'Velocity Profile along Y (x={x_center}, z={z_center})')
    axes[1, 0].grid(True, alpha=0.3)

    # Y-component
    axes[1, 1].plot(y_coords * 1e6, vy_profile * 1000, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Y Position (μm)')
    axes[1, 1].set_ylabel('Vy (mm/s)')
    axes[1, 1].set_title('Vy Component along Y')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'velocity_profiles.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved velocity profiles to: {output_file}")
    plt.close()

def plot_velocity_distribution(stats, output_dir):
    """Plot velocity magnitude distribution histogram."""
    v_mag = stats['v_mag']

    # Filter out near-zero velocities for better histogram
    active_v = v_mag[v_mag > VELOCITY_THRESHOLD]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # All velocities
    axes[0].hist(v_mag * 1000, bins=100, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Velocity Magnitude (mm/s)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Velocity Distribution (All Cells)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Active flow only
    if len(active_v) > 0:
        axes[1].hist(active_v * 1000, bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Velocity Magnitude (mm/s)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Velocity Distribution (Active Flow, v > {VELOCITY_THRESHOLD:.1e} m/s)')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No active flow detected',
                    ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()
    output_file = Path(output_dir) / 'velocity_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved velocity distribution to: {output_file}")
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

        # Analyze velocity field
        stats = analyze_velocity_field(mesh, vtk_file.stem)
        if stats is None:
            continue

        # Plot profiles
        plot_velocity_profiles(mesh, output_dir)

        # Plot distribution
        plot_velocity_distribution(stats, output_dir)

    # Time series analysis: track maximum velocity
    print(f"\n{'='*60}")
    print("Time Series Analysis: Maximum Velocity")
    print('='*60)

    timesteps = []
    max_velocities = []
    mean_velocities = []

    for vtk_file in vtk_files:
        mesh = pv.read(vtk_file)
        if 'Velocity' not in mesh.array_names:
            continue

        velocity = mesh['Velocity']
        v_mag = np.linalg.norm(velocity, axis=1)

        # Extract timestep from filename (e.g., marangoni_flow_010000.vtk -> 10000)
        timestep = int(vtk_file.stem.split('_')[-1])
        timesteps.append(timestep)
        max_velocities.append(v_mag.max())
        mean_velocities.append(v_mag.mean())

    # Plot time series
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timesteps, np.array(max_velocities) * 1000, 'b-o', label='Max Velocity', linewidth=2)
    ax.plot(timesteps, np.array(mean_velocities) * 1000, 'r-s', label='Mean Velocity', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_title('Velocity Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = output_dir / 'velocity_time_series.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved time series to: {output_file}")
    plt.close()

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print('='*60)

    return 0

if __name__ == '__main__':
    sys.exit(main())
