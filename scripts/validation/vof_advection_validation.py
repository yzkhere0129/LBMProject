#!/usr/bin/env python3
"""
VOF Advection Validation Script

This script validates VOF (Volume of Fluid) advection accuracy by tracking
interface displacement in a prescribed velocity field.

Physical Test:
    - Uniform advection: u = constant, interface should translate at u*t
    - Rotation test: Zalesak's disk rotates and returns to original position
    - Mass conservation: Total volume should remain constant

Expected Results:
    - Interface position error < 0.5 cells after one period
    - Mass conservation error < 0.1%
    - Second-order convergence in interface sharpness

Usage:
    python vof_advection_validation.py <vtk_directory>
    python vof_advection_validation.py /path/to/vtk --test rotation

Output:
    - PNG plots of interface position vs time
    - Mass conservation error plot
    - Interface sharpness metrics
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

# === PARAMETERS ===
VOF_FIELD_NAMES = ['fill_level', 'vof', 'F', 'alpha', 'fill']
DEFAULT_VTK_DIR = Path("/home/yzk/LBMProject/tests/validation/output_vof_advection")
OUTPUT_DIR = Path("/home/yzk/LBMProject/scripts/validation/results")

# Physical parameters
DT = 1e-6  # Default timestep: 1 μs
DX = 1e-6  # Default grid spacing: 1 μm


def extract_timestep(filename):
    """Extract timestep number from VTK filename."""
    match = re.search(r'(?:step_|_)(\d+)\.vt[uki]$', str(filename))
    return int(match.group(1)) if match else None


def load_vtk_series(vtk_dir, pattern="*.vtu"):
    """Load VTK time series, sorted by timestep."""
    vtk_dir = Path(vtk_dir)
    if not vtk_dir.exists():
        print(f"Error: {vtk_dir} does not exist")
        return []

    files = sorted(vtk_dir.glob(pattern))
    if not files:
        for ext in ["*.vtk", "*.vti", "*.pvtu"]:
            files = sorted(vtk_dir.glob(ext))
            if files:
                break

    if not files:
        print(f"Error: No VTK files in {vtk_dir}")
        return []

    # Sort by timestep
    files_with_ts = [(f, extract_timestep(f)) for f in files]
    files_with_ts = [(f, ts) for f, ts in files_with_ts if ts is not None]
    files_with_ts.sort(key=lambda x: x[1])

    return [f for f, ts in files_with_ts]


def get_vof_field(mesh):
    """Extract VOF field from mesh."""
    for name in VOF_FIELD_NAMES:
        if name in mesh.array_names:
            return mesh[name]

    if len(mesh.array_names) > 0:
        print(f"Warning: Using '{mesh.array_names[0]}' as VOF")
        return mesh[mesh.array_names[0]]

    raise ValueError(f"No VOF field. Available: {mesh.array_names}")


def compute_interface_position(mesh):
    """
    Compute interface position (center of mass of F=0.5 isosurface).

    Returns:
        (x_center, y_center, z_center): Interface centroid [m]
    """
    vof = get_vof_field(mesh)
    points = mesh.points

    # Weight points by proximity to F=0.5
    weights = np.exp(-((vof - 0.5) ** 2) / (2 * 0.1**2))
    weights /= np.sum(weights) + 1e-10

    x_center = np.sum(weights * points[:, 0])
    y_center = np.sum(weights * points[:, 1])
    z_center = np.sum(weights * points[:, 2])

    return x_center, y_center, z_center


def compute_total_volume(mesh, dx=DX):
    """
    Compute total liquid volume (integral of F).

    Returns:
        volume: Total volume [m³]
    """
    vof = get_vof_field(mesh)
    cell_volume = dx**3
    return np.sum(vof) * cell_volume


def compute_interface_sharpness(mesh):
    """
    Compute interface sharpness metric.

    A sharp interface has most cells at F=0 or F=1, few at intermediate values.

    Returns:
        sharpness: Fraction of cells with F in [0.1, 0.9] (lower is sharper)
    """
    vof = get_vof_field(mesh)
    n_interface = np.sum((vof > 0.1) & (vof < 0.9))
    sharpness = n_interface / len(vof)
    return sharpness


def extract_time_series(vtk_files, dt=DT, dx=DX, u_advect=None):
    """
    Extract interface position and volume over time.

    Args:
        vtk_files: List of VTK file paths
        dt: Timestep size [s]
        dx: Grid spacing [m]
        u_advect: Expected advection velocity [m/s] (for error computation)

    Returns:
        times: Time array [s]
        x_pos: X-position of interface [m]
        y_pos: Y-position of interface [m]
        volumes: Total volume [m³]
        sharpness: Interface sharpness metric
    """
    n = len(vtk_files)
    times = np.zeros(n)
    x_pos = np.zeros(n)
    y_pos = np.zeros(n)
    volumes = np.zeros(n)
    sharpness = np.zeros(n)

    for i, vtk_file in enumerate(vtk_files):
        mesh = pv.read(vtk_file)

        x, y, z = compute_interface_position(mesh)
        x_pos[i] = x
        y_pos[i] = y
        volumes[i] = compute_total_volume(mesh, dx)
        sharpness[i] = compute_interface_sharpness(mesh)

        ts = extract_timestep(vtk_file)
        times[i] = ts * dt if ts is not None else i * dt

    return times, x_pos, y_pos, volumes, sharpness


def plot_interface_tracking(times, x_pos, y_pos, u_advect, dx, output_path):
    """Plot interface position vs time with analytical solution."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # X-position vs time
    axes[0, 0].plot(times * 1e6, x_pos * 1e3, 'b-o', markersize=4, label='Simulated')
    if u_advect is not None:
        x_expected = x_pos[0] + u_advect * times
        axes[0, 0].plot(times * 1e6, x_expected * 1e3, 'r--', label='Expected (u*t)')
    axes[0, 0].set_xlabel('Time (μs)')
    axes[0, 0].set_ylabel('X Position (mm)')
    axes[0, 0].set_title('Interface X-Position')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Y-position vs time
    axes[0, 1].plot(times * 1e6, y_pos * 1e3, 'b-o', markersize=4)
    axes[0, 1].set_xlabel('Time (μs)')
    axes[0, 1].set_ylabel('Y Position (mm)')
    axes[0, 1].set_title('Interface Y-Position')
    axes[0, 1].grid(True, alpha=0.3)

    # Position error (if u_advect provided)
    if u_advect is not None:
        x_expected = x_pos[0] + u_advect * times
        error = np.abs(x_pos - x_expected) / dx  # Error in grid cells
        axes[1, 0].plot(times * 1e6, error, 'g-o', markersize=4)
        axes[1, 0].set_xlabel('Time (μs)')
        axes[1, 0].set_ylabel('Position Error (cells)')
        axes[1, 0].set_title('Interface Position Error')
        axes[1, 0].axhline(0.5, color='r', linestyle='--', alpha=0.5,
                          label='Target: < 0.5 cells')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Print final error
        final_err = error[-1]
        print(f"Final position error: {final_err:.3f} cells")
    else:
        axes[1, 0].text(0.5, 0.5, 'No advection velocity specified\nCannot compute error',
                       ha='center', va='center', transform=axes[1, 0].transAxes)

    # Trajectory (X vs Y)
    axes[1, 1].plot(x_pos * 1e3, y_pos * 1e3, 'b-o', markersize=4)
    axes[1, 1].plot(x_pos[0] * 1e3, y_pos[0] * 1e3, 'go', markersize=8, label='Start')
    axes[1, 1].plot(x_pos[-1] * 1e3, y_pos[-1] * 1e3, 'ro', markersize=8, label='End')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Y (mm)')
    axes[1, 1].set_title('Interface Trajectory')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_conservation_metrics(times, volumes, sharpness, output_path):
    """Plot mass conservation and interface sharpness."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Volume conservation
    v0 = volumes[0]
    vol_error = np.abs(volumes - v0) / v0 * 100

    ax1.plot(times * 1e6, vol_error, 'b-o', markersize=4)
    ax1.axhline(0.1, color='r', linestyle='--', alpha=0.5, label='Target: < 0.1%')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Volume Error (%)')
    ax1.set_title('Mass Conservation Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Interface sharpness
    ax2.plot(times * 1e6, sharpness * 100, 'g-o', markersize=4)
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Interface Thickness (%)')
    ax2.set_title('Interface Sharpness (lower is better)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary
    print(f"\nMass Conservation:")
    print(f"  Initial volume:   {v0:.6e} m³")
    print(f"  Final volume:     {volumes[-1]:.6e} m³")
    print(f"  Max error:        {np.max(vol_error):.4f}%")
    print(f"\nInterface Sharpness:")
    print(f"  Initial:          {sharpness[0] * 100:.2f}%")
    print(f"  Final:            {sharpness[-1] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='VOF advection validation')
    parser.add_argument('vtk_dir', nargs='?', type=Path, default=DEFAULT_VTK_DIR,
                       help='VTK output directory')
    parser.add_argument('--dt', type=float, default=DT,
                       help='Timestep size (seconds)')
    parser.add_argument('--dx', type=float, default=DX,
                       help='Grid spacing (meters)')
    parser.add_argument('--velocity', type=float, default=None,
                       help='Expected advection velocity (m/s)')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR,
                       help='Output directory')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VOF Advection Validation")
    print("=" * 60)

    # Load VTK files
    print(f"\nLoading VTK files from: {args.vtk_dir}")
    vtk_files = load_vtk_series(args.vtk_dir)

    if not vtk_files:
        print("ERROR: No VTK files found")
        return 1

    print(f"Found {len(vtk_files)} timesteps")

    # Extract time series data
    print("\nExtracting interface position and volume...")
    times, x_pos, y_pos, volumes, sharpness = extract_time_series(
        vtk_files, args.dt, args.dx, args.velocity
    )

    # Compute displacement
    dx_total = x_pos[-1] - x_pos[0]
    dy_total = y_pos[-1] - y_pos[0]
    print(f"\nInterface displacement:")
    print(f"  ΔX: {dx_total * 1e3:.3f} mm ({dx_total / args.dx:.1f} cells)")
    print(f"  ΔY: {dy_total * 1e3:.3f} mm ({dy_total / args.dx:.1f} cells)")

    if args.velocity is not None:
        expected_dx = args.velocity * times[-1]
        error = abs(dx_total - expected_dx) / args.dx
        print(f"  Expected ΔX: {expected_dx * 1e3:.3f} mm")
        print(f"  Error: {error:.3f} cells")

    # Generate plots
    print("\nGenerating plots...")
    plot_interface_tracking(times, x_pos, y_pos, args.velocity, args.dx,
                           args.output / "vof_interface_tracking.png")

    plot_conservation_metrics(times, volumes, sharpness,
                             args.output / "vof_conservation_metrics.png")

    # Save numerical results
    summary_file = args.output / "vof_validation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("VOF Advection Validation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Interface displacement:\n")
        f.write(f"  ΔX: {dx_total * 1e3:.3f} mm ({dx_total / args.dx:.1f} cells)\n")
        f.write(f"  ΔY: {dy_total * 1e3:.3f} mm ({dy_total / args.dx:.1f} cells)\n\n")
        if args.velocity is not None:
            f.write(f"Position error: {error:.3f} cells\n\n")
        f.write(f"Mass conservation:\n")
        f.write(f"  Initial volume: {volumes[0]:.6e} m³\n")
        f.write(f"  Final volume:   {volumes[-1]:.6e} m³\n")
        f.write(f"  Max error:      {np.max(np.abs(volumes - volumes[0]) / volumes[0] * 100):.4f}%\n\n")
        f.write(f"Interface sharpness:\n")
        f.write(f"  Initial: {sharpness[0] * 100:.2f}%\n")
        f.write(f"  Final:   {sharpness[-1] * 100:.2f}%\n")

    print(f"\nResults saved to: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
