#!/usr/bin/env python3
"""
Thermal Validation: LBM vs walberla FD Comparison

This script compares thermal LBM results against walberla finite difference solver.
Extracts temperature profiles, peak evolution, and error metrics from VTK output.

Usage:
    python thermal_walberla_comparison.py --lbm <lbm_vtk_dir> --walberla <walberla_vtk_dir>
    python thermal_walberla_comparison.py  # Uses default paths

Output:
    - PNG figures showing temperature evolution and spatial profiles
    - Numerical error metrics (RMSE, peak error, convergence order)
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

# === PARAMETERS (modify these) ===
DEFAULT_WALBERLA_DIR = Path("/home/yzk/walberla/build/apps/showcases/LaserHeating/vtk_out/laser_heating")
DEFAULT_LBM_DIR = Path("/home/yzk/LBMProject/tests/validation/output_thermal_walberla")
OUTPUT_DIR = Path("/home/yzk/LBMProject/scripts/validation/results")

# Physical parameters
DT = 100e-9  # Default timestep: 100 ns
TEMP_FIELD_NAMES = ['T', 'temperature', 'Temperature', 'scalar']


def extract_timestep(filename):
    """Extract timestep number from VTK filename."""
    match = re.search(r'(?:step_|_)(\d+)\.vt[uki]$', str(filename))
    return int(match.group(1)) if match else None


def load_vtk_series(vtk_dir, pattern="*.vtu"):
    """Load VTK time series, sorted by timestep."""
    vtk_dir = Path(vtk_dir)
    if not vtk_dir.exists():
        print(f"Warning: {vtk_dir} does not exist")
        return []

    files = sorted(vtk_dir.glob(pattern))
    if not files:
        for ext in ["*.vtk", "*.vti", "*.pvtu"]:
            files = sorted(vtk_dir.glob(ext))
            if files:
                break

    if not files:
        return []

    # Sort by timestep number
    files_with_ts = [(f, extract_timestep(f)) for f in files]
    files_with_ts = [(f, ts) for f, ts in files_with_ts if ts is not None]
    files_with_ts.sort(key=lambda x: x[1])

    return [f for f, ts in files_with_ts]


def get_temperature(mesh):
    """Extract temperature field from VTK mesh."""
    for name in TEMP_FIELD_NAMES:
        if name in mesh.array_names:
            return mesh[name]

    if len(mesh.array_names) > 0:
        print(f"Warning: Using '{mesh.array_names[0]}' as temperature")
        return mesh[mesh.array_names[0]]

    raise ValueError(f"No temperature field. Available: {mesh.array_names}")


def extract_peak_evolution(vtk_files, dt=DT):
    """
    Extract peak temperature evolution over time.

    Returns:
        times (array): Time points [s]
        peaks (array): Peak temperatures [K]
        means (array): Mean temperatures [K]
    """
    if not vtk_files:
        return None, None, None

    n = len(vtk_files)
    times = np.zeros(n)
    peaks = np.zeros(n)
    means = np.zeros(n)

    for i, vtk_file in enumerate(vtk_files):
        mesh = pv.read(vtk_file)
        temp = get_temperature(mesh)

        peaks[i] = np.max(temp)
        means[i] = np.mean(temp)

        ts = extract_timestep(vtk_file)
        times[i] = ts * dt if ts is not None else i * dt

    return times, peaks, means


def interpolate_to_grid(mesh_src, target_points):
    """Interpolate mesh data to target grid using nearest neighbor."""
    temp_src = get_temperature(mesh_src)
    tree = cKDTree(mesh_src.points)
    distances, indices = tree.query(target_points)
    return temp_src[indices]


def compute_errors(vtk_lbm, vtk_ref):
    """
    Compute spatial error metrics between LBM and reference.

    Returns:
        rmse: Root mean square error [K]
        max_error: Maximum absolute error [K]
        rel_error: Relative error [%]
    """
    mesh_lbm = pv.read(vtk_lbm)
    mesh_ref = pv.read(vtk_ref)

    temp_lbm = get_temperature(mesh_lbm)
    temp_ref_interp = interpolate_to_grid(mesh_ref, mesh_lbm.points)

    diff = temp_lbm - temp_ref_interp
    rmse = np.sqrt(np.mean(diff**2))
    max_error = np.max(np.abs(diff))

    peak = max(np.max(temp_lbm), np.max(temp_ref_interp))
    rel_error = (max_error / peak) * 100 if peak > 0 else 0

    return rmse, max_error, rel_error


def plot_peak_evolution(times_lbm, peaks_lbm, times_ref, peaks_ref, output_path):
    """Plot peak temperature evolution comparison."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Temperature vs time
    ax1.plot(times_lbm * 1e6, peaks_lbm, 'b-o', label='LBM', markersize=4)
    ax1.plot(times_ref * 1e6, peaks_ref, 'r-s', label='walberla FD', markersize=4)
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Peak Temperature (K)')
    ax1.set_title('Peak Temperature Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative error
    if len(times_lbm) == len(times_ref):
        rel_err = np.abs(peaks_lbm - peaks_ref) / peaks_ref * 100
        ax2.plot(times_lbm * 1e6, rel_err, 'g-o', markersize=4)
        ax2.set_xlabel('Time (μs)')
        ax2.set_ylabel('Relative Error (%)')
        ax2.set_title('Peak Temperature Error')
        ax2.grid(True, alpha=0.3)

        # Print peak error
        max_err = np.max(rel_err)
        ax2.axhline(max_err, color='r', linestyle='--', alpha=0.5,
                   label=f'Max: {max_err:.2f}%')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Time series lengths differ\nCannot compute error',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_spatial_profile(vtk_lbm, vtk_ref, time_us, output_path):
    """Plot centerline temperature profiles."""
    mesh_lbm = pv.read(vtk_lbm)
    mesh_ref = pv.read(vtk_ref)

    temp_lbm = get_temperature(mesh_lbm)
    temp_ref = get_temperature(mesh_ref)

    # Compute domain center
    bounds_lbm = mesh_lbm.bounds
    cx = (bounds_lbm[0] + bounds_lbm[1]) / 2
    cy = (bounds_lbm[2] + bounds_lbm[3]) / 2
    cz = bounds_lbm[5]  # Surface

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # X-direction profile (along surface)
    tol = (bounds_lbm[1] - bounds_lbm[0]) / 100

    mask_lbm = (np.abs(mesh_lbm.points[:, 1] - cy) < tol) & \
               (np.abs(mesh_lbm.points[:, 2] - cz) < tol)
    x_lbm = mesh_lbm.points[mask_lbm, 0] * 1e3  # Convert to mm
    t_lbm = temp_lbm[mask_lbm]
    idx = np.argsort(x_lbm)

    mask_ref = (np.abs(mesh_ref.points[:, 1] - cy) < tol) & \
               (np.abs(mesh_ref.points[:, 2] - cz) < tol)
    x_ref = mesh_ref.points[mask_ref, 0] * 1e3
    t_ref = temp_ref[mask_ref]
    idx_ref = np.argsort(x_ref)

    axes[0].plot(x_lbm[idx], t_lbm[idx], 'b-o', label='LBM', markersize=3)
    axes[0].plot(x_ref[idx_ref], t_ref[idx_ref], 'r-s', label='walberla FD', markersize=3)
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title('Surface Centerline Profile')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Z-direction profile (depth)
    mask_lbm_z = (np.abs(mesh_lbm.points[:, 0] - cx) < tol) & \
                 (np.abs(mesh_lbm.points[:, 1] - cy) < tol)
    z_lbm = mesh_lbm.points[mask_lbm_z, 2] * 1e3
    t_lbm_z = temp_lbm[mask_lbm_z]
    idx_z = np.argsort(z_lbm)

    mask_ref_z = (np.abs(mesh_ref.points[:, 0] - cx) < tol) & \
                 (np.abs(mesh_ref.points[:, 1] - cy) < tol)
    z_ref = mesh_ref.points[mask_ref_z, 2] * 1e3
    t_ref_z = temp_ref[mask_ref_z]
    idx_ref_z = np.argsort(z_ref)

    axes[1].plot(z_lbm[idx_z], t_lbm_z[idx_z], 'b-o', label='LBM', markersize=3)
    axes[1].plot(z_ref[idx_ref_z], t_ref_z[idx_ref_z], 'r-s', label='walberla FD', markersize=3)
    axes[1].set_xlabel('Z (mm)')
    axes[1].set_ylabel('Temperature (K)')
    axes[1].set_title('Depth Profile at Laser Center')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Temperature Profiles at t = {time_us:.1f} μs', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare LBM vs walberla thermal solutions')
    parser.add_argument('--lbm', type=Path, default=DEFAULT_LBM_DIR,
                       help='LBM VTK directory')
    parser.add_argument('--walberla', type=Path, default=DEFAULT_WALBERLA_DIR,
                       help='walberla VTK directory')
    parser.add_argument('--dt', type=float, default=DT,
                       help='Timestep size (seconds)')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR,
                       help='Output directory for results')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Thermal Validation: LBM vs walberla FD")
    print("=" * 60)

    # Load VTK files
    print("\nLoading VTK files...")
    vtk_lbm = load_vtk_series(args.lbm)
    vtk_ref = load_vtk_series(args.walberla)

    if not vtk_ref:
        print(f"ERROR: No reference files in {args.walberla}")
        return 1

    if not vtk_lbm:
        print(f"ERROR: No LBM files in {args.lbm}")
        return 1

    print(f"  LBM:      {len(vtk_lbm)} files")
    print(f"  walberla: {len(vtk_ref)} files")

    # Extract peak evolution
    print("\nExtracting temperature evolution...")
    times_lbm, peaks_lbm, means_lbm = extract_peak_evolution(vtk_lbm, args.dt)
    times_ref, peaks_ref, means_ref = extract_peak_evolution(vtk_ref, args.dt)

    # Summary statistics
    print("\nPeak Temperatures:")
    print(f"  LBM:      {np.max(peaks_lbm):.1f} K at t = {times_lbm[np.argmax(peaks_lbm)] * 1e6:.1f} μs")
    print(f"  walberla: {np.max(peaks_ref):.1f} K at t = {times_ref[np.argmax(peaks_ref)] * 1e6:.1f} μs")

    peak_err = abs(np.max(peaks_lbm) - np.max(peaks_ref)) / np.max(peaks_ref) * 100
    print(f"  Error:    {peak_err:.2f}%")

    # Generate plots
    print("\nGenerating plots...")
    plot_peak_evolution(times_lbm, peaks_lbm, times_ref, peaks_ref,
                       args.output / "thermal_peak_evolution.png")

    # Spatial comparison at peak time
    peak_idx_ref = np.argmax(peaks_ref)
    peak_idx_lbm = np.argmin(np.abs(times_lbm - times_ref[peak_idx_ref]))

    plot_spatial_profile(vtk_lbm[peak_idx_lbm], vtk_ref[peak_idx_ref],
                        times_ref[peak_idx_ref] * 1e6,
                        args.output / "thermal_spatial_profiles.png")

    # Compute spatial errors
    print("\nSpatial Error Metrics:")
    rmse, max_err, rel_err = compute_errors(vtk_lbm[peak_idx_lbm], vtk_ref[peak_idx_ref])
    print(f"  RMSE:         {rmse:.2f} K")
    print(f"  Max Error:    {max_err:.2f} K")
    print(f"  Rel. Error:   {rel_err:.2f}%")

    # Save summary
    summary_file = args.output / "thermal_comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Thermal LBM vs walberla FD Comparison\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Peak Temperatures:\n")
        f.write(f"  LBM:      {np.max(peaks_lbm):.1f} K\n")
        f.write(f"  walberla: {np.max(peaks_ref):.1f} K\n")
        f.write(f"  Error:    {peak_err:.2f}%\n\n")
        f.write(f"Spatial Errors at peak time:\n")
        f.write(f"  RMSE:      {rmse:.2f} K\n")
        f.write(f"  Max Error: {max_err:.2f} K\n")
        f.write(f"  Rel. Err:  {rel_err:.2f}%\n")

    print(f"\nResults saved to: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
