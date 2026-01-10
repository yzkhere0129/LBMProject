#!/usr/bin/env python3
"""
Compare thermal LBM (LBMProject) vs walberla FD simulation results.

Analyzes VTK output from both simulations to compare:
- Temperature evolution over time
- Spatial temperature distributions
- Peak temperature accuracy
- Error metrics

Expected VTK locations:
- walberla: /home/yzk/walberla/build/apps/showcases/LaserHeating/vtk_out/
- LBMProject: (to be specified based on test output location)
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import re

# === PARAMETERS ===
WALBERLA_VTK_DIR = Path("/home/yzk/walberla/build/apps/showcases/LaserHeating/vtk_out/laser_heating")
LBMPROJECT_VTK_DIR = Path("/home/yzk/LBMProject/tests/validation/output_thermal_walberla")  # Adjust if needed

# NOTE: If LBMProject test doesn't output VTK yet, you can add VTK output to
# test_thermal_walberla_match.cu by including the VTK writer from thermal_lbm.h

# Physical parameters matching the test
NX, NY, NZ = 200, 200, 100
DX = 2.0e-6  # 2 μm grid spacing
DT = 100e-9  # 100 ns timestep

DOMAIN_SIZE_X = NX * DX  # 400 μm
DOMAIN_SIZE_Y = NY * DX  # 400 μm
DOMAIN_SIZE_Z = NZ * DX  # 200 μm

LASER_CENTER_X = DOMAIN_SIZE_X / 2.0
LASER_CENTER_Y = DOMAIN_SIZE_Y / 2.0
LASER_CENTER_Z = DOMAIN_SIZE_Z  # Surface at top

# Expected results from test
EXPECTED_LBM_PEAK = 4017.0  # K
EXPECTED_WALBERLA_PEAK = 4099.0  # K
EXPECTED_PEAK_TIME = 50.0e-6  # 50 μs

OUTPUT_DIR = Path("/home/yzk/LBMProject/scripts/comparison_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Temperature field name (may vary between outputs)
TEMP_FIELD_NAMES = ['T', 'temperature', 'Temperature', 'scalar']


def extract_timestep_from_filename(filename):
    """Extract timestep number from VTK filename."""
    # Common patterns: file_001.vtu, simulation_step_001.vti, etc.
    match = re.search(r'(?:step_|_)(\d+)\.vt[uki]$', str(filename))
    if match:
        return int(match.group(1))
    return None


def load_vtk_timeseries(vtk_dir, pattern="*.vtu"):
    """Load all VTK files in directory, sorted by timestep."""
    vtk_dir = Path(vtk_dir)
    if not vtk_dir.exists():
        print(f"Warning: Directory {vtk_dir} does not exist")
        return []

    files = sorted(vtk_dir.glob(pattern))
    if not files:
        # Try other common extensions
        for ext in ["*.vtk", "*.vti", "*.pvtu"]:
            files = sorted(vtk_dir.glob(ext))
            if files:
                break

    if not files:
        print(f"Warning: No VTK files found in {vtk_dir}")
        return []

    # Sort by timestep number in filename
    files_with_ts = [(f, extract_timestep_from_filename(f)) for f in files]
    files_with_ts = [(f, ts) for f, ts in files_with_ts if ts is not None]
    files_with_ts.sort(key=lambda x: x[1])

    print(f"Found {len(files_with_ts)} VTK files in {vtk_dir.name}")
    return [f for f, ts in files_with_ts]


def get_temperature_field(mesh):
    """Extract temperature field from VTK mesh, trying common field names."""
    for name in TEMP_FIELD_NAMES:
        if name in mesh.array_names:
            return mesh[name]

    # If not found, try the first scalar field
    if len(mesh.array_names) > 0:
        print(f"Warning: Standard temp field not found. Using '{mesh.array_names[0]}'")
        return mesh[mesh.array_names[0]]

    raise ValueError(f"No temperature field found. Available: {mesh.array_names}")


def extract_temperature_stats(vtk_files, dt=DT):
    """
    Extract temperature statistics from VTK timeseries.

    Args:
        vtk_files: List of tuples (file_path, timestep_number)
        dt: Timestep size (default: 100 ns from test)

    Returns:
        times: array of times (s)
        max_temps: array of maximum temperatures (K)
        mean_temps: array of mean temperatures (K)
        center_temps: array of temperatures at laser center
    """
    if not vtk_files:
        return None, None, None, None

    n_files = len(vtk_files)
    max_temps = np.zeros(n_files)
    mean_temps = np.zeros(n_files)
    center_temps = np.zeros(n_files)
    times = np.zeros(n_files)

    for i, vtk_file in enumerate(vtk_files):
        mesh = pv.read(vtk_file)
        temp = get_temperature_field(mesh)

        max_temps[i] = np.max(temp)
        mean_temps[i] = np.mean(temp)

        # Extract temperature at laser center (if grid is structured)
        try:
            points = mesh.points
            # Find point closest to laser center
            distances = np.sqrt(
                (points[:, 0] - LASER_CENTER_X)**2 +
                (points[:, 1] - LASER_CENTER_Y)**2 +
                (points[:, 2] - LASER_CENTER_Z)**2
            )
            center_idx = np.argmin(distances)
            center_temps[i] = temp[center_idx]
        except:
            center_temps[i] = max_temps[i]  # Fallback

        # Compute time from timestep number extracted from filename
        timestep = extract_timestep_from_filename(vtk_file)
        if timestep is not None:
            times[i] = timestep * dt
        else:
            times[i] = i * dt

    return times, max_temps, mean_temps, center_temps


def interpolate_mesh_to_grid(mesh_src, mesh_target_points):
    """
    Interpolate unstructured mesh data to target grid points.

    Returns interpolated temperature array.
    """
    temp_src = get_temperature_field(mesh_src)
    points_src = mesh_src.points

    # Use nearest neighbor interpolation for robustness
    from scipy.spatial import cKDTree
    tree = cKDTree(points_src)
    distances, indices = tree.query(mesh_target_points)

    return temp_src[indices]


def compute_spatial_error(vtk_file1, vtk_file2):
    """
    Compute spatial error metrics between two VTK files.

    Returns:
        rmse: Root mean square error (K)
        max_error: Maximum absolute error (K)
        mean_error: Mean absolute error (K)
        relative_error: Relative error (%)
    """
    mesh1 = pv.read(vtk_file1)
    mesh2 = pv.read(vtk_file2)

    temp1 = get_temperature_field(mesh1)

    # Interpolate mesh2 to mesh1 grid
    temp2_interp = interpolate_mesh_to_grid(mesh2, mesh1.points)

    # Compute errors
    diff = temp1 - temp2_interp
    rmse = np.sqrt(np.mean(diff**2))
    max_error = np.max(np.abs(diff))
    mean_error = np.mean(np.abs(diff))

    # Relative error based on peak temperature
    peak_temp = max(np.max(temp1), np.max(temp2_interp))
    relative_error = (max_error / peak_temp) * 100 if peak_temp > 0 else 0

    return rmse, max_error, mean_error, relative_error


def plot_temperature_evolution(times_lbm, max_temps_lbm, times_wb, max_temps_wb):
    """Plot temperature evolution comparison."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Temperature vs time
    ax1.plot(times_lbm * 1e6, max_temps_lbm, 'b-o', label='LBMProject', markersize=4)
    ax1.plot(times_wb * 1e6, max_temps_wb, 'r-s', label='walberla FD', markersize=4)
    ax1.axhline(EXPECTED_LBM_PEAK, color='b', linestyle='--', alpha=0.5, label=f'Expected LBM: {EXPECTED_LBM_PEAK:.0f} K')
    ax1.axhline(EXPECTED_WALBERLA_PEAK, color='r', linestyle='--', alpha=0.5, label=f'Expected walberla: {EXPECTED_WALBERLA_PEAK:.0f} K')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Peak Temperature (K)')
    ax1.set_title('Temperature Evolution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative error over time
    if len(times_lbm) == len(times_wb):
        rel_error = np.abs(max_temps_lbm - max_temps_wb) / max_temps_wb * 100
        ax2.plot(times_lbm * 1e6, rel_error, 'g-o', markersize=4)
        ax2.axhline(2.01, color='k', linestyle='--', alpha=0.5, label='Expected: 2.01%')
        ax2.set_xlabel('Time (μs)')
        ax2.set_ylabel('Relative Error (%)')
        ax2.set_title('Peak Temperature Error Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Time arrays have different lengths\nCannot compute error',
                ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "temperature_evolution.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def plot_spatial_distribution_comparison(vtk_lbm, vtk_wb, time_us):
    """Plot spatial temperature distributions at a specific time."""
    mesh_lbm = pv.read(vtk_lbm)
    mesh_wb = pv.read(vtk_wb)

    temp_lbm = get_temperature_field(mesh_lbm)
    temp_wb = get_temperature_field(mesh_wb)

    fig = plt.figure(figsize=(15, 5))

    # LBMProject
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(mesh_lbm.points[:, 0] * 1e3,
                           mesh_lbm.points[:, 1] * 1e3,
                           mesh_lbm.points[:, 2] * 1e3,
                           c=temp_lbm, cmap='hot', s=1)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'LBMProject\nPeak: {np.max(temp_lbm):.0f} K')
    plt.colorbar(scatter1, ax=ax1, label='Temperature (K)', shrink=0.5)

    # walberla
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(mesh_wb.points[:, 0] * 1e3,
                           mesh_wb.points[:, 1] * 1e3,
                           mesh_wb.points[:, 2] * 1e3,
                           c=temp_wb, cmap='hot', s=1)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title(f'walberla FD\nPeak: {np.max(temp_wb):.0f} K')
    plt.colorbar(scatter2, ax=ax2, label='Temperature (K)', shrink=0.5)

    # Difference
    temp_wb_interp = interpolate_mesh_to_grid(mesh_wb, mesh_lbm.points)
    diff = temp_lbm - temp_wb_interp

    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(mesh_lbm.points[:, 0] * 1e3,
                           mesh_lbm.points[:, 1] * 1e3,
                           mesh_lbm.points[:, 2] * 1e3,
                           c=diff, cmap='RdBu_r', s=1,
                           vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    ax3.set_title(f'Difference (LBM - walberla)\nMax: {np.max(np.abs(diff)):.0f} K')
    plt.colorbar(scatter3, ax=ax3, label='ΔT (K)', shrink=0.5)

    plt.suptitle(f'Spatial Temperature Distribution at t = {time_us:.1f} μs', fontsize=14)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"spatial_comparison_t{time_us:.0f}us.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def plot_centerline_profiles(vtk_lbm, vtk_wb, time_us):
    """Plot temperature profiles along centerlines."""
    mesh_lbm = pv.read(vtk_lbm)
    mesh_wb = pv.read(vtk_wb)

    temp_lbm = get_temperature_field(mesh_lbm)
    temp_wb = get_temperature_field(mesh_wb)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # X-direction profile (along laser path, at surface)
    y_tol = DOMAIN_SIZE_Y / 100
    z_tol = DOMAIN_SIZE_Z / 100

    # LBM X-profile
    mask_lbm_x = (np.abs(mesh_lbm.points[:, 1] - LASER_CENTER_Y) < y_tol) & \
                 (np.abs(mesh_lbm.points[:, 2] - LASER_CENTER_Z) < z_tol)
    x_lbm = mesh_lbm.points[mask_lbm_x, 0] * 1e3
    t_lbm_x = temp_lbm[mask_lbm_x]
    sort_idx = np.argsort(x_lbm)

    # walberla X-profile
    mask_wb_x = (np.abs(mesh_wb.points[:, 1] - LASER_CENTER_Y) < y_tol) & \
                (np.abs(mesh_wb.points[:, 2] - LASER_CENTER_Z) < z_tol)
    x_wb = mesh_wb.points[mask_wb_x, 0] * 1e3
    t_wb_x = temp_wb[mask_wb_x]
    sort_idx_wb = np.argsort(x_wb)

    axes[0].plot(x_lbm[sort_idx], t_lbm_x[sort_idx], 'b-o', label='LBMProject', markersize=3)
    axes[0].plot(x_wb[sort_idx_wb], t_wb_x[sort_idx_wb], 'r-s', label='walberla', markersize=3)
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title('X-direction (surface centerline)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Z-direction profile (depth, at laser center)
    x_tol = DOMAIN_SIZE_X / 100

    mask_lbm_z = (np.abs(mesh_lbm.points[:, 0] - LASER_CENTER_X) < x_tol) & \
                 (np.abs(mesh_lbm.points[:, 1] - LASER_CENTER_Y) < y_tol)
    z_lbm = mesh_lbm.points[mask_lbm_z, 2] * 1e3
    t_lbm_z = temp_lbm[mask_lbm_z]
    sort_idx = np.argsort(z_lbm)

    mask_wb_z = (np.abs(mesh_wb.points[:, 0] - LASER_CENTER_X) < x_tol) & \
                (np.abs(mesh_wb.points[:, 1] - LASER_CENTER_Y) < y_tol)
    z_wb = mesh_wb.points[mask_wb_z, 2] * 1e3
    t_wb_z = temp_wb[mask_wb_z]
    sort_idx_wb = np.argsort(z_wb)

    axes[1].plot(z_lbm[sort_idx], t_lbm_z[sort_idx], 'b-o', label='LBMProject', markersize=3)
    axes[1].plot(z_wb[sort_idx_wb], t_wb_z[sort_idx_wb], 'r-s', label='walberla', markersize=3)
    axes[1].set_xlabel('Z (mm)')
    axes[1].set_ylabel('Temperature (K)')
    axes[1].set_title('Z-direction (depth at laser center)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Radial profile (from laser center, at surface)
    mask_lbm_surf = np.abs(mesh_lbm.points[:, 2] - LASER_CENTER_Z) < z_tol
    r_lbm = np.sqrt((mesh_lbm.points[mask_lbm_surf, 0] - LASER_CENTER_X)**2 +
                    (mesh_lbm.points[mask_lbm_surf, 1] - LASER_CENTER_Y)**2) * 1e3
    t_lbm_r = temp_lbm[mask_lbm_surf]

    mask_wb_surf = np.abs(mesh_wb.points[:, 2] - LASER_CENTER_Z) < z_tol
    r_wb = np.sqrt((mesh_wb.points[mask_wb_surf, 0] - LASER_CENTER_X)**2 +
                   (mesh_wb.points[mask_wb_surf, 1] - LASER_CENTER_Y)**2) * 1e3
    t_wb_r = temp_wb[mask_wb_surf]

    axes[2].scatter(r_lbm, t_lbm_r, c='b', s=10, alpha=0.5, label='LBMProject')
    axes[2].scatter(r_wb, t_wb_r, c='r', s=10, alpha=0.5, label='walberla')
    axes[2].set_xlabel('Radial distance from laser (mm)')
    axes[2].set_ylabel('Temperature (K)')
    axes[2].set_title('Radial profile (surface)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Temperature Profiles at t = {time_us:.1f} μs', fontsize=14)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"centerline_profiles_t{time_us:.0f}us.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main analysis workflow."""
    print("=" * 60)
    print("Thermal LBM vs walberla FD Comparison")
    print("=" * 60)

    # Load VTK files
    print("\n1. Loading VTK files...")
    vtk_files_wb = load_vtk_timeseries(WALBERLA_VTK_DIR)
    vtk_files_lbm = load_vtk_timeseries(LBMPROJECT_VTK_DIR)

    if not vtk_files_wb:
        print(f"ERROR: No walberla VTK files found in {WALBERLA_VTK_DIR}")
        return

    if not vtk_files_lbm:
        print(f"WARNING: No LBMProject VTK files found in {LBMPROJECT_VTK_DIR}")
        print("Please update LBMPROJECT_VTK_DIR path and rerun.")
        print("Continuing with walberla analysis only...")

    # Extract temperature evolution
    print("\n2. Extracting temperature statistics...")
    times_wb, max_temps_wb, mean_temps_wb, center_temps_wb = extract_temperature_stats(vtk_files_wb)

    if vtk_files_lbm:
        times_lbm, max_temps_lbm, mean_temps_lbm, center_temps_lbm = extract_temperature_stats(vtk_files_lbm)
    else:
        # Create dummy data for plotting
        times_lbm = times_wb
        max_temps_lbm = np.zeros_like(max_temps_wb)

    # Print summary statistics
    print("\n3. Summary Statistics:")
    print(f"   walberla:")
    print(f"     Peak temperature: {np.max(max_temps_wb):.1f} K at t = {times_wb[np.argmax(max_temps_wb)] * 1e6:.1f} μs")
    print(f"     Expected: {EXPECTED_WALBERLA_PEAK:.1f} K at {EXPECTED_PEAK_TIME * 1e6:.1f} μs")

    if vtk_files_lbm:
        print(f"   LBMProject:")
        print(f"     Peak temperature: {np.max(max_temps_lbm):.1f} K at t = {times_lbm[np.argmax(max_temps_lbm)] * 1e6:.1f} μs")
        print(f"     Expected: {EXPECTED_LBM_PEAK:.1f} K at {EXPECTED_PEAK_TIME * 1e6:.1f} μs")

        # Error calculation
        peak_idx_lbm = np.argmax(max_temps_lbm)
        peak_idx_wb = np.argmax(max_temps_wb)
        error = abs(max_temps_lbm[peak_idx_lbm] - max_temps_wb[peak_idx_wb]) / max_temps_wb[peak_idx_wb] * 100
        print(f"   Peak temperature error: {error:.2f}% (expected: 2.01%)")

    # Generate plots
    print("\n4. Generating comparison plots...")

    # Temperature evolution
    plot_temperature_evolution(times_lbm, max_temps_lbm, times_wb, max_temps_wb)

    # Find timestep closest to peak (50 μs)
    peak_idx_wb = np.argmin(np.abs(times_wb - EXPECTED_PEAK_TIME))

    if vtk_files_lbm:
        peak_idx_lbm = np.argmin(np.abs(times_lbm - EXPECTED_PEAK_TIME))

        # Spatial comparison at peak time
        print(f"   Comparing spatial distributions at t = {times_wb[peak_idx_wb] * 1e6:.1f} μs...")
        plot_spatial_distribution_comparison(
            vtk_files_lbm[peak_idx_lbm],
            vtk_files_wb[peak_idx_wb],
            times_wb[peak_idx_wb] * 1e6
        )

        # Centerline profiles
        print(f"   Extracting centerline profiles...")
        plot_centerline_profiles(
            vtk_files_lbm[peak_idx_lbm],
            vtk_files_wb[peak_idx_wb],
            times_wb[peak_idx_wb] * 1e6
        )

        # Error analysis
        print("\n5. Computing spatial error metrics...")
        rmse, max_error, mean_error, rel_error = compute_spatial_error(
            vtk_files_lbm[peak_idx_lbm],
            vtk_files_wb[peak_idx_wb]
        )
        print(f"   RMSE: {rmse:.2f} K")
        print(f"   Maximum error: {max_error:.2f} K")
        print(f"   Mean absolute error: {mean_error:.2f} K")
        print(f"   Relative error: {rel_error:.2f}%")

    # Save numerical results
    output_file = OUTPUT_DIR / "comparison_summary.txt"
    with open(output_file, 'w') as f:
        f.write("Thermal LBM vs walberla FD Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"walberla Peak: {np.max(max_temps_wb):.1f} K at {times_wb[np.argmax(max_temps_wb)] * 1e6:.1f} μs\n")
        if vtk_files_lbm:
            f.write(f"LBMProject Peak: {np.max(max_temps_lbm):.1f} K at {times_lbm[np.argmax(max_temps_lbm)] * 1e6:.1f} μs\n")
            f.write(f"Peak Error: {error:.2f}%\n")
            f.write(f"\nSpatial Error Metrics at t = {times_wb[peak_idx_wb] * 1e6:.1f} μs:\n")
            f.write(f"  RMSE: {rmse:.2f} K\n")
            f.write(f"  Max Error: {max_error:.2f} K\n")
            f.write(f"  Mean Error: {mean_error:.2f} K\n")
            f.write(f"  Relative Error: {rel_error:.2f}%\n")

    print(f"\n6. Results saved to: {OUTPUT_DIR}")
    print(f"   - temperature_evolution.png")
    if vtk_files_lbm:
        print(f"   - spatial_comparison_t{times_wb[peak_idx_wb] * 1e6:.0f}us.png")
        print(f"   - centerline_profiles_t{times_wb[peak_idx_wb] * 1e6:.0f}us.png")
    print(f"   - comparison_summary.txt")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
