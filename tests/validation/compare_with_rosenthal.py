#!/usr/bin/env python3
"""
Compare Case 5 LBM simulation with Rosenthal analytical solution.
Extracts temperature profiles and computes L2 error norms.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add vtk_comparison tools to path
sys.path.insert(0, '/home/yzk/LBMProject/benchmark/vtk_comparison')
from vtk_unified_reader import VTKUnifiedReader

# === PARAMETERS ===
VTK_DIR = "/home/yzk/LBMProject/tests/validation/output_laser_melting_senior"
OUTPUT_DIR = "/home/yzk/LBMProject/tests/validation/analysis_case5"

# Select timestep for detailed comparison (from previous analysis: timestep 1700 has peak)
COMPARISON_TIMESTEP = "001700"

# Physical parameters for Ti-6Al-4V
T_AMBIENT = 300.0  # K
T_MELTING = 1923.0  # K
T_BOILING = 3560.0  # K

# Laser and material parameters
LASER_POWER = 200.0  # W
BEAM_RADIUS = 50e-6  # m (50 um)
SCAN_VELOCITY = 1.0  # m/s
ABSORPTION = 0.3  # absorptivity

# Material properties (Ti-6Al-4V)
THERMAL_CONDUCTIVITY = 21.0  # W/(m*K) at ~2000K
DENSITY = 4110.0  # kg/m^3
SPECIFIC_HEAT = 670.0  # J/(kg*K)
THERMAL_DIFFUSIVITY = THERMAL_CONDUCTIVITY / (DENSITY * SPECIFIC_HEAT)  # m^2/s

def rosenthal_3d_moving_point_source(x, y, z, t, x0=0, y0=0, z0=0):
    """
    3D moving point source Rosenthal solution.

    Parameters:
        x, y, z: position (m)
        t: time (s)
        x0, y0, z0: initial laser position (m)
    """
    # Effective power (accounting for absorption)
    P_eff = ABSORPTION * LASER_POWER

    # Moving coordinate system (laser moving in +x direction)
    x_laser = x0 + SCAN_VELOCITY * t
    x_rel = x - x_laser
    y_rel = y - y0
    z_rel = z - z0

    # Distance from moving heat source
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

    # Avoid singularity
    r = np.maximum(r, 1e-9)

    # Peclet number
    Pe = SCAN_VELOCITY * r / (2 * THERMAL_DIFFUSIVITY)

    # Rosenthal solution
    T = T_AMBIENT + (P_eff / (2 * np.pi * THERMAL_CONDUCTIVITY * r)) * \
        np.exp(-Pe * (1 + x_rel / r))

    return T

def read_vtk_field(filename):
    """Read temperature field from VTK file."""
    print(f"Reading: {filename}")
    reader = VTKUnifiedReader(verbose=False)
    mesh = reader.read(filename)

    # Extract coordinates and temperature
    points = mesh.points
    if 'Temperature' in mesh.point_data:
        temperature = mesh.point_data['Temperature']
    elif 'temperature' in mesh.point_data:
        temperature = mesh.point_data['temperature']
    else:
        available = list(mesh.point_data.keys())
        raise ValueError(f"Temperature field not found. Available fields: {available}")

    print(f"  Grid points: {len(points)}")
    print(f"  Temperature range: {temperature.min():.2f} - {temperature.max():.2f} K")

    return points, temperature

def extract_centerline_profile(points, temperature, axis='x', y_center=None, z_center=None):
    """Extract temperature profile along centerline."""
    if y_center is None:
        y_center = np.median(points[:, 1])
    if z_center is None:
        z_center = np.median(points[:, 2])

    # Find points near centerline (within small tolerance)
    tol_y = np.diff(np.unique(points[:, 1])).min() if len(np.unique(points[:, 1])) > 1 else 1e-6
    tol_z = np.diff(np.unique(points[:, 2])).min() if len(np.unique(points[:, 2])) > 1 else 1e-6

    mask = (np.abs(points[:, 1] - y_center) < tol_y) & \
           (np.abs(points[:, 2] - z_center) < tol_z)

    if mask.sum() == 0:
        print(f"WARNING: No points found on centerline at y={y_center}, z={z_center}")
        # Relax tolerance
        tol_y *= 2
        tol_z *= 2
        mask = (np.abs(points[:, 1] - y_center) < tol_y) & \
               (np.abs(points[:, 2] - z_center) < tol_z)
        print(f"  Relaxed tolerance, found {mask.sum()} points")

    centerline_points = points[mask, 0]
    centerline_temps = temperature[mask]

    # Sort by position
    sort_idx = np.argsort(centerline_points)

    return centerline_points[sort_idx], centerline_temps[sort_idx]

def extract_depth_profile(points, temperature, x_center=None, y_center=None):
    """Extract temperature profile along depth (z-direction)."""
    if x_center is None:
        x_center = np.median(points[:, 0])
    if y_center is None:
        y_center = np.median(points[:, 1])

    # Find points near vertical line
    tol_x = np.diff(np.unique(points[:, 0])).min() if len(np.unique(points[:, 0])) > 1 else 1e-6
    tol_y = np.diff(np.unique(points[:, 1])).min() if len(np.unique(points[:, 1])) > 1 else 1e-6

    mask = (np.abs(points[:, 0] - x_center) < tol_x) & \
           (np.abs(points[:, 1] - y_center) < tol_y)

    if mask.sum() == 0:
        # Relax tolerance
        tol_x *= 2
        tol_y *= 2
        mask = (np.abs(points[:, 0] - x_center) < tol_x) & \
               (np.abs(points[:, 1] - y_center) < tol_y)

    depth_points = points[mask, 2]
    depth_temps = temperature[mask]

    # Sort by depth (descending, from surface)
    sort_idx = np.argsort(-depth_points)

    return depth_points[sort_idx], depth_temps[sort_idx]

def compute_l2_error(T_lbm, T_analytical):
    """Compute L2 relative error norm."""
    numerator = np.sqrt(np.mean((T_lbm - T_analytical)**2))
    denominator = np.sqrt(np.mean(T_analytical**2))

    if denominator < 1e-10:
        return np.nan

    return numerator / denominator

def main():
    """Main comparison workflow."""
    print("=" * 80)
    print("CASE 5: LBM vs ROSENTHAL ANALYTICAL SOLUTION")
    print("=" * 80)

    # Find VTK file for comparison timestep
    vtk_file = Path(VTK_DIR) / f"temperature_{COMPARISON_TIMESTEP}.vtk.vtk"

    if not vtk_file.exists():
        print(f"ERROR: VTK file not found: {vtk_file}")
        return

    # Read LBM data
    points, temp_lbm = read_vtk_field(str(vtk_file))

    # Estimate physical time from timestep
    # Assuming dt = 1e-8 s (typical LBM timestep for this scale)
    timestep_num = int(COMPARISON_TIMESTEP)
    dt = 1e-8  # s
    t_physical = timestep_num * dt
    print(f"\nPhysical time: {t_physical*1e6:.2f} us")

    # Get domain center (likely laser position)
    x_center = np.median(points[:, 0])
    y_center = np.median(points[:, 1])
    z_surface = np.max(points[:, 2])  # Surface is at max z

    print(f"\nDomain center: ({x_center*1e6:.2f}, {y_center*1e6:.2f}) um")
    print(f"Surface at z = {z_surface*1e6:.2f} um")

    # Extract centerline profile (along x-axis, at surface)
    print("\n" + "-" * 80)
    print("EXTRACTING CENTERLINE PROFILE (X-AXIS)")
    print("-" * 80)
    x_profile, T_lbm_centerline = extract_centerline_profile(
        points, temp_lbm, axis='x', y_center=y_center, z_center=z_surface
    )

    # Compute analytical solution along centerline
    T_analytical_centerline = rosenthal_3d_moving_point_source(
        x_profile, y_center, z_surface, t_physical,
        x0=x_center, y0=y_center, z0=z_surface
    )

    # Extract depth profile (along z-axis, at domain center)
    print("\n" + "-" * 80)
    print("EXTRACTING DEPTH PROFILE (Z-AXIS)")
    print("-" * 80)
    z_profile, T_lbm_depth = extract_depth_profile(
        points, temp_lbm, x_center=x_center, y_center=y_center
    )

    # Compute analytical solution along depth
    T_analytical_depth = rosenthal_3d_moving_point_source(
        x_center, y_center, z_profile, t_physical,
        x0=x_center, y0=y_center, z0=z_surface
    )

    # Compute errors
    print("\n" + "-" * 80)
    print("ERROR ANALYSIS")
    print("-" * 80)

    l2_centerline = compute_l2_error(T_lbm_centerline, T_analytical_centerline)
    l2_depth = compute_l2_error(T_lbm_depth, T_analytical_depth)

    print(f"L2 relative error (centerline): {l2_centerline*100:.2f}%")
    print(f"L2 relative error (depth):      {l2_depth*100:.2f}%")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Centerline profile
    ax = axes[0]
    ax.plot(x_profile*1e6, T_lbm_centerline, 'b-', linewidth=2, label='LBM Simulation')
    ax.plot(x_profile*1e6, T_analytical_centerline, 'r--', linewidth=2, label='Rosenthal Solution')
    ax.axhline(y=T_MELTING, color='g', linestyle=':', linewidth=1.5, label=f'Melting ({T_MELTING:.0f} K)')
    ax.set_xlabel('X Position (um)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(f'Centerline Temperature Profile\nL2 Error: {l2_centerline*100:.2f}%',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Depth profile
    ax = axes[1]
    ax.plot(z_profile*1e6, T_lbm_depth, 'b-', linewidth=2, label='LBM Simulation')
    ax.plot(z_profile*1e6, T_analytical_depth, 'r--', linewidth=2, label='Rosenthal Solution')
    ax.axhline(y=T_MELTING, color='g', linestyle=':', linewidth=1.5, label=f'Melting ({T_MELTING:.0f} K)')
    ax.set_xlabel('Depth - Z Position (um)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(f'Depth Temperature Profile\nL2 Error: {l2_depth*100:.2f}%',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Depth increases downward

    plt.tight_layout()

    # Save
    output_file = f"{OUTPUT_DIR}/rosenthal_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Timestep analyzed: {COMPARISON_TIMESTEP} (t = {t_physical*1e6:.2f} us)")
    print(f"LBM peak temperature: {temp_lbm.max():.2f} K")
    print(f"Analytical peak temperature: {T_analytical_centerline.max():.2f} K")
    print(f"\nOverall L2 error: {max(l2_centerline, l2_depth)*100:.2f}%")

    if max(l2_centerline, l2_depth) < 0.15:
        print("VALIDATION: PASS (< 15% error threshold)")
    else:
        print("VALIDATION: FAIL (> 15% error threshold)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
