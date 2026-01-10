#!/usr/bin/env python3
"""
Lid-Driven Cavity Validation

This script validates LBM fluid solver against benchmark lid-driven cavity
solutions from Ghia et al. (1982). The lid-driven cavity is a fundamental
CFD validation case featuring:
    - Steady-state incompressible flow
    - Primary and secondary vortices
    - Benchmark reference data at Re=100, 400, 1000, 3200, 5000, 7500, 10000

Physical Basis:
    Square cavity with moving top lid (u=U_lid, v=0)
    No-slip walls on remaining boundaries
    Reynolds number: Re = U_lid * L / nu

Expected Results:
    - Centerline velocity profiles match Ghia et al.
    - L∞ error < 1% for Re=100 (well-resolved)
    - L∞ error < 2% for Re=400 (adequate resolution)
    - L∞ error < 5% for Re≥1000 (requires fine grid)

Reference:
    Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for
    incompressible flow using the Navier-Stokes equations and a multigrid method.
    Journal of Computational Physics, 48(3), 387-411.

Usage:
    python fluid_lid_driven_cavity_analysis.py --vtk <vtk_file> --re 100
    python fluid_lid_driven_cavity_analysis.py  # Uses default Re=100

Output:
    - lid_driven_cavity_re{Re}_profiles.png: u(y) and v(x) vs Ghia
    - lid_driven_cavity_re{Re}_summary.txt: Error metrics and pass/fail
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# === PARAMETERS ===
DEFAULT_VTK_DIR = Path("/home/yzk/LBMProject/tests/validation/output_lid_cavity")
OUTPUT_DIR = Path("/home/yzk/LBMProject/scripts/validation/results")

VEL_FIELD_NAMES = ['Velocity', 'velocity', 'vel', 'u']

# Error thresholds for different Reynolds numbers
ERROR_THRESHOLDS = {
    100: 1.0,   # Well-resolved, expect <1% error
    400: 2.0,   # Adequate resolution, <2%
    1000: 5.0,  # Requires finer grid, <5%
}


# === GHIA ET AL. (1982) REFERENCE DATA ===
# Benchmark data for centerline velocities at various Reynolds numbers

GHIA_DATA = {
    100: {
        # u-velocity along vertical centerline (x=0.5)
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                       0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                       0.9688, 0.9766, 1.0000]),
        'u': np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                       -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                       0.68717, 0.73722, 0.78871, 0.84123, 1.00000]),

        # v-velocity along horizontal centerline (y=0.5)
        'x': np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                       0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                       0.9609, 0.9688, 1.0000]),
        'v': np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                       0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                       -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]),
    },

    400: {
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                       0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                       0.9688, 0.9766, 1.0000]),
        'u': np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
                       -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
                       0.55892, 0.61756, 0.68439, 0.75837, 1.00000]),

        'x': np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                       0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                       0.9609, 0.9688, 1.0000]),
        'v': np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124,
                       0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.38598,
                       -0.22847, -0.19254, -0.15663, -0.12146, 0.00000]),
    },

    1000: {
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                       0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                       0.9688, 0.9766, 1.0000]),
        'u': np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                       -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                       0.46604, 0.51117, 0.57492, 0.65928, 1.00000]),

        'x': np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                       0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                       0.9609, 0.9688, 1.0000]),
        'v': np.array([0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                       0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
                       -0.39188, -0.33714, -0.27669, -0.21388, 0.00000]),
    },
}


def get_velocity(mesh):
    """Extract velocity field from VTK mesh."""
    for name in VEL_FIELD_NAMES:
        if name in mesh.array_names:
            vel = mesh[name]
            # Ensure it's 3D (N, 3)
            if vel.ndim == 1:
                vel = vel.reshape(-1, 3)
            return vel

    raise ValueError(f"No velocity field. Available: {mesh.array_names}")


def extract_centerline_profile(mesh, direction='vertical'):
    """
    Extract velocity profile along cavity centerline.

    Args:
        mesh: PyVista mesh with velocity data
        direction: 'vertical' (x=0.5, extract u(y)) or 'horizontal' (y=0.5, extract v(x))

    Returns:
        coords: Normalized coordinates (y or x) in [0, 1]
        vel_component: Velocity component (u or v) normalized by U_lid
    """
    vel = get_velocity(mesh)
    points = mesh.points

    # Determine domain bounds and normalize coordinates
    bounds = mesh.bounds
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    L_x = x_max - x_min
    L_y = y_max - y_min

    # Normalize U_lid (assume top wall has max velocity)
    U_lid = np.max(np.abs(vel[:, 0]))  # Maximum u-velocity
    if U_lid < 1e-10:
        U_lid = 1.0  # Fallback

    if direction == 'vertical':
        # Extract u-velocity along vertical centerline (x = 0.5)
        x_center = (x_min + x_max) / 2.0
        tol = L_x * 0.02  # 2% tolerance

        mask = np.abs(points[:, 0] - x_center) < tol
        y_coords = (points[mask, 1] - y_min) / L_y  # Normalize to [0, 1]
        u_vel = vel[mask, 0] / U_lid

        # Sort by y
        sort_idx = np.argsort(y_coords)
        return y_coords[sort_idx], u_vel[sort_idx]

    elif direction == 'horizontal':
        # Extract v-velocity along horizontal centerline (y = 0.5)
        y_center = (y_min + y_max) / 2.0
        tol = L_y * 0.02

        mask = np.abs(points[:, 1] - y_center) < tol
        x_coords = (points[mask, 0] - x_min) / L_x
        v_vel = vel[mask, 1] / U_lid

        # Sort by x
        sort_idx = np.argsort(x_coords)
        return x_coords[sort_idx], v_vel[sort_idx]

    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")


def interpolate_to_reference(coords_sim, vel_sim, coords_ref):
    """Interpolate simulation data to reference coordinate points."""
    return np.interp(coords_ref, coords_sim, vel_sim)


def compute_errors(vel_sim, vel_ref):
    """
    Compute L2 and L∞ errors.

    Returns:
        l2_error: L2 norm of error
        linf_error: Maximum absolute error
        rel_linf: Relative L∞ error (%)
    """
    diff = vel_sim - vel_ref
    l2_error = np.sqrt(np.mean(diff**2))
    linf_error = np.max(np.abs(diff))

    # Relative error w.r.t. maximum velocity magnitude
    vel_range = np.max(np.abs(vel_ref))
    rel_linf = (linf_error / vel_range * 100) if vel_range > 0 else 0

    return l2_error, linf_error, rel_linf


def plot_profiles(coords_u_sim, u_sim, coords_v_sim, v_sim, re, ghia_data, output_path):
    """Plot velocity profiles vs Ghia reference data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # U-velocity along vertical centerline
    axes[0].plot(u_sim, coords_u_sim, 'b-', linewidth=2, label='LBM Simulation')
    axes[0].plot(ghia_data['u'], ghia_data['y'], 'ro', markersize=8,
                markerfacecolor='none', markeredgewidth=2, label='Ghia et al. (1982)')
    axes[0].set_xlabel('u / U_lid', fontsize=12)
    axes[0].set_ylabel('y / L', fontsize=12)
    axes[0].set_title(f'U-Velocity Along Vertical Centerline\n(x = 0.5, Re = {re})',
                     fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.6, 1.2)
    axes[0].set_ylim(0, 1)

    # V-velocity along horizontal centerline
    axes[1].plot(coords_v_sim, v_sim, 'b-', linewidth=2, label='LBM Simulation')
    axes[1].plot(ghia_data['x'], ghia_data['v'], 'ro', markersize=8,
                markerfacecolor='none', markeredgewidth=2, label='Ghia et al. (1982)')
    axes[1].set_xlabel('x / L', fontsize=12)
    axes[1].set_ylabel('v / U_lid', fontsize=12)
    axes[1].set_title(f'V-Velocity Along Horizontal Centerline\n(y = 0.5, Re = {re})',
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-0.6, 0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Lid-driven cavity validation against Ghia et al. (1982)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vtk', type=Path, default=DEFAULT_VTK_DIR,
                       help='VTK file or directory (uses final timestep)')
    parser.add_argument('--re', type=int, default=100, choices=[100, 400, 1000],
                       help='Reynolds number')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Pass/fail error threshold (%) [default: Re-dependent]')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR,
                       help='Output directory')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Set threshold based on Re if not provided
    if args.threshold is None:
        args.threshold = ERROR_THRESHOLDS.get(args.re, 5.0)

    print("=" * 70)
    print("Lid-Driven Cavity Validation")
    print("=" * 70)
    print(f"\nReynolds number: Re = {args.re}")
    print(f"Error threshold: {args.threshold}%")

    # Load reference data
    if args.re not in GHIA_DATA:
        print(f"ERROR: No Ghia reference data for Re={args.re}")
        print(f"Available: {list(GHIA_DATA.keys())}")
        return 1

    ghia = GHIA_DATA[args.re]
    print(f"\nGhia reference data: {len(ghia['y'])} points (u-profile), "
          f"{len(ghia['x'])} points (v-profile)")

    # Load VTK file
    vtk_path = Path(args.vtk)
    if vtk_path.is_dir():
        # Find final timestep
        vtk_files = sorted(vtk_path.glob("*.vtu"))
        if not vtk_files:
            vtk_files = sorted(vtk_path.glob("*.vtk"))
        if not vtk_files:
            print(f"ERROR: No VTK files in {vtk_path}")
            return 1
        vtk_file = vtk_files[-1]
        print(f"\nLoading final timestep: {vtk_file.name}")
    else:
        vtk_file = vtk_path
        print(f"\nLoading: {vtk_file}")

    if not vtk_file.exists():
        print(f"ERROR: File not found: {vtk_file}")
        return 1

    mesh = pv.read(vtk_file)
    print(f"  Points: {mesh.n_points}")
    print(f"  Arrays: {mesh.array_names}")

    # Extract centerline profiles
    print("\nExtracting centerline profiles...")
    y_sim, u_sim = extract_centerline_profile(mesh, direction='vertical')
    x_sim, v_sim = extract_centerline_profile(mesh, direction='horizontal')

    print(f"  Vertical centerline: {len(y_sim)} points")
    print(f"  Horizontal centerline: {len(x_sim)} points")

    # Interpolate to Ghia reference points
    u_sim_interp = interpolate_to_reference(y_sim, u_sim, ghia['y'])
    v_sim_interp = interpolate_to_reference(x_sim, v_sim, ghia['x'])

    # Compute errors
    print("\nComputing errors...")
    l2_u, linf_u, rel_linf_u = compute_errors(u_sim_interp, ghia['u'])
    l2_v, linf_v, rel_linf_v = compute_errors(v_sim_interp, ghia['v'])

    print(f"\nU-velocity errors:")
    print(f"  L2:   {l2_u:.6f}")
    print(f"  L∞:   {linf_u:.6f}")
    print(f"  Rel. L∞: {rel_linf_u:.3f}%")

    print(f"\nV-velocity errors:")
    print(f"  L2:   {l2_v:.6f}")
    print(f"  L∞:   {linf_v:.6f}")
    print(f"  Rel. L∞: {rel_linf_v:.3f}%")

    # Overall error (maximum of u and v)
    max_error = max(rel_linf_u, rel_linf_v)
    print(f"\nMaximum relative L∞ error: {max_error:.3f}%")

    # Pass/fail check
    if max_error <= args.threshold:
        status = "PASS"
        status_symbol = "✓"
    else:
        status = "FAIL"
        status_symbol = "✗"

    print(f"\n  {status_symbol} Result: {status} (threshold: {args.threshold}%)")

    # Generate plot
    print("\nGenerating plot...")
    plot_profiles(y_sim, u_sim, x_sim, v_sim, args.re, ghia,
                 args.output / f"lid_driven_cavity_re{args.re}_profiles.png")

    # Save summary
    summary_file = args.output / f"lid_driven_cavity_re{args.re}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Lid-Driven Cavity Validation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Reynolds number: Re = {args.re}\n")
        f.write(f"Reference: Ghia et al. (1982)\n\n")
        f.write("U-velocity (vertical centerline x=0.5):\n")
        f.write(f"  L2 error:         {l2_u:.6f}\n")
        f.write(f"  L∞ error:         {linf_u:.6f}\n")
        f.write(f"  Relative L∞:      {rel_linf_u:.3f}%\n\n")
        f.write("V-velocity (horizontal centerline y=0.5):\n")
        f.write(f"  L2 error:         {l2_v:.6f}\n")
        f.write(f"  L∞ error:         {linf_v:.6f}\n")
        f.write(f"  Relative L∞:      {rel_linf_v:.3f}%\n\n")
        f.write(f"Maximum error:      {max_error:.3f}%\n")
        f.write(f"Threshold:          {args.threshold}%\n")
        f.write(f"\nValidation: {status}\n")

    print(f"\nResults saved to: {args.output}")
    print(f"  - lid_driven_cavity_re{args.re}_profiles.png")
    print(f"  - lid_driven_cavity_re{args.re}_summary.txt")
    print("=" * 70)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    exit(main())
