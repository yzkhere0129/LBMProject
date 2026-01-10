#!/usr/bin/env python3
"""
Fluid Grid Convergence Analysis

This script analyzes grid convergence for fluid simulations by comparing
velocity field solutions at multiple grid resolutions. Computes convergence
order and Grid Convergence Index (GCI).

Physical Basis:
    For a p-th order accurate scheme, the discretization error scales as:
        E(dx) ≈ C * dx^p

    Taking logarithms:
        log(E) ≈ log(C) + p * log(dx)

    The slope of log(E) vs log(dx) gives the convergence order p.

Expected Results:
    - LBM (D3Q19, BGK): p ≈ 2.0 (second-order in space)
    - LBM (D3Q27, MRT): p ≈ 2.0 to 2.5
    - Error should decrease monotonically with grid refinement
    - GCI provides uncertainty estimate

Usage:
    # From VTK files
    python fluid_grid_convergence_analysis.py <vtk_dir1> <vtk_dir2> <vtk_dir3>

    # From pre-computed errors
    python fluid_grid_convergence_analysis.py --resolutions 32,64,128 --errors 1e-3,2.5e-4,6e-5

Output:
    - fluid_grid_convergence.png: Log-log convergence plot
    - fluid_error_reduction.png: Error reduction factors
    - fluid_convergence_summary.txt: Order, GCI, and pass/fail
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
from scipy.spatial import cKDTree

# === PARAMETERS ===
OUTPUT_DIR = Path("/home/yzk/LBMProject/scripts/validation/results")
VEL_FIELD_NAMES = ['Velocity', 'velocity', 'vel', 'u']


def extract_timestep(filename):
    """Extract timestep from filename."""
    match = re.search(r'(?:step_|_)(\d+)\.vt[uki]$', str(filename))
    return int(match.group(1)) if match else None


def load_vtk_final(vtk_dir):
    """Load final timestep VTK file from directory."""
    vtk_dir = Path(vtk_dir)
    if not vtk_dir.exists():
        return None

    files = []
    for ext in ["*.vtu", "*.vtk", "*.vti"]:
        files.extend(vtk_dir.glob(ext))

    if not files:
        return None

    # Get last timestep
    files_with_ts = [(f, extract_timestep(f)) for f in files]
    files_with_ts = [(f, ts) for f, ts in files_with_ts if ts is not None]
    if not files_with_ts:
        return files[-1]  # Fallback to last file alphabetically

    files_with_ts.sort(key=lambda x: x[1])
    return files_with_ts[-1][0]


def get_velocity(mesh):
    """Extract velocity field from mesh."""
    for name in VEL_FIELD_NAMES:
        if name in mesh.array_names:
            vel = mesh[name]
            if vel.ndim == 1:
                vel = vel.reshape(-1, 3)
            return vel

    raise ValueError(f"No velocity field. Available: {mesh.array_names}")


def compute_l2_error(mesh_fine, mesh_coarse, analytical_func=None):
    """
    Compute L2 error between fine and coarse grid solutions.

    If analytical_func is provided, compute error vs analytical solution.
    Otherwise, compute difference between fine and coarse grids (uses coarse
    grid interpolated to fine grid points).

    Returns:
        l2_error: L2 norm of velocity error
    """
    vel_fine = get_velocity(mesh_fine)

    if analytical_func is not None:
        # Compute error vs analytical solution
        points = mesh_fine.points
        vel_analytical = np.array([analytical_func(p[0], p[1], p[2]) for p in points])
        error = vel_fine - vel_analytical
    else:
        # Interpolate coarse to fine grid
        vel_coarse = get_velocity(mesh_coarse)
        tree = cKDTree(mesh_coarse.points)
        distances, indices = tree.query(mesh_fine.points)
        vel_coarse_interp = vel_coarse[indices]
        error = vel_fine - vel_coarse_interp

    # L2 norm of velocity magnitude error
    error_mag = np.sqrt(np.sum(error**2, axis=1))
    l2_error = np.sqrt(np.mean(error_mag**2))

    return l2_error


def compute_linf_error(mesh_fine, mesh_coarse, analytical_func=None):
    """Compute maximum velocity magnitude error."""
    vel_fine = get_velocity(mesh_fine)

    if analytical_func is not None:
        points = mesh_fine.points
        vel_analytical = np.array([analytical_func(p[0], p[1], p[2]) for p in points])
        error = vel_fine - vel_analytical
    else:
        vel_coarse = get_velocity(mesh_coarse)
        tree = cKDTree(mesh_coarse.points)
        distances, indices = tree.query(mesh_fine.points)
        vel_coarse_interp = vel_coarse[indices]
        error = vel_fine - vel_coarse_interp

    error_mag = np.sqrt(np.sum(error**2, axis=1))
    return np.max(error_mag)


def estimate_grid_spacing(mesh):
    """Estimate characteristic grid spacing from mesh."""
    bounds = mesh.bounds
    n_points = mesh.n_points

    # For structured grid, estimate cells per dimension
    domain_volume = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
    if domain_volume > 0:
        dx = (domain_volume / n_points) ** (1.0/3.0)
    else:
        # 2D case
        domain_area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
        dx = np.sqrt(domain_area / n_points)

    return dx


def compute_convergence_order(dx_values, error_values):
    """
    Compute convergence order from grid spacings and errors.

    Uses least-squares fit to log(E) vs log(dx).

    Returns:
        p: Convergence order
        C: Constant in E = C * dx^p
        r_squared: Coefficient of determination (goodness of fit)
    """
    if len(dx_values) != len(error_values):
        raise ValueError("dx and error arrays must have same length")

    if len(dx_values) < 2:
        raise ValueError("Need at least 2 data points")

    # Least-squares fit: log(E) = log(C) + p * log(dx)
    log_dx = np.log(dx_values)
    log_E = np.log(error_values)

    A = np.vstack([log_dx, np.ones(len(log_dx))]).T
    result = np.linalg.lstsq(A, log_E, rcond=None)
    p, log_C = result[0]
    C = np.exp(log_C)

    # Compute R^2
    log_E_fit = p * log_dx + log_C
    ss_res = np.sum((log_E - log_E_fit)**2)
    ss_tot = np.sum((log_E - np.mean(log_E))**2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return p, C, r_squared


def compute_gci(E_fine, E_coarse, r, p, safety_factor=1.25):
    """
    Compute Grid Convergence Index (GCI).

    GCI estimates discretization error and provides a measure of
    how close the solution is to the asymptotic range.

    Args:
        E_fine: Error on fine grid
        E_coarse: Error on coarse grid
        r: Grid refinement ratio (dx_coarse / dx_fine)
        p: Order of convergence
        safety_factor: Safety factor (1.25 for 3+ grids, 3.0 for 2 grids)

    Returns:
        GCI: Grid convergence index (%)
    """
    if E_fine == 0:
        return 0.0

    GCI = safety_factor * abs((E_fine - E_coarse) / E_fine) / (r**p - 1) * 100
    return GCI


def plot_convergence(dx_values, error_values, p, C, r_squared, output_path):
    """Plot log-log convergence curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Log-log plot
    ax1.loglog(dx_values, error_values, 'bo-', markersize=10, linewidth=2.5,
              label='Simulation', markeredgewidth=1.5, markeredgecolor='darkblue')

    # Fitted line
    dx_fit = np.logspace(np.log10(dx_values.min()), np.log10(dx_values.max()), 100)
    E_fit = C * dx_fit**p
    ax1.loglog(dx_fit, E_fit, 'r--', linewidth=2.5,
              label=f'Fit: E = {C:.2e} · Δx^{p:.2f} (R²={r_squared:.4f})')

    # Reference slopes
    dx_ref_min = dx_values.min() * 0.5
    dx_ref_max = dx_values.max() * 2.0
    E_ref_at_min = error_values[0]

    # 1st order reference
    E_1st_min = E_ref_at_min
    E_1st_max = E_ref_at_min * (dx_ref_max / dx_values[0])**1
    ax1.loglog([dx_values[0], dx_ref_max], [E_1st_min, E_1st_max],
              'k:', alpha=0.5, linewidth=1.5, label='1st order')

    # 2nd order reference
    E_2nd_min = E_ref_at_min
    E_2nd_max = E_ref_at_min * (dx_ref_max / dx_values[0])**2
    ax1.loglog([dx_values[0], dx_ref_max], [E_2nd_min, E_2nd_max],
              'k-.', alpha=0.5, linewidth=1.5, label='2nd order')

    ax1.set_xlabel('Grid Spacing Δx (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('L2 Velocity Error (m/s)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Grid Convergence Study\nOrder p = {p:.3f}',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

    # Linear plot (error vs inverse grid spacing)
    inv_dx = 1.0 / dx_values
    ax2.plot(inv_dx, error_values, 'bo-', markersize=10, linewidth=2.5,
            markeredgewidth=1.5, markeredgecolor='darkblue')
    ax2.set_xlabel('Grid Resolution 1/Δx (m⁻¹)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('L2 Velocity Error (m/s)', fontsize=12, fontweight='bold')
    ax2.set_title('Error vs Grid Resolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_reduction(dx_values, error_values, output_path):
    """Plot error reduction factor between successive grids."""
    if len(dx_values) < 2:
        return

    refinement_ratios = dx_values[:-1] / dx_values[1:]
    error_ratios = error_values[:-1] / error_values[1:]

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(error_ratios))
    bars = ax.bar(x_pos, error_ratios, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Color bars based on expected reduction
    for i, (bar, ratio, r) in enumerate(zip(bars, error_ratios, refinement_ratios)):
        expected_2nd = r**2
        if ratio >= expected_2nd * 0.8:
            bar.set_color('green')
        elif ratio >= expected_2nd * 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # Reference lines
    avg_r = np.mean(refinement_ratios)
    ax.axhline(avg_r**1, color='blue', linestyle=':', linewidth=2,
              label=f'1st order ({avg_r:.1f}x reduction)')
    ax.axhline(avg_r**2, color='red', linestyle='--', linewidth=2,
              label=f'2nd order ({avg_r**2:.1f}x reduction)')

    x_labels = [f'{dx_values[i]*1e6:.1f}→{dx_values[i+1]*1e6:.1f} μm\n(r={refinement_ratios[i]:.2f})'
                for i in range(len(refinement_ratios))]

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
    ax.set_ylabel('Error Reduction Factor', fontsize=12, fontweight='bold')
    ax.set_title('Error Reduction Between Successive Grids', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Fluid grid convergence analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('vtk_dirs', nargs='*', type=Path,
                       help='VTK directories (coarse to fine)')
    parser.add_argument('--resolutions', type=str,
                       help='Grid resolutions (comma-separated, e.g., 32,64,128)')
    parser.add_argument('--errors', type=str,
                       help='L2 errors (comma-separated, e.g., 1e-3,2.5e-4)')
    parser.add_argument('--domain-length', type=float, default=1.0,
                       help='Domain length (meters)')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR,
                       help='Output directory')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Fluid Grid Convergence Analysis")
    print("=" * 70)

    # Option 1: Load from VTK files
    if args.vtk_dirs:
        print(f"\nLoading VTK files from {len(args.vtk_dirs)} directories...")
        meshes = []
        dx_values = []

        for vtk_dir in args.vtk_dirs:
            vtk_file = load_vtk_final(vtk_dir)
            if vtk_file is None:
                print(f"ERROR: No VTK in {vtk_dir}")
                return 1

            mesh = pv.read(vtk_file)
            meshes.append(mesh)

            # Estimate dx from mesh
            dx = estimate_grid_spacing(mesh)
            dx_values.append(dx)

            print(f"  {vtk_dir.name}: {mesh.n_points} points, Δx ≈ {dx*1e6:.3f} μm")

        # Compute errors (use finest grid as reference)
        print("\nComputing errors relative to finest grid...")
        error_values = []
        finest_mesh = meshes[-1]

        for i, mesh in enumerate(meshes[:-1]):
            error = compute_l2_error(finest_mesh, mesh)
            error_values.append(error)
            print(f"  Grid {i+1}: L2 error = {error:.6e} m/s")

        # Add zero error for finest grid
        error_values.append(0.0)
        dx_values = np.array(dx_values)
        error_values = np.array(error_values)

        # Remove finest grid for convergence analysis
        dx_values = dx_values[:-1]
        error_values = error_values[:-1]

    # Option 2: Use provided resolutions and errors
    elif args.resolutions and args.errors:
        resolutions = [int(x) for x in args.resolutions.split(',')]
        errors = [float(x) for x in args.errors.split(',')]

        if len(resolutions) != len(errors):
            print("ERROR: Number of resolutions must match number of errors")
            return 1

        dx_values = np.array([args.domain_length / n for n in resolutions])
        error_values = np.array(errors)

        print(f"\nUsing provided data:")
        for i, (n, dx, err) in enumerate(zip(resolutions, dx_values, error_values)):
            print(f"  Grid {i+1}: {n} cells, Δx = {dx*1e6:.3f} μm, error = {err:.6e} m/s")

    else:
        print("ERROR: Must provide either VTK directories or --resolutions and --errors")
        return 1

    # Check monotonic decrease
    if not np.all(np.diff(error_values) < 0):
        print("\nWARNING: Errors are not monotonically decreasing!")
        print("This suggests:")
        print("  - Solution not fully converged to steady state")
        print("  - Roundoff error dominating (over-refinement)")
        print("  - Possible implementation issue")

    # Compute convergence order
    print("\nComputing convergence order...")
    p, C, r_squared = compute_convergence_order(dx_values, error_values)

    print(f"\nConvergence Order:  p = {p:.3f}")
    print(f"Error constant:     C = {C:.3e}")
    print(f"Fit quality:        R² = {r_squared:.6f}")

    # Interpret results
    if 1.9 <= p <= 2.3:
        print("  ✓ Second-order convergence (PASS)")
        status = "PASS"
    elif 1.5 <= p <= 1.9:
        print("  ~ Nearly second-order (acceptable)")
        status = "ACCEPTABLE"
    elif 0.9 <= p <= 1.5:
        print("  ! First-order convergence (marginal)")
        status = "MARGINAL"
    else:
        print(f"  ✗ Unexpected convergence order (FAIL)")
        status = "FAIL"

    # Compute GCI for adjacent grids
    if len(dx_values) >= 2:
        print("\nGrid Convergence Index (GCI):")
        safety_factor = 1.25 if len(dx_values) >= 3 else 3.0
        print(f"  Safety factor: {safety_factor}")

        for i in range(len(dx_values) - 1):
            r = dx_values[i] / dx_values[i+1]
            gci = compute_gci(error_values[i+1], error_values[i], r, p, safety_factor)
            print(f"  GCI_{i+1},{i+2} (r={r:.2f}): {gci:.2f}%")

    # Richardson extrapolation
    if len(dx_values) >= 2:
        E_fine = error_values[-1]
        E_coarse = error_values[-2]
        r = dx_values[-2] / dx_values[-1]
        richardson_correction = abs((r**p * E_fine - E_coarse) / (r**p - 1))
        print(f"\nRichardson extrapolation:")
        print(f"  Estimated discretization error: {richardson_correction:.6e} m/s")

    # Generate plots
    print("\nGenerating plots...")
    plot_convergence(dx_values, error_values, p, C, r_squared,
                    args.output / "fluid_grid_convergence.png")

    if len(dx_values) >= 2:
        plot_error_reduction(dx_values, error_values,
                           args.output / "fluid_error_reduction.png")

    # Save summary
    summary_file = args.output / "fluid_convergence_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Fluid Grid Convergence Analysis Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("Grid resolutions:\n")
        for i, (dx, err) in enumerate(zip(dx_values, error_values)):
            n_equiv = int(args.domain_length / dx)
            f.write(f"  Grid {i+1}: Δx = {dx*1e6:.3f} μm (~{n_equiv} cells), "
                   f"error = {err:.6e} m/s\n")
        f.write(f"\nConvergence order:  p = {p:.3f}\n")
        f.write(f"Error constant:     C = {C:.3e}\n")
        f.write(f"Fit quality:        R² = {r_squared:.6f}\n\n")

        f.write(f"Result: {status}\n")
        if 1.9 <= p <= 2.3:
            f.write("  Second-order spatial accuracy confirmed.\n")
        elif p < 1.5:
            f.write("  WARNING: Lower than expected convergence order.\n")
            f.write("  Consider checking timestep convergence and solver settings.\n")

    print(f"\nResults saved to: {args.output}")
    print(f"  - fluid_grid_convergence.png")
    print(f"  - fluid_error_reduction.png")
    print(f"  - fluid_convergence_summary.txt")
    print("=" * 70)

    return 0 if status in ["PASS", "ACCEPTABLE"] else 1


if __name__ == "__main__":
    exit(main())
