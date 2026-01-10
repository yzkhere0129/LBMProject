#!/usr/bin/env python3
"""
Grid Convergence Analysis

This script analyzes grid convergence by comparing simulation results at
multiple grid resolutions. Computes convergence order and Richardson
extrapolation estimates.

Physical Basis:
    For a p-th order accurate scheme, the discretization error scales as:
        E(dx) ≈ C * dx^p
    Taking logarithms:
        log(E) ≈ log(C) + p * log(dx)
    The slope of log(E) vs log(dx) gives the convergence order p.

Expected Results:
    - Second-order schemes: p ≈ 2.0 (1.8 < p < 2.2)
    - First-order schemes: p ≈ 1.0 (0.8 < p < 1.2)
    - Error should decrease monotonically with grid refinement

Usage:
    python grid_convergence_analysis.py <vtk_dir1> <vtk_dir2> <vtk_dir3> ...
    python grid_convergence_analysis.py --resolutions 25,50,100,200 --errors 1e-2,2.5e-3,6e-4,1.5e-4

Output:
    - PNG log-log convergence plot
    - Computed convergence order
    - Richardson extrapolation estimate
    - Grid convergence index (GCI)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pyvista as pv
import re
from scipy.spatial import cKDTree

# === PARAMETERS ===
OUTPUT_DIR = Path("/home/yzk/LBMProject/scripts/validation/results")
TEMP_FIELD_NAMES = ['T', 'temperature', 'Temperature', 'scalar']


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


def get_field(mesh, field_names):
    """Extract field from mesh trying multiple possible names."""
    for name in field_names:
        if name in mesh.array_names:
            return mesh[name]

    if len(mesh.array_names) > 0:
        return mesh[mesh.array_names[0]]

    raise ValueError(f"No field found. Available: {mesh.array_names}")


def compute_l2_error(mesh_fine, mesh_coarse, analytical_func=None):
    """
    Compute L2 error between fine and coarse grid solutions.

    If analytical_func is provided, compute error vs analytical solution.
    Otherwise, compute difference between fine and coarse grids.

    Returns:
        l2_error: L2 norm of error
    """
    field_fine = get_field(mesh_fine, TEMP_FIELD_NAMES)

    if analytical_func is not None:
        # Compute error vs analytical solution
        points = mesh_fine.points
        analytical = np.array([analytical_func(p[0], p[1], p[2]) for p in points])
        error = field_fine - analytical
    else:
        # Interpolate coarse to fine grid
        field_coarse = get_field(mesh_coarse, TEMP_FIELD_NAMES)
        tree = cKDTree(mesh_coarse.points)
        distances, indices = tree.query(mesh_fine.points)
        field_coarse_interp = field_coarse[indices]
        error = field_fine - field_coarse_interp

    l2_error = np.sqrt(np.mean(error**2))
    return l2_error


def compute_max_error(mesh, analytical_func=None, reference_mesh=None):
    """Compute maximum absolute error."""
    field = get_field(mesh, TEMP_FIELD_NAMES)

    if analytical_func is not None:
        points = mesh.points
        analytical = np.array([analytical_func(p[0], p[1], p[2]) for p in points])
        error = np.abs(field - analytical)
    elif reference_mesh is not None:
        field_ref = get_field(reference_mesh, TEMP_FIELD_NAMES)
        tree = cKDTree(reference_mesh.points)
        distances, indices = tree.query(mesh.points)
        field_ref_interp = field_ref[indices]
        error = np.abs(field - field_ref_interp)
    else:
        raise ValueError("Must provide either analytical_func or reference_mesh")

    return np.max(error)


def compute_convergence_order(dx_values, error_values):
    """
    Compute convergence order from grid spacings and errors.

    Uses least-squares fit to log(E) vs log(dx).

    Returns:
        p: Convergence order
        C: Constant in E = C * dx^p
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

    return p, C


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
        safety_factor: Safety factor (typically 1.25 or 3.0)

    Returns:
        GCI: Grid convergence index (%)
    """
    GCI = safety_factor * abs((E_fine - E_coarse) / E_fine) / (r**p - 1) * 100
    return GCI


def plot_convergence(dx_values, error_values, p, C, output_path):
    """Plot log-log convergence curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Log-log plot
    ax1.loglog(dx_values * 1e6, error_values, 'bo-', markersize=8, linewidth=2,
              label='Simulation data')

    # Fitted line
    dx_fit = np.array([dx_values[0], dx_values[-1]])
    E_fit = C * dx_fit**p
    ax1.loglog(dx_fit * 1e6, E_fit, 'r--', linewidth=2,
              label=f'Fit: E = {C:.2e} * dx^{p:.2f}')

    # Reference slopes
    dx_ref = np.array([dx_values[0] * 0.5, dx_values[-1] * 2])
    E_ref_1st = error_values[0] * (dx_ref / dx_values[0])**1
    E_ref_2nd = error_values[0] * (dx_ref / dx_values[0])**2

    ax1.loglog(dx_ref * 1e6, E_ref_1st, 'k:', alpha=0.5, label='1st order slope')
    ax1.loglog(dx_ref * 1e6, E_ref_2nd, 'k-.', alpha=0.5, label='2nd order slope')

    ax1.set_xlabel('Grid Spacing (μm)')
    ax1.set_ylabel('L2 Error (K)')
    ax1.set_title(f'Grid Convergence Study\nOrder p = {p:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Linear plot
    ax2.plot(1.0 / dx_values, error_values, 'bo-', markersize=8, linewidth=2)
    ax2.set_xlabel('Grid Resolution (1/dx) [m⁻¹]')
    ax2.set_ylabel('L2 Error (K)')
    ax2.set_title('Error vs Grid Resolution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_reduction(dx_values, error_values, output_path):
    """Plot error reduction factor between successive grids."""
    if len(dx_values) < 2:
        return

    refinement_ratios = dx_values[:-1] / dx_values[1:]
    error_ratios = error_values[:-1] / error_values[1:]

    fig, ax = plt.subplots(figsize=(8, 6))

    x_labels = [f'{dx_values[i]*1e6:.1f}→{dx_values[i+1]*1e6:.1f} μm'
                for i in range(len(dx_values)-1)]

    ax.bar(range(len(error_ratios)), error_ratios, alpha=0.7, edgecolor='black')
    ax.axhline(2.0, color='r', linestyle='--', label='2nd order (4x reduction)')
    ax.axhline(4.0, color='g', linestyle='--', label='Expected for 2x refinement')
    ax.set_xticks(range(len(error_ratios)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Error Reduction Ratio')
    ax.set_title('Error Reduction Between Successive Grids')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Grid convergence analysis')
    parser.add_argument('vtk_dirs', nargs='*', type=Path,
                       help='VTK directories (coarse to fine)')
    parser.add_argument('--resolutions', type=str,
                       help='Grid resolutions (comma-separated, e.g., 25,50,100,200)')
    parser.add_argument('--errors', type=str,
                       help='L2 errors (comma-separated, e.g., 1e-2,2.5e-3)')
    parser.add_argument('--domain-length', type=float, default=200e-6,
                       help='Domain length (meters)')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR,
                       help='Output directory')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Grid Convergence Analysis")
    print("=" * 60)

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
            bounds = mesh.bounds
            n_points = mesh.n_points
            dx = (bounds[1] - bounds[0]) / np.cbrt(n_points)
            dx_values.append(dx)

            print(f"  {vtk_dir.name}: {mesh.n_points} points, dx ≈ {dx*1e6:.2f} μm")

        # Compute errors (use finest grid as reference)
        print("\nComputing errors...")
        error_values = []
        finest_mesh = meshes[-1]

        for i, mesh in enumerate(meshes[:-1]):
            error = compute_l2_error(finest_mesh, mesh)
            error_values.append(error)
            print(f"  Grid {i+1}: L2 error = {error:.6e} K")

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

        dx_values = np.array([args.domain_length / (n - 1) for n in resolutions])
        error_values = np.array(errors)

        print(f"\nUsing provided data:")
        for i, (n, dx, err) in enumerate(zip(resolutions, dx_values, error_values)):
            print(f"  Grid {i+1}: {n} cells, dx = {dx*1e6:.2f} μm, error = {err:.6e}")

    else:
        print("ERROR: Must provide either VTK directories or --resolutions and --errors")
        return 1

    # Check monotonic decrease
    if not np.all(np.diff(error_values) < 0):
        print("\nWARNING: Errors are not monotonically decreasing!")
        print("This suggests:")
        print("  - Solution not converged")
        print("  - Roundoff error dominating")
        print("  - Bug in implementation")

    # Compute convergence order
    print("\nComputing convergence order...")
    p, C = compute_convergence_order(dx_values, error_values)

    print(f"\nConvergence Order: p = {p:.3f}")
    print(f"Error constant:    C = {C:.3e}")

    # Interpret results
    if 1.8 <= p <= 2.2:
        print("  ✓ Second-order convergence (PASS)")
    elif 0.8 <= p <= 1.2:
        print("  ✓ First-order convergence (acceptable)")
    else:
        print(f"  ✗ Unexpected convergence order (FAIL)")

    # Compute GCI for adjacent grids
    if len(dx_values) >= 2:
        print("\nGrid Convergence Index (GCI):")
        for i in range(len(dx_values) - 1):
            r = dx_values[i] / dx_values[i+1]
            gci = compute_gci(error_values[i+1], error_values[i], r, p)
            print(f"  GCI_{i+1},{i+2}: {gci:.2f}%")

    # Richardson extrapolation
    if len(dx_values) >= 2:
        # Estimate solution at dx=0
        E_fine = error_values[-1]
        E_coarse = error_values[-2]
        r = dx_values[-2] / dx_values[-1]
        richardson_correction = (r**p * E_fine - E_coarse) / (r**p - 1)
        print(f"\nRichardson extrapolation correction: {richardson_correction:.6e} K")

    # Generate plots
    print("\nGenerating plots...")
    plot_convergence(dx_values, error_values, p, C,
                    args.output / "grid_convergence.png")

    if len(dx_values) >= 2:
        plot_error_reduction(dx_values, error_values,
                           args.output / "error_reduction.png")

    # Save summary
    summary_file = args.output / "convergence_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Grid Convergence Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Grid resolutions:\n")
        for i, (dx, err) in enumerate(zip(dx_values, error_values)):
            f.write(f"  Grid {i+1}: dx = {dx*1e6:.2f} μm, error = {err:.6e} K\n")
        f.write(f"\nConvergence order: p = {p:.3f}\n")
        f.write(f"Error constant:    C = {C:.3e}\n\n")

        if 1.8 <= p <= 2.2:
            f.write("Result: Second-order convergence (PASS)\n")
        elif 0.8 <= p <= 1.2:
            f.write("Result: First-order convergence\n")
        else:
            f.write(f"Result: Unexpected order p={p:.2f} (FAIL)\n")

    print(f"\nResults saved to: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
