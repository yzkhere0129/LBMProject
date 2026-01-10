#!/usr/bin/env python3
"""
Taylor-Green Vortex Validation

This script validates LBM fluid solver against the analytical Taylor-Green vortex
solution. The Taylor-Green vortex is a canonical test for incompressible flow
solvers, featuring:
    - Exact analytical solution
    - Exponential decay of kinetic energy
    - Tests momentum diffusion and vorticity

Physical Basis:
    Initial velocity: u = U0 * sin(kx) * cos(ky)
                      v = -U0 * cos(kx) * sin(ky)
    Kinetic energy:   E(t) = E0 * exp(-2 * nu * k^2 * t)

Expected Results:
    - Kinetic energy decays exponentially
    - Decay rate matches analytical: -2 * nu * k^2
    - Error < 5% for adequate resolution and timestep

Usage:
    python fluid_taylor_green_analysis.py --vtk <vtk_dir>
    python fluid_taylor_green_analysis.py  # Uses default path

Output:
    - taylor_green_energy_decay.png: E(t) vs analytical
    - taylor_green_velocity_profiles.png: Velocity field at key times
    - taylor_green_summary.txt: Numerical metrics and pass/fail
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

# === PARAMETERS ===
DEFAULT_VTK_DIR = Path("/home/yzk/LBMProject/tests/validation/output_taylor_green")
OUTPUT_DIR = Path("/home/yzk/LBMProject/scripts/validation/results")

# Physical parameters (typical values, override via CLI)
NU = 0.01          # Kinematic viscosity [m^2/s]
U0 = 1.0           # Initial velocity magnitude [m/s]
L = 2.0 * np.pi    # Domain size [m]
DT = 0.001         # Timestep [s]
ERROR_THRESHOLD = 5.0  # Pass/fail threshold [%]

VEL_FIELD_NAMES = ['Velocity', 'velocity', 'vel', 'u']


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


def compute_kinetic_energy(mesh, dx=None):
    """
    Compute total kinetic energy: E = 0.5 * sum(u^2 + v^2 + w^2) * dV

    Args:
        mesh: PyVista mesh with velocity data
        dx: Grid spacing [m] (if None, estimated from mesh)

    Returns:
        E: Total kinetic energy [J] (per unit mass)
    """
    vel = get_velocity(mesh)

    # Compute velocity magnitude squared
    vel_mag_sq = np.sum(vel**2, axis=1)

    # Estimate cell volume
    if dx is None:
        bounds = mesh.bounds
        n_points = mesh.n_points
        # Estimate dx as cube root of average volume per point
        domain_volume = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
        dx = (domain_volume / n_points) ** (1.0/3.0)

    dV = dx**3

    # Total kinetic energy (per unit mass, assuming rho=1)
    E = 0.5 * np.sum(vel_mag_sq) * dV

    return E


def analytical_energy(t, E0, nu, k):
    """
    Analytical kinetic energy for Taylor-Green vortex.

    E(t) = E0 * exp(-2 * nu * k^2 * t)
    """
    return E0 * np.exp(-2.0 * nu * k**2 * t)


def compute_decay_rate(times, energies):
    """
    Fit exponential decay: E(t) = E0 * exp(-lambda * t)

    Returns:
        lambda_fit: Fitted decay rate
        E0_fit: Fitted initial energy
    """
    # Linear fit to log(E) = log(E0) - lambda * t
    log_E = np.log(energies)
    A = np.vstack([times, np.ones(len(times))]).T
    result = np.linalg.lstsq(A, log_E, rcond=None)
    lambda_fit, log_E0 = result[0]
    lambda_fit = -lambda_fit  # Convert to positive decay rate
    E0_fit = np.exp(log_E0)

    return lambda_fit, E0_fit


def plot_energy_decay(times, energies, nu, k, output_path):
    """Plot kinetic energy decay vs analytical solution."""
    # Analytical solution
    E0_analytical = energies[0]
    lambda_analytical = 2.0 * nu * k**2
    E_analytical = analytical_energy(times, E0_analytical, nu, k)

    # Fitted decay rate
    lambda_fit, E0_fit = compute_decay_rate(times, energies)
    E_fit = E0_fit * np.exp(-lambda_fit * times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Linear scale
    ax1.plot(times, energies, 'bo-', markersize=6, linewidth=2,
            label='LBM Simulation', alpha=0.8)
    ax1.plot(times, E_analytical, 'r--', linewidth=2,
            label=f'Analytical: E₀ exp(-2νk²t)')
    ax1.plot(times, E_fit, 'g:', linewidth=2,
            label=f'Fitted: E₀ exp(-λt), λ={lambda_fit:.6f}')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Kinetic Energy (J/kg)', fontsize=12)
    ax1.set_title('Taylor-Green Vortex: Energy Decay', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.semilogy(times, energies, 'bo-', markersize=6, linewidth=2,
                label='LBM Simulation', alpha=0.8)
    ax2.semilogy(times, E_analytical, 'r--', linewidth=2,
                label=f'Analytical (λ={lambda_analytical:.6f})')
    ax2.semilogy(times, E_fit, 'g:', linewidth=2,
                label=f'Fitted (λ={lambda_fit:.6f})')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Kinetic Energy (J/kg) [log]', fontsize=12)
    ax2.set_title('Exponential Decay (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    # Add error annotation
    rel_error = abs(lambda_fit - lambda_analytical) / lambda_analytical * 100
    ax2.text(0.05, 0.05,
            f'Decay rate error: {rel_error:.2f}%\n'
            f'Expected: λ = 2νk² = {lambda_analytical:.6f}\n'
            f'Fitted:   λ = {lambda_fit:.6f}',
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return lambda_fit, lambda_analytical, rel_error


def plot_velocity_profiles(vtk_files, times, nu, k, U0, L, output_path):
    """Plot velocity profiles at initial and final timesteps."""
    if len(vtk_files) < 2:
        print("Warning: Need at least 2 timesteps for profile comparison")
        return

    # Select initial and final times
    indices = [0, len(vtk_files) // 2, -1]

    fig, axes = plt.subplots(2, len(indices), figsize=(14, 8))

    for col, idx in enumerate(indices):
        mesh = pv.read(vtk_files[idx])
        vel = get_velocity(mesh)
        points = mesh.points

        t = times[idx]

        # Extract centerline velocities (y=L/2 or closest)
        y_center = L / 2.0
        tol = L / 100.0
        mask_y = np.abs(points[:, 1] - y_center) < tol

        x_line = points[mask_y, 0]
        u_line = vel[mask_y, 0]
        v_line = vel[mask_y, 1]

        # Sort by x
        sort_idx = np.argsort(x_line)
        x_line = x_line[sort_idx]
        u_line = u_line[sort_idx]
        v_line = v_line[sort_idx]

        # Analytical solution at time t
        x_analytical = np.linspace(0, L, 200)
        decay = np.exp(-nu * k**2 * t)
        u_analytical = U0 * np.sin(k * x_analytical) * np.cos(k * y_center) * decay
        v_analytical = -U0 * np.cos(k * x_analytical) * np.sin(k * y_center) * decay

        # Plot u-velocity
        axes[0, col].plot(x_line, u_line, 'bo-', markersize=4, label='LBM')
        axes[0, col].plot(x_analytical, u_analytical, 'r--', linewidth=2, label='Analytical')
        axes[0, col].set_xlabel('x (m)')
        axes[0, col].set_ylabel('u (m/s)')
        axes[0, col].set_title(f't = {t:.4f} s')
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        # Plot v-velocity
        axes[1, col].plot(x_line, v_line, 'go-', markersize=4, label='LBM')
        axes[1, col].plot(x_analytical, v_analytical, 'r--', linewidth=2, label='Analytical')
        axes[1, col].set_xlabel('x (m)')
        axes[1, col].set_ylabel('v (m/s)')
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)

    plt.suptitle('Taylor-Green Velocity Profiles (y = L/2)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Taylor-Green vortex validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vtk', type=Path, default=DEFAULT_VTK_DIR,
                       help='VTK output directory')
    parser.add_argument('--nu', type=float, default=NU,
                       help='Kinematic viscosity (m^2/s)')
    parser.add_argument('--U0', type=float, default=U0,
                       help='Initial velocity magnitude (m/s)')
    parser.add_argument('--L', type=float, default=L,
                       help='Domain size (m)')
    parser.add_argument('--dt', type=float, default=DT,
                       help='Timestep (s)')
    parser.add_argument('--threshold', type=float, default=ERROR_THRESHOLD,
                       help='Pass/fail error threshold (%)')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR,
                       help='Output directory')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Taylor-Green Vortex Validation")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Viscosity:  ν = {args.nu} m²/s")
    print(f"  Initial U:  U₀ = {args.U0} m/s")
    print(f"  Domain:     L = {args.L} m")
    print(f"  Timestep:   Δt = {args.dt} s")

    k = 2.0 * np.pi / args.L
    print(f"  Wave number: k = 2π/L = {k:.6f} rad/m")
    print(f"  Expected decay rate: λ = 2νk² = {2.0 * args.nu * k**2:.6f} s⁻¹")

    # Load VTK files
    print(f"\nLoading VTK files from: {args.vtk}")
    vtk_files = load_vtk_series(args.vtk)

    if not vtk_files:
        print(f"ERROR: No VTK files found in {args.vtk}")
        return 1

    print(f"  Found {len(vtk_files)} timesteps")

    # Extract kinetic energy evolution
    print("\nComputing kinetic energy evolution...")
    times = np.zeros(len(vtk_files))
    energies = np.zeros(len(vtk_files))

    for i, vtk_file in enumerate(vtk_files):
        mesh = pv.read(vtk_file)
        energies[i] = compute_kinetic_energy(mesh)

        ts = extract_timestep(vtk_file)
        times[i] = ts * args.dt if ts is not None else i * args.dt

        if i == 0 or i == len(vtk_files) - 1:
            print(f"  t = {times[i]:.6f} s: E = {energies[i]:.8e} J/kg")

    # Compute decay rate and error
    print("\nAnalyzing decay rate...")
    lambda_analytical = 2.0 * args.nu * k**2
    lambda_fit, E0_fit = compute_decay_rate(times, energies)
    rel_error = abs(lambda_fit - lambda_analytical) / lambda_analytical * 100

    print(f"  Analytical decay rate: λ = {lambda_analytical:.8f} s⁻¹")
    print(f"  Fitted decay rate:     λ = {lambda_fit:.8f} s⁻¹")
    print(f"  Relative error:        {rel_error:.3f}%")

    # Pass/fail check
    if rel_error <= args.threshold:
        status = "PASS"
        status_symbol = "✓"
    else:
        status = "FAIL"
        status_symbol = "✗"

    print(f"\n  {status_symbol} Result: {status} (threshold: {args.threshold}%)")

    # Generate plots
    print("\nGenerating plots...")
    lambda_fit_plot, lambda_analytical_plot, error_plot = plot_energy_decay(
        times, energies, args.nu, k,
        args.output / "taylor_green_energy_decay.png")

    plot_velocity_profiles(
        vtk_files, times, args.nu, k, args.U0, args.L,
        args.output / "taylor_green_velocity_profiles.png")

    # Save summary
    summary_file = args.output / "taylor_green_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Taylor-Green Vortex Validation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write("Physical Parameters:\n")
        f.write(f"  Kinematic viscosity: ν = {args.nu} m²/s\n")
        f.write(f"  Initial velocity:    U₀ = {args.U0} m/s\n")
        f.write(f"  Domain size:         L = {args.L} m\n")
        f.write(f"  Wave number:         k = 2π/L = {k:.6f} rad/m\n")
        f.write(f"  Timestep:            Δt = {args.dt} s\n\n")
        f.write("Results:\n")
        f.write(f"  Number of timesteps:     {len(vtk_files)}\n")
        f.write(f"  Simulation time:         {times[-1]:.6f} s\n")
        f.write(f"  Initial energy:          E₀ = {energies[0]:.8e} J/kg\n")
        f.write(f"  Final energy:            E = {energies[-1]:.8e} J/kg\n")
        f.write(f"  Energy ratio:            E/E₀ = {energies[-1]/energies[0]:.6f}\n\n")
        f.write("Decay Rate Analysis:\n")
        f.write(f"  Analytical (expected):   λ = 2νk² = {lambda_analytical:.8f} s⁻¹\n")
        f.write(f"  Fitted (simulation):     λ = {lambda_fit:.8f} s⁻¹\n")
        f.write(f"  Absolute error:          Δλ = {abs(lambda_fit - lambda_analytical):.8f} s⁻¹\n")
        f.write(f"  Relative error:          {rel_error:.3f}%\n\n")
        f.write(f"Validation: {status} (threshold: {args.threshold}%)\n")

    print(f"\nResults saved to: {args.output}")
    print(f"  - taylor_green_energy_decay.png")
    print(f"  - taylor_green_velocity_profiles.png")
    print(f"  - taylor_green_summary.txt")
    print("=" * 70)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    exit(main())
