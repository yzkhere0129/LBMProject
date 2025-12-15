#!/usr/bin/env python3
"""
Poiseuille Flow Comparison Script
==================================

Compare Poiseuille flow results between LBMProject and WalBerla
with analytical solution validation.

Author: LBMProject Team
Date: 2025-12-04
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import from compare_vtk_files
sys.path.insert(0, str(Path(__file__).parent))
from compare_vtk_files import VTKData


def analytical_poiseuille_velocity(y, H, dp_dx, mu):
    """
    Analytical solution for Poiseuille flow velocity profile.

    Args:
        y: Position across channel height
        H: Channel height
        dp_dx: Pressure gradient
        mu: Dynamic viscosity

    Returns:
        Velocity at position y
    """
    return -(dp_dx / (2 * mu)) * y * (H - y)


def extract_centerline_velocity(vtk: VTKData, axis='z'):
    """
    Extract velocity profile along channel centerline.

    Args:
        vtk: VTKData object
        axis: Flow direction ('x', 'y', or 'z')

    Returns:
        positions, velocities
    """
    vel = vtk.get_field('Velocity') or vtk.get_field('u')
    if vel is None:
        return None, None

    # Reshape to grid
    grid = vtk.reshape_to_grid(vel)
    nx, ny, nz = vtk.dimensions

    # Extract centerline based on flow direction
    if axis == 'z':
        # Flow in Z, extract at center of X-Y plane
        centerline = grid[:, ny//2, nx//2, 2]  # Z-component
        positions = np.arange(nz) * vtk.spacing[2]
    elif axis == 'x':
        # Flow in X, extract at center of Y-Z plane
        centerline = grid[nz//2, ny//2, :, 0]  # X-component
        positions = np.arange(nx) * vtk.spacing[0]
    else:  # y
        # Flow in Y, extract at center of X-Z plane
        centerline = grid[nz//2, :, nx//2, 1]  # Y-component
        positions = np.arange(ny) * vtk.spacing[1]

    return positions, centerline


def compare_poiseuille_flows(lbm_file: str, walberla_file: str,
                            output_dir: str = '.'):
    """
    Compare Poiseuille flow results from LBMProject and WalBerla.

    Args:
        lbm_file: Path to LBMProject VTK file
        walberla_file: Path to WalBerla VTK file
        output_dir: Directory for output plots
    """
    print("=" * 80)
    print("POISEUILLE FLOW COMPARISON")
    print("=" * 80)

    # Load VTK files
    print(f"\nLoading LBMProject file: {lbm_file}")
    vtk_lbm = VTKData(lbm_file)

    print(f"Loading WalBerla file: {walberla_file}")
    vtk_walberla = VTKData(walberla_file)

    # Extract velocity profiles
    print("\nExtracting velocity profiles...")
    pos_lbm, vel_lbm = extract_centerline_velocity(vtk_lbm, axis='z')
    pos_walberla, vel_walberla = extract_centerline_velocity(vtk_walberla, axis='z')

    if vel_lbm is None or vel_walberla is None:
        print("Error: Could not extract velocity profiles")
        return

    # Compute analytical solution (example parameters)
    H = pos_lbm[-1]  # Channel height
    dp_dx = -1e-5    # Pressure gradient (example)
    mu = 1.0         # Dynamic viscosity (example)

    y_analytical = np.linspace(0, H, 100)
    vel_analytical = analytical_poiseuille_velocity(y_analytical, H, dp_dx, mu)

    # Normalize velocities for comparison
    vel_lbm_norm = vel_lbm / np.max(np.abs(vel_lbm))
    vel_walberla_norm = vel_walberla / np.max(np.abs(vel_walberla))
    vel_analytical_norm = vel_analytical / np.max(np.abs(vel_analytical))

    # Compute errors
    print("\nComputing error metrics...")
    l2_error = np.linalg.norm(vel_lbm_norm - vel_walberla_norm) / np.sqrt(len(vel_lbm_norm))
    max_error = np.max(np.abs(vel_lbm_norm - vel_walberla_norm))
    rmse = np.sqrt(np.mean((vel_lbm_norm - vel_walberla_norm)**2))

    print(f"  L2 Error:   {l2_error:.6e}")
    print(f"  Max Error:  {max_error:.6e}")
    print(f"  RMSE:       {rmse:.6e}")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Velocity profiles
    ax = axes[0, 0]
    ax.plot(pos_lbm, vel_lbm, 'b-', label='LBMProject', linewidth=2)
    ax.plot(pos_walberla, vel_walberla, 'r--', label='WalBerla', linewidth=2)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Normalized profiles with analytical
    ax = axes[0, 1]
    ax.plot(pos_lbm/H, vel_lbm_norm, 'b-', label='LBMProject', linewidth=2)
    ax.plot(pos_walberla/pos_walberla[-1], vel_walberla_norm, 'r--', label='WalBerla', linewidth=2)
    ax.plot(y_analytical/H, vel_analytical_norm, 'k:', label='Analytical', linewidth=2)
    ax.set_xlabel('Normalized Position (y/H)')
    ax.set_ylabel('Normalized Velocity')
    ax.set_title('Normalized Profiles vs Analytical Solution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Absolute difference
    ax = axes[1, 0]
    diff = np.abs(vel_lbm - vel_walberla)
    ax.plot(pos_lbm, diff, 'g-', linewidth=2)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Absolute Difference (m/s)')
    ax.set_title('Absolute Velocity Difference')
    ax.grid(True, alpha=0.3)

    # Plot 4: Relative error
    ax = axes[1, 1]
    rel_error = np.abs(vel_lbm_norm - vel_walberla_norm)
    ax.plot(pos_lbm/H, rel_error, 'orange', linewidth=2)
    ax.set_xlabel('Normalized Position (y/H)')
    ax.set_ylabel('Relative Error')
    ax.set_title('Relative Error Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'poiseuille_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()

    # Create error metrics plot
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['L2 Error', 'Max Error', 'RMSE']
    values = [l2_error, max_error, rmse]

    bars = ax.bar(metrics, values, color=['blue', 'red', 'green'], alpha=0.7)
    ax.set_ylabel('Error Magnitude')
    ax.set_title('Poiseuille Flow Error Metrics: LBMProject vs WalBerla')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.3e}',
               ha='center', va='bottom')

    plt.tight_layout()
    output_path = Path(output_dir) / 'poiseuille_error_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Error metrics plot saved to: {output_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare Poiseuille flow results between LBMProject and WalBerla'
    )
    parser.add_argument('lbm_file', help='LBMProject VTK file')
    parser.add_argument('walberla_file', help='WalBerla VTK file')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for plots')

    args = parser.parse_args()

    compare_poiseuille_flows(args.lbm_file, args.walberla_file, args.output_dir)


if __name__ == '__main__':
    main()
