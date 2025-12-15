#!/usr/bin/env python3
"""
Marangoni Flow Visualization Script
====================================

Advanced visualization for Marangoni convection flows from LBMProject.
Shows temperature gradients, velocity fields, and surface tension effects.

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


def visualize_marangoni_flow(vtk_file: str, output_dir: str = '.'):
    """
    Create comprehensive visualization of Marangoni flow.

    Args:
        vtk_file: Path to VTK file
        output_dir: Directory for output plots
    """
    print("=" * 80)
    print("MARANGONI FLOW VISUALIZATION")
    print("=" * 80)
    print(f"\nLoading: {vtk_file}")

    vtk = VTKData(vtk_file)

    # Extract fields
    velocity = vtk.get_field('Velocity')
    temperature = vtk.get_field('Temperature')
    liquid_fraction = vtk.get_field('LiquidFraction')
    fill_level = vtk.get_field('FillLevel')

    if velocity is None or temperature is None:
        print("Error: Required fields not found in VTK file")
        return

    print(f"Dimensions: {vtk.dimensions}")
    print(f"Fields found: {list(vtk.point_data.keys())}")

    # Reshape data to grid
    nx, ny, nz = vtk.dimensions
    vel_grid = vtk.reshape_to_grid(velocity)
    temp_grid = vtk.reshape_to_grid(temperature)

    # Compute velocity magnitude
    vel_mag = np.linalg.norm(velocity, axis=1)
    vel_mag_grid = vtk.reshape_to_grid(vel_mag)

    # Get middle slice indices
    mid_x = nx // 2
    mid_y = ny // 2
    mid_z = nz // 2

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. Temperature field - XY plane (top view)
    ax1 = plt.subplot(2, 3, 1)
    temp_xy = temp_grid[mid_z, :, :]
    im1 = ax1.imshow(temp_xy.T, cmap='hot', origin='lower', aspect='auto')
    ax1.set_title('Temperature Field (XY plane, top view)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Temperature (K)')

    # 2. Temperature field - XZ plane (side view)
    ax2 = plt.subplot(2, 3, 2)
    temp_xz = temp_grid[:, mid_y, :]
    im2 = ax2.imshow(temp_xz.T, cmap='hot', origin='lower', aspect='auto')
    ax2.set_title('Temperature Field (XZ plane, side view)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    plt.colorbar(im2, ax=ax2, label='Temperature (K)')

    # 3. Velocity magnitude - XY plane
    ax3 = plt.subplot(2, 3, 3)
    vel_xy = vel_mag_grid[mid_z, :, :]
    im3 = ax3.imshow(vel_xy.T, cmap='viridis', origin='lower', aspect='auto')
    ax3.set_title('Velocity Magnitude (XY plane)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3, label='Velocity (m/s)')

    # 4. Velocity vectors - XY plane
    ax4 = plt.subplot(2, 3, 4)
    # Subsample for quiver plot
    skip = max(1, nx // 20)
    X, Y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip))
    U = vel_grid[mid_z, ::skip, ::skip, 0].T
    V = vel_grid[mid_z, ::skip, ::skip, 1].T
    im4 = ax4.imshow(temp_xy.T, cmap='hot', origin='lower', aspect='auto', alpha=0.5)
    ax4.quiver(X, Y, U, V, color='white', alpha=0.8)
    ax4.set_title('Velocity Vectors with Temperature (XY plane)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')

    # 5. Velocity magnitude - XZ plane
    ax5 = plt.subplot(2, 3, 5)
    vel_xz = vel_mag_grid[:, mid_y, :]
    im5 = ax5.imshow(vel_xz.T, cmap='viridis', origin='lower', aspect='auto')
    ax5.set_title('Velocity Magnitude (XZ plane)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Z')
    plt.colorbar(im5, ax=ax5, label='Velocity (m/s)')

    # 6. Temperature gradient
    ax6 = plt.subplot(2, 3, 6)
    grad_temp = np.gradient(temp_xz)
    grad_mag = np.sqrt(grad_temp[0]**2 + grad_temp[1]**2)
    im6 = ax6.imshow(grad_mag.T, cmap='plasma', origin='lower', aspect='auto')
    ax6.set_title('Temperature Gradient Magnitude (XZ plane)')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Z')
    plt.colorbar(im6, ax=ax6, label='∇T (K/m)')

    plt.suptitle(f'Marangoni Flow Analysis: {Path(vtk_file).name}', fontsize=14)
    plt.tight_layout()

    output_path = Path(output_dir) / 'marangoni_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

    # Create surface analysis plot if fill level available
    if fill_level is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        fill_grid = vtk.reshape_to_grid(fill_level)

        # Surface reconstruction
        ax = axes[0, 0]
        fill_xz = fill_grid[:, mid_y, :]
        im = ax.imshow(fill_xz.T, cmap='Blues', origin='lower', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Liquid Fill Level (XZ plane)')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax, label='Fill Level')

        # Surface temperature
        ax = axes[0, 1]
        # Find surface indices
        surface_mask = (fill_grid > 0.4) & (fill_grid < 0.6)
        temp_surface = np.where(surface_mask, temp_grid, np.nan)
        im = ax.imshow(temp_surface[mid_z, :, :].T, cmap='hot', origin='lower', aspect='auto')
        ax.set_title('Surface Temperature (XY plane)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Temperature (K)')

        # Surface velocity
        ax = axes[1, 0]
        vel_surface = np.where(surface_mask, vel_mag_grid, np.nan)
        im = ax.imshow(vel_surface[mid_z, :, :].T, cmap='viridis', origin='lower', aspect='auto')
        ax.set_title('Surface Velocity (XY plane)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Velocity (m/s)')

        # Liquid fraction
        if liquid_fraction is not None:
            ax = axes[1, 1]
            lf_grid = vtk.reshape_to_grid(liquid_fraction)
            im = ax.imshow(lf_grid[mid_z, :, :].T, cmap='RdYlBu', origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set_title('Liquid Fraction (XY plane)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Liquid Fraction')

        plt.suptitle('Marangoni Flow Surface Analysis', fontsize=14)
        plt.tight_layout()

        output_path = Path(output_dir) / 'marangoni_surface_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Surface analysis saved to: {output_path}")
        plt.close()

    # Statistics
    print("\n" + "=" * 80)
    print("FIELD STATISTICS")
    print("=" * 80)
    print(f"Temperature:  min={np.min(temperature):.2f} K, "
          f"max={np.max(temperature):.2f} K, "
          f"mean={np.mean(temperature):.2f} K")
    print(f"Velocity:     min={np.min(vel_mag):.4e} m/s, "
          f"max={np.max(vel_mag):.4e} m/s, "
          f"mean={np.mean(vel_mag):.4e} m/s")

    if liquid_fraction is not None:
        print(f"Liquid Fraction: min={np.min(liquid_fraction):.3f}, "
              f"max={np.max(liquid_fraction):.3f}, "
              f"mean={np.mean(liquid_fraction):.3f}")

    print("=" * 80)


def compare_marangoni_timesteps(vtk_files: list, output_dir: str = '.'):
    """
    Compare Marangoni flow evolution across multiple timesteps.

    Args:
        vtk_files: List of VTK file paths (time-ordered)
        output_dir: Directory for output plots
    """
    print("=" * 80)
    print("MARANGONI FLOW TIME EVOLUTION")
    print("=" * 80)

    n_files = len(vtk_files)
    fig, axes = plt.subplots(n_files, 3, figsize=(15, 5*n_files))

    if n_files == 1:
        axes = axes.reshape(1, -1)

    for i, vtk_file in enumerate(vtk_files):
        print(f"\nProcessing: {vtk_file}")
        vtk = VTKData(vtk_file)

        velocity = vtk.get_field('Velocity')
        temperature = vtk.get_field('Temperature')

        if velocity is None or temperature is None:
            continue

        vel_mag = np.linalg.norm(velocity, axis=1)
        vel_mag_grid = vtk.reshape_to_grid(vel_mag)
        temp_grid = vtk.reshape_to_grid(temperature)

        mid_z = vtk.dimensions[2] // 2

        # Temperature
        im1 = axes[i, 0].imshow(temp_grid[mid_z, :, :].T, cmap='hot',
                               origin='lower', aspect='auto')
        axes[i, 0].set_title(f't={i}: Temperature')
        axes[i, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[i, 0], label='T (K)')

        # Velocity magnitude
        im2 = axes[i, 1].imshow(vel_mag_grid[mid_z, :, :].T, cmap='viridis',
                               origin='lower', aspect='auto')
        axes[i, 1].set_title(f't={i}: Velocity Magnitude')
        plt.colorbar(im2, ax=axes[i, 1], label='|v| (m/s)')

        # Combined view
        axes[i, 2].imshow(temp_grid[mid_z, :, :].T, cmap='hot',
                         origin='lower', aspect='auto', alpha=0.5)

        # Add velocity vectors
        nx, ny, nz = vtk.dimensions
        skip = max(1, nx // 15)
        vel_grid = vtk.reshape_to_grid(velocity)
        X, Y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip))
        U = vel_grid[mid_z, ::skip, ::skip, 0].T
        V = vel_grid[mid_z, ::skip, ::skip, 1].T
        axes[i, 2].quiver(X, Y, U, V, color='white', alpha=0.8)
        axes[i, 2].set_title(f't={i}: Combined View')

        if i == n_files - 1:
            axes[i, 0].set_xlabel('X')
            axes[i, 1].set_xlabel('X')
            axes[i, 2].set_xlabel('X')

    plt.suptitle('Marangoni Flow Time Evolution', fontsize=14)
    plt.tight_layout()

    output_path = Path(output_dir) / 'marangoni_time_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nTime evolution plot saved to: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize Marangoni flow from VTK files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('vtk_files', nargs='+', help='VTK file(s) to visualize')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for plots')
    parser.add_argument('--time-evolution', action='store_true',
                       help='Create time evolution plot (requires multiple files)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.time_evolution and len(args.vtk_files) > 1:
        compare_marangoni_timesteps(args.vtk_files, str(output_dir))
    else:
        for vtk_file in args.vtk_files:
            visualize_marangoni_flow(vtk_file, str(output_dir))


if __name__ == '__main__':
    main()
