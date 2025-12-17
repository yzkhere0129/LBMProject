#!/usr/bin/env python3
"""
Quick visualization script for Marangoni flow VTK files.

Usage:
    python3 quick_visualize.py <vtk_file> [output_name]

Example:
    python3 quick_visualize.py ../../build/phase6_test2c_visualization/marangoni_flow_005000.vtk final_state
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

class VTKReader:
    """Lightweight VTK structured points reader."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.dimensions = None
        self.origin = None
        self.spacing = None
        self.n_points = None
        self.arrays = {}

    def read(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('DIMENSIONS'):
                self.dimensions = tuple(map(int, line.split()[1:4]))
                self.n_points = np.prod(self.dimensions)

            elif line.startswith('ORIGIN'):
                self.origin = tuple(map(float, line.split()[1:4]))

            elif line.startswith('SPACING'):
                self.spacing = tuple(map(float, line.split()[1:4]))

            elif line.startswith('VECTORS'):
                array_name = line.split()[1]
                i += 1
                data = []
                for _ in range(self.n_points):
                    data.append([float(x) for x in lines[i].strip().split()])
                    i += 1
                self.arrays[array_name] = np.array(data)
                continue

            elif line.startswith('SCALARS'):
                array_name = line.split()[1]
                i += 1
                if lines[i].strip().startswith('LOOKUP_TABLE'):
                    i += 1
                data = [float(lines[i + j].strip()) for j in range(self.n_points)]
                self.arrays[array_name] = np.array(data)
                i += self.n_points
                continue

            i += 1

        return self

def visualize_vtk(vtk_file, output_name=None):
    """Generate quick visualization of VTK file."""

    print(f"Loading: {vtk_file}")
    reader = VTKReader(vtk_file).read()

    nx, ny, nz = reader.dimensions
    print(f"Grid: {nx} x {ny} x {nz}")

    # Reshape data to 3D
    temp = reader.arrays['Temperature'].reshape(nz, ny, nx)
    vel = reader.arrays['Velocity']
    vx = vel[:, 0].reshape(nz, ny, nx)
    vy = vel[:, 1].reshape(nz, ny, nx)
    vz = vel[:, 2].reshape(nz, ny, nx)
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Find interesting z-slice (where max velocity is)
    z_max_vel = np.unravel_index(np.argmax(vel_mag), vel_mag.shape)[0]

    # Create coordinate arrays
    x = (reader.origin[0] + np.arange(nx) * reader.spacing[0]) * 1e6
    y = (reader.origin[1] + np.arange(ny) * reader.spacing[1]) * 1e6
    z = (reader.origin[2] + np.arange(nz) * reader.spacing[2]) * 1e6
    X, Y = np.meshgrid(x, y)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Temperature at max velocity slice
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X, Y, temp[z_max_vel, :, :], levels=20, cmap='hot')
    plt.colorbar(im1, ax=ax1, label='Temperature (K)')
    ax1.set_title(f'Temperature (z={z[z_max_vel]:.1f} µm)', fontweight='bold')
    ax1.set_xlabel('x (µm)')
    ax1.set_ylabel('y (µm)')
    ax1.set_aspect('equal')

    # Velocity magnitude at max velocity slice
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(X, Y, vel_mag[z_max_vel, :, :], levels=20, cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='|V| (m/s)')
    ax2.set_title(f'Velocity Magnitude (z={z[z_max_vel]:.1f} µm)', fontweight='bold')
    ax2.set_xlabel('x (µm)')
    ax2.set_ylabel('y (µm)')
    ax2.set_aspect('equal')

    # Velocity vectors at max velocity slice
    ax3 = fig.add_subplot(gs[0, 2])
    skip = max(1, nx // 16)
    ax3.contourf(X, Y, vel_mag[z_max_vel, :, :], levels=20, cmap='viridis', alpha=0.6)
    ax3.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               vx[z_max_vel, ::skip, ::skip], vy[z_max_vel, ::skip, ::skip],
               scale=2.0, width=0.003, color='white')
    ax3.set_title(f'Velocity Vectors (z={z[z_max_vel]:.1f} µm)', fontweight='bold')
    ax3.set_xlabel('x (µm)')
    ax3.set_ylabel('y (µm)')
    ax3.set_aspect('equal')

    # XZ cross-section at domain center
    y_mid = ny // 2
    X_xz, Z_xz = np.meshgrid(x, z)

    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.contourf(X_xz, Z_xz, temp[:, y_mid, :], levels=20, cmap='hot')
    plt.colorbar(im4, ax=ax4, label='Temperature (K)')
    ax4.set_title(f'Temperature XZ (y={y[y_mid]:.1f} µm)', fontweight='bold')
    ax4.set_xlabel('x (µm)')
    ax4.set_ylabel('z (µm)')

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.contourf(X_xz, Z_xz, vel_mag[:, y_mid, :], levels=20, cmap='viridis')
    plt.colorbar(im5, ax=ax5, label='|V| (m/s)')
    ax5.set_title(f'Velocity Magnitude XZ (y={y[y_mid]:.1f} µm)', fontweight='bold')
    ax5.set_xlabel('x (µm)')
    ax5.set_ylabel('z (µm)')

    # Statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    stats_text = f"""
    STATISTICS

    Temperature:
      Min:  {np.min(temp):.1f} K
      Max:  {np.max(temp):.1f} K
      Mean: {np.mean(temp):.1f} K
      Std:  {np.std(temp):.1f} K

    Velocity Magnitude:
      Min:  {np.min(vel_mag):.4f} m/s
      Max:  {np.max(vel_mag):.4f} m/s
      Mean: {np.mean(vel_mag):.4f} m/s
      Std:  {np.std(vel_mag):.4f} m/s

    Grid:
      Dimensions: {nx} x {ny} x {nz}
      Spacing: {reader.spacing[0]*1e6:.2f} µm
      Domain: {x[-1]-x[0]:.1f} x {y[-1]-y[0]:.1f} x {z[-1]-z[0]:.1f} µm³

    Max velocity location:
      z = {z[z_max_vel]:.1f} µm
    """

    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)

    # Finalize
    vtk_name = Path(vtk_file).stem
    fig.suptitle(f'Quick Visualization: {vtk_name}', fontsize=14, fontweight='bold')

    if output_name is None:
        output_name = vtk_name

    output_path = f"{output_name}_quick_viz.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Print summary to console
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"Temperature: {np.min(temp):.1f} - {np.max(temp):.1f} K")
    print(f"Velocity:    {np.min(vel_mag):.4f} - {np.max(vel_mag):.4f} m/s")
    print(f"Max velocity at z = {z[z_max_vel]:.1f} µm")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 quick_visualize.py <vtk_file> [output_name]")
        print("\nExample:")
        print("  python3 quick_visualize.py ../../build/phase6_test2c_visualization/marangoni_flow_005000.vtk")
        sys.exit(1)

    vtk_file = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None

    visualize_vtk(vtk_file, output_name)
