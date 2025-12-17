#!/usr/bin/env python3
"""
Diagnostic analysis of velocity field structure.

Investigates why radial flow analysis shows 0% outward flow by examining
velocity field at multiple z-heights and producing detailed diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the VTK reader from the main analysis script
sys.path.insert(0, '/home/yzk/LBMProject/analysis/marangoni_perfect')

# === PARAMETERS ===
VTK_FILE = "/home/yzk/LBMProject/build/phase6_test2c_visualization/marangoni_flow_005000.vtk"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/marangoni_perfect"

# === VTK READER ===
class VTKStructuredPointsReader:
    """Simple reader for VTK structured points format."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.dimensions = None
        self.origin = None
        self.spacing = None
        self.n_points = None
        self.arrays = {}

    def read(self):
        """Read VTK file and parse data."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('DIMENSIONS'):
                parts = line.split()
                self.dimensions = tuple(map(int, parts[1:4]))
                self.n_points = np.prod(self.dimensions)

            elif line.startswith('ORIGIN'):
                parts = line.split()
                self.origin = tuple(map(float, parts[1:4]))

            elif line.startswith('SPACING'):
                parts = line.split()
                self.spacing = tuple(map(float, parts[1:4]))

            elif line.startswith('VECTORS'):
                parts = line.split()
                array_name = parts[1]
                i += 1
                data = []
                for _ in range(self.n_points):
                    vec_line = lines[i].strip().split()
                    data.append([float(x) for x in vec_line])
                    i += 1
                self.arrays[array_name] = np.array(data)
                continue

            elif line.startswith('SCALARS'):
                parts = line.split()
                array_name = parts[1]
                i += 1
                if lines[i].strip().startswith('LOOKUP_TABLE'):
                    i += 1
                data = []
                for _ in range(self.n_points):
                    data.append(float(lines[i].strip()))
                    i += 1
                self.arrays[array_name] = np.array(data)
                continue

            i += 1

        return self

    def get_points(self):
        """Generate point coordinates from structured grid."""
        nx, ny, nz = self.dimensions
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing

        x = ox + np.arange(nx) * dx
        y = oy + np.arange(ny) * dy
        z = oz + np.arange(nz) * dz

        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        return points

# === ANALYSIS ===

def main():
    print("="*70)
    print("VELOCITY FIELD DIAGNOSTIC ANALYSIS")
    print("="*70)

    # Load data
    print(f"\nLoading: {VTK_FILE}")
    reader = VTKStructuredPointsReader(VTK_FILE)
    reader.read()

    nx, ny, nz = reader.dimensions
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    print(f"Spacing: {reader.spacing[0]*1e6:.2f} µm")
    print(f"Domain size: {nx*reader.spacing[0]*1e6:.1f} x {ny*reader.spacing[1]*1e6:.1f} x {nz*reader.spacing[2]*1e6:.1f} µm³")

    # Get data
    points = reader.get_points()
    velocity = reader.arrays['Velocity']
    temperature = reader.arrays['Temperature']

    # Reshape to 3D grid
    vx = velocity[:, 0].reshape(nz, ny, nx)
    vy = velocity[:, 1].reshape(nz, ny, nx)
    vz = velocity[:, 2].reshape(nz, ny, nx)
    temp_3d = temperature.reshape(nz, ny, nx)

    # Velocity magnitude
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    print("\n" + "="*70)
    print("VELOCITY STATISTICS")
    print("="*70)

    print(f"\nVelocity X component:")
    print(f"  Min: {np.min(vx):.6f} m/s")
    print(f"  Max: {np.max(vx):.6f} m/s")
    print(f"  Mean: {np.mean(vx):.6f} m/s")
    print(f"  Std: {np.std(vx):.6f} m/s")

    print(f"\nVelocity Y component:")
    print(f"  Min: {np.min(vy):.6f} m/s")
    print(f"  Max: {np.max(vy):.6f} m/s")
    print(f"  Mean: {np.mean(vy):.6f} m/s")
    print(f"  Std: {np.std(vy):.6f} m/s")

    print(f"\nVelocity Z component:")
    print(f"  Min: {np.min(vz):.6f} m/s")
    print(f"  Max: {np.max(vz):.6f} m/s")
    print(f"  Mean: {np.mean(vz):.6f} m/s")
    print(f"  Std: {np.std(vz):.6f} m/s")

    print(f"\nVelocity Magnitude:")
    print(f"  Min: {np.min(vel_mag):.6f} m/s")
    print(f"  Max: {np.max(vel_mag):.6f} m/s")
    print(f"  Mean: {np.mean(vel_mag):.6f} m/s")
    print(f"  Std: {np.std(vel_mag):.6f} m/s")

    # Analyze at different z-heights
    print("\n" + "="*70)
    print("VELOCITY AT DIFFERENT Z-HEIGHTS")
    print("="*70)

    z_fractions = [0.25, 0.50, 0.75, 0.90, 0.95]
    for z_frac in z_fractions:
        z_idx = int(z_frac * (nz - 1))
        z_val = reader.origin[2] + z_idx * reader.spacing[2]

        vx_slice = vx[z_idx, :, :]
        vy_slice = vy[z_idx, :, :]
        vz_slice = vz[z_idx, :, :]
        vel_mag_slice = vel_mag[z_idx, :, :]

        print(f"\nz = {z_val*1e6:.2f} µm (z_fraction = {z_frac:.2f}):")
        print(f"  Vx: min={np.min(vx_slice):.6f}, max={np.max(vx_slice):.6f}, mean={np.mean(vx_slice):.6f}")
        print(f"  Vy: min={np.min(vy_slice):.6f}, max={np.max(vy_slice):.6f}, mean={np.mean(vy_slice):.6f}")
        print(f"  Vz: min={np.min(vz_slice):.6f}, max={np.max(vz_slice):.6f}, mean={np.mean(vz_slice):.6f}")
        print(f"  |V|: min={np.min(vel_mag_slice):.6f}, max={np.max(vel_mag_slice):.6f}, mean={np.mean(vel_mag_slice):.6f}")

        # Count non-zero velocities
        non_zero = vel_mag_slice > 1e-6
        print(f"  Non-zero velocity points: {np.sum(non_zero)} / {vel_mag_slice.size} ({100*np.sum(non_zero)/vel_mag_slice.size:.1f}%)")

    # Find location of maximum velocity
    print("\n" + "="*70)
    print("MAXIMUM VELOCITY LOCATION")
    print("="*70)

    idx_max = np.unravel_index(np.argmax(vel_mag), vel_mag.shape)
    z_max, y_max, x_max = idx_max
    x_coord = reader.origin[0] + x_max * reader.spacing[0]
    y_coord = reader.origin[1] + y_max * reader.spacing[1]
    z_coord = reader.origin[2] + z_max * reader.spacing[2]

    print(f"\nMax velocity location:")
    print(f"  Grid indices: ({x_max}, {y_max}, {z_max})")
    print(f"  Physical coords: ({x_coord*1e6:.2f}, {y_coord*1e6:.2f}, {z_coord*1e6:.2f}) µm")
    print(f"  Velocity: ({vx[z_max, y_max, x_max]:.6f}, {vy[z_max, y_max, x_max]:.6f}, {vz[z_max, y_max, x_max]:.6f}) m/s")
    print(f"  Magnitude: {vel_mag[z_max, y_max, x_max]:.6f} m/s")
    print(f"  Temperature: {temp_3d[z_max, y_max, x_max]:.2f} K")

    # Find location of maximum temperature
    idx_temp_max = np.unravel_index(np.argmax(temp_3d), temp_3d.shape)
    z_tmax, y_tmax, x_tmax = idx_temp_max
    print(f"\nMax temperature location:")
    print(f"  Grid indices: ({x_tmax}, {y_tmax}, {z_tmax})")
    print(f"  Temperature: {temp_3d[z_tmax, y_tmax, x_tmax]:.2f} K")

    # === VISUALIZATION ===
    print("\n" + "="*70)
    print("GENERATING DIAGNOSTIC FIGURES")
    print("="*70)

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    x = (reader.origin[0] + np.arange(nx) * reader.spacing[0]) * 1e6
    y = (reader.origin[1] + np.arange(ny) * reader.spacing[1]) * 1e6
    X, Y = np.meshgrid(x, y)

    z_levels = [0.25, 0.50, 0.75]

    for i, z_frac in enumerate(z_levels):
        z_idx = int(z_frac * (nz - 1))
        z_val = (reader.origin[2] + z_idx * reader.spacing[2]) * 1e6

        # Column 1: Velocity magnitude
        ax = axes[i, 0]
        im = ax.contourf(X, Y, vel_mag[z_idx, :, :], levels=20, cmap='viridis')
        ax.contour(X, Y, vel_mag[z_idx, :, :], levels=5, colors='k', linewidths=0.5, alpha=0.3)
        plt.colorbar(im, ax=ax, label='|V| (m/s)')
        ax.set_title(f'Velocity Magnitude (z={z_val:.1f} µm)', fontsize=10, fontweight='bold')
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_aspect('equal')

        # Column 2: Temperature
        ax = axes[i, 1]
        im = ax.contourf(X, Y, temp_3d[z_idx, :, :], levels=20, cmap='hot')
        ax.contour(X, Y, temp_3d[z_idx, :, :], levels=5, colors='k', linewidths=0.5, alpha=0.3)
        plt.colorbar(im, ax=ax, label='T (K)')
        ax.set_title(f'Temperature (z={z_val:.1f} µm)', fontsize=10, fontweight='bold')
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_aspect('equal')

        # Column 3: Velocity vectors
        ax = axes[i, 2]
        skip = 4
        ax.contourf(X, Y, vel_mag[z_idx, :, :], levels=20, cmap='viridis', alpha=0.6)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  vx[z_idx, ::skip, ::skip], vy[z_idx, ::skip, ::skip],
                  scale=2.0, width=0.003, headwidth=3, headlength=4, color='white')
        ax.set_title(f'Velocity Vectors (z={z_val:.1f} µm)', fontsize=10, fontweight='bold')
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_aspect('equal')

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/diagnostic_velocity_slices.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # === XZ and YZ CROSS-SECTIONS ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    z = (reader.origin[2] + np.arange(nz) * reader.spacing[2]) * 1e6

    # XZ slice at domain center (y = ny/2)
    y_mid = ny // 2
    X_xz, Z_xz = np.meshgrid(x, z)

    ax = axes[0, 0]
    im = ax.contourf(X_xz, Z_xz, vel_mag[:, y_mid, :], levels=20, cmap='viridis')
    plt.colorbar(im, ax=ax, label='|V| (m/s)')
    ax.set_title(f'Velocity Magnitude - XZ Slice (y={y[y_mid]:.1f} µm)', fontweight='bold')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('z (µm)')

    ax = axes[0, 1]
    im = ax.contourf(X_xz, Z_xz, temp_3d[:, y_mid, :], levels=20, cmap='hot')
    plt.colorbar(im, ax=ax, label='T (K)')
    ax.set_title(f'Temperature - XZ Slice (y={y[y_mid]:.1f} µm)', fontweight='bold')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('z (µm)')

    # YZ slice at domain center (x = nx/2)
    x_mid = nx // 2
    Y_yz, Z_yz = np.meshgrid(y, z)

    ax = axes[1, 0]
    im = ax.contourf(Y_yz, Z_yz, vel_mag[:, :, x_mid], levels=20, cmap='viridis')
    plt.colorbar(im, ax=ax, label='|V| (m/s)')
    ax.set_title(f'Velocity Magnitude - YZ Slice (x={x[x_mid]:.1f} µm)', fontweight='bold')
    ax.set_xlabel('y (µm)')
    ax.set_ylabel('z (µm)')

    ax = axes[1, 1]
    im = ax.contourf(Y_yz, Z_yz, temp_3d[:, :, x_mid], levels=20, cmap='hot')
    plt.colorbar(im, ax=ax, label='T (K)')
    ax.set_title(f'Temperature - YZ Slice (x={x[x_mid]:.1f} µm)', fontweight='bold')
    ax.set_xlabel('y (µm)')
    ax.set_ylabel('z (µm)')

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/diagnostic_cross_sections.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print("\n" + "="*70)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - diagnostic_velocity_slices.png")
    print("  - diagnostic_cross_sections.png")

if __name__ == "__main__":
    main()
