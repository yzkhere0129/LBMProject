#!/usr/bin/env python3
"""
Create dummy VTK files for testing the verification script.

This creates synthetic VTK data with known properties to test
all validation checks in verify_simulation_correctness.py
"""

import numpy as np
from pathlib import Path
import struct


def write_legacy_vtk(filename: str, nx: int, ny: int, nz: int,
                     temperature: np.ndarray, fill_level: np.ndarray):
    """Write a legacy VTK structured points file."""
    ncells = nx * ny * nz

    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Test simulation data\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")

        # Dimensions
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0.0 0.0 0.0\n")
        f.write("SPACING 1.0 1.0 1.0\n")

        # Point data
        f.write(f"POINT_DATA {ncells}\n")

        # Temperature field
        f.write("SCALARS temperature float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in temperature.flatten():
            f.write(f"{val}\n")

        # Fill level field
        f.write("SCALARS fill_level float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in fill_level.flatten():
            f.write(f"{val}\n")


def create_test_case_valid():
    """Create valid test data."""
    nx, ny, nz = 20, 20, 20
    ncells = nx * ny * nz

    # Create reasonable temperature field (300-1800 K)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Gaussian hot spot in center
    r_sq = (X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2
    temperature = 300.0 + 1500.0 * np.exp(-20 * r_sq)

    # Valid fill level (0-1)
    fill_level = 0.5 + 0.5 * np.exp(-10 * r_sq)

    return nx, ny, nz, temperature, fill_level


def create_test_case_invalid():
    """Create data with known issues."""
    nx, ny, nz = 20, 20, 20
    ncells = nx * ny * nz

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Temperature with issues
    r_sq = (X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2
    temperature = 300.0 + 1500.0 * np.exp(-20 * r_sq)

    # Add NaN values
    temperature[5:8, 5:8, 5:8] = np.nan

    # Add unrealistic high temperature
    temperature[12:15, 12:15, 12:15] = 100000.0

    # Fill level with issues
    fill_level = 0.5 + 0.5 * np.exp(-10 * r_sq)

    # Add out-of-bounds values
    fill_level[0:3, 0:3, 0:3] = -0.1
    fill_level[17:20, 17:20, 17:20] = 1.5

    return nx, ny, nz, temperature, fill_level


def main():
    """Create test VTK files."""
    output_dir = Path('/home/yzk/LBMProject/scripts/test_vtk_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating test VTK files...")

    # Valid case - multiple timesteps for conservation checks
    print("  Creating valid timesteps...")
    nx, ny, nz, temp, fill = create_test_case_valid()

    for ts in [0, 10, 20, 30]:
        # Add small variation to simulate time evolution
        temp_ts = temp + ts * 0.1 * np.random.randn(*temp.shape)
        fill_ts = np.clip(fill + ts * 0.001 * np.random.randn(*fill.shape), 0, 1)

        filename = output_dir / f'valid_{ts:04d}.vtk'
        write_legacy_vtk(str(filename), nx, ny, nz, temp_ts, fill_ts)
        print(f"    Created {filename}")

    # Invalid case - single timestep to test field validation
    print("  Creating invalid timestep...")
    nx, ny, nz, temp, fill = create_test_case_invalid()
    filename = output_dir / 'invalid_0000.vtk'
    write_legacy_vtk(str(filename), nx, ny, nz, temp, fill)
    print(f"    Created {filename}")

    print(f"\nTest data created in: {output_dir}")
    print("\nYou can now test the verification script:")
    print(f"  python3 /home/yzk/LBMProject/scripts/verify_simulation_correctness.py {output_dir}")

    # Also print example of expected structure
    print(f"\n--- Example VTK file structure (first 20 lines of {filename}) ---")
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            print(line.rstrip())


if __name__ == '__main__':
    main()
