#!/usr/bin/env python3
"""
Quick diagnostic: Check temperature profile at peak heating time (t=49 μs)
"""

import numpy as np
from pathlib import Path

def read_vtk_structured_points(filename):
    """Read VTK structured points file (ASCII format)"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    nx, ny, nz = None, None, None
    dx, dy, dz = None, None, None
    data_start_line = None

    for i, line in enumerate(lines):
        if line.startswith('DIMENSIONS'):
            parts = line.split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
        elif line.startswith('SPACING'):
            parts = line.split()
            dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
        elif line.startswith('LOOKUP_TABLE'):
            data_start_line = i + 1
            break

    if data_start_line is None:
        raise ValueError(f"Could not find data in {filename}")

    # Read data
    data_lines = lines[data_start_line:]
    data = []
    for line in data_lines:
        values = line.strip().split()
        data.extend([float(v) for v in values])

    data = np.array(data)

    return {
        'nx': nx, 'ny': ny, 'nz': nz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'data': data.reshape((nz, ny, nx))
    }

def main():
    # Find file at t=49 μs (step 49000)
    output_dir = Path("/home/yzk/LBMProject/tests/validation/output_laser_melting_senior")
    vtk_file = output_dir / "temperature_049000.vtk.vtk"

    if not vtk_file.exists():
        print(f"ERROR: File not found: {vtk_file}")
        return

    print("=" * 70)
    print("TEMPERATURE PROFILE DIAGNOSTIC (t = 49 μs, peak heating)")
    print("=" * 70)

    vtk_data = read_vtk_structured_points(str(vtk_file))
    temp = vtk_data['data']
    nx, ny, nz = vtk_data['nx'], vtk_data['ny'], vtk_data['nz']
    dx, dy, dz = vtk_data['dx'], vtk_data['dy'], vtk_data['dz']

    print(f"\nGrid: {nx} × {ny} × {nz}")
    print(f"Spacing: dx={dx*1e6:.2f} μm, dy={dy*1e6:.2f} μm, dz={dz*1e6:.2f} μm")
    print(f"Domain: {nx*dx*1e6:.1f} × {ny*dy*1e6:.1f} × {nz*dz*1e6:.1f} μm³")

    # Overall statistics
    print(f"\nOverall Temperature Statistics:")
    print(f"  Min: {np.min(temp):.1f} K")
    print(f"  Max: {np.max(temp):.1f} K")
    print(f"  Mean: {np.mean(temp):.1f} K")
    print(f"  Std: {np.std(temp):.1f} K")

    # Find location of max temperature
    max_idx = np.argmax(temp)
    max_k, max_j, max_i = np.unravel_index(max_idx, temp.shape)
    print(f"\nMax temperature location:")
    print(f"  Grid indices: i={max_i}, j={max_j}, k={max_k}")
    print(f"  Physical location: x={max_i*dx*1e6:.1f} μm, y={max_j*dy*1e6:.1f} μm, z={max_k*dz*1e6:.1f} μm")

    # Laser center should be at x=75 μm, y=150 μm (center of domain)
    center_i = nx // 2
    center_j = ny // 2
    print(f"\nExpected laser center:")
    print(f"  Grid indices: i={center_i}, j={center_j}")
    print(f"  Physical location: x={center_i*dx*1e6:.1f} μm, y={center_j*dy*1e6:.1f} μm")

    # Temperature profile along vertical (z) axis at laser center
    print(f"\nVertical temperature profile at laser center (i={center_i}, j={center_j}):")
    print("  k    z(μm)   T(K)")
    print("  " + "-"*30)
    for k in range(nz-1, -1, -5):  # Top to bottom, every 5 cells
        z = k * dz * 1e6
        T = temp[k, center_j, center_i]
        print(f"  {k:2d}   {z:6.1f}   {T:7.1f}")

    # Check if laser spot is visible (radial profile at surface)
    print(f"\nRadial temperature profile at surface (k={nz-1}):")
    print("  r(μm)   T(K)")
    print("  " + "-"*20)
    for r_um in [0, 10, 20, 30, 40, 50]:
        # Find approximate cell at radius r
        r_cells = int(r_um / (dx * 1e6))
        if r_cells < nx // 2:
            i = center_i + r_cells
            j = center_j
            T = temp[nz-1, j, i]
            print(f"  {r_um:5.0f}   {T:7.1f}")

    # Count cells above certain temperatures
    print(f"\nTemperature distribution:")
    for T_threshold in [1900, 1500, 1000, 500]:
        count = np.sum(temp > T_threshold)
        percent = 100.0 * count / temp.size
        print(f"  Cells > {T_threshold} K: {count:6d} ({percent:5.2f}%)")

    # Material properties
    T_liquidus = 1923.0
    T_solidus = 1878.0
    print(f"\nMaterial properties (Ti6Al4V):")
    print(f"  T_solidus:  {T_solidus:.0f} K")
    print(f"  T_liquidus: {T_liquidus:.0f} K")
    print(f"  Max T / T_liquidus: {np.max(temp) / T_liquidus:.3f}")

    if np.max(temp) < T_liquidus:
        shortage = T_liquidus - np.max(temp)
        print(f"\n  WARNING: Peak temperature is {shortage:.1f} K below liquidus!")
        print(f"           No melting occurred.")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
