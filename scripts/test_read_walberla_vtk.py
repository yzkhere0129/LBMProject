#!/usr/bin/env python3
"""Quick diagnostic to verify walberla VTK files can be read."""

import pyvista as pv
import numpy as np
from pathlib import Path

# walberla VTK directory
vtk_dir = Path("/home/yzk/walberla/build/apps/showcases/LaserHeating/vtk_out/laser_heating")

print("=" * 60)
print("Walberla VTK File Diagnostic")
print("=" * 60)

# Find VTK files
vtk_files = sorted(vtk_dir.glob("*.vti"))
if not vtk_files:
    print(f"ERROR: No .vti files found in {vtk_dir}")
    exit(1)

print(f"\nFound {len(vtk_files)} VTK files")
print(f"First file: {vtk_files[0].name}")
print(f"Last file: {vtk_files[-1].name}")

# Read first file
print(f"\n--- Reading {vtk_files[0].name} ---")
mesh = pv.read(vtk_files[0])

print(f"Mesh type: {type(mesh)}")
print(f"Number of points: {mesh.n_points}")
print(f"Number of cells: {mesh.n_cells}")
print(f"Bounds (m): {mesh.bounds}")
print(f"Dimensions: {mesh.dimensions}")

print(f"\nAvailable data arrays:")
for name in mesh.array_names:
    data = mesh[name]
    print(f"  {name}: shape={data.shape}, dtype={data.dtype}")
    print(f"    min={np.min(data):.2f}, max={np.max(data):.2f}, mean={np.mean(data):.2f}")

# Read file at t=50 μs (step 500)
target_step = 500
target_file = vtk_dir / f"simulation_step_{target_step}.vti"

if target_file.exists():
    print(f"\n--- Reading {target_file.name} (t=50 μs) ---")
    mesh_peak = pv.read(target_file)

    # Try to get temperature field
    temp_field_names = ['T', 'temperature', 'Temperature', 'scalar']
    temp = None
    temp_name = None

    for name in temp_field_names:
        if name in mesh_peak.array_names:
            temp = mesh_peak[name]
            temp_name = name
            break

    if temp is None and len(mesh_peak.array_names) > 0:
        temp_name = mesh_peak.array_names[0]
        temp = mesh_peak[temp_name]
        print(f"Using first field '{temp_name}' as temperature")

    if temp is not None:
        print(f"Temperature field: '{temp_name}'")
        print(f"  Peak temperature: {np.max(temp):.1f} K")
        print(f"  Mean temperature: {np.mean(temp):.1f} K")
        print(f"  Min temperature: {np.min(temp):.1f} K")

        # Expected from test
        expected_peak = 4099.0
        error = abs(np.max(temp) - expected_peak) / expected_peak * 100
        print(f"\nExpected peak: {expected_peak:.1f} K")
        print(f"Error: {error:.2f}%")

        if error < 0.1:
            print("✓ Peak temperature matches walberla reference!")
        else:
            print("⚠ Peak temperature differs from expected value")
    else:
        print("ERROR: No temperature field found!")
else:
    print(f"\nFile {target_file.name} not found")
    print("Available steps:", [f.stem.split('_')[-1] for f in vtk_files[:10]], "...")

print("\n" + "=" * 60)
