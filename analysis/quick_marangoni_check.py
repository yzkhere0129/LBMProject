#!/usr/bin/env python3
"""
Quick Marangoni Flow Check Script

Rapidly analyze a single VTK file to extract key Marangoni metrics.
Useful for quick validation of simulation output.

Usage:
    python quick_marangoni_check.py [vtk_file]

If no file specified, analyzes the most recent marangoni_flow VTK file.
"""

import sys
import numpy as np
import pyvista as pv
from pathlib import Path
import glob

# === PARAMETERS ===
DEFAULT_VTK_DIR = "/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization"
DEFAULT_PATTERN = "marangoni_flow_*.vtk"

SOLIDUS_TEMP = 1650.0
LIQUIDUS_TEMP = 1700.0
LIQUID_THRESHOLD = 0.5

def analyze_vtk_quick(vtk_file):
    """Quick analysis of a single VTK file"""
    print(f"\n{'='*70}")
    print(f"QUICK MARANGONI ANALYSIS: {Path(vtk_file).name}")
    print(f"{'='*70}\n")

    # Read file
    try:
        mesh = pv.read(vtk_file)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return

    # Check fields
    print(f"Available fields: {mesh.array_names}")
    print(f"Grid dimensions: {mesh.dimensions}")
    print(f"Number of points: {mesh.n_points}\n")

    # Extract data
    velocity = mesh['Velocity']
    temperature = mesh['Temperature']
    liquid_frac = mesh['LiquidFraction']

    # Compute velocity magnitude
    vel_mag = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2 + velocity[:, 2]**2)

    # Liquid mask
    liquid_mask = liquid_frac > LIQUID_THRESHOLD
    n_liquid = np.sum(liquid_mask)

    # === KEY METRICS ===
    print(f"--- VELOCITY FIELD ---")
    print(f"  Max velocity (global):    {np.max(vel_mag):.4f} m/s")
    print(f"  Mean velocity (global):   {np.mean(vel_mag):.4f} m/s")

    if n_liquid > 0:
        print(f"  Max velocity (liquid):    {np.max(vel_mag[liquid_mask]):.4f} m/s")
        print(f"  Mean velocity (liquid):   {np.mean(vel_mag[liquid_mask]):.4f} m/s")

    print(f"\n--- TEMPERATURE FIELD ---")
    print(f"  Max temperature:          {np.max(temperature):.1f} K")
    print(f"  Min temperature:          {np.min(temperature):.1f} K")
    print(f"  Temperature range:        {np.max(temperature) - np.min(temperature):.1f} K")

    if n_liquid > 0:
        print(f"  Mean temp (liquid):       {np.mean(temperature[liquid_mask]):.1f} K")

    print(f"\n--- PHASE FIELD ---")
    print(f"  Liquid points:            {n_liquid} ({100*n_liquid/mesh.n_points:.2f}%)")
    print(f"  Max liquid fraction:      {np.max(liquid_frac):.3f}")

    # Surface analysis (if FillLevel available)
    if 'FillLevel' in mesh.array_names:
        fill_level = mesh['FillLevel']
        surface_mask = (fill_level >= 0.4) & (fill_level <= 0.6)
        n_surface = np.sum(surface_mask)

        if n_surface > 0:
            print(f"\n--- FREE SURFACE ---")
            print(f"  Surface points:           {n_surface}")
            print(f"  Max surface velocity:     {np.max(vel_mag[surface_mask]):.4f} m/s")
            print(f"  Mean surface velocity:    {np.mean(vel_mag[surface_mask]):.4f} m/s")
            print(f"  Surface temp range:       {np.min(temperature[surface_mask]):.1f} - {np.max(temperature[surface_mask]):.1f} K")

    # === MARANGONI ASSESSMENT ===
    print(f"\n{'='*70}")
    print(f"MARANGONI ASSESSMENT")
    print(f"{'='*70}\n")

    max_vel = np.max(vel_mag[liquid_mask]) if n_liquid > 0 else np.max(vel_mag)

    print(f"Expected Marangoni velocity: 0.5 - 2.0 m/s")
    print(f"Observed maximum velocity:   {max_vel:.4f} m/s")

    if max_vel >= 0.5 and max_vel <= 2.5:
        print(f"\nSTATUS: ✓ WITHIN EXPECTED RANGE")
    elif max_vel < 0.5:
        print(f"\nSTATUS: ✗ BELOW EXPECTED (weak Marangoni effect)")
    else:
        print(f"\nSTATUS: ! ABOVE EXPECTED (very strong convection)")

    print(f"\n{'='*70}\n")

def main():
    """Main execution"""
    if len(sys.argv) > 1:
        vtk_file = sys.argv[1]
    else:
        # Find most recent VTK file
        vtk_files = sorted(glob.glob(f"{DEFAULT_VTK_DIR}/{DEFAULT_PATTERN}"))
        if len(vtk_files) == 0:
            print(f"ERROR: No VTK files found in {DEFAULT_VTK_DIR}")
            print(f"Usage: {sys.argv[0]} [vtk_file]")
            return
        vtk_file = vtk_files[-1]
        print(f"Auto-selected most recent file: {Path(vtk_file).name}")

    if not Path(vtk_file).exists():
        print(f"ERROR: File not found: {vtk_file}")
        return

    analyze_vtk_quick(vtk_file)

if __name__ == "__main__":
    main()
