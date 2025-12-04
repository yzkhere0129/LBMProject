#!/usr/bin/env python3
"""
Simple velocity divergence analysis with only velocity and temperature fields.

This works with the old VTK format that only has Velocity and Temperature.
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import sys
import os

def read_vtk(filename):
    """Read structured grid VTK file"""
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def compute_divergence_3d(fx, fy, fz, nx, ny, nz, dx):
    """
    Compute divergence of 3D vector field.

    Uses central differences with periodic boundaries.
    """
    # Reshape to 3D grid (z-major, then y, then x)
    fx_3d = fx.reshape((nz, ny, nx))
    fy_3d = fy.reshape((nz, ny, nx))
    fz_3d = fz.reshape((nz, ny, nx))

    # Central differences (periodic)
    dfx_dx = (np.roll(fx_3d, -1, axis=2) - np.roll(fx_3d, 1, axis=2)) / (2 * dx)
    dfy_dy = (np.roll(fy_3d, -1, axis=1) - np.roll(fy_3d, 1, axis=1)) / (2 * dx)
    dfz_dz = (np.roll(fz_3d, -1, axis=0) - np.roll(fz_3d, 1, axis=0)) / (2 * dx)

    # Divergence
    div = dfx_dx + dfy_dy + dfz_dz

    return div.flatten()

def estimate_marangoni_divergence_from_temperature(temperature, nx, ny, nz, dx, dsigma_dT=-0.26e-3, h_interface=2.0):
    """
    Estimate ∇·F_marangoni from temperature gradients.

    Simplified: ∇·F_m ≈ (dσ/dT)/h · ∇²T (assuming |∇f| ≈ 1/h at interface)
    """
    T_3d = temperature.reshape((nz, ny, nx))

    # Compute Laplacian of temperature (∇²T)
    laplacian_T = np.zeros_like(T_3d)

    # X direction
    laplacian_T += (np.roll(T_3d, -1, axis=2) - 2*T_3d + np.roll(T_3d, 1, axis=2)) / (dx**2)

    # Y direction
    laplacian_T += (np.roll(T_3d, -1, axis=1) - 2*T_3d + np.roll(T_3d, 1, axis=1)) / (dx**2)

    # Z direction
    laplacian_T += (np.roll(T_3d, -1, axis=0) - 2*T_3d + np.roll(T_3d, 1, axis=0)) / (dx**2)

    # Estimate divergence of Marangoni force
    # ∇·F ≈ (dσ/dT) / h * ∇²T
    div_F_marangoni = (dsigma_dT / h_interface) * laplacian_T

    return div_F_marangoni.flatten(), laplacian_T

def estimate_darcy_divergence_from_velocity(ux, uy, uz, temperature, T_solidus, T_liquidus, nx, ny, nz, dx, darcy_coeff=1e7):
    """
    Estimate ∇·F_darcy from velocity and temperature.

    Liquid fraction estimated from temperature:
      fl = 0 if T < T_solidus
      fl = (T - T_solidus)/(T_liquidus - T_solidus) if T_solidus < T < T_liquidus
      fl = 1 if T > T_liquidus

    Darcy force: F = -C·(1-fl)²/(fl³+ε)·u
    """
    # Estimate liquid fraction from temperature
    liquid_fraction = np.zeros_like(temperature)
    solid_mask = temperature < T_solidus
    mushy_mask = (temperature >= T_solidus) & (temperature <= T_liquidus)
    liquid_mask = temperature > T_liquidus

    liquid_fraction[solid_mask] = 0.0
    liquid_fraction[mushy_mask] = (temperature[mushy_mask] - T_solidus) / (T_liquidus - T_solidus)
    liquid_fraction[liquid_mask] = 1.0

    # Darcy damping factor
    eps = 1e-3
    damping_factor = -darcy_coeff * (1 - liquid_fraction)**2 / (liquid_fraction**3 + eps)

    # Darcy force
    Fx_darcy = damping_factor * ux
    Fy_darcy = damping_factor * uy
    Fz_darcy = damping_factor * uz

    # Divergence
    div_darcy = compute_divergence_3d(Fx_darcy, Fy_darcy, Fz_darcy, nx, ny, nz, dx)

    return div_darcy, liquid_fraction, damping_factor

def main():
    # Try to find VTK file
    vtk_candidates = [
        "lpbf_realistic/lpbf_001000.vtk",
        "build/lpbf_realistic/lpbf_001000.vtk",
        "build/lpbf_realistic_backup_20251115_203414/lpbf_009000.vtk",
    ]

    vtk_file = None
    for candidate in vtk_candidates:
        if os.path.exists(candidate):
            vtk_file = candidate
            break

    if vtk_file is None:
        print("ERROR: No VTK file found")
        sys.exit(1)

    # Configuration
    dx = 2.0e-6  # meters
    T_solidus = 1878.0  # K (Ti6Al4V)
    T_liquidus = 1928.0  # K

    print("=" * 70)
    print("VELOCITY DIVERGENCE ANALYSIS (Simplified)")
    print("=" * 70)
    print()

    # Read VTK
    print(f"Reading: {vtk_file}")
    data = read_vtk(vtk_file)

    # Extract dimensions
    dims = data.GetDimensions()
    nx, ny, nz = dims[0], dims[1], dims[2]
    num_cells = nx * ny * nz

    print(f"Grid: {nx} x {ny} x {nz} = {num_cells} cells")
    print(f"Cell size: {dx * 1e6:.2f} μm")
    print()

    # Extract fields
    point_data = data.GetPointData()

    temperature = vtk_to_numpy(point_data.GetArray('Temperature'))
    velocity = vtk_to_numpy(point_data.GetArray('Velocity'))

    ux = velocity[:, 0]
    uy = velocity[:, 1]
    uz = velocity[:, 2]

    print("Fields:")
    print(f"  Temperature: [{temperature.min():.1f}, {temperature.max():.1f}] K")
    print(f"  Velocity: max = {np.max(np.linalg.norm(velocity, axis=1)) * 1e3:.3f} mm/s")
    print()

    # Compute velocity divergence
    print("Computing velocity divergence...")
    div_u = compute_divergence_3d(ux, uy, uz, nx, ny, nz, dx)

    print(f"\nVelocity divergence (∇·u):")
    print(f"  max |∇·u| = {np.max(np.abs(div_u)):.3e} s^-1")
    print(f"  mean |∇·u| = {np.mean(np.abs(div_u)):.3e} s^-1")
    print(f"  RMS ∇·u = {np.sqrt(np.mean(div_u**2)):.3e} s^-1")

    # Find where divergence is largest
    max_div_idx = np.argmax(np.abs(div_u))
    k_max = max_div_idx // (nx * ny)
    j_max = (max_div_idx % (nx * ny)) // nx
    i_max = max_div_idx % nx

    print(f"\n  Location of max |∇·u|: ({i_max}, {j_max}, {k_max})")
    print(f"    Temperature: {temperature[max_div_idx]:.1f} K")
    print(f"    Velocity: ({ux[max_div_idx]*1e3:.2f}, {uy[max_div_idx]*1e3:.2f}, {uz[max_div_idx]*1e3:.2f}) mm/s")
    print()

    # Estimate Marangoni contribution
    print("Estimating Marangoni force divergence...")
    div_marangoni, laplacian_T = estimate_marangoni_divergence_from_temperature(
        temperature, nx, ny, nz, dx
    )

    print(f"\nMarangoni force divergence estimate:")
    print(f"  max |∇·F_marangoni| = {np.max(np.abs(div_marangoni)):.3e} N/m^4")
    print(f"  mean |∇·F_marangoni| = {np.mean(np.abs(div_marangoni)):.3e} N/m^4")

    # Where is temperature Laplacian largest?
    max_laplacian_idx = np.argmax(np.abs(laplacian_T.flatten()))
    k_lap = max_laplacian_idx // (nx * ny)
    j_lap = (max_laplacian_idx % (nx * ny)) // nx
    i_lap = max_laplacian_idx % nx
    print(f"  Location of max |∇²T|: ({i_lap}, {j_lap}, {k_lap})")
    print(f"    ∇²T = {laplacian_T.flatten()[max_laplacian_idx]:.3e} K/m²")
    print()

    # Estimate Darcy contribution
    print("Estimating Darcy damping force divergence...")
    div_darcy, liquid_fraction, damping_factor = estimate_darcy_divergence_from_velocity(
        ux, uy, uz, temperature, T_solidus, T_liquidus, nx, ny, nz, dx
    )

    print(f"\nDarcy force divergence estimate:")
    print(f"  max |∇·F_darcy| = {np.max(np.abs(div_darcy)):.3e} N/m^4")
    print(f"  mean |∇·F_darcy| = {np.mean(np.abs(div_darcy)):.3e} N/m^4")

    # Statistics by phase
    solid_mask = liquid_fraction < 0.01
    mushy_mask = (liquid_fraction >= 0.01) & (liquid_fraction <= 0.99)
    liquid_mask = liquid_fraction > 0.99

    print(f"\nPhase distribution:")
    print(f"  Solid (fl < 0.01): {np.sum(solid_mask)} cells ({100*np.sum(solid_mask)/num_cells:.1f}%)")
    print(f"  Mushy (0.01 ≤ fl ≤ 0.99): {np.sum(mushy_mask)} cells ({100*np.sum(mushy_mask)/num_cells:.1f}%)")
    print(f"  Liquid (fl > 0.99): {np.sum(liquid_mask)} cells ({100*np.sum(liquid_mask)/num_cells:.1f}%)")

    if np.any(mushy_mask):
        print(f"\n  In mushy zone:")
        print(f"    max |∇·F_darcy| = {np.max(np.abs(div_darcy[mushy_mask])):.3e} N/m^4")
        print(f"    max |damping_factor| = {np.max(np.abs(damping_factor[mushy_mask])):.3e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    max_div_m = np.max(np.abs(div_marangoni))
    max_div_d = np.max(np.abs(div_darcy))

    print(f"\n{'Source':<30} {'max |∇·F|':<20}")
    print("-" * 70)
    print(f"{'Marangoni (estimated)':<30} {max_div_m:<20.3e}")
    print(f"{'Darcy damping (estimated)':<30} {max_div_d:<20.3e}")
    print()

    if max_div_d > 2 * max_div_m:
        print("CULPRIT: Darcy damping")
        print("\nPROBLEM: F_darcy = -C·(1-fl)²/(fl³+ε)·u has divergence")
        print("         ∇·F_darcy = -C·∇·[(1-fl)²/(fl³+ε)·u]")
        print("         This is NON-ZERO when ∇fl ≠ 0 (at solid-liquid interface)")
        print("\nSOLUTION:")
        print("  1. Reduce darcy_coefficient from 1e7 to 1e6")
        print("  2. Use divergence-free Darcy formulation")
        print("  3. Smooth liquid fraction field")
    elif max_div_m > 2 * max_div_d:
        print("CULPRIT: Marangoni force")
        print("\nPROBLEM: Strong temperature gradients from laser")
        print("\nSOLUTION:")
        print("  1. Verify interface localization")
        print("  2. Smooth temperature field")
    else:
        print("BOTH forces contribute to divergence")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
