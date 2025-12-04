#!/usr/bin/env python3
"""
Isolate which force field contributes most to velocity divergence.

This script:
1. Reads VTK output from LPBF simulation
2. Extracts velocity and temperature fields
3. Computes divergence of velocity
4. Estimates divergence contribution from each force component

Run after simulation step 1000.
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

    # Central differences
    dfx_dx = np.zeros_like(fx_3d)
    dfy_dy = np.zeros_like(fy_3d)
    dfz_dz = np.zeros_like(fz_3d)

    # X derivative (periodic)
    dfx_dx[:, :, :] = (np.roll(fx_3d, -1, axis=2) - np.roll(fx_3d, 1, axis=2)) / (2 * dx)

    # Y derivative (periodic)
    dfy_dy[:, :, :] = (np.roll(fy_3d, -1, axis=1) - np.roll(fy_3d, 1, axis=1)) / (2 * dx)

    # Z derivative (periodic)
    dfz_dz[:, :, :] = (np.roll(fz_3d, -1, axis=0) - np.roll(fz_3d, 1, axis=0)) / (2 * dx)

    # Divergence
    div = dfx_dx + dfy_dy + dfz_dz

    return div.flatten()

def estimate_marangoni_force_divergence(temperature, fill_level, nx, ny, nz, dx, dsigma_dT=-0.26e-3, h_interface=2.0):
    """
    Estimate divergence of Marangoni force.

    Marangoni force: F = (dσ/dT) * ∇_s T * |∇f| / h
    where ∇_s T is tangential temperature gradient

    Simplified: We'll compute ∇·(|∇f| * ∇T) as an approximation
    """
    T_3d = temperature.reshape((nz, ny, nx))
    f_3d = fill_level.reshape((nz, ny, nx))

    # Compute |∇f|
    grad_f_x = (np.roll(f_3d, -1, axis=2) - np.roll(f_3d, 1, axis=2)) / (2 * dx)
    grad_f_y = (np.roll(f_3d, -1, axis=1) - np.roll(f_3d, 1, axis=1)) / (2 * dx)
    grad_f_z = (np.roll(f_3d, -1, axis=0) - np.roll(f_3d, 1, axis=0)) / (2 * dx)
    grad_f_mag = np.sqrt(grad_f_x**2 + grad_f_y**2 + grad_f_z**2)

    # Compute ∇T
    grad_T_x = (np.roll(T_3d, -1, axis=2) - np.roll(T_3d, 1, axis=2)) / (2 * dx)
    grad_T_y = (np.roll(T_3d, -1, axis=1) - np.roll(T_3d, 1, axis=1)) / (2 * dx)
    grad_T_z = (np.roll(T_3d, -1, axis=0) - np.roll(T_3d, 1, axis=0)) / (2 * dx)

    # Marangoni force components (simplified - not computing tangential projection)
    coeff = dsigma_dT * grad_f_mag / h_interface
    Fx_marangoni = coeff * grad_T_x
    Fy_marangoni = coeff * grad_T_y
    Fz_marangoni = coeff * grad_T_z

    # Compute divergence
    div_marangoni = compute_divergence_3d(
        Fx_marangoni.flatten(),
        Fy_marangoni.flatten(),
        Fz_marangoni.flatten(),
        nx, ny, nz, dx
    )

    return div_marangoni, Fx_marangoni.flatten(), Fy_marangoni.flatten(), Fz_marangoni.flatten()

def estimate_darcy_force_divergence(ux, uy, uz, liquid_fraction, nx, ny, nz, dx, darcy_coeff=1e7):
    """
    Estimate divergence of Darcy damping force.

    F_darcy = -C * (1-fl)^2 / (fl^3 + ε) * u
    """
    eps = 1e-3
    damping_factor = -darcy_coeff * (1 - liquid_fraction)**2 / (liquid_fraction**3 + eps)

    Fx_darcy = damping_factor * ux
    Fy_darcy = damping_factor * uy
    Fz_darcy = damping_factor * uz

    div_darcy = compute_divergence_3d(Fx_darcy, Fy_darcy, Fz_darcy, nx, ny, nz, dx)

    return div_darcy, Fx_darcy, Fy_darcy, Fz_darcy

def main():
    # Configuration
    # Try multiple possible locations
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
        print(f"ERROR: No VTK file found. Tried:")
        for c in vtk_candidates:
            print(f"  - {c}")
        print("Please run simulation first:")
        print("  cd /home/yzk/LBMProject/build")
        print("  ./visualize_lpbf_marangoni_realistic")
        sys.exit(1)

    dx = 2.0e-6  # Cell size in meters

    print("=" * 70)
    print("FORCE DIVERGENCE ISOLATION ANALYSIS")
    print("=" * 70)
    print()

    # Read VTK
    print(f"Reading: {vtk_file}")
    data = read_vtk(vtk_file)

    # Extract grid dimensions
    dims = data.GetDimensions()
    nx, ny, nz = dims[0], dims[1], dims[2]
    num_cells = nx * ny * nz

    print(f"Grid: {nx} x {ny} x {nz} = {num_cells} cells")
    print(f"Cell size: {dx * 1e6:.2f} μm")
    print()

    # Extract fields
    point_data = data.GetPointData()

    temperature = vtk_to_numpy(point_data.GetArray('Temperature'))
    fill_level = vtk_to_numpy(point_data.GetArray('FillLevel'))
    phase = vtk_to_numpy(point_data.GetArray('Phase'))
    velocity = vtk_to_numpy(point_data.GetArray('Velocity'))

    ux = velocity[:, 0]
    uy = velocity[:, 1]
    uz = velocity[:, 2]

    print("Fields extracted:")
    print(f"  Temperature: [{temperature.min():.1f}, {temperature.max():.1f}] K")
    print(f"  Fill level: [{fill_level.min():.3f}, {fill_level.max():.3f}]")
    print(f"  Velocity magnitude: max = {np.max(np.linalg.norm(velocity, axis=1)) * 1e3:.3f} mm/s")
    print()

    # Compute velocity divergence
    print("Computing velocity divergence...")
    div_u = compute_divergence_3d(ux, uy, uz, nx, ny, nz, dx)

    print(f"Velocity divergence statistics:")
    print(f"  max |∇·u| = {np.max(np.abs(div_u)):.3e} s^-1")
    print(f"  mean |∇·u| = {np.mean(np.abs(div_u)):.3e} s^-1")
    print(f"  std |∇·u| = {np.std(np.abs(div_u)):.3e} s^-1")
    print()

    # Estimate liquid fraction from phase (0=solid, 1=mushy, 2=liquid)
    # For Darcy: liquid_fraction = 0 (solid), 0.5 (mushy), 1 (liquid)
    liquid_fraction = np.zeros_like(phase)
    liquid_fraction[phase < 0.5] = 0.0   # Solid
    liquid_fraction[(phase >= 0.5) & (phase < 1.5)] = 0.5   # Mushy
    liquid_fraction[phase >= 1.5] = 1.0   # Liquid

    # Estimate Marangoni force divergence
    print("Estimating Marangoni force divergence...")
    div_marangoni, Fx_m, Fy_m, Fz_m = estimate_marangoni_force_divergence(
        temperature, fill_level, nx, ny, nz, dx
    )

    print(f"Marangoni force divergence:")
    print(f"  max |∇·F_marangoni| = {np.max(np.abs(div_marangoni)):.3e} N/m^4")
    print(f"  mean |∇·F_marangoni| = {np.mean(np.abs(div_marangoni)):.3e} N/m^4")

    # Where is Marangoni force largest?
    interface_mask = (fill_level > 0.01) & (fill_level < 0.99)
    if np.any(interface_mask):
        div_m_interface = div_marangoni[interface_mask]
        print(f"  At interface (0.01 < f < 0.99):")
        print(f"    max |∇·F_marangoni| = {np.max(np.abs(div_m_interface)):.3e} N/m^4")
        print(f"    mean |∇·F_marangoni| = {np.mean(np.abs(div_m_interface)):.3e} N/m^4")
    print()

    # Estimate Darcy force divergence
    print("Estimating Darcy damping force divergence...")
    div_darcy, Fx_d, Fy_d, Fz_d = estimate_darcy_force_divergence(
        ux, uy, uz, liquid_fraction, nx, ny, nz, dx
    )

    print(f"Darcy force divergence:")
    print(f"  max |∇·F_darcy| = {np.max(np.abs(div_darcy)):.3e} N/m^4")
    print(f"  mean |∇·F_darcy| = {np.mean(np.abs(div_darcy)):.3e} N/m^4")

    # Where is Darcy force largest?
    mushy_mask = liquid_fraction < 0.99
    if np.any(mushy_mask):
        div_d_mushy = div_darcy[mushy_mask]
        print(f"  In mushy/solid regions (fl < 0.99):")
        print(f"    max |∇·F_darcy| = {np.max(np.abs(div_d_mushy)):.3e} N/m^4")
        print(f"    mean |∇·F_darcy| = {np.mean(np.abs(div_d_mushy)):.3e} N/m^4")
    print()

    # Summary comparison
    print("=" * 70)
    print("SUMMARY: DIVERGENCE COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Source':<25} {'max |∇·F| [N/m^4]':<20} {'Ratio to max':<15}")
    print("-" * 70)

    max_div_u = np.max(np.abs(div_u))
    max_div_m = np.max(np.abs(div_marangoni))
    max_div_d = np.max(np.abs(div_darcy))

    max_overall = max(max_div_m, max_div_d)

    print(f"{'Velocity (∇·u)':<25} {max_div_u:<20.3e} {'(reference)':<15}")
    print(f"{'Marangoni':<25} {max_div_m:<20.3e} {max_div_m/max_overall:<15.2f}")
    print(f"{'Darcy damping':<25} {max_div_d:<20.3e} {max_div_d/max_overall:<15.2f}")
    print()

    # Determine culprit
    if max_div_d > 2 * max_div_m:
        print("CULPRIT: Darcy damping force has largest divergence!")
        print()
        print("REASON: Darcy force F = -C·(1-fl)²/(fl³+ε)·u is proportional to")
        print("        velocity, and coefficient varies sharply at solid-liquid interface.")
        print()
        print("RECOMMENDATION:")
        print("  1. Reduce darcy_coefficient from 1e7 to 1e6 or 1e5")
        print("  2. Smooth liquid fraction field to reduce ∇fl")
        print("  3. Clamp Darcy force magnitude to prevent extreme values")
    elif max_div_m > 2 * max_div_d:
        print("CULPRIT: Marangoni force has largest divergence!")
        print()
        print("REASON: Marangoni force F ∝ |∇f|·∇T has strong gradients near")
        print("        laser-heated interface region.")
        print()
        print("RECOMMENDATION:")
        print("  1. Verify interface localization (should only apply at 0.01 < f < 0.99)")
        print("  2. Reduce dsigma_dT magnitude")
        print("  3. Smooth temperature gradients")
    else:
        print("MIXED: Both forces contribute significantly to divergence.")
        print()
        print("RECOMMENDATION: Address both Darcy and Marangoni forces.")

    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
