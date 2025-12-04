#!/usr/bin/env python3
"""Quick divergence check - just compute ∇·u"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import sys

def read_vtk(filename):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def compute_divergence(ux, uy, uz, nx, ny, nz, dx):
    """Compute ∇·u using central differences"""
    ux_3d = ux.reshape((nz, ny, nx))
    uy_3d = uy.reshape((nz, ny, nx))
    uz_3d = uz.reshape((nz, ny, nx))

    # Central differences (periodic BC)
    dux_dx = (np.roll(ux_3d, -1, axis=2) - np.roll(ux_3d, 1, axis=2)) / (2 * dx)
    duy_dy = (np.roll(uy_3d, -1, axis=1) - np.roll(uy_3d, 1, axis=1)) / (2 * dx)
    duz_dz = (np.roll(uz_3d, -1, axis=0) - np.roll(uz_3d, 1, axis=0)) / (2 * dx)

    div = dux_dx + duy_dy + duz_dz
    return div.flatten()

if len(sys.argv) < 2:
    print("Usage: python3 quick_divergence_check.py <vtk_file>")
    sys.exit(1)

filename = sys.argv[1]
dx = 2.0e-6

print(f"Reading: {filename}")
data = read_vtk(filename)

dims = data.GetDimensions()
nx, ny, nz = dims[0], dims[1], dims[2]

velocity = vtk_to_numpy(data.GetPointData().GetArray('Velocity'))
ux, uy, uz = velocity[:, 0], velocity[:, 1], velocity[:, 2]

div = compute_divergence(ux, uy, uz, nx, ny, nz, dx)

print(f"\nVelocity Divergence:")
print(f"  max |∇·u| = {np.max(np.abs(div)):.3e} s^-1")
print(f"  mean |∇·u| = {np.mean(np.abs(div)):.3e} s^-1")
print(f"  RMS ∇·u = {np.sqrt(np.mean(div**2)):.3e} s^-1")

# Count cells with large divergence
threshold = 1000.0  # s^-1
large_div = np.abs(div) > threshold
print(f"\nCells with |∇·u| > {threshold:.0f} s^-1: {np.sum(large_div)} ({100*np.sum(large_div)/len(div):.2f}%)")
