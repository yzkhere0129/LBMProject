#!/usr/bin/env python3
"""Measure wake centerline (= y-coordinate of velocity-defect minimum) downstream.

For Cl > 0: wake center drifts DOWN (y < y_LE)
For Cl < 0: wake center drifts UP   (y > y_LE)
"""
import sys, glob, numpy as np

vtks = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'output_naca_a8_kurtulus/snap_*.vtk'))

def load_vtk(path):
    with open(path) as f: lines = f.readlines()
    nx=ny=nz=0; dx=0; ds=-1
    for i, ln in enumerate(lines):
        p = ln.split()
        if p[:1]==['DIMENSIONS']: nx,ny,nz=int(p[1]),int(p[2]),int(p[3])
        elif p[:1]==['SPACING']: dx=float(p[1])
        elif p[:1]==['VECTORS']: ds=i+1; break
    n=nx*ny*nz
    v=np.empty((n,3),dtype=np.float32)
    for k,ln in enumerate(lines[ds:ds+n]):
        p=ln.split()
        v[k,0]=float(p[0]); v[k,1]=float(p[1]); v[k,2]=float(p[2])
    return v.reshape(nz,ny,nx,3), dx, nx, ny, nz

# Time-averaged u_y in wake at multiple x cross-sections
print("Measuring wake centerline...")
# Reference: y_LE = 10.0
y_LE = 10.0
# Downstream samples at x_probe = 12, 13, 14, 15, 16 (chord lengths behind airfoil)
x_probes = [12.0, 13.0, 14.0, 15.0]

mean_uy = {x: [] for x in x_probes}
wake_y  = {x: [] for x in x_probes}
mean_uy_band = {x: [] for x in x_probes}

for vtk in vtks:
    v, dx, nx, ny, nz = load_vtk(vtk)
    kz = nz//2
    ux = v[kz,...,0]; uy = v[kz,...,1]
    speed = np.hypot(ux, uy)
    for xp in x_probes:
        i = int(xp/dx)
        # Wake region near airfoil center: y in [9.0, 11.0]
        j_lo, j_hi = int(9.0/dx), int(11.0/dx)
        column = speed[j_lo:j_hi, i]
        # Find min speed (= wake center)
        j_min = j_lo + np.argmin(column)
        wake_y[xp].append((j_min + 0.5)*dx)
        # Mean u_y in band
        mean_uy_band[xp].append(uy[j_lo:j_hi, i].mean())

for xp in x_probes:
    wy = np.asarray(wake_y[xp])
    uyb = np.asarray(mean_uy_band[xp])
    print(f"x={xp:.1f}: wake_y_center mean={wy.mean():.3f} (relative to LE: {wy.mean()-y_LE:+.3f}),"
          f" mean u_y in [9,11]={uyb.mean():+.5f}")

print()
print("Verdict:")
print("  + wake center BELOW y_LE (Δy < 0) => wake bends down => Cl > 0 (positive lift)")
print("  + wake center ABOVE y_LE (Δy > 0) => wake bends up   => Cl < 0 (negative lift)")
