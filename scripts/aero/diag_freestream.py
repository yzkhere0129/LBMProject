#!/usr/bin/env python3
"""Check if there's any u_y in the upstream freestream far from the airfoil."""
import sys, numpy as np
vtk = sys.argv[1] if len(sys.argv) > 1 else 'output_naca_a8_kurtulus/snap_0032000.vtk'
with open(vtk) as f: lines = f.readlines()
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
v=v.reshape(nz,ny,nx,3)
ux,uy=v[nz//2,...,0], v[nz//2,...,1]
print(f"Domain {nx}x{ny}x{nz}, dx={dx}")
print()
print("Mean u_y at fixed x cross-sections (ALL y in domain):")
for xp in [0.5, 1.0, 2.0, 5.0, 8.0, 9.5, 10.0, 12.0, 15.0, 20.0, 25.0]:
    i = int(xp/dx)
    if 0 <= i < nx:
        col = uy[:, i]
        ux_col = ux[:, i]
        print(f"  x={xp:5.1f} (i={i:4d}): u_y mean={col.mean():+.6f}, "
              f"u_y std={col.std():.6f}, u_x mean={ux_col.mean():+.6f}")

# Check u_y as a function of y at x=8 (well upstream of airfoil at x=10)
i_up = int(8.0/dx)
print()
print(f"\nu_y(y) profile at x=8.0 (i={i_up}, well upstream of airfoil at x=10):")
for jp in [50, 100, 200, 300, 400, 500, 600, 700, 750]:
    if 0 <= jp < ny:
        print(f"  j={jp:4d} (y={jp*dx:.2f}): u_x={ux[jp, i_up]:+.6f}, u_y={uy[jp, i_up]:+.6f}")
