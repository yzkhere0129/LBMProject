#!/usr/bin/env python3
"""
Diagnostic: read a VTK snapshot, print actual min/max/mean of velocity
fields at specific locations to verify inlet condition + flow profile.
"""
import sys
import re
import numpy as np


def read_vtk_image(path):
    with open(path, "rb") as f:
        data = f.read()
    text = data.decode("latin-1", errors="ignore")
    m = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)", text)
    nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"VECTORS\s+\S+\s+(\w+)\s*\n", text)
    body = text[m.end():]
    n = nx * ny * nz
    vals = np.fromstring(body, sep=" ", count=n * 3, dtype=np.float64)
    vels = vals.reshape((nz, ny, nx, 3))  # k, j, i, component
    return nx, ny, nz, vels


def main():
    if len(sys.argv) < 2:
        print("Usage: diag_velocity.py <snap.vtk>")
        sys.exit(1)
    path = sys.argv[1]
    nx, ny, nz, vels = read_vtk_image(path)
    u = vels[..., 0]
    v = vels[..., 1]
    w = vels[..., 2]
    speed = np.sqrt(u**2 + v**2 + w**2)

    print(f"=== Velocity diagnostics: {path}  (nx,ny,nz)={nx,ny,nz} ===\n")
    print(f"u_x:  min={u.min():.5f}  max={u.max():.5f}  mean={u.mean():.5f}")
    print(f"u_y:  min={v.min():.5f}  max={v.max():.5f}  mean={v.mean():.5f}")
    print(f"u_z:  min={w.min():.5f}  max={w.max():.5f}  mean={w.mean():.5f}")
    print(f"|u|:  min={speed.min():.5f}  max={speed.max():.5f}  mean={speed.mean():.5f}")

    print("\n--- Inlet face (i=0) profile ---")
    # u_x along centerline of z (k=nz/2), all j
    k_mid = nz // 2
    print("j     y_phys   u_x@(i=0)    u_x@(i=10)   u_x@(i=nx/2)   u_x@(i=nx-2)")
    j_samples = [0, ny // 4, ny // 2, 3 * ny // 4, ny - 1]
    for j in j_samples:
        y = j  # in cell index units (LU)
        line = f"{j:3d}   {y:6d}   "
        for ix in [0, 10, nx // 2, nx - 2]:
            line += f"{u[k_mid, j, ix]:.5f}      "
        print(line)

    # Compute density and y-integrated mass flux at several x cross-sections
    print("\n--- Density and mass flux along x ---")
    print("(From velocity field alone we can't get rho, but assume rho≈1 to estimate flux)")
    print("x  flux=Σ_j u_x(j, k_mid)  (LU, 'mass' flux per unit z)")
    for ix in [0, 1, 5, 10, nx // 4, nx // 2, 3 * nx // 4, nx - 2, nx - 1]:
        flux = float(u[k_mid, :, ix].sum())
        print(f"  i={ix:4d}  flux={flux:.4f}")

    print("\n--- Predicted parabolic inlet for comparison ---")
    print("Parabolic u_x(y) = 4·u_max·y(H-y)/H² with u_max=0.05, H=(ny-1)*dx_LU")
    print("(in LU units, dx_LU=1)")
    H = ny - 1
    for j in j_samples:
        y_lu = j
        u_pred = 4.0 * 0.05 * y_lu * (H - y_lu) / (H * H)
        print(f"  j={j}: u_x_predicted = {u_pred:.5f}")


if __name__ == "__main__":
    main()
