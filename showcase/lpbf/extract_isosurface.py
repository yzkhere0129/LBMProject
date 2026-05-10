#!/usr/bin/env python3
"""Extract VOF=0.5 isosurface + T at vertices from a single VTK keyframe.

Output: a self-contained .npz so the 3D figure can be re-rendered without
the original 600 MB VTK.
"""
import sys
import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes


def main(vtk_path, step, npz_out):
    m = pv.read(vtk_path)
    nx, ny, nz = m.dimensions
    dx, dy, dz = m.spacing
    ox, oy, oz = m.origin
    f = np.asarray(m.point_data["fill_level"]).reshape((nx, ny, nz), order="F")
    T = np.asarray(m.point_data["temperature"]).reshape((nx, ny, nz), order="F")

    verts, _, _, _ = marching_cubes(f, level=0.5, spacing=(dx, dy, dz))
    verts += np.array([ox, oy, oz])
    verts *= 1e6  # → μm

    # Laser-relative coords: laser axis at the moving spot for this step
    laser_x_um = 500.0 + 0.8 * step * 80e-9 * 1e6  # 80ns·dt, 0.8m/s scan
    z_sub_um = 80 * 2.0
    y_center_um = (ny - 1) * 2.0 / 2.0
    verts_rel = verts.copy()
    verts_rel[:, 0] -= laser_x_um
    verts_rel[:, 1] -= y_center_um
    verts_rel[:, 2] -= z_sub_um

    # Trilinear sample T at each vertex (for colouring)
    abs_x = (verts_rel[:, 0] + laser_x_um) * 1e-6
    abs_y = (verts_rel[:, 1] + y_center_um) * 1e-6
    abs_z = (verts_rel[:, 2] + z_sub_um) * 1e-6
    i = np.clip(((abs_x - ox) / dx).astype(int), 0, nx - 1)
    j = np.clip(((abs_y - oy) / dy).astype(int), 0, ny - 1)
    k = np.clip(((abs_z - oz) / dz).astype(int), 0, nz - 1)
    T_v = T[i, j, k]

    # Crop to viewing window to keep file size bounded
    in_view = ((np.abs(verts_rel[:, 0]) < 250) &
               (np.abs(verts_rel[:, 1]) < 70) &
               (verts_rel[:, 2] > -150))
    verts_rel = verts_rel[in_view]
    T_v = T_v[in_view]

    np.savez_compressed(
        npz_out,
        verts_um=verts_rel.astype(np.float32),
        T_K=T_v.astype(np.float32),
        step=step,
        dt_s=80e-9,
        scan_v_ms=0.8,
        dx_um=2.0,
    )
    print(f"{vtk_path} step={step} → {npz_out}")
    print(f"  {len(verts_rel):,} surface vertices, T range "
          f"{T_v.min():.0f}–{T_v.max():.0f} K")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
