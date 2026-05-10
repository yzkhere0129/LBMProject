"""
extract_sections.py — Extract longitudinal/transverse/top-down cross-sections
from a 316L LPBF VTK keyframe and pickle them for downstream rendering.

Usage:
    python3 extract_sections.py <vtk_path> <output_npz>
"""
import sys
import numpy as np
import pyvista as pv


def main(vtk_path: str, npz_out: str) -> None:
    g = pv.read(vtk_path)
    nx, ny, nz = g.dimensions
    bounds = g.bounds
    dx = (bounds.x_max - bounds.x_min) / (nx - 1)

    def field(name: str) -> np.ndarray:
        arr = g.point_data[name]
        if arr.ndim == 1:
            return arr.reshape((nz, ny, nx))
        return arr.reshape((nz, ny, nx, -1))

    T = field("temperature")
    fl = field("fill_level")
    velocity = field("velocity")
    speed = np.linalg.norm(velocity, axis=-1)

    j_mid = ny // 2
    long_T = T[:, j_mid, :]
    long_fl = fl[:, j_mid, :]
    long_speed = speed[:, j_mid, :]

    laser_x_phys = 0.4e-3 + 0.8 * 240e-6
    i_pool = int(round(laser_x_phys / dx)) - 30
    i_pool = max(0, min(nx - 1, i_pool))
    trans_T = T[:, :, i_pool]
    trans_fl = fl[:, :, i_pool]
    trans_speed = speed[:, :, i_pool]

    fl_x = fl.mean(axis=(0, 1))
    interior_mask = fl_x > 0.5
    if interior_mask.any():
        substrate_top_idx_per_col = []
        for i in range(nx):
            col = fl[:, j_mid, i]
            top = np.argmax(col[::-1] > 0.5)
            if (col[::-1] > 0.5).any():
                substrate_top_idx_per_col.append(nz - 1 - top)
        if substrate_top_idx_per_col:
            substrate_top_k = int(np.median(substrate_top_idx_per_col))
        else:
            substrate_top_k = nz // 2
    else:
        substrate_top_k = nz // 2

    k_top = max(0, min(nz - 1, substrate_top_k - 2))
    top_T = T[k_top, :, :]
    top_speed = speed[k_top, :, :]
    top_fl = fl[k_top, :, :]

    np.savez_compressed(
        npz_out,
        bounds=np.array(bounds),
        dx=dx,
        nx=nx, ny=ny, nz=nz,
        j_mid=j_mid,
        i_pool=i_pool,
        substrate_top_k=substrate_top_k,
        long_T=long_T.astype(np.float32),
        long_fl=long_fl.astype(np.float32),
        long_speed=long_speed.astype(np.float32),
        trans_T=trans_T.astype(np.float32),
        trans_fl=trans_fl.astype(np.float32),
        trans_speed=trans_speed.astype(np.float32),
        top_T=top_T.astype(np.float32),
        top_speed=top_speed.astype(np.float32),
        top_fl=top_fl.astype(np.float32),
    )
    print(f"Saved {npz_out}")
    print(f"  long shape: {long_T.shape}  (z, x)")
    print(f"  trans shape: {trans_T.shape}  (z, y)")
    print(f"  top shape: {top_T.shape}  (y, x)  at k={k_top}")
    print(f"  T range: [{T.min():.0f}, {T.max():.0f}] K")
    print(f"  speed max: {speed.max():.2f} m/s")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
