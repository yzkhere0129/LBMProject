#!/usr/bin/env python3
"""Flow3D vs LBM 3D melt-pool comparison harness.

Compares LBM volumetric VTK (ImageData with fill_level / temperature / velocity)
against a Flow3D free-surface PolyData snapshot (triangle mesh + Temperature).

Outputs a JSON + CSV row with quantitative metrics:

  * symmetric_hausdorff_um       max-min surface distance (both directions)
  * mean_chamfer_um              average min-distance, both directions
  * surface_T_rms_K              RMS of (T_LBM_at_F3D_pts - T_F3D)
  * surface_T_p95_K              95-percentile |ΔT|
  * pool_L_F3D_um, pool_W..., pool_D_um   from Flow3D T>=T_liq bbox
  * pool_L_LBM_um, ...                    from LBM T>=T_liq bbox
  * pool_volume_F3D_um3 / pool_volume_LBM_um3 (proxy: bbox volume)
  * Tmax_F3D_K, Tmax_LBM_K
  * vmax_LBM_m_s                 LBM only (Flow3D PolyData has no velocity)

Coordinate transform:
  LBM_xyz = F3D_xyz + offset_xyz
  (default offset assumes LBM laser starts at LBM-coord (100,75,80)μm and
  Flow3D laser starts at F3D-coord (0,0,0). Override with --offset.)

USAGE
  python compare.py <flow3d.vtk> <lbm.vtk> --out comparison.json \
      [--offset 0.0005,0.0002,0.0001975]  # meters
"""
import argparse, json, os, sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

T_LIQUIDUS_316L = 1697.15  # K (Flow3D prepin tl1)


def load_flow3d(path):
    m = pv.read(path)
    if not isinstance(m, pv.PolyData):
        raise RuntimeError(f"Expected Flow3D PolyData, got {type(m).__name__}")
    pts = np.asarray(m.points, dtype=np.float64)  # (N, 3) meters
    T = np.asarray(m.point_data['Temperature'], dtype=np.float64)  # (N,)
    return pts, T, m


def load_lbm(path):
    m = pv.read(path)
    dims = np.array(m.dimensions, dtype=np.int64)        # (nx, ny, nz)
    spacing = np.array(m.spacing, dtype=np.float64)      # (dx, dy, dz)
    origin = np.array(m.origin, dtype=np.float64)
    f = np.asarray(m.point_data['fill_level'], dtype=np.float64).reshape(dims, order='F')
    T = np.asarray(m.point_data['temperature'], dtype=np.float64).reshape(dims, order='F')
    if 'velocity' in m.point_data.keys():
        u = np.asarray(m.point_data['velocity'], dtype=np.float64).reshape((*dims, 3), order='F')
    else:
        u = None
    return dict(dims=dims, spacing=spacing, origin=origin, f=f, T=T, u=u)


def lbm_surface_marching_cubes(lbm, level=0.5):
    """Return (verts_xyz_m, faces) of fill_level=0.5 isosurface in LBM coords (meters)."""
    dx, dy, dz = lbm['spacing']
    ox, oy, oz = lbm['origin']
    f = lbm['f']
    if (f.min() > level) or (f.max() < level):
        return None, None
    verts, faces, _, _ = marching_cubes(f, level=level, spacing=(dx, dy, dz))
    verts = verts + np.array([ox, oy, oz])  # (Nv, 3) meters
    return verts.astype(np.float64), faces.astype(np.int64)


def trilinear_sample(field, dims, spacing, origin, query_xyz):
    """Sample 3D scalar `field` (shape dims, F-order) at query_xyz (in meters).
    Out-of-bounds returns NaN."""
    nx, ny, nz = dims
    dx, dy, dz = spacing
    ox, oy, oz = origin
    rx = (query_xyz[:, 0] - ox) / dx
    ry = (query_xyz[:, 1] - oy) / dy
    rz = (query_xyz[:, 2] - oz) / dz
    ix0 = np.floor(rx).astype(np.int64)
    iy0 = np.floor(ry).astype(np.int64)
    iz0 = np.floor(rz).astype(np.int64)
    fx = rx - ix0
    fy = ry - iy0
    fz = rz - iz0
    out = np.full(query_xyz.shape[0], np.nan, dtype=np.float64)
    valid = (ix0 >= 0) & (ix0 < nx - 1) & (iy0 >= 0) & (iy0 < ny - 1) & (iz0 >= 0) & (iz0 < nz - 1)
    if not valid.any():
        return out
    ix0v, iy0v, iz0v = ix0[valid], iy0[valid], iz0[valid]
    fxv, fyv, fzv = fx[valid], fy[valid], fz[valid]
    # 8 corners (F-order indexing: field[i, j, k])
    c000 = field[ix0v,   iy0v,   iz0v]
    c100 = field[ix0v+1, iy0v,   iz0v]
    c010 = field[ix0v,   iy0v+1, iz0v]
    c110 = field[ix0v+1, iy0v+1, iz0v]
    c001 = field[ix0v,   iy0v,   iz0v+1]
    c101 = field[ix0v+1, iy0v,   iz0v+1]
    c011 = field[ix0v,   iy0v+1, iz0v+1]
    c111 = field[ix0v+1, iy0v+1, iz0v+1]
    c00 = c000 * (1 - fxv) + c100 * fxv
    c10 = c010 * (1 - fxv) + c110 * fxv
    c01 = c001 * (1 - fxv) + c101 * fxv
    c11 = c011 * (1 - fxv) + c111 * fxv
    c0 = c00 * (1 - fyv) + c10 * fyv
    c1 = c01 * (1 - fyv) + c11 * fyv
    out[valid] = c0 * (1 - fzv) + c1 * fzv
    return out


def melt_pool_bbox(points, T, T_liq):
    mask = T >= T_liq
    if not mask.any():
        return dict(L=0.0, W=0.0, D=0.0, Tmax=float(T.max()), n_pts=0)
    sub = points[mask]
    L = (sub[:, 0].max() - sub[:, 0].min()) * 1e6  # μm
    W = (sub[:, 1].max() - sub[:, 1].min()) * 1e6
    D = (sub[:, 2].max() - sub[:, 2].min()) * 1e6
    return dict(L=L, W=W, D=D, Tmax=float(T.max()), n_pts=int(mask.sum()))


def melt_pool_bbox_lbm(lbm, T_liq):
    T = lbm['T']
    f = lbm['f']
    mask = (T >= T_liq) & (f > 0.5)  # liquid + has metal
    if not mask.any():
        return dict(L=0.0, W=0.0, D=0.0, Tmax=float(T.max()), n_cells=0)
    nx, ny, nz = lbm['dims']
    dx, dy, dz = lbm['spacing']
    ox, oy, oz = lbm['origin']
    ii, jj, kk = np.where(mask)
    L = (ii.max() - ii.min() + 1) * dx * 1e6
    W = (jj.max() - jj.min() + 1) * dy * 1e6
    D = (kk.max() - kk.min() + 1) * dz * 1e6
    return dict(L=L, W=W, D=D, Tmax=float(T.max()), n_cells=int(mask.sum()))


def hausdorff_chamfer(A, B):
    """Symmetric Hausdorff + symmetric mean chamfer (meters)."""
    if A is None or len(A) == 0 or B is None or len(B) == 0:
        return dict(haus=np.nan, chamfer=np.nan, dAB_max=np.nan, dBA_max=np.nan)
    treeB = cKDTree(B)
    dAB, _ = treeB.query(A, k=1)  # for each a in A, dist to nearest b in B
    treeA = cKDTree(A)
    dBA, _ = treeA.query(B, k=1)
    haus = max(dAB.max(), dBA.max())
    cham = 0.5 * (dAB.mean() + dBA.mean())
    return dict(haus=haus, chamfer=cham, dAB_max=float(dAB.max()), dBA_max=float(dBA.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('flow3d_vtk')
    ap.add_argument('lbm_vtk')
    ap.add_argument('--offset', default='0,0,0',
                    help='Offset to add to Flow3D coords to map them into LBM coord frame, meters, "x,y,z"')
    ap.add_argument('--t_liq', type=float, default=T_LIQUIDUS_316L)
    ap.add_argument('--out', default='-', help='Output JSON path, or "-" for stdout')
    ap.add_argument('--csv-append', default=None,
                    help='If given, append a row to this CSV')
    ap.add_argument('--label', default='', help='Run label for CSV row')
    args = ap.parse_args()

    offset = np.array([float(x) for x in args.offset.split(',')])

    # Load
    f3d_pts, f3d_T, f3d_mesh = load_flow3d(args.flow3d_vtk)
    lbm = load_lbm(args.lbm_vtk)

    # Apply offset to Flow3D coords (now both in LBM frame)
    f3d_pts_aligned = f3d_pts + offset

    # Crop Flow3D points to LBM domain bbox (the F3D run can have a wider scan
    # span than the LBM domain; comparing far points yields fake constant
    # Hausdorff dominated by domain-extent mismatch).
    nx, ny, nz = lbm['dims']
    dx_lbm, dy_lbm, dz_lbm = lbm['spacing']
    ox, oy, oz = lbm['origin']
    lbm_bbox_min = np.array([ox, oy, oz])
    lbm_bbox_max = lbm_bbox_min + np.array([(nx-1)*dx_lbm, (ny-1)*dy_lbm, (nz-1)*dz_lbm])
    in_lbm = ((f3d_pts_aligned >= lbm_bbox_min) & (f3d_pts_aligned <= lbm_bbox_max)).all(axis=1)
    f3d_pts_in = f3d_pts_aligned[in_lbm]
    f3d_T_in = f3d_T[in_lbm]
    n_f3d_in_lbm = int(in_lbm.sum())

    # LBM iso-surface
    verts, faces = lbm_surface_marching_cubes(lbm, level=0.5)

    # Surface metrics — use only F3D points inside the LBM domain
    if verts is not None and len(f3d_pts_in) > 0:
        hc = hausdorff_chamfer(f3d_pts_in, verts)
    else:
        hc = dict(haus=np.nan, chamfer=np.nan, dAB_max=np.nan, dBA_max=np.nan)

    # Sample LBM T at Flow3D vertices (for surface temperature comparison)
    # Use only the F3D points inside the LBM domain (cropped above).
    T_lbm_at_f3d = trilinear_sample(lbm['T'], lbm['dims'], lbm['spacing'], lbm['origin'],
                                     f3d_pts_in)
    valid = np.isfinite(T_lbm_at_f3d)
    if valid.any():
        dT = T_lbm_at_f3d[valid] - f3d_T_in[valid]
        T_rms = float(np.sqrt(np.mean(dT ** 2)))
        T_p95 = float(np.percentile(np.abs(dT), 95))
        T_max_err = float(np.max(np.abs(dT)))
        n_valid = int(valid.sum())
        n_total = int(valid.size)
    else:
        T_rms, T_p95, T_max_err, n_valid, n_total = (np.nan, np.nan, np.nan, 0, int(valid.size))

    # Pool bbox — use only F3D points within LBM domain (avoid spurious far track)
    f3d_bbox = melt_pool_bbox(f3d_pts_in, f3d_T_in, args.t_liq)
    lbm_bbox = melt_pool_bbox_lbm(lbm, args.t_liq)

    # LBM v_max (Flow3D doesn't have velocity)
    if lbm['u'] is not None:
        u = lbm['u']
        umag = np.sqrt(np.sum(u ** 2, axis=-1))
        vmax_lbm = float(umag.max())
        # convert to m/s if it's in lattice units (u_phys = u_lat * dx/dt) — assume already physical
        # actually FluidLBM writes physical-unit velocity to VTK
    else:
        vmax_lbm = float('nan')

    metrics = dict(
        flow3d=os.path.basename(args.flow3d_vtk),
        lbm=os.path.basename(args.lbm_vtk),
        offset_m=offset.tolist(),
        t_liq_K=args.t_liq,
        n_f3d_pts=len(f3d_pts),
        n_f3d_in_lbm=n_f3d_in_lbm,
        n_lbm_iso_verts=(len(verts) if verts is not None else 0),
        haus_um=hc['haus'] * 1e6,
        chamfer_um=hc['chamfer'] * 1e6,
        dAB_max_um=hc['dAB_max'] * 1e6 if not np.isnan(hc['dAB_max']) else np.nan,
        dBA_max_um=hc['dBA_max'] * 1e6 if not np.isnan(hc['dBA_max']) else np.nan,
        surface_T_rms_K=T_rms,
        surface_T_p95_K=T_p95,
        surface_T_maxerr_K=T_max_err,
        surface_T_n_valid=n_valid,
        surface_T_n_total=n_total,
        pool_L_F3D_um=f3d_bbox['L'],
        pool_W_F3D_um=f3d_bbox['W'],
        pool_D_F3D_um=f3d_bbox['D'],
        Tmax_F3D_K=f3d_bbox['Tmax'],
        n_pts_above_Tliq_F3D=f3d_bbox['n_pts'],
        pool_L_LBM_um=lbm_bbox['L'],
        pool_W_LBM_um=lbm_bbox['W'],
        pool_D_LBM_um=lbm_bbox['D'],
        Tmax_LBM_K=lbm_bbox['Tmax'],
        n_cells_above_Tliq_LBM=lbm_bbox['n_cells'],
        vmax_LBM_m_s=vmax_lbm,
    )

    out_str = json.dumps(metrics, indent=2)
    if args.out == '-':
        print(out_str)
    else:
        with open(args.out, 'w') as fh:
            fh.write(out_str + '\n')

    if args.csv_append:
        row_keys = list(metrics.keys())
        # Add label as first column
        write_header = not os.path.exists(args.csv_append) or os.path.getsize(args.csv_append) == 0
        with open(args.csv_append, 'a') as fh:
            if write_header:
                fh.write('label,' + ','.join(row_keys) + '\n')
            row_vals = [args.label]
            for k in row_keys:
                v = metrics[k]
                if isinstance(v, list):
                    row_vals.append('"' + ';'.join(f'{x:g}' for x in v) + '"')
                elif isinstance(v, float):
                    row_vals.append(f'{v:.6g}')
                else:
                    row_vals.append(str(v))
            fh.write(','.join(row_vals) + '\n')


if __name__ == '__main__':
    main()
