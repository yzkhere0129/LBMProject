#!/usr/bin/env python3
"""Phase 2 — strict global mass-conservation diagnostic.

F3D enables `if_vol_corr=1` (1 μs corrective period). LBM has
`enable_vof_mass_correction=false`. If LBM is leaking metal mass
between t0 and t1, the trailing-zone groove can never refill no
matter how strong capillary is.

Computes total metal mass = Σ ρ_eff(T) · fill · dx³  over the entire
domain. ρ_eff(T) uses material density-vs-T table from F3D prepin
(Mills 316L) for accuracy: 7950 @ 298 K, 7411 @ 1373 K, 7269 @ 1658,
7236 @ 1723 (solidus end), 6881 @ 1723.15 (liquidus side, density jump),
6765 @ 1873.

Usage:
    python check_mass_conservation.py <vtk_t0> <vtk_t1>
"""
import sys
import numpy as np
import pyvista as pv

# Mills 316L density table (T_K, rho_kg_m3) from F3D prepin.
RHO_TABLE = np.array([
    [298.15, 7950],   [373.15, 7921], [473.15, 7880], [573.15, 7833],
    [673.15, 7785],   [773.15, 7735], [873.15, 7681], [973.15, 7628],
    [1073.15, 7575],  [1173.15, 7520], [1273.15, 7462], [1373.15, 7411],
    [1473.15, 7361],  [1573.15, 7311], [1658.15, 7269], [1723.15, 7236],
    # Density jump at melting (latent volume change)
    [1723.151, 6881], [1773.15, 6842], [1873.15, 6765],
])

def rho_of_T(T_array):
    return np.interp(T_array, RHO_TABLE[:, 0], RHO_TABLE[:, 1])

def total_metal_mass(path):
    m = pv.read(path)
    nx, ny, nz = m.dimensions
    dx, dy, dz = m.spacing
    cell_vol = dx * dy * dz  # m³

    fill = np.asarray(m.point_data['fill_level']).reshape((nx, ny, nz), order='F')
    T = np.asarray(m.point_data['temperature']).reshape((nx, ny, nz), order='F')

    rho = rho_of_T(T)
    # Metal mass per cell = ρ(T) · fill · V_cell
    mass_per_cell = rho * fill * cell_vol
    total_mass = float(mass_per_cell.sum())

    fill_sum = float(fill.sum())
    n_full = int((fill > 0.95).sum())
    n_partial = int(((fill > 0.05) & (fill < 0.95)).sum())
    return total_mass, fill_sum, n_full, n_partial, (nx, ny, nz)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)

    p0, p1 = sys.argv[1], sys.argv[2]
    M0, F0, n_full0, n_part0, dims = total_metal_mass(p0)
    M1, F1, n_full1, n_part1, _    = total_metal_mass(p1)

    dM = M1 - M0
    rel = dM / M0 * 100.0

    print(f"Domain: {dims[0]}×{dims[1]}×{dims[2]}")
    print(f"")
    print(f"{'Frame':>30} {'Mass [g]':>12} {'Σfill':>10} {'n_full':>10} {'n_partial':>10}")
    print(f"{p0:>30} {M0*1e3:>12.6f} {F0:>10.0f} {n_full0:>10} {n_part0:>10}")
    print(f"{p1:>30} {M1*1e3:>12.6f} {F1:>10.0f} {n_full1:>10} {n_part1:>10}")
    print(f"")
    print(f"Δ mass         : {dM*1e3:+.6f} g  ({rel:+.3f} %)")
    print(f"Δ Σfill        : {F1-F0:+.1f} cells")
    print(f"")

    # Verdict
    if abs(rel) > 3.0:
        print(f"VERDICT: SIGNIFICANT MASS LOSS ({rel:+.3f}%)")
        print(f"  Likely numerical leakage in VOF advection or evap mass-loss kernel.")
        print(f"  Recommend: enable_vof_mass_correction=true and rerun.")
    elif abs(rel) > 1.0:
        print(f"VERDICT: MODERATE MASS DRIFT ({rel:+.3f}%)")
        print(f"  Within tolerance for forward integration; not the dominant cause.")
    else:
        print(f"VERDICT: GOOD CONSERVATION ({rel:+.3f}%)")
        print(f"  Mass loss ruled out as cause of trailing groove.")
