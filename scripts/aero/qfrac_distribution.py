"""Reproduce driver's qfrac generation in Python to quantify upper bound
on QMIN clamp effect.

Driver algorithm (obstacle_geometry.h:makeNacaQFraction):
  For each fluid cell (i,j,k):
    For each q ∈ {1..26}:
      neighbour = (i+ex[q], j+ey[q], k+ez[q])
      if mask[neighbour] is solid:
        scan along link from cell center toward solid in 64 substeps,
        bisect 20× to locate wall crossing.
        qfrac = fraction of link from cell-center to wall.

Reports:
  - Histogram of qfrac values
  - Count of links with qfrac < 0.05 (would be clamped UP)
  - Count of links with qfrac > 0.95 (would be clamped DOWN)
  - Median, mean, std
  - Spatial map of clamped vs unclamped links
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"

# D3Q27 direction vectors
EX = [0, 1,-1, 0, 0, 0, 0,  1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0,
      1,-1, 1,-1, 1,-1, 1,-1]
EY = [0, 0, 0, 1,-1, 0, 0,  1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1,
      1, 1,-1,-1, 1, 1,-1,-1]
EZ = [0, 0, 0, 0, 0, 1,-1,  0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1,
      1, 1, 1, 1,-1,-1,-1,-1]

# Geometry params (matching G2000 run)
D_PER_CHORD = 80
THICK_PCT = 0.12
ALPHA_DEG = 8.0
ALPHA_RAD = np.radians(ALPHA_DEG)
CHORD = 1.0
XLE_OVER_C = 10.0
LY_OVER_C = 20.0

# Driver rotation convention: std aero, α>0 nose-up
COS_A =  np.cos(ALPHA_RAD)
SIN_A = -np.sin(ALPHA_RAD)


def naca_yt(s: float, t: float = THICK_PCT) -> float:
    if s < 0.0 or s > 1.0:
        return -1.0
    return 5.0 * t * (0.2969 * np.sqrt(s)
                      - 0.1260 * s
                      - 0.3516 * s * s
                      + 0.2843 * s ** 3
                      - 0.1015 * s ** 4)


def inside_airfoil(xw: float, yw: float, xle: float, yle: float) -> bool:
    xr =  COS_A * (xw - xle) + SIN_A * (yw - yle)
    yr = -SIN_A * (xw - xle) + COS_A * (yw - yle)
    s = xr / CHORD
    if s < 0.0 or s > 1.0:
        return False
    return abs(yr) <= naca_yt(s) * CHORD


def compute_qfrac_link(xc: float, yc: float, dxL: float, dyL: float,
                       xle: float, yle: float,
                       n_scan: int = 64, n_bisect: int = 20) -> float:
    idx_first = -1
    for n in range(1, n_scan + 1):
        s = n / n_scan
        if inside_airfoil(xc + s * dxL, yc + s * dyL, xle, yle):
            idx_first = n
            break
    if idx_first < 0:
        return 0.5
    s_lo = (idx_first - 1) / n_scan
    s_hi = idx_first / n_scan
    for _ in range(n_bisect):
        sm = 0.5 * (s_lo + s_hi)
        if inside_airfoil(xc + sm * dxL, yc + sm * dyL, xle, yle):
            s_hi = sm
        else:
            s_lo = sm
    qf = 0.5 * (s_lo + s_hi)
    return max(qf, 1e-6)


def read_mask_zmid(path: str):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    return np.array([[int(v) for v in l.split()] for l in lines if l.strip()],
                    dtype=np.uint8)


def main():
    mask_path = os.path.join(ROOT, "output_univ_G2000", "mask_zmid.txt")
    mask = read_mask_zmid(mask_path)
    ny, nx = mask.shape
    print(f"Mask shape: ny={ny} nx={nx}")
    dx = CHORD / D_PER_CHORD
    xle = XLE_OVER_C * CHORD
    yle = ny * dx / 2.0
    print(f"LE world pos: ({xle:.4f}, {yle:.4f})")

    # Iterate all fluid cells with solid neighbour (in z-mid plane)
    # For z-extruded mask we only need 2D mask; z-direction contributes
    # to link length but not inside test.
    qf_values = []
    qf_links_per_cell = []
    le_region_qfs = []  # qfrac values within 5 cells of LE
    qfs_position = []  # (i, j, q, qf) for spatial map

    # Find LE row indices to define "LE region"
    solid_idx = np.argwhere(mask > 0)
    i_LE = int(solid_idx[:, 1].min())
    j_LE_candidates = solid_idx[solid_idx[:, 1] == i_LE, 0]
    j_LE = int(j_LE_candidates.mean())
    print(f"LE pixel: i={i_LE} j={j_LE}")

    n_clamped_low = 0
    n_clamped_high = 0
    n_total_links = 0
    n_links_in_LE_band = 0
    n_clamped_low_LE = 0

    # Bounding box around airfoil with 3-cell margin
    i_min = max(0, int(solid_idx[:, 1].min()) - 3)
    i_max = min(nx, int(solid_idx[:, 1].max()) + 4)
    j_min = max(0, int(solid_idx[:, 0].min()) - 3)
    j_max = min(ny, int(solid_idx[:, 0].max()) + 4)
    print(f"Scanning bbox: i∈[{i_min},{i_max}) j∈[{j_min},{j_max})")

    for j in range(j_min, j_max):
        for i in range(i_min, i_max):
            if mask[j, i] != 0:
                continue
            xc = (i + 0.5) * dx
            yc = (j + 0.5) * dx
            cell_qfs = []
            for q in range(1, 27):
                ex, ey, ez = EX[q], EY[q], EZ[q]
                # z-extruded, so z-component doesn't change inside test;
                # we just need to know neighbour is solid (which depends
                # only on x,y for our extruded mask).
                ni, nj = i + ex, j + ey
                if not (0 <= ni < nx and 0 <= nj < ny):
                    continue
                if mask[nj, ni] == 0:
                    continue
                # Solid neighbour. Compute qfrac.
                dxL, dyL = ex * dx, ey * dx
                qf = compute_qfrac_link(xc, yc, dxL, dyL, xle, yle)
                qf_values.append(qf)
                cell_qfs.append(qf)
                n_total_links += 1
                if qf < 0.05:
                    n_clamped_low += 1
                if qf > 0.95:
                    n_clamped_high += 1
                # LE band: within 5 cells of LE pixel
                if abs(i - i_LE) <= 5 and abs(j - j_LE) <= 5:
                    le_region_qfs.append(qf)
                    n_links_in_LE_band += 1
                    if qf < 0.05:
                        n_clamped_low_LE += 1
            qf_links_per_cell.append(len(cell_qfs))

    qf_arr = np.array(qf_values)
    print(f"\n=== Overall qfrac stats ===")
    print(f"Total wall-adjacent links: {n_total_links}")
    print(f"Links per cell distribution: mean={np.mean(qf_links_per_cell):.2f}, "
          f"max={max(qf_links_per_cell)}")
    print(f"qfrac range: [{qf_arr.min():.4f}, {qf_arr.max():.4f}]")
    print(f"qfrac mean: {qf_arr.mean():.4f}, median: {np.median(qf_arr):.4f}")
    print(f"\n=== Clamp impact ===")
    print(f"Links with qf<0.05 (clamped UP to 0.05): {n_clamped_low}/{n_total_links} = "
          f"{100*n_clamped_low/n_total_links:.1f}%")
    print(f"Links with qf>0.95 (clamped DOWN to 0.95): {n_clamped_high}/{n_total_links} = "
          f"{100*n_clamped_high/n_total_links:.1f}%")
    print(f"\n=== LE region (±5 cells of LE pixel) ===")
    print(f"Total LE-band links: {n_links_in_LE_band}")
    print(f"LE-band links with qf<0.05: {n_clamped_low_LE}/{n_links_in_LE_band} = "
          f"{100*n_clamped_low_LE/max(n_links_in_LE_band,1):.1f}%")

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(qf_arr, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0.05, color='red', ls='--', label='QMIN=0.05')
    axes[0].axvline(0.95, color='red', ls='--', label='QMAX=0.95')
    axes[0].set_xlabel('qfrac')
    axes[0].set_ylabel('# links')
    axes[0].set_title(f'All wall-adjacent links (N={n_total_links})\n'
                       f'{100*n_clamped_low/n_total_links:.1f}% clamped low, '
                       f'{100*n_clamped_high/n_total_links:.1f}% clamped high')
    axes[0].legend()
    axes[0].grid(True, ls=':', alpha=0.5)

    if le_region_qfs:
        axes[1].hist(le_region_qfs, bins=30, edgecolor='black',
                     alpha=0.7, color='orange')
        axes[1].axvline(0.05, color='red', ls='--', label='QMIN=0.05')
        axes[1].axvline(0.95, color='red', ls='--')
        axes[1].set_xlabel('qfrac (LE band ±5 cells)')
        axes[1].set_ylabel('# links')
        axes[1].set_title(f'LE region links (N={n_links_in_LE_band})\n'
                          f'{100*n_clamped_low_LE/max(n_links_in_LE_band,1):.1f}% clamped low')
        axes[1].legend()
        axes[1].grid(True, ls=':', alpha=0.5)

    plt.tight_layout()
    out = os.path.join(ROOT, "naca_qfrac_distribution.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
