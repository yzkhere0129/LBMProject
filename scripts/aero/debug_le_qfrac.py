"""D1 debug: replicate the driver's QBB q-fraction calculation in Python,
focusing on the NACA0012 LE region.

Driver algorithm (from aero_naca0012_cumulant.cu):
  For each fluid cell, for each q in [1, 26]:
    neighbour = (i+ex[q], j+ey[q], k+ez[q])
    if neighbour is solid:
       qfrac[link] = ratio of distance from cell-center to wall over
                     distance to neighbour center (= |c_q| in lattice units).

We approximate the wall position via bisection along the lattice link
between cell center and solid neighbour, using the NACA0012 thickness
formula in airfoil frame, then rotated by α.

Reports for cells whose distance to LE < 5 cells:
  - number of fluid cells
  - number of QBB links per cell
  - histogram of q-fraction values
  - which cells are stair-only (no q∈(0,1))
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"

# D3Q27 direction vectors (from driver / lattice header)
EX = [0, 1,-1, 0, 0, 0, 0,
      1,-1, 1,-1, 1,-1, 1,-1,
      0, 0, 0, 0,
      1,-1, 1,-1, 1,-1, 1,-1]
EY = [0, 0, 0, 1,-1, 0, 0,
      1, 1,-1,-1, 0, 0, 0, 0,
      1,-1, 1,-1,
      1, 1,-1,-1, 1, 1,-1,-1]
EZ = [0, 0, 0, 0, 0, 1,-1,
      0, 0, 0, 0,
      1, 1,-1,-1,
      1,-1, 1,-1,
      1, 1, 1, 1,-1,-1,-1,-1]


def read_mask_zmid(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    return np.array([[int(v) for v in l.split()] for l in lines if l.strip()],
                    dtype=np.uint8)


def main():
    # use G2000_vtk (same mask geometry as G500_vtk; both NACA0012 D/dx=80 α=8)
    mask_path = os.path.join(ROOT, "output_univ_G2000_vtk", "mask_zmid.txt")
    mask = read_mask_zmid(mask_path)
    ny, nx = mask.shape
    print(f"Mask shape: ny={ny} nx={nx}")
    print(f"Solid cells in z-mid: {mask.sum()}")

    # NACA0012 LE position from driver: x_LE = 10·chord, y_LE = (ny·dx)/2
    # D/dx=80 chord=1m → dx=1/80=0.0125 (in m, but driver uses dimensionless 'm'
    # where 1 chord = 1 'm'). So we work in cell units.
    # LE cell index (i_LE, j_LE) = (10·80, ny/2) = (800, ny/2)
    # But the geometry was rotated by α=8° around mid-chord (driver convention).
    # Find LE = leftmost solid cell.
    solid_idx = np.argwhere(mask > 0)  # (j, i)
    j_arr = solid_idx[:, 0]
    i_arr = solid_idx[:, 1]
    i_LE = int(i_arr.min())   # leftmost solid column
    i_TE = int(i_arr.max())
    j_LE_candidates = solid_idx[i_arr == i_LE, 0]
    j_LE = int(j_LE_candidates.mean())
    print(f"LE pixel: i={i_LE} j={j_LE}")
    print(f"TE pixel: i={i_TE}")
    print(f"Effective chord in cells: {i_TE - i_LE} (expected 80)")

    # Focus on LE region: window of ±10 cells around (i_LE, j_LE), z=0 only
    half = 10
    i0, i1 = i_LE - half, i_LE + half + 1
    j0, j1 = j_LE - half, j_LE + half + 1
    sub_mask = mask[j0:j1, i0:i1]
    print(f"\nLE window {i1-i0}×{j1-j0}:")
    for jj in range(sub_mask.shape[0] - 1, -1, -1):
        row = ''.join('█' if v else '·' for v in sub_mask[jj])
        print(f"  {row}")

    # Compute qfrac for each fluid cell in LE window
    # qfrac formula: for each q ∈ {1..26}, find link from cell (i,j) to (i+ex,j+ey,k+ez)
    # If neighbour is solid, qfrac ∈ (0, 1] is fraction of link inside fluid.
    # Approximation: bisection along link to find wall crossing.
    # But we don't have analytic NACA0012 surface here.
    # Easier: use sub-cell sampling of mask. Subdivide each cell into NxN sub-cells
    # using the mask's binary state — gives ~1/N resolution on qfrac.
    # For pure mask comparison we'd need higher-res mask, which the driver builds
    # internally. So we COUNT how many QBB-active links exist per cell instead,
    # and check that count matches expected QBB behaviour.

    # An "active QBB link" is a fluid-to-solid neighbour link.
    fluid_cells = []
    link_counts = []
    for j_loc in range(sub_mask.shape[0]):
        for i_loc in range(sub_mask.shape[1]):
            if sub_mask[j_loc, i_loc] != 0:
                continue
            i_glob, j_glob = i0 + i_loc, j0 + j_loc
            count = 0
            for q in range(1, 27):
                ex, ey, ez = EX[q], EY[q], EZ[q]
                if ez != 0:    # z-stencil: only z-mid mask available, skip out-of-plane
                    continue
                ni, nj = i_glob + ex, j_glob + ey
                if 0 <= ni < nx and 0 <= nj < ny and mask[nj, ni] != 0:
                    count += 1
            fluid_cells.append((i_glob, j_glob))
            link_counts.append(count)

    link_counts = np.array(link_counts)
    print(f"\nLE-region fluid cells: {len(fluid_cells)}")
    print(f"Cells with ≥1 QBB link: {(link_counts > 0).sum()}")
    print(f"Cells with 0 QBB links: {(link_counts == 0).sum()}")
    print(f"Link count distribution (1..max):")
    for k in range(1, max(8, link_counts.max() + 1) if link_counts.max() > 0 else 8):
        n = (link_counts == k).sum()
        print(f"  {k} links: {n} cells")

    # Check resolution at the LE point itself
    # NACA0012: y(x) = ±0.6·t·(0.2969√x − 0.1260 x − 0.3516 x² + 0.2843 x³ − 0.1015 x⁴)
    # t=0.12, LE radius = 1.1019·t² = 1.1019·0.0144 = 0.01587c ≈ 1.27 cells at D/dx=80
    LE_radius_c = 1.1019 * 0.12 ** 2
    print(f"\nNACA0012 LE radius = {LE_radius_c:.4f} chord = {LE_radius_c*80:.2f} cells (D/dx=80)")

    # Sub-cell sampling estimate of qfrac at LE — count solid sub-cells in 8×8 sub-grid
    # for each cell in window
    N_sub = 16
    # but we don't have analytic geometry here — only mask. So estimate q by
    # interpolating: q[link] ~ (distance from cell-center to nearest solid
    # along link) / 1 cell.

    # Simple proxy: for each fluid cell adjacent to solid, the q value
    # depends on where the wall actually is. With the binary mask we can
    # only say q ≈ 0.5 (cell center to neighbour center, wall midway), so
    # all stair-type. The actual driver computes q via sub-cell ray-march
    # against analytic NACA — different from what we can re-derive here.

    # Plot the LE region with mask + fluid link counts
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(sub_mask, origin='lower', cmap='gray_r', alpha=0.4,
              extent=(i0-0.5, i1-0.5, j0-0.5, j1-0.5))
    for (i_glob, j_glob), c in zip(fluid_cells, link_counts):
        if c > 0:
            color = 'red' if c >= 3 else ('orange' if c == 2 else 'yellow')
            ax.scatter(i_glob, j_glob, s=120, c=color, edgecolors='k', linewidth=0.5)
            ax.text(i_glob, j_glob, str(c), ha='center', va='center',
                    fontsize=7, color='black')
    ax.set_title(f'NACA0012 LE region (D/dx=80, α=8°)\n'
                 f'Coloured cells = fluid cells with QBB links to solid\n'
                 f'Number = link count (red≥3, orange=2, yellow=1)\n'
                 f'LE radius = {LE_radius_c*80:.2f} cells')
    ax.set_xlabel('i (cell index)')
    ax.set_ylabel('j (cell index)')
    ax.scatter(i_LE, j_LE, marker='+', s=400, color='cyan',
               label=f'LE detected at ({i_LE},{j_LE})')
    ax.legend()
    out = os.path.join(ROOT, "naca_le_qfrac_debug.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
