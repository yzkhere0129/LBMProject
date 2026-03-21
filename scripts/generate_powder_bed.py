#!/usr/bin/env python3
"""
316L powder bed — dense packing with numpy-vectorized overlap check.

Uses brute-force numpy vectorized distance for fast overlap detection.
For N<5000 particles this is faster than Python spatial hash loops.
"""

import numpy as np
import struct
from pathlib import Path

DX = 2.0e-6
NX, NY, NZ = 500, 150, 65
Z_SUB = 30    # substrate top: 60 μm
Z_POW = 45    # doctor blade: 90 μm → 30 μm powder layer

Lx, Ly = NX * DX, NY * DX
z_sub_m = Z_SUB * DX
z_pow_m = Z_POW * DX

D50, D90 = 25.0e-6, 45.0e-6
MU_LN = np.log(D50)
SIGMA_LN = np.log(D90 / D50) / 1.2816

TARGET_PACKING = 0.55
OVERLAP_TOL = 0.95


def generate():
    print("=" * 60, flush=True)
    print("  316L Powder Bed — Dense Packing (numpy vectorized)", flush=True)
    print("=" * 60, flush=True)
    print(f"Domain: {Lx*1e6:.0f}×{Ly*1e6:.0f}×{(z_pow_m-z_sub_m)*1e6:.0f} μm", flush=True)
    print(f"Target: ≥{TARGET_PACKING*100:.0f}%\n", flush=True)

    rng = np.random.default_rng(42)
    powder_vol = Lx * Ly * (z_pow_m - z_sub_m)

    # Pre-allocate arrays (max ~5000 particles)
    MAX_P = 30000
    cx = np.empty(MAX_P)
    cy = np.empty(MAX_P)
    cz = np.empty(MAX_P)
    cr = np.empty(MAX_P)
    n = 0

    def try_place_batch(radii, batch_att=50):
        """Try to place multiple particles. Returns number placed."""
        nonlocal n
        placed = 0
        for rad in radii:
            z_lo = z_sub_m + rad
            z_hi = z_pow_m - rad
            if z_hi <= z_lo:
                continue

            success = False
            for _ in range(batch_att):
                x = rng.uniform(rad, Lx - rad)
                y = rng.uniform(rad, Ly - rad)
                z = rng.uniform(z_lo, z_hi)

                if n == 0:
                    success = True
                else:
                    # Vectorized distance check against all existing particles
                    dx = cx[:n] - x
                    dy = cy[:n] - y
                    dz = cz[:n] - z
                    dist2 = dx*dx + dy*dy + dz*dz
                    min_sep = (cr[:n] + rad) * OVERLAP_TOL
                    if not np.any(dist2 < min_sep * min_sep):
                        success = True

                if success:
                    cx[n] = x; cy[n] = y; cz[n] = z; cr[n] = rad
                    n += 1
                    placed += 1
                    break
        return placed

    def packing():
        if n == 0: return 0.0
        return np.sum((4/3) * np.pi * cr[:n]**3) / powder_vol

    # Phase 1: Large (25-40 μm) — wider range to fit in 40μm layer
    print("Phase 1: Large (25-40 μm)...", flush=True)
    radii = []
    for _ in range(500):
        while True:
            d = rng.lognormal(MU_LN, SIGMA_LN)
            if 25e-6 <= d <= 40e-6: break
        radii.append(d/2)
    placed = try_place_batch(radii, 80)
    print(f"  +{placed}, packing={packing()*100:.1f}%, n={n}", flush=True)

    # Phase 2: Medium (12-25 μm)
    print("Phase 2: Medium (12-25 μm)...", flush=True)
    radii = []
    for _ in range(2000):
        while True:
            d = rng.lognormal(MU_LN, SIGMA_LN)
            if 12e-6 <= d <= 25e-6: break
        radii.append(d/2)
    placed = try_place_batch(radii, 100)
    print(f"  +{placed}, packing={packing()*100:.1f}%, n={n}", flush=True)

    # Phase 3: Small (5-12 μm) — many attempts
    print("Phase 3: Small (5-12 μm)...", flush=True)
    total_placed = 0
    for batch in range(20):
        radii = []
        for _ in range(500):
            while True:
                d = rng.lognormal(MU_LN, SIGMA_LN)
                if 5e-6 <= d <= 12e-6: break
            radii.append(d/2)
        placed = try_place_batch(radii, 80)
        total_placed += placed
        if placed == 0:
            print(f"  Saturated at batch {batch}", flush=True)
            break
        if packing() >= TARGET_PACKING:
            break
        if batch % 5 == 4:
            print(f"  batch {batch+1}: +{total_placed}, packing={packing()*100:.1f}%, n={n}", flush=True)
    print(f"  +{total_placed}, packing={packing()*100:.1f}%, n={n}", flush=True)

    # Phase 4: Tiny (3-6 μm)
    if packing() < TARGET_PACKING:
        print("Phase 4: Tiny (3-6 μm)...", flush=True)
        total_placed = 0
        for batch in range(30):
            radii = [rng.uniform(1.5e-6, 3e-6) for _ in range(500)]
            placed = try_place_batch(radii, 60)
            total_placed += placed
            if placed == 0:
                print(f"  Saturated at batch {batch}", flush=True)
                break
            if packing() >= TARGET_PACKING:
                break
            if batch % 10 == 9:
                print(f"  batch {batch+1}: +{total_placed}, packing={packing()*100:.1f}%, n={n}", flush=True)
        print(f"  +{total_placed}, packing={packing()*100:.1f}%, n={n}", flush=True)

    d_um = cr[:n] * 2e6
    density = packing()
    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL: {n} particles, {density*100:.1f}% packing", flush=True)
    print(f"  PSD: D10={np.percentile(d_um,10):.1f} D50={np.percentile(d_um,50):.1f} "
          f"D90={np.percentile(d_um,90):.1f} μm", flush=True)
    print(f"{'='*60}", flush=True)

    return cx[:n].copy(), cy[:n].copy(), cz[:n].copy(), cr[:n].copy(), n


def voxelize(px, py, pz, pr, n):
    print(f"\nVoxelizing {n} particles...", flush=True)
    fill = np.zeros((NZ, NY, NX), dtype=np.float32)
    fill[:Z_SUB, :, :] = 1.0

    for p in range(n):
        cx, cy, cz, cr = px[p], py[p], pz[p], pr[p]
        i0 = max(0, int((cx-cr)/DX)-1); i1 = min(NX-1, int((cx+cr)/DX)+1)
        j0 = max(0, int((cy-cr)/DX)-1); j1 = min(NY-1, int((cy+cr)/DX)+1)
        k0 = max(0, int((cz-cr)/DX)-1); k1 = min(NZ-1, int((cz+cr)/DX)+1)

        # Vectorized per-particle voxelization
        ii = np.arange(i0, i1+1)
        jj = np.arange(j0, j1+1)
        kk = np.arange(k0, k1+1)
        I, J, K = np.meshgrid(ii, jj, kk, indexing='ij')
        DXv = (I+0.5)*DX - cx
        DYv = (J+0.5)*DX - cy
        DZv = (K+0.5)*DX - cz
        dist = np.sqrt(DXv**2 + DYv**2 + DZv**2)

        inside = dist <= cr - DX*0.5
        shell = (~inside) & (dist <= cr + DX*0.5)
        frac = np.where(shell, np.clip((cr + DX*0.5 - dist) / DX, 0, 1), 0)

        for li in range(len(ii)):
            for lj in range(len(jj)):
                for lk in range(len(kk)):
                    gi, gj, gk = ii[li], jj[lj], kk[lk]
                    if inside[li, lj, lk]:
                        fill[gk, gj, gi] = 1.0
                    elif shell[li, lj, lk]:
                        fill[gk, gj, gi] = max(fill[gk, gj, gi], frac[li, lj, lk])

    fill = np.clip(fill, 0.0, 1.0)
    print(f"Powder zone mean fill: {fill[Z_SUB:Z_POW].mean():.3f}", flush=True)
    return fill


def visualize(px, py, pz, pr, n, fill):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    ax.axhspan(0, z_sub_m*1e6, color='#8B0000', alpha=0.8)
    ax.axhline(z_sub_m*1e6, color='cyan', ls='--', lw=1, label='Substrate')
    ax.axhline(z_pow_m*1e6, color='lime', ls='--', lw=1, label='Blade')
    mid_y = Ly / 2
    for p in range(n):
        dy = py[p] - mid_y
        if abs(dy) <= pr[p]:
            sr = np.sqrt(max(0, pr[p]**2 - dy**2))
            if sr > DX:
                circ = Circle((px[p]*1e6, pz[p]*1e6), sr*1e6,
                              fc='#c0c0c0', ec='white', lw=0.3, alpha=0.9)
                ax.add_patch(circ)
    ax.set_xlim(0, Lx*1e6); ax.set_ylim(40, 120)
    ax.set_aspect('equal')
    ax.set_xlabel('x [μm]'); ax.set_ylabel('z [μm]')
    ax.set_title('XZ Side View'); ax.legend(fontsize=8)

    ax = axes[1]
    ax.set_facecolor('#1a1a2e')
    for p in range(n):
        circ = Circle((px[p]*1e6, py[p]*1e6), pr[p]*1e6,
                      fc='#c0c0c0', ec='white', lw=0.2, alpha=0.85)
        ax.add_patch(circ)
    ax.set_xlim(0, Lx*1e6); ax.set_ylim(0, Ly*1e6)
    ax.set_aspect('equal')
    ax.set_xlabel('x [μm]'); ax.set_ylabel('y [μm]')
    ax.set_title(f'XY Top View ({n} particles)')

    ax = axes[2]
    z_um = np.arange(NZ) * DX * 1e6
    ax.plot(fill.mean(axis=(1,2)), z_um, 'b-', lw=2)
    ax.axhline(z_sub_m*1e6, color='cyan', ls='--', lw=1)
    ax.axhline(z_pow_m*1e6, color='lime', ls='--', lw=1)
    ax.set_xlabel('Mean fill'); ax.set_ylabel('z [μm]')
    ax.set_title('Packing Profile'); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1)

    powder_vol = Lx * Ly * (z_pow_m - z_sub_m)
    density = np.sum((4/3)*np.pi*pr**3) / powder_vol
    d_um = pr * 2e6
    fig.suptitle(f'316L: {n} particles, {density*100:.0f}% packing, '
                 f'D50={np.median(d_um):.0f}μm',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig('/home/yzk/LBMProject/scripts/viz/powder_bed_preview.png',
                dpi=150, bbox_inches='tight')
    print("Saved visualization", flush=True)


def main():
    px, py, pz, pr, n = generate()
    fill = voxelize(px, py, pz, pr, n)

    outdir = Path("/home/yzk/LBMProject/output_powder_bed")
    outdir.mkdir(exist_ok=True, parents=True)

    with open(outdir / "powder_bed_fill_level.bin", 'wb') as f:
        f.write(struct.pack('iii', NX, NY, NZ))
        f.write(fill.astype(np.float32).tobytes())
    print(f"Saved binary fill_level", flush=True)

    with open(outdir / "particle_list.csv", 'w') as f:
        f.write("cx_um,cy_um,cz_um,radius_um\n")
        for p in range(n):
            f.write(f"{px[p]*1e6:.3f},{py[p]*1e6:.3f},"
                    f"{pz[p]*1e6:.3f},{pr[p]*1e6:.3f}\n")

    visualize(px, py, pz, pr, n, fill)


if __name__ == "__main__":
    main()
