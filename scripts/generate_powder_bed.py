#!/usr/bin/env python3
"""
316L powder bed — multi-pass size-segregated packing to 45%+ density.

Strategy: Largest-first with interstice filling
  Pass 1: Large particles (35-55 μm) — form the scaffold
  Pass 2: Medium particles (20-35 μm) — fill first-order gaps
  Pass 3: Small particles (8-20 μm) — jam into every remaining crevice
  Each pass: massive blind attempts until no more fit
  Loop until packing >= 45%
"""

import numpy as np
import struct
from pathlib import Path

DX = 2.0e-6
NX, NY, NZ = 500, 150, 75
Z_SUB = 30   # substrate top in cells (60 μm)
Z_POW = 50   # doctor blade in cells (100 μm)

Lx, Ly = NX * DX, NY * DX
z_sub_m = Z_SUB * DX
z_pow_m = Z_POW * DX

# PSD parameters
D50, D90 = 30.0e-6, 45.0e-6
MU_LN = np.log(D50)
SIGMA_LN = np.log(D90 / D50) / 1.2816

TARGET_PACKING = 0.45


class PackedBed:
    def __init__(self):
        self.rng = np.random.default_rng(42)
        # Store as numpy arrays for fast vectorized overlap checks
        self.x = np.empty(0, dtype=np.float64)
        self.y = np.empty(0, dtype=np.float64)
        self.z = np.empty(0, dtype=np.float64)
        self.r = np.empty(0, dtype=np.float64)
        self.n = 0

    def _add(self, cx, cy, cz, cr):
        self.x = np.append(self.x, cx)
        self.y = np.append(self.y, cy)
        self.z = np.append(self.z, cz)
        self.r = np.append(self.r, cr)
        self.n += 1

    def _overlaps_any(self, cx, cy, cz, cr):
        """Vectorized overlap check against all existing particles."""
        if self.n == 0:
            return False
        dx = self.x - cx
        dy = self.y - cy
        dz = self.z - cz
        dist2 = dx*dx + dy*dy + dz*dz
        min_sep = (self.r + cr) * 0.97  # 3% overlap tolerance
        return np.any(dist2 < min_sep * min_sep)

    def _gravity_z(self, cx, cy, cr):
        """Find resting z via gravity fall. Returns cz."""
        best_z = z_sub_m + cr  # Floor: on substrate

        if self.n > 0:
            dx = self.x - cx
            dy = self.y - cy
            horiz2 = dx*dx + dy*dy
            contact_r = self.r + cr
            contact2 = contact_r * contact_r

            # Which existing particles are horizontally close enough to support?
            mask = horiz2 < contact2
            if np.any(mask):
                gap2 = contact2[mask] - horiz2[mask]
                z_on = self.z[mask] + np.sqrt(gap2)
                z_top = z_on.max()
                if z_top > best_z:
                    best_z = z_top

        return best_z

    def try_place(self, cr, max_xy_attempts=200):
        """Try to place a particle of radius cr. Returns True if placed."""
        for _ in range(max_xy_attempts):
            cx = self.rng.uniform(cr, Lx - cr)
            cy = self.rng.uniform(cr, Ly - cr)
            cz = self._gravity_z(cx, cy, cr)

            # Doctor blade: reject if center above powder top
            if cz > z_pow_m:
                continue

            # Must be above substrate
            if cz - cr < z_sub_m - DX * 0.1:
                continue

            if not self._overlaps_any(cx, cy, cz, cr):
                self._add(cx, cy, cz, cr)
                return True

        return False

    def total_volume(self):
        return np.sum((4.0/3.0) * np.pi * self.r**3) if self.n > 0 else 0.0

    def packing_density(self):
        powder_vol = Lx * Ly * (z_pow_m - z_sub_m)
        return self.total_volume() / powder_vol


def generate():
    print("=" * 60)
    print("  316L Powder Bed — Size-Segregated Interstice Packing")
    print("=" * 60)
    print(f"Domain: {Lx*1e6:.0f}×{Ly*1e6:.0f}×{(z_pow_m-z_sub_m)*1e6:.0f} μm powder layer")
    print(f"Target: ≥{TARGET_PACKING*100:.0f}% packing\n")

    bed = PackedBed()
    powder_vol = Lx * Ly * (z_pow_m - z_sub_m)
    target_vol = TARGET_PACKING * powder_vol

    # Phase 1: Place large scaffold (limited count — leave room for small)
    large_vol_target = 0.60 * target_vol
    print("\n--- Phase 1: Large scaffold (35-55 μm, 60% of volume) ---")
    placed_large = 0
    for _ in range(80000):
        while True:
            d = bed.rng.lognormal(MU_LN, SIGMA_LN)
            if 35e-6 <= d <= 55e-6:
                break
        if bed.try_place(d/2, 50):
            placed_large += 1
        if bed.total_volume() >= large_vol_target:
            break
    print(f"  +{placed_large} large, packing={bed.packing_density()*100:.1f}%")

    # Phase 2: Medium particles fill first-order gaps
    print("\n--- Phase 2: Medium fill (20-35 μm) ---")
    placed_med = 0
    for _ in range(150000):
        while True:
            d = bed.rng.lognormal(MU_LN, SIGMA_LN)
            if 20e-6 <= d <= 35e-6:
                break
        if bed.try_place(d/2, 80):
            placed_med += 1
        if bed.packing_density() >= TARGET_PACKING:
            break
    print(f"  +{placed_med} medium, packing={bed.packing_density()*100:.1f}%")

    # Phase 3: Small particles jam into every crevice
    print("\n--- Phase 3: Small interstice fill (8-20 μm) ---")
    placed_small = 0
    consec_fail = 0
    for attempt in range(500000):
        while True:
            d = bed.rng.lognormal(MU_LN, SIGMA_LN)
            if 8e-6 <= d <= 20e-6:
                break
        if bed.try_place(d/2, 100):
            placed_small += 1
            consec_fail = 0
        else:
            consec_fail += 1

        if attempt % 100000 == 0 and attempt > 0:
            print(f"  Attempt {attempt}: +{placed_small} small, "
                  f"packing={bed.packing_density()*100:.1f}%")

        # Stop if can't place any more (10000 consecutive failures)
        if consec_fail > 10000:
            print(f"  Saturated at attempt {attempt} (10k consecutive failures)")
            break

        if bed.packing_density() >= TARGET_PACKING:
            print(f"  TARGET {TARGET_PACKING*100:.0f}% REACHED!")
            break

    print(f"  +{placed_small} small, packing={bed.packing_density()*100:.1f}%")

    # Final stats
    density = bed.packing_density()
    d_um = bed.r * 2e6
    z_um = bed.z * 1e6

    print(f"\n{'='*60}")
    print(f"  FINAL: {bed.n} particles, {density*100:.1f}% packing")
    print(f"  PSD: D10={np.percentile(d_um,10):.1f} D50={np.percentile(d_um,50):.1f} "
          f"D90={np.percentile(d_um,90):.1f} μm")
    print(f"  Z centers: {z_um.min():.1f} – {z_um.max():.1f} μm")
    print(f"{'='*60}")

    return bed


def voxelize(bed):
    print(f"\nVoxelizing {bed.n} particles...")
    fill = np.zeros((NZ, NY, NX), dtype=np.float32)
    fill[:Z_SUB, :, :] = 1.0

    for p in range(bed.n):
        cx, cy, cz, cr = bed.x[p], bed.y[p], bed.z[p], bed.r[p]
        i0 = max(0, int((cx-cr)/DX)-1); i1 = min(NX-1, int((cx+cr)/DX)+1)
        j0 = max(0, int((cy-cr)/DX)-1); j1 = min(NY-1, int((cy+cr)/DX)+1)
        k0 = max(0, int((cz-cr)/DX)-1); k1 = min(NZ-1, int((cz+cr)/DX)+1)

        for k in range(k0, k1+1):
            dz = (k+0.5)*DX - cz
            dz2 = dz*dz
            for j in range(j0, j1+1):
                dy = (j+0.5)*DX - cy
                dy2 = dy*dy
                for i in range(i0, i1+1):
                    dx = (i+0.5)*DX - cx
                    dist = np.sqrt(dx*dx + dy2 + dz2)
                    if dist <= cr - DX*0.5:
                        fill[k, j, i] = 1.0
                    elif dist <= cr + DX*0.5:
                        frac = (cr + DX*0.5 - dist) / DX
                        fill[k, j, i] = max(fill[k, j, i], min(1.0, frac))

    fill = np.clip(fill, 0.0, 1.0)
    print(f"Powder zone mean fill: {fill[Z_SUB:Z_POW].mean():.3f}")
    return fill


def visualize(bed, fill):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # XZ side view with true circles
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    ax.axhspan(0, z_sub_m*1e6, color='#8B0000', alpha=0.8)
    ax.axhline(z_sub_m*1e6, color='cyan', ls='--', lw=1, label='Substrate')
    ax.axhline(z_pow_m*1e6, color='lime', ls='--', lw=1, label='Blade')
    mid_y = Ly / 2
    for p in range(bed.n):
        dy = bed.y[p] - mid_y
        if abs(dy) <= bed.r[p]:
            sr = np.sqrt(max(0, bed.r[p]**2 - dy**2))
            if sr > DX:
                circ = Circle((bed.x[p]*1e6, bed.z[p]*1e6), sr*1e6,
                              fc='#c0c0c0', ec='white', lw=0.3, alpha=0.9)
                ax.add_patch(circ)
    ax.set_xlim(0, Lx*1e6); ax.set_ylim(40, 120)
    ax.set_aspect('equal')
    ax.set_xlabel('x [μm]'); ax.set_ylabel('z [μm]')
    ax.set_title('XZ Side View'); ax.legend(fontsize=8)

    # XY top view with true circles
    ax = axes[1]
    ax.set_facecolor('#1a1a2e')
    for p in range(bed.n):
        circ = Circle((bed.x[p]*1e6, bed.y[p]*1e6), bed.r[p]*1e6,
                      fc='#c0c0c0', ec='white', lw=0.2, alpha=0.85)
        ax.add_patch(circ)
    ax.set_xlim(0, Lx*1e6); ax.set_ylim(0, Ly*1e6)
    ax.set_aspect('equal')
    ax.set_xlabel('x [μm]'); ax.set_ylabel('y [μm]')
    ax.set_title(f'XY Top View ({bed.n} particles)')

    # Packing profile
    ax = axes[2]
    z_um = np.arange(NZ) * DX * 1e6
    ax.plot(fill.mean(axis=(1,2)), z_um, 'b-', lw=2)
    ax.axhline(z_sub_m*1e6, color='cyan', ls='--', lw=1)
    ax.axhline(z_pow_m*1e6, color='lime', ls='--', lw=1)
    ax.set_xlabel('Mean fill'); ax.set_ylabel('z [μm]')
    ax.set_title('Packing Profile'); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1)

    density = bed.packing_density()
    d_um = bed.r * 2e6
    fig.suptitle(f'316L: {bed.n} particles, {density*100:.0f}% packing, '
                 f'D50={np.median(d_um):.0f}μm',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig('/home/yzk/LBMProject/scripts/viz/powder_bed_preview.png',
                dpi=150, bbox_inches='tight')
    print("Saved visualization")


def main():
    bed = generate()
    fill = voxelize(bed)

    outdir = Path("/home/yzk/LBMProject/output_powder_bed")
    outdir.mkdir(exist_ok=True, parents=True)

    with open(outdir / "powder_bed_fill_level.bin", 'wb') as f:
        f.write(struct.pack('iii', NX, NY, NZ))
        f.write(fill.astype(np.float32).tobytes())
    print(f"Saved binary fill_level")

    with open(outdir / "particle_list.csv", 'w') as f:
        f.write("cx_um,cy_um,cz_um,radius_um\n")
        for p in range(bed.n):
            f.write(f"{bed.x[p]*1e6:.3f},{bed.y[p]*1e6:.3f},"
                    f"{bed.z[p]*1e6:.3f},{bed.r[p]*1e6:.3f}\n")

    visualize(bed, fill)


if __name__ == "__main__":
    main()
