#!/usr/bin/env python3
"""Locate where VOF mass-correction redistributes mass.

Phase-2 minus Phase-1 column-mass Δm map: identifies which zone (scan-start
splash, side ridges, trailing centerline) received the correction-kernel mass.

Usage:
    python3 scripts/diagnostics/locate_phase2_mass.py <phase1.vtk> <phase2.vtk> \
        [z0_um=160] [laser_x_um=auto] [laser_start_um=500]

If phase2.vtk does not exist yet, run the line_scan_316L app with
enable_vof_mass_correction=true and save a frame at the same step as phase1.
"""
import sys, os
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── zone thresholds (μm) ─────────────────────────────────────────────────────
SCAN_START_X_MAX  = 700.0   # x < this => scan-start splash zone (absolute)
CENTERLINE_Y_HALF = 15.0    # |y - y_mid| <=  this => centerline
SIDE_RIDGE_Y_MIN  = 30.0    # |y - y_mid| >=  this => side ridge
TRAILING_MARGIN   = 100.0   # trailing zone: x < laser_x - this
V_SCAN_MPS        = 0.8     # m/s (for auto laser_x inference)
DT_NS             = 80.0    # ns

def load(path):
    m   = pv.read(path)
    dims = np.array(m.dimensions, dtype=np.int64)
    sp   = np.array(m.spacing,    dtype=np.float64)
    org  = np.array(m.origin,     dtype=np.float64)
    f    = np.asarray(m.point_data["fill_level"]).reshape(dims, order="F")
    vz   = (np.asarray(m.point_data["velocity"]).reshape((*dims, 3), order="F")[..., 2]
            if "velocity" in m.point_data.keys() else None)
    return dims, sp, org, f, vz

def infer_laser_x(path, start_um):
    try:
        step  = int(path.rsplit("_", 1)[-1].split(".")[0])
        t_us  = step * DT_NS * 1e-3
        return start_um + V_SCAN_MPS * t_us
    except (ValueError, IndexError):
        return float("nan")

def zone_sums(dm, x_um, y_um, y_mid_idx, lx_um):
    ym = y_um[y_mid_idx]
    dy = np.abs(y_um - ym)
    ss = x_um[:, None] < SCAN_START_X_MAX
    tr = x_um[:, None] < (lx_um - TRAILING_MARGIN)
    cx = dy[None, :] <= CENTERLINE_Y_HALF
    sr = dy[None, :] >= SIDE_RIDGE_Y_MIN
    def s(mask): return float(dm[mask & (dm > 0)].sum())
    return {
        "scan_start"         : s(ss),
        "trailing_center"    : s(tr & cx & ~ss),
        "trailing_side_ridge": s(tr & sr & ~ss),
        "active_front"       : s(~tr),
    }

def save_png(dm, x_um, y_um, lx_um, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = max(np.abs(dm).max(), 1e-12)
    ax.pcolormesh(x_um, y_um, dm.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ym = (y_um.min() + y_um.max()) / 2
    if np.isfinite(lx_um):
        ax.axvline(lx_um, color="yellow", lw=1.5, label=f"laser {lx_um:.0f} μm")
        ax.axvline(lx_um - TRAILING_MARGIN, color="lime", lw=1, ls="--", label="trailing margin")
    ax.axvline(SCAN_START_X_MAX, color="orange", lw=1, ls=":", label="scan-start zone edge")
    ax.axhspan(ym - CENTERLINE_Y_HALF, ym + CENTERLINE_Y_HALF, alpha=0.15, color="cyan")
    for yb, yt in ((y_um.min(), ym - SIDE_RIDGE_Y_MIN), (ym + SIDE_RIDGE_Y_MIN, y_um.max())):
        ax.axhspan(yb, yt, alpha=0.12, color="magenta")
    ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")
    ax.set_title("Δm = Phase-2 − Phase-1 column mass (fill·dz)  |  Red=Phase-2 gained")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Heatmap saved: {path}")

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(0)

    p1, p2 = sys.argv[1], sys.argv[2]
    laser_start_um = float(sys.argv[5]) if len(sys.argv) > 5 else 500.0

    missing = [p for p in (p1, p2) if not os.path.exists(p)]
    if missing:
        print(f"ERROR: file(s) not found: {missing}")
        print("Phase-2 VTKs not yet generated — see docstring for harness instructions.")
        sys.exit(0)

    dims, sp, org, f1, _  = load(p1)
    dims2, sp2, _, f2, vz2 = load(p2)
    nx, ny, nz = dims
    dx, dy_sp, dz = sp
    ox, oy, oz    = org
    x_um = (ox + np.arange(nx) * dx) * 1e6
    y_um = (oy + np.arange(ny) * dy_sp) * 1e6
    y_mid = ny // 2

    if not np.array_equal(dims, dims2):
        print(f"WARNING: grid dims differ {dims} vs {dims2}")

    # per-column Δm (fill × dz, units μm)
    dm = (f2.sum(axis=2) - f1.sum(axis=2)) * dz * 1e6

    lx_um = (float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "auto"
             else infer_laser_x(p1, laser_start_um))

    total_gain = float(dm[dm > 0].sum())
    total_loss = float(dm[dm < 0].sum())
    total_net  = float(dm.sum())

    print(f"\nPhase-1 Σfill: {f1.sum():.2f}   Phase-2 Σfill: {f2.sum():.2f}   "
          f"Δ = {f2.sum()-f1.sum():+.3f} cells")
    print(f"Δm total  net: {total_net:+.5f} μm   gain: {total_gain:.5f}   loss: {total_loss:.5f}")
    print(f"Laser x = {lx_um:.1f} μm\n")

    zs = zone_sums(dm, x_um, y_um, y_mid, lx_um)
    print(f"{'Zone':<28} {'Gain (μm)':>12}  {'% of total gain':>16}")
    print("-" * 62)
    for name, gain in zs.items():
        pct = gain / total_gain * 100 if total_gain > 0 else float("nan")
        pct_s = f"{pct:6.1f}%" if np.isfinite(pct) else "   NaN"
        print(f"  {name:<26} {gain:>12.5f}  {pct_s:>16}")

    if total_gain > 0:
        sc = zs["scan_start"] / total_gain * 100
        sr = zs["trailing_side_ridge"] / total_gain * 100
        tc = zs["trailing_center"] / total_gain * 100
        print(f"\nVERDICT: correction mass went to —")
        print(f"  scan-start splash  {sc:5.1f}%  (should be ~0)")
        print(f"  trailing side ridge{sr:5.1f}%  (should be ~0)")
        print(f"  trailing centerline{tc:5.1f}%  (should be ~100)")
    else:
        print("\nVERDICT: no net mass gain in Phase-2. Verify correction is enabled.")

    # top-10 gaining columns
    v_factor = dx / (DT_NS * 1e-9)       # LU → m/s
    flat_top  = np.argsort(dm.ravel())[::-1][:10]
    ii, jj    = np.unravel_index(flat_top, dm.shape)
    print(f"\nTop-10 columns by Δm gain:")
    print(f"  {'x μm':>7} {'y μm':>7} {'Δm μm':>9} {'v_z m/s':>10} {'x-laser μm':>12}")
    print("  " + "-" * 50)
    for i, j in zip(ii, jj):
        vz_val = float("nan")
        if vz2 is not None:
            mask2 = f2[i, j, :] > 0.5
            if mask2.any():
                k = int(nz - 1 - np.argmax(mask2[::-1]))
                vz_val = float(vz2[i, j, k]) * v_factor
        dist = x_um[i] - lx_um
        vz_s = f"{vz_val:+9.3f}" if np.isfinite(vz_val) else f"{'--':>9}"
        print(f"  {x_um[i]:>7.1f} {y_um[j]:>7.1f} {dm[i,j]:>9.5f} {vz_s:>10} {dist:>12.1f}")

    out = os.path.join(os.path.dirname(os.path.abspath(p1)), "delta_mass_map.png")
    save_png(dm, x_um, y_um, lx_um, out)

if __name__ == "__main__":
    main()
