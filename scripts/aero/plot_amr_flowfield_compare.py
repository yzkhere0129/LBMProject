"""LE flow field comparison: AMR-OFF vs AMR-ON at settled state (step 30000).

Reads VTK snaps from both runs, plots vorticity ω_z and |u| at LE
zoom window. The KEY visual question is whether AMR introduces
sharper LE suction peak (lit signature) vs the smeared baseline.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"


def parse_vtk(path):
    with open(path) as f:
        text = f.read()
    nx = ny = nz = None
    dx = 1.0
    for line in text.splitlines():
        if line.startswith("DIMENSIONS"):
            p = line.split(); nx, ny, nz = int(p[1]), int(p[2]), int(p[3])
        elif line.startswith("SPACING"):
            dx = float(line.split()[1])
        elif line.startswith("VECTORS"):
            break
    idx = text.find("VECTORS velocity float")
    body = text[idx:].split("\n", 1)[1]
    vals = np.fromstring(body, sep=" ")
    n = nx * ny * nz
    arr = vals[: 3 * n].reshape(nz, ny, nx, 3)
    return arr, (nx, ny, nz), dx


def parse_mask(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    return np.array([[int(v) for v in l.split()] for l in lines if l.strip()])


def naca0012_outline_world(alpha_deg, n=400):
    """Analytical NACA0012 surface in (x - x_LE, y - y_LE) world frame.

    Mirrors the stamper convention in include/physics/aero/obstacle_geometry.h:
    world→chord uses M = R(α); so chord→world uses M^-1 = R(-α). α>0 = nose-up
    means upper-surface points end up below the chord-axis projection in world.
    Returns (xs_world, ys_world) closed polyline (upper, then lower reversed).
    """
    t = 0.12
    a = np.deg2rad(alpha_deg)
    ca, sa = np.cos(a), np.sin(a)
    s = np.linspace(0.0, 1.0, n)
    sqrt_s = np.sqrt(s)
    yt = 5.0 * t * (0.2969 * sqrt_s
                    - 0.1260 * s
                    - 0.3516 * s * s
                    + 0.2843 * s ** 3
                    - 0.1036 * s ** 4)  # closed TE
    # upper surface (xr, +yt), lower (xr, -yt) — chord-frame
    xr_up, yr_up = s, +yt
    xr_lo, yr_lo = s[::-1], -yt[::-1]
    xr = np.concatenate([xr_up, xr_lo])
    yr = np.concatenate([yr_up, yr_lo])
    # chord → world via R(-α) (inverse of stamper M)
    xw = ca * xr + sa * yr
    yw = -sa * xr + ca * yr
    return xw, yw


def compute_vort_umag(arr):
    kmid = arr.shape[0] // 2
    ux = arr[kmid, :, :, 0]; uy = arr[kmid, :, :, 1]
    dx_uy = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / 2
    dy_ux = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / 2
    omega = dx_uy - dy_ux
    umag = np.sqrt(ux*ux + uy*uy)
    return omega, umag


def main():
    cases = [
        ("AMR-OFF",       "output_30k_amr_off",           "C0"),
        ("AMR-ON bilin",  "output_30k_amr_on_bilin_time", "C2"),
    ]
    snap_file = "snap_0030000.vtk"

    found_cases = []
    for label, subdir, color in cases:
        vtk = os.path.join(ROOT, subdir, snap_file)
        mask_p = os.path.join(ROOT, subdir, "mask_zmid.txt")
        if not os.path.exists(vtk):
            print(f"SKIP {label}: {vtk} missing"); continue
        arr, dims, dx = parse_vtk(vtk)
        mask = parse_mask(mask_p)
        omega, umag = compute_vort_umag(arr)
        found_cases.append((label, color, dims, dx, omega, umag, mask))

    if len(found_cases) < 2:
        print("Need both AMR-OFF and AMR-ON VTKs to compare.")
        return

    # Wake-inclusive window: -0.5c upstream to +4.0c downstream of LE,
    # ±1.0c laterally (covers ~1 vortex-shedding wavelength at Re=2000).
    # The numerical domain is 30c x 20c — this is a zoom, not the full domain.
    nx, ny, nz = found_cases[0][2]
    dx = found_cases[0][3]
    le_x = 10.0; le_y = ny * dx / 2.0
    x_lo_c, x_hi_c = -0.5, +4.0
    y_lo_c, y_hi_c = -1.0, +1.0
    i_lo = max(0, int((le_x + x_lo_c) / dx))
    i_hi = min(nx, int((le_x + x_hi_c) / dx))
    j_lo = max(0, int((le_y + y_lo_c) / dx))
    j_hi = min(ny, int((le_y + y_hi_c) / dx))

    naca_x, naca_y = naca0012_outline_world(alpha_deg=8.0, n=400)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    extent = (i_lo*dx - le_x, (i_hi-1)*dx - le_x,
              j_lo*dx - le_y, (j_hi-1)*dx - le_y)

    # First pass: find common vmax for fair comparison
    om_all = np.concatenate([f[4][j_lo:j_hi, i_lo:i_hi].ravel() for f in found_cases])
    um_all = np.concatenate([f[5][j_lo:j_hi, i_lo:i_hi].ravel() for f in found_cases])
    vmax_om = max(np.percentile(np.abs(om_all), 99.0), 1e-4)
    vmax_um = max(np.percentile(um_all, 99.5), 0.05)

    for row, (label, color, dims, dx_, omega, umag, mask) in enumerate(found_cases):
        om_w = omega[j_lo:j_hi, i_lo:i_hi]
        um_w = umag[j_lo:j_hi, i_lo:i_hi]
        im1 = axes[row, 0].imshow(om_w, extent=extent, origin='lower',
                                   cmap='RdBu_r', vmin=-vmax_om, vmax=vmax_om, aspect='equal')
        axes[row, 0].set_title(f"{label}  ω_z (step 30000)")
        plt.colorbar(im1, ax=axes[row, 0], shrink=0.85, label="ω_z (LU)")
        im2 = axes[row, 1].imshow(um_w, extent=extent, origin='lower',
                                   cmap='viridis', vmin=0, vmax=vmax_um, aspect='equal')
        axes[row, 1].set_title(f"{label}  |u| (freestream≈0.05)")
        plt.colorbar(im2, ax=axes[row, 1], shrink=0.85, label="|u| (LU)")
        # Use analytical NACA0012 surface — the stamper's binary mask gives
        # a stair-stepped contour at D/dx=80; the analytical curve is what
        # the QBB sub-cell qfrac BC actually approximates.
        for ax in axes[row, :]:
            ax.fill(naca_x, naca_y, facecolor='#222222', edgecolor='k',
                    linewidth=1.0, alpha=0.85, zorder=5)
            ax.set_xlabel("x − x_LE (chord)")
            ax.set_ylabel("y − y_LE (chord)")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

    fig.suptitle("NACA0012 settled flow field (step 30000) — AMR-OFF vs AMR-ON (bilinear+time interp)\n"
                 "Window: -0.5c..+4.0c × ±1.0c around LE  (sim domain is 30c×20c — this is a wake-zoom)\n"
                 "Airfoil outline = analytical NACA0012 (stamper's actual sub-cell shape via qfrac)",
                 fontsize=11, y=1.005)
    plt.tight_layout()
    out = os.path.join(ROOT, "images", "amr_30k_flowfield_compare.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
