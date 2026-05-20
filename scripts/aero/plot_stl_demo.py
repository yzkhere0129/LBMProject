"""STL demo analysis: forces.csv summary + VTK flow field plot.

Usage:
  python3 scripts/aero/plot_stl_demo.py <output_dir> [<label>]

Outputs:
  images/<label>_forces.png        — Cd/Cl/mass time series
  images/<label>_flowfield.png     — VTK mid-z slice ω_z + |u| with silhouette outline

Reusable for any --shape stl run; F-18 and F-35 use the same path.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"


def parse_vtk(path):
    """Read structured-grid legacy VTK; return velocity (nz, ny, nx, 3), dims, dx."""
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
    if idx < 0: return None, None, None
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


def main(out_dir, label=None):
    if label is None:
        label = os.path.basename(out_dir).replace("output_", "")
    forces_path = os.path.join(out_dir, "forces.csv")
    if not os.path.exists(forces_path):
        print(f"ERROR: {forces_path} missing")
        sys.exit(1)

    d = np.loadtxt(forces_path, delimiter=",", skiprows=1)
    if d.ndim < 2 or d.shape[0] < 5:
        print(f"too few samples in {forces_path}")
        sys.exit(1)
    step, t, Cd, Cl, mass = d[:, 0], d[:, 1], d[:, 7], d[:, 8], d[:, 9]

    s = max(int(len(step) * 2 / 3), 1)
    Cd_s, Cl_s = Cd[s:].mean(), Cl[s:].mean()
    Cd_rms, Cl_rms = Cd[s:].std(ddof=1), Cl[s:].std(ddof=1)
    mass_drift = (mass[-1] - mass[0]) / mass[0]

    print(f"=== {label} ===")
    print(f"  N samples:   {len(step)}")
    print(f"  Cl_settled:  {Cl_s:+.4f} ± {Cl_rms:.4f}  (last 1/3 mean)")
    print(f"  Cd_settled:  {Cd_s:+.4f} ± {Cd_rms:.4f}")
    print(f"  L/D:         {Cl_s/Cd_s if Cd_s != 0 else 0:.3f}")
    print(f"  mass drift:  {mass_drift:+.3e}")
    print(f"  NaN check:   {bool(np.isnan(d).any())}")

    # ===== Plot 1: forces time series =====
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(step, Cl, lw=1.0, color='C0', alpha=0.85)
    axes[0].axhline(Cl_s, ls='--', color='C0', alpha=0.5,
                    label=f'settled mean {Cl_s:+.3f}')
    axes[0].set_ylabel("Cl"); axes[0].legend(loc='lower right')
    axes[0].grid(True, ls=':', alpha=0.4)

    axes[1].plot(step, Cd, lw=1.0, color='C1', alpha=0.85)
    axes[1].axhline(Cd_s, ls='--', color='C1', alpha=0.5,
                    label=f'settled mean {Cd_s:+.3f}')
    axes[1].set_ylabel("Cd"); axes[1].legend(loc='lower right')
    axes[1].grid(True, ls=':', alpha=0.4)

    drift = (mass - mass[0]) / mass[0]
    axes[2].plot(step, drift, lw=1.0, color='gray')
    axes[2].axhspan(-1e-5, 1e-5, color='C2', alpha=0.15, label='FP32 noise band')
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("Mass drift (rel)")
    axes[2].legend(loc='lower right')
    axes[2].grid(True, ls=':', alpha=0.4)

    fig.suptitle(f"{label} — forces & mass time series", fontweight='bold')
    plt.tight_layout()
    out1 = os.path.join(ROOT, "images", f"{label}_forces.png")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ===== Plot 2: VTK flow field (last snapshot if any) =====
    vtks = sorted([f for f in os.listdir(out_dir) if f.startswith("snap_") and f.endswith(".vtk")])
    if not vtks:
        print("No VTK files in output dir — skipping flow-field plot.")
        return
    vtk_path = os.path.join(out_dir, vtks[-1])
    arr, dims, dx = parse_vtk(vtk_path)
    if arr is None:
        print("VTK parse failed"); return
    nx, ny, nz = dims
    kmid = nz // 2
    ux = arr[kmid, :, :, 0]; uy = arr[kmid, :, :, 1]
    umag = np.sqrt(ux*ux + uy*uy)
    dx_uy = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / 2
    dy_ux = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / 2
    omega = dx_uy - dy_ux

    # Find solid bbox from mask file (for zoom)
    mask = parse_mask(os.path.join(out_dir, "mask_zmid.txt"))
    if mask is not None and (mask == 1).any():
        ys_s, xs_s = np.where(mask == 1)
        cx, cy = (xs_s.min() + xs_s.max()) / 2, (ys_s.min() + ys_s.max()) / 2
        ext = max(xs_s.max() - xs_s.min(), ys_s.max() - ys_s.min())
        # Zoom: 2× model bbox to include wake
        zoom = max(int(2.5 * ext), 100)
        i_lo = max(0, int(cx) - zoom // 2)
        i_hi = min(nx, int(cx) + zoom)
        j_lo = max(0, int(cy) - zoom // 2)
        j_hi = min(ny, int(cy) + zoom // 2)
    else:
        i_lo, i_hi, j_lo, j_hi = 0, nx, 0, ny

    fig2, axes = plt.subplots(1, 2, figsize=(18, 7))
    extent_w = (i_lo * dx, i_hi * dx, j_lo * dx, j_hi * dx)
    om_w = omega[j_lo:j_hi, i_lo:i_hi]
    um_w = umag [j_lo:j_hi, i_lo:i_hi]
    vmax_om = max(np.percentile(np.abs(om_w), 99.5), 1e-4)
    vmax_um = max(np.percentile(um_w, 99.5), 0.05)

    im1 = axes[0].imshow(om_w, extent=extent_w, origin='lower',
                          cmap='RdBu_r', vmin=-vmax_om, vmax=vmax_om, aspect='equal')
    axes[0].set_title("ω_z (vorticity, mid-z slab)")
    plt.colorbar(im1, ax=axes[0], shrink=0.85)

    im2 = axes[1].imshow(um_w, extent=extent_w, origin='lower',
                          cmap='viridis', vmin=0, vmax=vmax_um, aspect='equal')
    axes[1].set_title("|u| (speed, mid-z slab)")
    plt.colorbar(im2, ax=axes[1], shrink=0.85)

    # Overlay silhouette outline from mask
    if mask is not None:
        mc = mask[j_lo:j_hi, i_lo:i_hi]
        xs_grid = np.linspace(extent_w[0], extent_w[1], mc.shape[1])
        ys_grid = np.linspace(extent_w[2], extent_w[3], mc.shape[0])
        for ax in axes:
            ax.contour(xs_grid, ys_grid, mc, levels=[0.5], colors='k', linewidths=1.0)

    for ax in axes:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    fig2.suptitle(f"{label} — flow field at step {step[-1]:.0f}", fontweight='bold')
    plt.tight_layout()
    out2 = os.path.join(ROOT, "images", f"{label}_flowfield.png")
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    print(f"Saved: {out2}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_stl_demo.py <output_dir> [<label>]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
