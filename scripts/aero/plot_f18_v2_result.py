"""Visualize F-18 v2 demo with flow direction annotated.

Reads output_f18_v2_demo/{forces.csv, snap_*.vtk, mask_zmid.txt} and
produces images/f18_v2_{forces,flowfield,wake}.png with clear flow
direction overlay.
"""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"
OUT_DIR = os.path.join(ROOT, "output_f18_v2_demo")


def parse_vtk(p):
    with open(p) as f: text = f.read()
    nx = ny = nz = None; dx = 1.0
    for line in text.splitlines():
        if line.startswith("DIMENSIONS"): nx, ny, nz = [int(x) for x in line.split()[1:4]]
        elif line.startswith("SPACING"):  dx = float(line.split()[1])
        elif line.startswith("VECTORS"):  break
    idx = text.find("VECTORS velocity float")
    body = text[idx:].split("\n", 1)[1]
    vals = np.fromstring(body, sep=" ")
    arr = vals[:3*nx*ny*nz].reshape(nz, ny, nx, 3)
    return arr, (nx, ny, nz), dx


def parse_mask(p):
    if not os.path.exists(p): return None
    with open(p) as f: lines = [l for l in f if not l.startswith("#")]
    return np.array([[int(v) for v in l.split()] for l in lines if l.strip()])


def main():
    # ===== Forces =====
    fp = os.path.join(OUT_DIR, "forces.csv")
    if not os.path.exists(fp):
        print(f"NO forces.csv yet at {fp}"); return
    d = np.loadtxt(fp, delimiter=",", skiprows=1)
    step, t, Cd, Cl, mass = d[:,0], d[:,1], d[:,7], d[:,8], d[:,9]
    s = int(len(step) * 2/3)
    Cd_s, Cl_s = Cd[s:].mean(), Cl[s:].mean()

    print(f"F-18 v2 demo — {len(step)} samples")
    print(f"  Cd_settled (last 1/3): {Cd_s:+.4f}")
    print(f"  Cl_settled:            {Cl_s:+.4f}  (should be ≈ 0 at α=0 by symmetry)")
    print(f"  mass drift:            {(mass[-1]-mass[0])/mass[0]:+.3e}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(step, Cd, lw=1, color='C1'); axes[0].axhline(Cd_s, ls='--', color='C1', alpha=0.4, label=f'mean {Cd_s:+.3f}')
    axes[0].set_ylabel("Cd"); axes[0].legend(); axes[0].grid(True, ls=':', alpha=0.4)
    axes[1].plot(step, Cl, lw=1, color='C0'); axes[1].axhline(Cl_s, ls='--', color='C0', alpha=0.4, label=f'mean {Cl_s:+.4f}')
    axes[1].axhline(0, ls=':', color='gray')
    axes[1].set_ylabel("Cl"); axes[1].set_xlabel("step"); axes[1].legend(); axes[1].grid(True, ls=':', alpha=0.4)
    fig.suptitle(f"F/A-18E silhouette v2 demo (Re=2000, α=0) — Cl should ≈ 0 by symmetry", fontweight='bold')
    plt.tight_layout()
    out1 = os.path.join(ROOT, "images", "f18_v2_forces.png")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    plt.savefig(out1, dpi=120, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ===== Flow field =====
    vtks = sorted([f for f in os.listdir(OUT_DIR) if f.startswith("snap_") and f.endswith(".vtk")])
    if not vtks:
        print("no VTK files"); return
    arr, dims, dx = parse_vtk(os.path.join(OUT_DIR, vtks[-1]))
    nx, ny, nz = dims; kmid = nz // 2
    ux, uy = arr[kmid,:,:,0], arr[kmid,:,:,1]
    umag = np.sqrt(ux*ux + uy*uy)
    dx_uy = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / 2
    dy_ux = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / 2
    omega = dx_uy - dy_ux

    mask = parse_mask(os.path.join(OUT_DIR, "mask_zmid.txt"))
    # Zoom around F-18 (x=10-11 nose-tail, plus wake to x=14)
    i_lo, i_hi = int(8/dx), int(14.5/dx)
    j_lo, j_hi = int(2/dx), int(8/dx)

    fig2, axes = plt.subplots(2, 1, figsize=(15, 10))
    extent_w = (i_lo*dx, i_hi*dx, j_lo*dx, j_hi*dx)
    om_w = omega[j_lo:j_hi, i_lo:i_hi]
    um_w = umag[j_lo:j_hi, i_lo:i_hi]
    vmax_om = max(np.percentile(np.abs(om_w), 99), 1e-4)
    vmax_um = max(np.percentile(um_w, 99.5), 0.05)

    im1 = axes[0].imshow(om_w, extent=extent_w, origin='lower',
                          cmap='RdBu_r', vmin=-vmax_om, vmax=vmax_om, aspect='equal')
    axes[0].set_title(f"F/A-18E v2 — ω_z (vorticity) at step {step[-1]:.0f}")
    plt.colorbar(im1, ax=axes[0], shrink=0.85)

    im2 = axes[1].imshow(um_w, extent=extent_w, origin='lower',
                          cmap='viridis', vmin=0, vmax=vmax_um, aspect='equal')
    axes[1].set_title(f"F/A-18E v2 — |u| (speed)  freestream≈0.05 LU")
    plt.colorbar(im2, ax=axes[1], shrink=0.85)

    # Overlay aircraft silhouette + flow arrow
    if mask is not None:
        mc = mask[j_lo:j_hi, i_lo:i_hi]
        xs = np.linspace(extent_w[0], extent_w[1], mc.shape[1])
        ys = np.linspace(extent_w[2], extent_w[3], mc.shape[0])
        for ax in axes:
            ax.contour(xs, ys, mc, levels=[0.5], colors='k', linewidths=1.0)
            # Flow direction arrow
            ax.annotate('', xy=(9.5, 5.0), xytext=(8.5, 5.0),
                        arrowprops=dict(arrowstyle='->', color='C3', lw=3, alpha=0.7))
            ax.text(8.5, 5.5, 'FLOW +x', fontsize=11, fontweight='bold', color='C3')
            ax.text(10.0, 4.5, 'nose', fontsize=10, color='k', alpha=0.6)
            ax.text(11.0, 4.5, 'tail', fontsize=10, color='k', alpha=0.6)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    fig2.suptitle("F/A-18E silhouette v2 (oriented) — wake should trail downstream (+x)",
                  fontweight='bold')
    plt.tight_layout()
    out2 = os.path.join(ROOT, "images", "f18_v2_flowfield.png")
    plt.savefig(out2, dpi=120, bbox_inches='tight')
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
