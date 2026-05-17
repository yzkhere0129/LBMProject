"""Compare G500 vs G2000 flowfields at settled state (step 30000).

Layout: 2 rows × 2 cols
  row 0:  Re=500   ω_z   |  Re=500   |u|
  row 1:  Re=2000  ω_z   |  Re=2000  |u|
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"


def parse_vtk_structured(path):
    with open(path) as f:
        text = f.read()
    nx = ny = nz = None
    dx = dy = dz = 1.0
    for line in text.splitlines():
        if line.startswith("DIMENSIONS"):
            parts = line.split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
        elif line.startswith("SPACING"):
            parts = line.split()
            dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
        elif line.startswith("VECTORS"):
            break
    idx = text.find("VECTORS velocity float")
    body = text[idx:].split("\n", 1)[1]
    vals = np.fromstring(body, sep=" ")
    n = nx * ny * nz
    arr = vals[: 3 * n].reshape(nz, ny, nx, 3)
    return arr, (nx, ny, nz), (dx, dy, dz)


def parse_mask_zmid(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    return np.array([[int(v) for v in l.split()] for l in lines if l.strip()])


def settled_cl_cd(csv_path, frac=2.0 / 3):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    n = data.shape[0]
    s = int(n * frac)
    cd = data[s:, 7]
    cl = data[s:, 8]
    return cd.mean(), cd.std(ddof=1), cl.mean(), cl.std(ddof=1)


def compute_fields(arr):
    kmid = arr.shape[0] // 2
    ux = arr[kmid, :, :, 0]
    uy = arr[kmid, :, :, 1]
    duy_dx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / 2
    dux_dy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / 2
    omega = duy_dx - dux_dy
    umag = np.sqrt(ux * ux + uy * uy)
    return omega, umag


def crop_window(field, dx, le_x, le_y, x_lo_c, x_hi_c, y_lo_c, y_hi_c):
    ny, nx = field.shape
    i_lo = max(0, int((le_x + x_lo_c) / dx))
    i_hi = min(nx, int((le_x + x_hi_c) / dx))
    j_lo = max(0, int((le_y + y_lo_c) / dx))
    j_hi = min(ny, int((le_y + y_hi_c) / dx))
    extent = (i_lo * dx - le_x, (i_hi - 1) * dx - le_x,
              j_lo * dx - le_y, (j_hi - 1) * dx - le_y)
    return field[j_lo:j_hi, i_lo:i_hi], extent, (i_lo, j_lo, i_hi, j_hi)


def main():
    cases = [
        ("Re=500",  "output_univ_G500_vtk",   500),
        ("Re=2000", "output_univ_G2000_vtk", 2000),
    ]
    snap = "snap_0030000.vtk"

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    chord = 1.0
    x_lo_c, x_hi_c = -1.5, 5.0
    y_lo_c, y_hi_c = -2.5, 2.5

    for row, (label, subdir, re) in enumerate(cases):
        vtk_path = os.path.join(ROOT, subdir, snap)
        mask_path = os.path.join(ROOT, subdir, "mask_zmid.txt")
        csv_path = os.path.join(ROOT, subdir, "forces.csv")

        arr, (nx, ny, nz), (dx, _, _) = parse_vtk_structured(vtk_path)
        omega, umag = compute_fields(arr)
        mask = parse_mask_zmid(mask_path)
        cd_m, cd_s, cl_m, cl_s = settled_cl_cd(csv_path)

        le_x = 10.0 * chord
        le_y = ny * dx / 2.0

        om_w, ext, (i_lo, j_lo, _, _) = crop_window(
            omega, dx, le_x, le_y, x_lo_c, x_hi_c, y_lo_c, y_hi_c)
        um_w, _, _ = crop_window(
            umag, dx, le_x, le_y, x_lo_c, x_hi_c, y_lo_c, y_hi_c)

        vmax_om = max(np.percentile(np.abs(om_w), 99.0), 1e-4)
        vmax_um = max(np.percentile(um_w, 99.5), 0.05)

        ax_om = axes[row, 0]
        ax_um = axes[row, 1]

        im_om = ax_om.imshow(om_w, extent=ext, origin="lower",
                             cmap="RdBu_r", vmin=-vmax_om, vmax=vmax_om,
                             aspect="equal")
        ax_om.set_title(
            f"{label} (α=8°, D/dx=80, step 30000)\n"
            f"vorticity ω_z  —  Cl_settled = {cl_m:.3f}±{cl_s:.3f}",
            fontsize=10)
        plt.colorbar(im_om, ax=ax_om, shrink=0.85, label="ω_z (LU)")

        im_um = ax_um.imshow(um_w, extent=ext, origin="lower",
                             cmap="viridis", vmin=0, vmax=vmax_um,
                             aspect="equal")
        ax_um.set_title(
            f"{label}  —  |u| (LU, freestream ≈ 0.05)\n"
            f"Cd_settled = {cd_m:.3f}±{cd_s:.3f}",
            fontsize=10)
        plt.colorbar(im_um, ax=ax_um, shrink=0.85, label="|u| (LU)")

        # Overlay airfoil boundary from mask
        if mask is not None:
            mc = mask[j_lo:j_lo + om_w.shape[0],
                      i_lo:i_lo + om_w.shape[1]]
            xs = np.linspace(ext[0], ext[1], mc.shape[1])
            ys = np.linspace(ext[2], ext[3], mc.shape[0])
            for ax in (ax_om, ax_um):
                ax.contour(xs, ys, mc, levels=[0.5], colors="k", linewidths=1.2)

        for ax in (ax_om, ax_um):
            ax.set_xlabel("x − x_LE (chord)")
            ax.set_ylabel("y − y_LE (chord)")

    fig.suptitle(
        "NACA0012 flowfield comparison: Re=500 vs Re=2000 (α=8°, D/dx=80, Cumulant sparse, step 30000)\n"
        "ω increases at higher Re → stronger shedding, but LBM under-predicts Cl by 9% (Re=500) → 49% (Re=2000)",
        fontsize=11, y=1.005)
    plt.tight_layout()
    out_path = os.path.join(ROOT, "naca_flowfield_g500_g2000.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
