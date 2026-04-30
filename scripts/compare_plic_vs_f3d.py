#!/usr/bin/env python3
"""compare_plic_vs_f3d.py

Compare pool metrics (W / L / D / T_max) for:
  - Legacy LBM  (algebraic VOF)  from output_line_scan_316L/
  - PLIC LBM                     from output_line_scan_316L_plic/
  - F3D ground truth             from vtk-316L-150W-50um-V800mms/

F3D reference values (from MEMORY.md / experiment-calibrated):
  Pool W = 73 μm, Pool L = 438 μm, D_melt = 78 μm
  T_max ≈ 4013 K, v_max ≈ 7 m/s

Usage:
  python3 scripts/compare_plic_vs_f3d.py [--legacy DIR] [--plic DIR] [--f3d DIR] [--out FILE]

Outputs:
  - 3-row comparison table printed to stdout
  - Single PNG with 3 panels saved to --out (default: compare_plic_vs_f3d.png)

Dependencies: numpy, matplotlib, vtk  (pip install vtk)
"""

import argparse
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# F3D hard-coded ground-truth from MEMORY (experiment-calibrated by 学长)
F3D_GT = {
    "W_um":     73.0,
    "L_um":    438.0,
    "D_um":     78.0,
    "T_max_K": 4013.0,
    "v_max_ms":  7.0,
}

# 316L solidus for melt-pool boundary in LBM structured grids
T_SOLIDUS = 1658.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PLIC vs Legacy vs F3D pool metric comparison")
    p.add_argument("--legacy", default="output_line_scan_316L",
                   help="Directory containing legacy VTK files (line_scan_*.vtk)")
    p.add_argument("--plic",   default="output_line_scan_316L_plic",
                   help="Directory containing PLIC VTK files (line_scan_plic_*.vtk)")
    p.add_argument("--f3d",    default="/home/yzk/LBMProject/vtk-316L-150W-50um-V800mms",
                   help="Directory containing F3D reference VTK files")
    p.add_argument("--out",    default="compare_plic_vs_f3d.png",
                   help="Output PNG file path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LBM structured-grid VTK reader
# Returns: dict with keys W_um, L_um, D_um, T_max_K, v_max_ms
#          + raw numpy arrays fill_level, liquid_fraction, temperature,
#            vx, vy, vz, nx, ny, nz, dx_m for the visualisation panels
# ---------------------------------------------------------------------------
def load_lbm_vtk(path):
    """Parse a legacy/PLIC STRUCTURED_POINTS VTK file written by VTKWriter::writeFields."""
    with open(path, "r") as f:
        lines = f.readlines()

    # Parse header for grid dimensions and spacing
    nx = ny = nz = 1
    dx = 1.0
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            parts = line.split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
        if line.startswith("SPACING"):
            dx = float(line.split()[1])  # assume isotropic
        if line.startswith("POINT_DATA"):
            data_start = i
            break

    n_cells = nx * ny * nz
    fields = {}
    i = data_start
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("SCALARS") or line.startswith("VECTORS"):
            parts = line.split()
            fname = parts[1]
            is_vec = line.startswith("VECTORS")
            # skip LOOKUP_TABLE line for scalars
            if not is_vec:
                i += 2  # skip "LOOKUP_TABLE default"
            else:
                i += 1
            vals = []
            components = 3 if is_vec else 1
            needed = n_cells * components
            while len(vals) < needed and i < len(lines):
                vals.extend(map(float, lines[i].split()))
                i += 1
            arr = np.array(vals[:needed], dtype=np.float32)
            if is_vec:
                arr = arr.reshape(n_cells, 3)
            fields[fname] = arr
        else:
            i += 1

    fill    = fields.get("FillLevel",       np.zeros(n_cells, np.float32))
    lf      = fields.get("LiquidFraction",  np.zeros(n_cells, np.float32))
    temp    = fields.get("Temperature",     np.zeros(n_cells, np.float32))
    vel     = fields.get("Velocity",        np.zeros((n_cells, 3), np.float32))

    # Melt pool = metal cells (fill>0.5) that are melted (lf>0.5)
    mask = (fill > 0.5) & (lf > 0.5)
    n_melt = int(mask.sum())

    if n_melt == 0:
        # Fallback: use temperature threshold instead of liquid fraction
        mask = (fill > 0.5) & (temp > T_SOLIDUS)
        n_melt = int(mask.sum())

    # Reconstruct 3D index arrays for melt mask
    idx = np.where(mask)[0]
    iz = idx // (nx * ny)
    jy = (idx % (nx * ny)) // nx
    ix = idx % nx

    metrics = {}
    if n_melt == 0:
        metrics.update({"W_um": 0, "L_um": 0, "D_um": 0})
    else:
        metrics["L_um"] = float((ix.max() - ix.min() + 1) * dx * 1e6)
        metrics["W_um"] = float((jy.max() - jy.min() + 1) * dx * 1e6)
        # Depth: distance from free surface (z=0.8*nz) down to deepest melt cell
        z_surface_cells = 0.80 * nz
        metrics["D_um"] = float(max(0.0, (z_surface_cells - iz.min()) * dx * 1e6))

    T_all = temp[fill > 0.5]
    metrics["T_max_K"] = float(T_all.max()) if T_all.size > 0 else 0.0

    if vel.ndim == 2:
        vmag = np.linalg.norm(vel[fill > 0.5], axis=1)
        metrics["v_max_ms"] = float(vmag.max()) if vmag.size > 0 else 0.0
    else:
        metrics["v_max_ms"] = 0.0

    return metrics, fill, lf, temp, vel, nx, ny, nz, dx


# ---------------------------------------------------------------------------
# F3D POLYDATA VTK reader (surface mesh, T > T_solidus defines the melt pool)
# ---------------------------------------------------------------------------
def load_f3d_vtk(path):
    """
    Parse a Flow3D POLYDATA VTK surface-mesh file.
    Falls back to the hard-coded GT values from MEMORY if vtk module is unavailable.
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        print("  [WARN] vtk module not found — using hard-coded F3D ground-truth values.")
        return F3D_GT.copy(), None, None, None, None

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    pd = reader.GetOutput()

    pts = vtk_to_numpy(pd.GetPoints().GetData()).reshape(-1, 3)  # [m]
    pa  = pd.GetPointData()

    def get_arr(name):
        a = pa.GetAbstractArray(name)
        return vtk_to_numpy(a).astype(np.float32) if a is not None else None

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]  # [m]
    T    = get_arr("Temperature")
    vmag = get_arr("Velocity")

    if T is None:
        print("  [WARN] No Temperature field in F3D VTK — using hard-coded GT values.")
        return F3D_GT.copy(), None, None, None, None

    hot = T > T_SOLIDUS
    n_hot = int(hot.sum())

    metrics = {}
    if n_hot == 0:
        metrics.update({"W_um": 0, "L_um": 0, "D_um": 0})
    else:
        x_um = x * 1e6
        y_um = y * 1e6
        z_um = z * 1e6

        laser_x_um = float(x_um[np.argmax(T)])
        at_laser = hot & (np.abs(x_um - laser_x_um) < 30)
        if at_laser.sum() > 0:
            metrics["W_um"] = float(y_um[at_laser].max() - y_um[at_laser].min())
        else:
            metrics["W_um"] = float(y_um[hot].max() - y_um[hot].min())
        metrics["L_um"] = float(x_um[hot].max() - x_um[hot].min())
        metrics["D_um"] = float(max(0.0, -z_um[hot].min()))

    metrics["T_max_K"]  = float(T.max())
    metrics["v_max_ms"] = float(vmag.max()) if vmag is not None else F3D_GT["v_max_ms"]

    return metrics, x * 1e6, y * 1e6, z * 1e6, T


# ---------------------------------------------------------------------------
# Latest VTK file selection helpers
# ---------------------------------------------------------------------------
def latest_lbm_vtk(directory, pattern):
    """Return path to the VTK file with the highest step number."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        return None
    return files[-1]  # sorted lexicographically — last = highest step


def latest_f3d_vtk(directory):
    """Return path to F3D VTK at frame 100 (≈ t=2000 μs, mid-steady-state)."""
    # Prefer frame 100; fall back to highest available
    preferred = os.path.join(directory, "150W-800mms-50um_100.vtk")
    if os.path.isfile(preferred):
        return preferred
    files = sorted(glob.glob(os.path.join(directory, "150W-800mms-50um_*.vtk")))
    return files[-1] if files else None


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
def print_table(legacy_m, plic_m, f3d_m):
    gt = f3d_m  # row used for relative-error denominator

    def rel(val, ref):
        if ref > 0:
            return (val - ref) / ref * 100.0
        return float("nan")

    rows = [
        ("Legacy (algebraic VOF)", legacy_m),
        ("PLIC (full stack)",       plic_m),
        ("F3D ground truth",        f3d_m),
    ]

    header = f"{'Run':<28} {'W[μm]':>8} {'L[μm]':>8} {'D[μm]':>8} {'T_max[K]':>10} {'v_max[m/s]':>12}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for label, m in rows:
        print(f"{label:<28} {m['W_um']:>8.1f} {m['L_um']:>8.1f} {m['D_um']:>8.1f} "
              f"{m['T_max_K']:>10.0f} {m['v_max_ms']:>12.3f}")
    print(sep)

    print(f"\n{'Relative error vs F3D':<28} {'W':>8} {'L':>8} {'D':>8} {'T_max':>10} {'v_max':>12}")
    print(sep)
    for label, m in rows[:2]:
        rw = rel(m["W_um"],      gt["W_um"])
        rl = rel(m["L_um"],      gt["L_um"])
        rd = rel(m["D_um"],      gt["D_um"])
        rt = rel(m["T_max_K"],   gt["T_max_K"])
        rv = rel(m["v_max_ms"],  gt["v_max_ms"])
        print(f"{label:<28} {rw:>+7.1f}% {rl:>+7.1f}% {rd:>+7.1f}% {rt:>+9.1f}% {rv:>+11.1f}%")
    print(sep + "\n")

    # Delta between PLIC and Legacy (improvement check)
    print("PLIC improvement over legacy (positive = closer to F3D):")
    for key, label in [("W_um","W"), ("L_um","L"), ("D_um","D")]:
        err_leg  = abs(legacy_m[key] - gt[key])
        err_plic = abs(plic_m[key]   - gt[key])
        delta = err_leg - err_plic
        direction = "better" if delta > 0 else "worse"
        print(f"  {label}: legacy err={err_leg:.1f} μm, plic err={err_plic:.1f} μm  "
              f"(Δ={delta:+.1f} μm  {direction})")
    print()


# ---------------------------------------------------------------------------
# Visualisation panels
# ---------------------------------------------------------------------------
def make_panels(legacy_data, plic_data, f3d_data, out_path):
    """3-panel figure: top-down (XY), side (XZ), transverse (YZ)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Melt pool: Legacy vs PLIC vs F3D  (f=0.5 / T>T_solidus isosurface)",
                 fontsize=10)

    panel_titles = ["Top-down (XY, deepest melt z)", "Side (XZ, centerline y)", "Transverse (YZ, laser x)"]

    def project_lbm(fill, lf, nx, ny, nz):
        """Return 2D projections of the melt mask."""
        mask = (fill > 0.5) & (lf > 0.5)
        mask_3d = mask.reshape(nz, ny, nx)
        xy = mask_3d.any(axis=0)     # (ny, nx) — top-down
        xz = mask_3d.any(axis=1)     # (nz, nx) — side
        yz = mask_3d.any(axis=2)     # (nz, ny) — transverse
        return xy, xz, yz

    colors = {"Legacy": "royalblue", "PLIC": "tomato", "F3D": "green"}
    alpha  = 0.35

    for run_name, data in [("Legacy", legacy_data), ("PLIC", plic_data)]:
        if data is None:
            continue
        metrics, fill, lf, temp, vel, nx, ny, nz, dx = data
        if fill is None:
            continue
        xy, xz, yz = project_lbm(fill, lf, nx, ny, nz)
        c = colors[run_name]
        # top-down: x horizontal, y vertical
        axes[0].contourf(np.arange(nx)*dx*1e6, np.arange(ny)*dx*1e6,
                         xy.astype(float), levels=[0.5, 1.5], colors=[c], alpha=alpha)
        axes[0].contour( np.arange(nx)*dx*1e6, np.arange(ny)*dx*1e6,
                         xy.astype(float), levels=[0.5], colors=[c], linewidths=1)
        # side: x horizontal, z vertical
        axes[1].contourf(np.arange(nx)*dx*1e6, np.arange(nz)*dx*1e6,
                         xz.astype(float), levels=[0.5, 1.5], colors=[c], alpha=alpha)
        axes[1].contour( np.arange(nx)*dx*1e6, np.arange(nz)*dx*1e6,
                         xz.astype(float), levels=[0.5], colors=[c], linewidths=1)
        # transverse: y horizontal, z vertical
        axes[2].contourf(np.arange(ny)*dx*1e6, np.arange(nz)*dx*1e6,
                         yz.astype(float), levels=[0.5, 1.5], colors=[c], alpha=alpha)
        axes[2].contour( np.arange(ny)*dx*1e6, np.arange(nz)*dx*1e6,
                         yz.astype(float), levels=[0.5], colors=[c], linewidths=1)

    # F3D scatter (surface points with T > T_solidus)
    if f3d_data is not None:
        _, x_um, y_um, z_um, T_f3d = f3d_data
        if x_um is not None and T_f3d is not None:
            hot = T_f3d > T_SOLIDUS
            c = colors["F3D"]
            # Offset F3D coordinates to roughly align scan start
            x_plot = x_um[hot]
            y_plot = y_um[hot]
            z_plot = z_um[hot]
            s = 0.2
            axes[0].scatter(x_plot, y_plot, s=s, c=c, alpha=0.5, label="F3D")
            axes[1].scatter(x_plot, z_plot, s=s, c=c, alpha=0.5)
            axes[2].scatter(y_plot, z_plot, s=s, c=c, alpha=0.5)

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors[k], label=k, alpha=0.7)
               for k in ["Legacy", "PLIC", "F3D"]]
    axes[0].legend(handles=patches, fontsize=7, loc="upper right")

    for ax, title in zip(axes, panel_titles):
        ax.set_title(title, fontsize=8)
        ax.set_aspect("auto")

    axes[0].set_xlabel("x [μm]"); axes[0].set_ylabel("y [μm]")
    axes[1].set_xlabel("x [μm]"); axes[1].set_ylabel("z [μm]")
    axes[2].set_xlabel("y [μm]"); axes[2].set_ylabel("z [μm]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved panel figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("\n=== PLIC vs Legacy vs F3D Comparison ===\n")

    # --- Load legacy LBM ---
    legacy_vtk = latest_lbm_vtk(args.legacy, "line_scan_*.vtk")
    legacy_data = None
    if legacy_vtk:
        print(f"Legacy VTK  : {legacy_vtk}")
        try:
            metrics, fill, lf, temp, vel, nx, ny, nz, dx = load_lbm_vtk(legacy_vtk)
            legacy_data = (metrics, fill, lf, temp, vel, nx, ny, nz, dx)
            legacy_m = metrics
            print(f"  W={legacy_m['W_um']:.1f} μm  L={legacy_m['L_um']:.1f} μm  "
                  f"D={legacy_m['D_um']:.1f} μm  T_max={legacy_m['T_max_K']:.0f} K  "
                  f"v_max={legacy_m['v_max_ms']:.3f} m/s")
        except Exception as e:
            print(f"  [ERROR] Failed to load legacy VTK: {e}")
            legacy_m = {k: 0.0 for k in F3D_GT}
    else:
        print(f"Legacy VTK  : NOT FOUND in {args.legacy} — using zeros")
        legacy_m = {k: 0.0 for k in F3D_GT}

    # --- Load PLIC LBM ---
    plic_vtk = latest_lbm_vtk(args.plic, "line_scan_plic_*.vtk")
    plic_data = None
    if plic_vtk:
        print(f"PLIC VTK    : {plic_vtk}")
        try:
            metrics, fill, lf, temp, vel, nx, ny, nz, dx = load_lbm_vtk(plic_vtk)
            plic_data = (metrics, fill, lf, temp, vel, nx, ny, nz, dx)
            plic_m = metrics
            print(f"  W={plic_m['W_um']:.1f} μm  L={plic_m['L_um']:.1f} μm  "
                  f"D={plic_m['D_um']:.1f} μm  T_max={plic_m['T_max_K']:.0f} K  "
                  f"v_max={plic_m['v_max_ms']:.3f} m/s")
        except Exception as e:
            print(f"  [ERROR] Failed to load PLIC VTK: {e}")
            plic_m = {k: 0.0 for k in F3D_GT}
    else:
        print(f"PLIC VTK    : NOT FOUND in {args.plic} — using zeros")
        plic_m = {k: 0.0 for k in F3D_GT}

    # --- Load F3D reference ---
    f3d_vtk = latest_f3d_vtk(args.f3d)
    f3d_data = None
    if f3d_vtk:
        print(f"F3D VTK     : {f3d_vtk}")
        try:
            metrics, x_um, y_um, z_um, T_f3d = load_f3d_vtk(f3d_vtk)
            f3d_data = (metrics, x_um, y_um, z_um, T_f3d)
            f3d_m = metrics
            print(f"  W={f3d_m['W_um']:.1f} μm  L={f3d_m['L_um']:.1f} μm  "
                  f"D={f3d_m['D_um']:.1f} μm  T_max={f3d_m['T_max_K']:.0f} K")
        except Exception as e:
            print(f"  [WARN] Failed to load F3D VTK ({e}); using hard-coded GT.")
            f3d_m = F3D_GT.copy()
    else:
        print(f"F3D VTK     : NOT FOUND in {args.f3d} — using hard-coded ground-truth")
        f3d_m = F3D_GT.copy()

    # --- Print comparison table ---
    print_table(legacy_m, plic_m, f3d_m)

    # --- Visualisation ---
    make_panels(legacy_data, plic_data, f3d_data, args.out)


if __name__ == "__main__":
    main()
