#!/usr/bin/env python3
"""
Render flow snapshots from aero_schaefer_turek_3d VTK output.

For each snap_*.vtk:
  - extract velocity (u,v) on a z-mid plane
  - compute |u| and vorticity ω_z = ∂v/∂x - ∂u/∂y
  - draw 2 subplots: speed contour + vorticity contour with cylinder overlay

Outputs:
  <dir>/viz_flow.png    montage of N timesteps
  <dir>/viz_last.png    just the last snapshot zoomed near cylinder

Usage:
    python3 viz_flow.py <output_dir>
"""

import argparse
import os
import sys
import re
import glob

import numpy as np


def read_vtk_image(path):
    """Read a legacy STRUCTURED_POINTS VTK with VECTORS data, return
    (origin, spacing, dims, vectors[nx,ny,nz,3]). Minimal parser — just enough
    for the format aero_schaefer_turek_3d emits via VTKWriter::writeVectorField.
    """
    with open(path, "rb") as f:
        data = f.read()
    # Header is text; data may be ascii or binary depending on writer
    # Find "DIMENSIONS X Y Z"
    text = data.decode("latin-1", errors="ignore")
    m = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if not m:
        raise ValueError(f"DIMENSIONS not found in {path}")
    nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"ORIGIN\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", text)
    ox, oy, oz = (float(m.group(i + 1)) for i in range(3)) if m else (0.0, 0.0, 0.0)
    m = re.search(r"SPACING\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", text)
    dx, dy, dz = (float(m.group(i + 1)) for i in range(3)) if m else (1.0, 1.0, 1.0)

    # Find VECTORS section (ascii expected)
    m = re.search(r"VECTORS\s+\S+\s+(\w+)\s*\n", text)
    if not m:
        raise ValueError(f"VECTORS not found in {path}")
    dtype = m.group(1).lower()
    body = text[m.end():]
    n = nx * ny * nz
    if dtype in ("float", "double"):
        vals = np.fromstring(body, sep=" ", count=n * 3, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported VECTORS dtype: {dtype}")
    vels = vals.reshape((nz, ny, nx, 3))  # VTK convention: k slow, then j, then i
    return (ox, oy, oz), (dx, dy, dz), (nx, ny, nz), vels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir")
    ap.add_argument("--D", type=float, default=0.1)
    ap.add_argument("--cx", type=float, default=0.2)
    ap.add_argument("--cy", type=float, default=0.2)
    ap.add_argument("--ly", type=float, default=0.41)
    ap.add_argument("--max-snaps", type=int, default=9,
                    help="Max snapshots in montage (default 9 = 3x3 grid)")
    args = ap.parse_args()

    snap_paths = sorted(glob.glob(os.path.join(args.output_dir, "snap_*.vtk")))
    if not snap_paths:
        print(f"No snap_*.vtk in {args.output_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(snap_paths)} snapshots; rendering up to {args.max_snaps}")

    if len(snap_paths) > args.max_snaps:
        idx = np.linspace(0, len(snap_paths) - 1, args.max_snaps, dtype=int)
        snap_paths = [snap_paths[i] for i in idx]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n = len(snap_paths)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    # Two figures: speed and vorticity
    fig_s, axs_s = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 1.6))
    fig_v, axs_v = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 1.6))
    axs_s = np.atleast_2d(axs_s).reshape(-1)
    axs_v = np.atleast_2d(axs_v).reshape(-1)

    # Get spacing/extents from first file for consistent colorbar
    (ox0, oy0, _), (dx0, dy0, _), (nx0, ny0, nz0), v0 = read_vtk_image(snap_paths[0])
    extent = [ox0, ox0 + nx0 * dx0, oy0, oy0 + ny0 * dy0]

    # Two-pass for global vmax (speed) and vorticity range
    vmax_speed = 0.0
    vmax_vort  = 0.0
    cached = []
    for p in snap_paths:
        (ox, oy, _), (dx, dy, _), (nx, ny, nz), vels = read_vtk_image(p)
        u = vels[nz // 2, :, :, 0]   # z-mid plane
        v = vels[nz // 2, :, :, 1]
        speed = np.sqrt(u * u + v * v)
        # vorticity: ω_z = dv/dx - du/dy. Use np.gradient.
        dv_dx = np.gradient(v, dx, axis=1)
        du_dy = np.gradient(u, dy, axis=0)
        omega = dv_dx - du_dy
        cached.append((p, u, v, speed, omega))
        vmax_speed = max(vmax_speed, float(np.nanpercentile(speed, 99.5)))
        vmax_vort = max(vmax_vort, float(np.nanpercentile(np.abs(omega), 99.0)))

    # Render
    for idx, (p, u, v, speed, omega) in enumerate(cached):
        ax_s = axs_s[idx]
        ax_v = axs_v[idx]
        step_str = re.search(r"snap_(\d+)\.vtk$", p).group(1)
        im_s = ax_s.imshow(speed, origin="lower", extent=extent, aspect="equal",
                           cmap="viridis", vmin=0, vmax=vmax_speed)
        ax_s.add_patch(Circle((args.cx, args.cy), 0.5 * args.D, fill=True,
                              facecolor="white", edgecolor="black", linewidth=0.6))
        ax_s.set_title(f"|u|  step={step_str}", fontsize=8)
        ax_s.set_xticks([]); ax_s.set_yticks([])

        im_v = ax_v.imshow(omega, origin="lower", extent=extent, aspect="equal",
                           cmap="RdBu_r", vmin=-vmax_vort, vmax=vmax_vort)
        ax_v.add_patch(Circle((args.cx, args.cy), 0.5 * args.D, fill=True,
                              facecolor="white", edgecolor="black", linewidth=0.6))
        ax_v.set_title(f"ω_z  step={step_str}", fontsize=8)
        ax_v.set_xticks([]); ax_v.set_yticks([])

    # Hide unused subplots
    for k in range(n, len(axs_s)):
        axs_s[k].axis("off"); axs_v[k].axis("off")

    fig_s.suptitle(f"Speed |u| montage — {os.path.basename(args.output_dir)}", fontsize=10)
    fig_v.suptitle(f"Vorticity ω_z montage — {os.path.basename(args.output_dir)}", fontsize=10)
    fig_s.tight_layout()
    fig_v.tight_layout()
    speed_path = os.path.join(args.output_dir, "viz_speed.png")
    vort_path  = os.path.join(args.output_dir, "viz_vorticity.png")
    fig_s.savefig(speed_path, dpi=120)
    fig_v.savefig(vort_path, dpi=120)
    print(f"speed:     {speed_path}")
    print(f"vorticity: {vort_path}")

    # Last-snap zoom (near cylinder)
    p, u, v, speed, omega = cached[-1]
    fig_z, axs_z = plt.subplots(1, 2, figsize=(13, 4))
    zoom_extent = [args.cx - 2 * args.D, args.cx + 6 * args.D,
                   args.cy - 2 * args.D, args.cy + 2 * args.D]
    im0 = axs_z[0].imshow(speed, origin="lower", extent=extent, aspect="equal",
                          cmap="viridis", vmin=0, vmax=vmax_speed)
    axs_z[0].add_patch(Circle((args.cx, args.cy), 0.5 * args.D, fill=True,
                              facecolor="white", edgecolor="black", linewidth=1))
    axs_z[0].set_xlim(zoom_extent[0], zoom_extent[1])
    axs_z[0].set_ylim(zoom_extent[2], zoom_extent[3])
    axs_z[0].set_title("|u| zoom (last snap)")
    plt.colorbar(im0, ax=axs_z[0], shrink=0.8)

    im1 = axs_z[1].imshow(omega, origin="lower", extent=extent, aspect="equal",
                          cmap="RdBu_r", vmin=-vmax_vort, vmax=vmax_vort)
    axs_z[1].add_patch(Circle((args.cx, args.cy), 0.5 * args.D, fill=True,
                              facecolor="white", edgecolor="black", linewidth=1))
    axs_z[1].set_xlim(zoom_extent[0], zoom_extent[1])
    axs_z[1].set_ylim(zoom_extent[2], zoom_extent[3])
    axs_z[1].set_title("ω_z zoom (last snap)")
    plt.colorbar(im1, ax=axs_z[1], shrink=0.8)

    fig_z.tight_layout()
    zoom_path = os.path.join(args.output_dir, "viz_last_zoom.png")
    fig_z.savefig(zoom_path, dpi=140)
    print(f"zoom:      {zoom_path}")


if __name__ == "__main__":
    main()
