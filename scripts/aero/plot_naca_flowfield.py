"""Render z-mid vorticity + |u| around the NACA airfoil.

Modes:
  single:  python plot_naca_flowfield.py <vtk> <out.png> [mask_txt]
  compare: python plot_naca_flowfield.py --compare <vtk_a>:<title_a> <vtk_b>:<title_b> <out.png> [mask_txt]

VTK velocities are in lattice units (LU); freestream u_LU ≈ 0.05.
"""
from __future__ import annotations
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_vtk_structured(path):
    with open(path) as f:
        text = f.read()
    nx, ny, nz = None, None, None
    dx, dy, dz = 1.0, 1.0, 1.0
    for line in text.splitlines():
        if line.startswith('DIMENSIONS'):
            parts = line.split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
        elif line.startswith('SPACING'):
            parts = line.split()
            dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
        elif line.startswith('VECTORS'):
            break
    idx = text.find('VECTORS velocity float')
    body = text[idx:].split('\n', 1)[1]
    vals = np.fromstring(body, sep=' ')
    n = nx * ny * nz
    arr = vals[:3*n].reshape(nz, ny, nx, 3)
    return arr, (nx, ny, nz), (dx, dy, dz)

def parse_mask_zmid(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    return np.array([[int(v) for v in l.split()] for l in lines if l.strip()])

def compute_fields(arr, dx):
    kmid = arr.shape[0] // 2
    ux = arr[kmid, :, :, 0]
    uy = arr[kmid, :, :, 1]
    # vorticity ω_z = ∂u_y/∂x - ∂u_x/∂y, in LU/cell (one-step dt=1)
    duy_dx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / 2
    dux_dy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / 2
    omega = duy_dx - dux_dy
    umag = np.sqrt(ux**2 + uy**2)
    return ux, uy, omega, umag

def draw_panel(ax, field, extent, mask, vmin=None, vmax=None, cmap='RdBu_r',
               i_lo=0, j_lo=0):
    im = ax.imshow(field, extent=extent, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='equal')
    if mask is not None:
        # crop mask to same window
        mc = mask[j_lo:j_lo+field.shape[0], i_lo:i_lo+field.shape[1]]
        # Draw contour of mask boundary
        ax.contour(np.linspace(extent[0], extent[1], mc.shape[1]),
                   np.linspace(extent[2], extent[3], mc.shape[0]),
                   mc, levels=[0.5], colors='k', linewidths=1.0)
    return im

def render_one(vtk_path, mask_path, fig, gs_row, title=None, le_x=10.0):
    arr, (nx, ny, nz), (dx, _, _) = parse_vtk_structured(vtk_path)
    ux, uy, omega, umag = compute_fields(arr, dx)
    mask = parse_mask_zmid(mask_path) if mask_path else None

    # window: 2c upstream / 5c downstream / ±3c vertical around LE
    chord = 1.0
    le_y = ny * dx / 2 - (ny * dx) / 2 + ny * dx / 2  # placeholder; use mid
    # Better: LE position from driver default = (10, ny*dx/2)
    le_y = ny * dx / 2
    x_lo = le_x - 2 * chord
    x_hi = le_x + 5 * chord
    y_lo = le_y - 3 * chord
    y_hi = le_y + 3 * chord
    i_lo = max(0, int(x_lo / dx))
    i_hi = min(nx, int(x_hi / dx))
    j_lo = max(0, int(y_lo / dx))
    j_hi = min(ny, int(y_hi / dx))

    om_w = omega[j_lo:j_hi, i_lo:i_hi]
    um_w = umag [j_lo:j_hi, i_lo:i_hi]
    extent = (i_lo*dx, (i_hi-1)*dx, j_lo*dx, (j_hi-1)*dx)
    vmax_om = max(np.percentile(np.abs(om_w), 99.0), 1e-4)
    vmax_um = max(np.percentile(um_w, 99.5), 1e-4)

    ax_w = fig.add_subplot(gs_row[0])
    ax_v = fig.add_subplot(gs_row[1])
    im1 = draw_panel(ax_w, om_w, extent, mask, vmin=-vmax_om, vmax=vmax_om,
                     cmap='RdBu_r', i_lo=i_lo, j_lo=j_lo)
    ax_w.set_title(f'{title}  —  vorticity ω_z (LU)')
    plt.colorbar(im1, ax=ax_w, shrink=0.7)
    im2 = draw_panel(ax_v, um_w, extent, mask, vmin=0, vmax=vmax_um,
                     cmap='viridis', i_lo=i_lo, j_lo=j_lo)
    ax_v.set_title(f'|u| LU  (freestream ≈ 0.05)')
    plt.colorbar(im2, ax=ax_v, shrink=0.7)
    ax_w.plot(le_x, le_y, 'kx', ms=8)
    ax_v.plot(le_x, le_y, 'kx', ms=8)
    ax_w.set_ylabel('y [m]')
    ax_v.set_ylabel('y [m]')
    ax_v.set_xlabel('x [m]')

if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--compare':
        a_arg, b_arg, out_png, *rest = args[1:]
        mask_path = rest[0] if rest else None
        vtk_a, title_a = a_arg.split(':', 1)
        vtk_b, title_b = b_arg.split(':', 1)
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(4, 1, figure=fig, hspace=0.35)
        render_one(vtk_a, mask_path, fig, (fig.add_subplot, fig.add_subplot), title=title_a)
        plt.close(fig)
        # Use a different layout: 2x2
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        for col, (vtk, title) in enumerate([(vtk_a, title_a), (vtk_b, title_b)]):
            arr, (nx, ny, nz), (dx, _, _) = parse_vtk_structured(vtk)
            ux, uy, omega, umag = compute_fields(arr, dx)
            mask = parse_mask_zmid(mask_path) if (mask_path and col==0) else None
            le_x = 10.0
            le_y = ny * dx / 2
            chord = 1.0
            x_lo, x_hi = le_x-2, le_x+5
            y_lo, y_hi = le_y-3, le_y+3
            i_lo = max(0, int(x_lo/dx)); i_hi = min(nx, int(x_hi/dx))
            j_lo = max(0, int(y_lo/dx)); j_hi = min(ny, int(y_hi/dx))
            om_w = omega[j_lo:j_hi, i_lo:i_hi]
            um_w = umag [j_lo:j_hi, i_lo:i_hi]
            extent = (i_lo*dx, (i_hi-1)*dx, j_lo*dx, (j_hi-1)*dx)
            vmax_om = max(np.percentile(np.abs(om_w), 99.0), 1e-4)
            vmax_um = max(np.percentile(um_w, 99.5), 1e-4)
            im1 = axes[0,col].imshow(om_w, extent=extent, origin='lower',
                cmap='RdBu_r', vmin=-vmax_om, vmax=vmax_om, aspect='equal')
            axes[0,col].set_title(f'{title}\nvorticity ω_z')
            plt.colorbar(im1, ax=axes[0,col], shrink=0.7)
            im2 = axes[1,col].imshow(um_w, extent=extent, origin='lower',
                cmap='viridis', vmin=0, vmax=vmax_um, aspect='equal')
            axes[1,col].set_title('|u| LU (freestream ≈ 0.05)')
            plt.colorbar(im2, ax=axes[1,col], shrink=0.7)
            axes[0,col].plot(le_x, le_y, 'kx', ms=8, label='LE')
            axes[1,col].plot(le_x, le_y, 'kx', ms=8)
            axes[0,col].legend()
            axes[1,col].set_xlabel('x [m]')
            axes[0,col].set_ylabel('y [m]')
            axes[1,col].set_ylabel('y [m]')
        plt.tight_layout()
        plt.savefig(out_png, dpi=110)
        print(f'Saved: {out_png}')
    else:
        vtk = args[0]; out = args[1]
        mask_path = args[2] if len(args) > 2 else None
        from matplotlib.gridspec import GridSpec
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))
        arr, (nx, ny, nz), (dx, _, _) = parse_vtk_structured(vtk)
        ux, uy, omega, umag = compute_fields(arr, dx)
        mask = parse_mask_zmid(mask_path) if mask_path else None
        le_x = 10.0
        le_y = ny * dx / 2
        x_lo, x_hi = le_x-2, le_x+5
        y_lo, y_hi = le_y-3, le_y+3
        i_lo = max(0, int(x_lo/dx)); i_hi = min(nx, int(x_hi/dx))
        j_lo = max(0, int(y_lo/dx)); j_hi = min(ny, int(y_hi/dx))
        om_w = omega[j_lo:j_hi, i_lo:i_hi]
        um_w = umag [j_lo:j_hi, i_lo:i_hi]
        extent = (i_lo*dx, (i_hi-1)*dx, j_lo*dx, (j_hi-1)*dx)
        vmax_om = max(np.percentile(np.abs(om_w), 99.0), 1e-4)
        vmax_um = max(np.percentile(um_w, 99.5), 1e-4)
        im1 = axes[0].imshow(om_w, extent=extent, origin='lower',
            cmap='RdBu_r', vmin=-vmax_om, vmax=vmax_om, aspect='equal')
        axes[0].set_title(f'vorticity ω_z (LU,  max abs={vmax_om:.4f})')
        plt.colorbar(im1, ax=axes[0], shrink=0.7)
        im2 = axes[1].imshow(um_w, extent=extent, origin='lower',
            cmap='viridis', vmin=0, vmax=vmax_um, aspect='equal')
        axes[1].set_title(f'|u| LU  (max≈{vmax_um:.4f},  freestream≈0.05)')
        plt.colorbar(im2, ax=axes[1], shrink=0.7)
        if mask is not None:
            for ax in axes:
                mc = mask[j_lo:j_hi, i_lo:i_hi]
                ax.contour(np.linspace(extent[0], extent[1], mc.shape[1]),
                           np.linspace(extent[2], extent[3], mc.shape[0]),
                           mc, levels=[0.5], colors='k', linewidths=1.0)
        for ax in axes:
            ax.plot(le_x, le_y, 'kx', ms=8)
            ax.set_ylabel('y [m]')
        axes[1].set_xlabel('x [m]')
        fig.suptitle(os.path.basename(vtk), fontsize=10)
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print(f'Saved: {out}')
