#!/usr/bin/env python3
"""
Extract reference metrics from the F3D gold-standard VTK dataset for LBM-PLIC vs F3D comparison.

Dataset: 316L LPBF line scan — P=150 W, v=800 mm/s, powder layer 50 um.
         Calibrated Flow3D solver (Fresnel ray-tracing, keyhole mode).
Source:  /home/yzk/LBMProject/vtk-316L-150W-50um-V800mms/

Files are free-surface POLYDATA (VTK 5.1 ASCII), where every point sits on the
f=0.5 isosurface that Flow3D tracks.  Fields stored as POINT_DATA:
  Temperature [K], Velocity [m/s magnitude], X/Y/Z-velocity [m/s].

Methodology:
  - Read the highest-numbered snapshot (snap 100 = t=2.0 ms, quasi-steady state).
  - Pool geometry: bounding box of T > T_SOLIDUS surface points.
  - D_open: z-range of T > T_SOLIDUS points within 50 um of the laser axis.
  - Δz_far: median z of solidified centerline points at x = 150-330 um
            (scan-start transient decayed, laser not yet reached).
  - v_max surface: v_max restricted to z > z_substrate - 15 um band.

Outputs:
  - JSON: <dataset>/f3d_reference_metrics.json
  - PNG:  <dataset>/f3d_preview.png  (3-panel: top, side, transverse)
"""

import os
import sys
import json
import glob
import re
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================
DATASET_DIR   = "/home/yzk/LBMProject/vtk-316L-150W-50um-V800mms"
OUTPUT_JSON   = os.path.join(DATASET_DIR, "f3d_reference_metrics.json")
OUTPUT_PNG    = os.path.join(DATASET_DIR, "f3d_preview.png")

T_SOLIDUS     = 1674.15    # K  (316L solidus, matches prepin material)
T_LIQUIDUS    = 1697.15    # K  (316L liquidus)
V_SCAN        = 0.80       # m/s (scan speed along +x)
DT_VTK        = 20.0e-6   # s between snapshots (2 ms / 100 intervals)

# Geometric thresholds (physical units)
Z_SUBSTRATE_UM = 0.0       # nominal substrate top; computed from flat far-field
KEYHOLE_RADIUS_UM = 50.0   # search radius around laser axis for D_open measurement
TRAIL_X_MIN_UM = 150.0     # start of steady far-trail window (past scan-start splash)
TRAIL_X_MAX_UM = 330.0     # end of steady far-trail window
TRAIL_Y_MAX_UM = 30.0      # half-width of centerline band for Δz measurement
SURFACE_BAND_BELOW_UM = 15.0  # z > z_substrate - this = "surface band" for v_max_surf


def find_latest_vtk(directory):
    """Return path to the highest-numbered VTK file in directory."""
    pattern = os.path.join(directory, "*.vtk")
    files = glob.glob(pattern)
    if not files:
        return None

    def snap_number(p):
        m = re.search(r'_(\d+)\.vtk$', os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(files, key=snap_number)


def load_surface_polydata(vtk_path):
    """Read a Flow3D free-surface VTK (POLYDATA ASCII) via pyvista.

    Returns dict with numpy arrays: pts (N,3) in metres, T (N,), V (N,).
    Raises RuntimeError if expected fields are missing.
    """
    mesh = pv.read(vtk_path)
    pts = np.asarray(mesh.points)           # metres

    # Field names contain URL-encoded spaces in older F3D exports
    available = mesh.point_data.keys()
    def get_field(candidates):
        for c in candidates:
            if c in available:
                return np.asarray(mesh.point_data[c])
        return None

    T = get_field(["Temperature", "temperature"])
    V = get_field(["Velocity", "velocity"])
    Vx = get_field(["X-velocity", "X velocity"])
    Vy = get_field(["Y-velocity", "Y velocity"])
    Vz = get_field(["Z-velocity", "Z velocity"])

    if T is None:
        raise RuntimeError(f"Temperature field not found in {vtk_path}. "
                           f"Available: {list(available)}")
    if V is None and Vx is not None:
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    if V is None:
        raise RuntimeError(f"Velocity field not found in {vtk_path}. "
                           f"Available: {list(available)}")

    print(f"Loaded {os.path.basename(vtk_path)}: {len(pts):,} surface points")
    print(f"  T range: [{T.min():.1f}, {T.max():.1f}] K")
    print(f"  V range: [{V.min():.4f}, {V.max():.4f}] m/s")
    return {"pts": pts, "T": T, "V": V}


def compute_substrate_level(pts_um, T, z_col):
    """Estimate undisturbed substrate top z [um] from cold flat surface points."""
    cold_flat = (T < 400.0) & (z_col > -5.0) & (z_col < 10.0)
    if cold_flat.sum() < 100:
        print("  WARNING: few cold-flat points, defaulting z_substrate = 0.0 um")
        return 0.0
    z_sub = float(np.median(z_col[cold_flat]))
    print(f"  z_substrate estimated from {cold_flat.sum():,} cold flat points: {z_sub:.3f} um")
    return z_sub


def extract_metrics(data, z_substrate_um, snap_idx, t_sec):
    """Compute all reference metrics from the surface polydata."""
    pts  = data["pts"]
    T    = data["T"]
    V    = data["V"]

    x = pts[:, 0] * 1e6   # um
    y = pts[:, 1] * 1e6
    z = pts[:, 2] * 1e6

    laser_x_um = V_SCAN * t_sec * 1e6

    # ----------------------------------------------------------
    # T_max and its location
    # ----------------------------------------------------------
    i_tmax = int(np.argmax(T))
    T_max_K = float(T[i_tmax])
    T_max_loc_um = (float(x[i_tmax]), float(y[i_tmax]), float(z[i_tmax]))

    # Surface temperature at laser centroid (highest T within 20 um of laser axis)
    laser_spot = (np.abs(x - laser_x_um) < 20.0) & (np.abs(y) < 20.0)
    T_surf_laser_K = float(T[laser_spot].max()) if laser_spot.sum() > 0 else float("nan")

    # ----------------------------------------------------------
    # Pool geometry: bounding box of T > T_SOLIDUS points
    # ----------------------------------------------------------
    melt = T > T_SOLIDUS
    if melt.sum() < 10:
        print("  WARNING: fewer than 10 molten points found.")
        pool_L_um = pool_W_um = pool_D_melt_um = pool_D_open_um = 0.0
    else:
        mx, my, mz = x[melt], y[melt], z[melt]
        pool_L_um      = float(mx.max() - mx.min())
        pool_W_um      = float(my.max() - my.min())
        pool_D_melt_um = float(z_substrate_um - mz.min())   # depth from substrate top

        # D_open: z-span of molten region directly under the laser (narrow column)
        kh = melt & (np.abs(x - laser_x_um) < KEYHOLE_RADIUS_UM) & \
                    (np.abs(y) < KEYHOLE_RADIUS_UM)
        pool_D_open_um = float(z[kh].max() - z[kh].min()) if kh.sum() > 5 else float("nan")

    # ----------------------------------------------------------
    # Kinematic metrics
    # ----------------------------------------------------------
    v_max_global_ms = float(V.max())
    i_vmax = int(np.argmax(V))
    v_max_loc_um = (float(x[i_vmax]), float(y[i_vmax]), float(z[i_vmax]))

    # v_max in top surface band (top 3 cells at dx=5 um = 15 um)
    surf_band = z > (z_substrate_um - SURFACE_BAND_BELOW_UM)
    v_max_surf_ms = float(V[surf_band].max()) if surf_band.sum() > 0 else float("nan")

    # ----------------------------------------------------------
    # Δz_far: steady-state solidified-track elevation
    # Measure median z of the highest surface points in each 10-um x-bin
    # within the steady far-trail window [TRAIL_X_MIN, TRAIL_X_MAX].
    # Using the z_max-per-bin approach isolates the track top surface from
    # domain-wall panels that sit at z = -97.5 um in this POLYDATA.
    # ----------------------------------------------------------
    dz_far_samples = []
    for xb in np.arange(TRAIL_X_MIN_UM, TRAIL_X_MAX_UM, 10.0):
        col = (x > xb) & (x < xb + 10.0) & (np.abs(y) < TRAIL_Y_MAX_UM)
        if col.sum() > 0:
            z_top = float(z[col].max())
            dz_far_samples.append(z_top - z_substrate_um)

    dz_far_um = float(np.median(dz_far_samples)) if dz_far_samples else float("nan")

    metrics = {
        "_source_vtk":        os.path.basename(vtk_path),
        "_snap_index":        snap_idx,
        "_t_ms":              round(t_sec * 1e3, 3),
        "_laser_x_um":        round(laser_x_um, 1),
        "_z_substrate_um":    round(z_substrate_um, 3),
        "_methodology":       (
            "All geometry from T>T_SOLIDUS bounding box on f=0.5 isosurface polydata. "
            "D_open = z-span of molten points within 50um of laser axis. "
            "dz_far = median(z_top_per_10um_bin - z_substrate) over x=150-330um centerline."
        ),

        # Pool geometry
        "pool_L_um":          round(pool_L_um, 1),
        "pool_W_um":          round(pool_W_um, 1),
        "pool_D_melt_um":     round(pool_D_melt_um, 1),
        "pool_D_open_um":     round(pool_D_open_um, 1) if not np.isnan(pool_D_open_um) else None,
        "pool_DW_ratio":      round(pool_D_melt_um / pool_W_um, 3) if pool_W_um > 0 else None,

        # Steady-state track elevation
        "dz_far_um":          round(dz_far_um, 2) if not np.isnan(dz_far_um) else None,

        # Thermal metrics
        "T_max_K":            round(T_max_K, 1),
        "T_max_loc_um":       [round(v, 1) for v in T_max_loc_um],
        "T_surf_laser_K":     round(T_surf_laser_K, 1) if not np.isnan(T_surf_laser_K) else None,
        "T_solidus_K":        T_SOLIDUS,
        "T_liquidus_K":       T_LIQUIDUS,

        # Kinematic metrics
        "v_max_ms":           round(v_max_global_ms, 3),
        "v_max_loc_um":       [round(v, 1) for v in v_max_loc_um],
        "v_max_surface_ms":   round(v_max_surf_ms, 3) if not np.isnan(v_max_surf_ms) else None,

        # Units note
        "_units": {
            "pool_L_um":        "micrometres",
            "pool_W_um":        "micrometres",
            "pool_D_melt_um":   "micrometres (depth from substrate top)",
            "pool_D_open_um":   "micrometres (z-span of molten region at laser axis)",
            "dz_far_um":        "micrometres (solidified track height above substrate)",
            "T_*":              "Kelvin",
            "v_*":              "m/s",
        },
    }
    return metrics


def make_preview_png(data, metrics, z_substrate_um, out_path):
    """3-panel preview: top-down, side (x-z), transverse (y-z) at laser position."""
    pts = data["pts"]
    T   = data["T"]
    x = pts[:, 0] * 1e6
    y = pts[:, 1] * 1e6
    z = pts[:, 2] * 1e6
    laser_x = metrics["_laser_x_um"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = "inferno"
    vmin, vmax = 300, 4200

    # Panel 1: top-down (x-y), near-surface points
    ax = axes[0]
    top = z > (z_substrate_um - 10.0)
    sc = ax.scatter(x[top], y[top], c=T[top], s=0.5, cmap=cmap, vmin=vmin, vmax=vmax,
                    rasterized=True)
    # overlay T > T_SOLIDUS contour
    melt_top = top & (T > T_SOLIDUS)
    if melt_top.sum() > 0:
        ax.scatter(x[melt_top], y[melt_top], s=1, c="cyan", alpha=0.4, label="T>T_sol")
    ax.axvline(laser_x, color="lime", lw=1.2, ls="--", label=f"laser x={laser_x:.0f}um")
    ax.set_xlim(max(-100, x.min()), min(laser_x + 250, x.max()))
    ax.set_ylim(-120, 120)
    ax.set_xlabel("x [um]  (scan direction)")
    ax.set_ylabel("y [um]")
    ax.set_title("Top-down (z near surface)")
    ax.legend(fontsize=7, markerscale=3)
    ax.set_aspect("equal")

    # Panel 2: side view (x-z), centerline slab |y| < 15 um
    ax = axes[1]
    yslab = np.abs(y) < 15.0
    sc = ax.scatter(x[yslab], z[yslab], c=T[yslab], s=0.8, cmap=cmap,
                    vmin=vmin, vmax=vmax, rasterized=True)
    melt_yslab = yslab & (T > T_SOLIDUS)
    if melt_yslab.sum() > 0:
        ax.scatter(x[melt_yslab], z[melt_yslab], s=1, c="cyan", alpha=0.4)
    ax.axhline(z_substrate_um, color="gray", lw=0.8, ls=":", alpha=0.7, label="z_substrate")
    ax.axvline(laser_x, color="lime", lw=1.2, ls="--")
    ax.set_xlim(max(-100, x.min()), min(laser_x + 200, x.max()))
    ax.set_ylim(-160, 30)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("z [um]")
    ax.set_title("Side view |y|<15 um (keyhole + trail)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # Panel 3: transverse (y-z) at laser x ± 20 um
    ax = axes[2]
    xslab = np.abs(x - laser_x) < 20.0
    if xslab.sum() > 0:
        sc = ax.scatter(y[xslab], z[xslab], c=T[xslab], s=1.5, cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)
        melt_xslab = xslab & (T > T_SOLIDUS)
        if melt_xslab.sum() > 0:
            ax.scatter(y[melt_xslab], z[melt_xslab], s=2, c="cyan", alpha=0.5,
                       label="T>T_sol")
    ax.axhline(z_substrate_um, color="gray", lw=0.8, ls=":", alpha=0.7, label="z_substrate")
    ax.axvline(0, color="lime", lw=1.2, ls="--", label="y=0")
    ax.set_xlim(-120, 120)
    ax.set_ylim(-160, 30)
    ax.set_xlabel("y [um]")
    ax.set_ylabel("z [um]")
    ax.set_title(f"Transverse @ x_laser={laser_x:.0f} um")
    ax.legend(fontsize=7, markerscale=2)
    ax.set_aspect("equal")

    fig.colorbar(sc, ax=axes, label="T [K]", shrink=0.6, pad=0.01)
    fig.suptitle(
        f"F3D reference — 316L, P=150 W, v=800 mm/s, snap={metrics['_snap_index']} "
        f"(t={metrics['_t_ms']:.1f} ms)\n"
        f"L={metrics['pool_L_um']:.0f} um  W={metrics['pool_W_um']:.0f} um  "
        f"D_melt={metrics['pool_D_melt_um']:.0f} um  D_open={metrics['pool_D_open_um']} um  "
        f"T_max={metrics['T_max_K']:.0f} K  v_max={metrics['v_max_ms']:.2f} m/s  "
        f"Dz_far={metrics['dz_far_um']} um",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1.0, 0.93])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview PNG written: {out_path}")


# ============================================================
# MAIN
# ============================================================
if not os.path.isdir(DATASET_DIR):
    print(f"ERROR: dataset directory not found: {DATASET_DIR}", file=sys.stderr)
    sys.exit(1)

vtk_path = find_latest_vtk(DATASET_DIR)
if vtk_path is None:
    print(f"ERROR: no .vtk files found in {DATASET_DIR}", file=sys.stderr)
    sys.exit(1)

print(f"Using latest VTK: {vtk_path}")
m = re.search(r'_(\d+)\.vtk$', os.path.basename(vtk_path))
snap_idx = int(m.group(1)) if m else 0
t_sec = snap_idx * DT_VTK
print(f"Snapshot index: {snap_idx}  ->  t = {t_sec*1e3:.2f} ms")

data = load_surface_polydata(vtk_path)
x_um = data["pts"][:, 0] * 1e6
y_um = data["pts"][:, 1] * 1e6
z_um = data["pts"][:, 2] * 1e6

z_substrate_um = compute_substrate_level(data["pts"] * 1e6, data["T"], z_um)

print("\nComputing metrics...")
metrics = extract_metrics(data, z_substrate_um, snap_idx, t_sec)

print("\n=== F3D REFERENCE METRICS ===")
keys_to_print = [
    "pool_L_um", "pool_W_um", "pool_D_melt_um", "pool_D_open_um", "pool_DW_ratio",
    "dz_far_um",
    "T_max_K", "T_max_loc_um", "T_surf_laser_K",
    "v_max_ms", "v_max_loc_um", "v_max_surface_ms",
]
for k in keys_to_print:
    print(f"  {k:25s}: {metrics[k]}")

with open(OUTPUT_JSON, "w") as fh:
    json.dump(metrics, fh, indent=2)
print(f"\nJSON written: {OUTPUT_JSON}")

print("\nGenerating preview PNG...")
make_preview_png(data, metrics, z_substrate_um, OUTPUT_PNG)

print("\nDone.")
