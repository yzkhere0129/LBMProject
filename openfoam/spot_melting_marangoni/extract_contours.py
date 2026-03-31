#!/usr/bin/env python3
"""
Spot Melting Benchmark v1.1 — Contour Extraction & Coordinate Transform
========================================================================
Extracts T=1650 K (solidus) isotherms from OpenFOAM results on the z=0 plane,
converts to LBM coordinate system, and exports to CSV.

Coordinate mapping (OpenFOAM → LBM):
    X_lbm [µm] = x_of [m] × 1e6 + 100
    Z_lbm [µm] = y_of [m] × 1e6 + 150

    OpenFOAM origin (0,0) = laser center at surface
    LBM origin (100, 150) = same physical point
"""

import numpy as np
import os
import re
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
CASE_DIR = "/home/yzk/OpenFOAM/spot_melting_marangoni"
OUTPUT_DIR = os.path.join(CASE_DIR, "postProcessing", "contours")

TIMES_US = [25, 50, 60, 75]          # desired times [µs]
T_ISO    = 1650.0                     # solidus isotherm [K]

# Mesh — must match blockMeshDict
NX, NY, NZ = 100, 50, 100
X_MIN, X_MAX = -100e-6, 100e-6       # [m]
Y_MIN, Y_MAX = -100e-6, 0.0
Z_MIN, Z_MAX = -100e-6, 100e-6

DX = (X_MAX - X_MIN) / NX
DY = (Y_MAX - Y_MIN) / NY
DZ = (Z_MAX - Z_MIN) / NZ

# Cell centres (1-D arrays)
x_cc = np.linspace(X_MIN + DX / 2, X_MAX - DX / 2, NX)   # [m]
y_cc = np.linspace(Y_MIN + DY / 2, Y_MAX - DY / 2, NY)   # [m]
z_cc = np.linspace(Z_MIN + DZ / 2, Z_MAX - DZ / 2, NZ)   # [m]

# z=0 slice: pick nearest cell centre
K_SLICE = int(np.argmin(np.abs(z_cc)))
print(f"z-slice: k={K_SLICE}, z_centre = {z_cc[K_SLICE]*1e6:.1f} µm")

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def find_time_dir(t_us):
    """Return the OpenFOAM time directory name closest to t_us."""
    t_target = t_us * 1e-6
    best, best_err = None, 1e30
    for name in os.listdir(CASE_DIR):
        full = os.path.join(CASE_DIR, name)
        if not os.path.isdir(full):
            continue
        try:
            t_val = float(name)
        except ValueError:
            continue
        err = abs(t_val - t_target)
        if err < best_err:
            best, best_err = name, err
    if best_err > 1e-6:
        raise FileNotFoundError(
            f"No time dir within 1 µs of t={t_us} µs (best: {best}, err={best_err*1e6:.2f} µs)"
        )
    return best


def read_scalar_field(filepath):
    """Parse an OpenFOAM ASCII volScalarField and return a 1-D numpy array."""
    with open(filepath, "r") as f:
        text = f.read()

    # --- nonuniform field ---
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\(", text)
    if m:
        n = int(m.group(1))
        start = m.end()
        end = text.index(")", start)
        vals = np.fromstring(text[start:end], sep="\n", count=n)
        if vals.size != n:
            # fall back to whitespace split
            vals = np.array(text[start:end].split(), dtype=float)
        assert vals.size == n, f"Expected {n} values, got {vals.size}"
        return vals

    # --- uniform field ---
    m = re.search(r"internalField\s+uniform\s+([\d.eE+\-]+)", text)
    if m:
        return np.full(NX * NY * NZ, float(m.group(1)))

    raise ValueError(f"Cannot parse {filepath}")


def slice_z(field_1d, k):
    """
    Extract the 2-D slice at z-index k.

    OpenFOAM blockMesh ordering (single block):
        cell = i + NX*j + NX*NY*k
        i → x (fastest), j → y, k → z (slowest)

    Returns array of shape (NY, NX).
    """
    field_3d = field_1d.reshape((NZ, NY, NX))
    return field_3d[k]                           # shape (NY, NX)


def extract_contour(T2d, x_1d, y_1d, level):
    """
    Use matplotlib contour to find the level-set of T2d on the (x_1d, y_1d) grid.
    Returns a list of (N,2) arrays, each an ordered contour segment.
    Coordinates are in the same units as x_1d, y_1d.
    """
    fig, ax = plt.subplots()
    cs = ax.contour(x_1d, y_1d, T2d, levels=[level])
    plt.close(fig)

    segments = []
    for coll in cs.collections:
        for path in coll.get_paths():
            verts = path.vertices                # (N, 2) — already ordered
            segments.append(verts.copy())
    return segments


def of_to_lbm(x_of, y_of):
    """
    Convert OpenFOAM coords [m] → LBM coords [µm].

        X_lbm = x_of × 1e6 + 100
        Z_lbm = y_of × 1e6 + 150
    """
    return x_of * 1e6 + 100.0, y_of * 1e6 + 150.0


# ═══════════════════════════════════════════════════════════════════════════
# Main processing
# ═══════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Also produce a combined overview figure
fig_all, axes_all = plt.subplots(2, 2, figsize=(12, 10))
axes_flat = axes_all.flatten()

for idx, t_us in enumerate(TIMES_US):
    tdir = find_time_dir(t_us)
    T_path = os.path.join(CASE_DIR, tdir, "T")
    print(f"\n── t = {t_us} µs  (dir: {tdir}) ──")

    # 1. Read & slice
    T_all = read_scalar_field(T_path)
    T2d = slice_z(T_all, K_SLICE)                # (NY, NX) on (x_cc, y_cc) grid
    print(f"   T range on z=0 slice: [{T2d.min():.1f}, {T2d.max():.1f}] K")

    # 2. Extract solidus contour
    if T2d.max() < T_ISO:
        print(f"   ⚠  T_max < {T_ISO} K — no solidus contour at this time.")
        # Write empty CSV
        csv_path = os.path.join(OUTPUT_DIR, f"openfoam_contour_{t_us}us.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["X_um", "Z_um"])
        continue

    segments = extract_contour(T2d, x_cc, y_cc, T_ISO)  # coords in [m]
    print(f"   Solidus contour segments: {len(segments)}")

    # Concatenate all segments (usually just one for a single melt pool)
    if len(segments) == 0:
        print("   ⚠  contour extraction returned empty — skipping.")
        continue

    all_pts_of = np.vstack(segments)              # (N, 2), columns = x_of, y_of  [m]

    # 3. Coordinate transform → LBM µm
    X_lbm, Z_lbm = of_to_lbm(all_pts_of[:, 0], all_pts_of[:, 1])

    # 4. Order points for smooth curve
    #    Strategy: sort by polar angle from melt pool centre (approx)
    x_cen = np.mean(X_lbm)
    z_cen = np.mean(Z_lbm)
    angles = np.arctan2(Z_lbm - z_cen, X_lbm - x_cen)
    order = np.argsort(angles)
    X_lbm = X_lbm[order]
    Z_lbm = Z_lbm[order]

    print(f"   LBM coords  X ∈ [{X_lbm.min():.1f}, {X_lbm.max():.1f}] µm")
    print(f"               Z ∈ [{Z_lbm.min():.1f}, {Z_lbm.max():.1f}] µm")
    print(f"   Points: {len(X_lbm)}")

    # 5. Write CSV
    csv_path = os.path.join(OUTPUT_DIR, f"openfoam_contour_{t_us}us.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X_um", "Z_um"])
        for xi, zi in zip(X_lbm, Z_lbm):
            writer.writerow([f"{xi:.4f}", f"{zi:.4f}"])
    print(f"   → {csv_path}")

    # 6. Plot on overview figure
    ax = axes_flat[idx]
    # Background: temperature colourmap
    X_grid, Z_grid = of_to_lbm(*np.meshgrid(x_cc, y_cc))
    pcm = ax.pcolormesh(X_grid, Z_grid, T2d, cmap="hot", shading="auto",
                        vmin=300, vmax=min(T2d.max(), 5000))
    ax.plot(X_lbm, Z_lbm, "c-", linewidth=2, label=f"T={T_ISO:.0f} K")
    ax.set_title(f"t = {t_us} µs", fontsize=13)
    ax.set_xlabel("X [µm]")
    ax.set_ylabel("Z [µm]  (depth)")
    ax.set_xlim(30, 170)
    ax.set_ylim(50, 155)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    fig_all.colorbar(pcm, ax=ax, label="T [K]", shrink=0.8)

fig_all.suptitle(
    "Spot Melting Benchmark v1.1 — Solidus isotherm (T=1650 K) on z=0 plane\n"
    "Coordinate system: LBM (µm), surface at Z=150",
    fontsize=14,
)
fig_all.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "solidus_overview.png")
fig_all.savefig(fig_path, dpi=200)
plt.close(fig_all)
print(f"\n── Overview figure saved → {fig_path}")

# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  CONTOUR SUMMARY (T = 1650 K solidus, LBM µm coordinates)")
print("=" * 65)
print(f"  {'t [µs]':>8}  {'X_min':>8}  {'X_max':>8}  {'Z_min':>8}  {'Z_max':>8}  {'N_pts':>6}")
print("-" * 65)
for t_us in TIMES_US:
    csv_path = os.path.join(OUTPUT_DIR, f"openfoam_contour_{t_us}us.csv")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.size == 0:
        print(f"  {t_us:>8}  {'(no contour)':>40}")
    else:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        print(f"  {t_us:>8}  {data[:,0].min():>8.1f}  {data[:,0].max():>8.1f}  "
              f"{data[:,1].min():>8.1f}  {data[:,1].max():>8.1f}  {len(data):>6}")
print("=" * 65)
print("Done.")
