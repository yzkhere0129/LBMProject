#!/usr/bin/env python3
"""Verify Track-B weight w = max(+(∇f·v), 0) vs Track-A w = max(v_z, 0).

Hypothesis:
  Track-A (w_A = max(v_z, 0)) fires on side-ridge cells because v_z > 0 as
  the melt is pushed up/outward by recoil.  Track-B (w_B = max(+∇f·v, 0))
  where ∇f points TOWARD the liquid (f=1 in liquid), so ∇f·v > 0 means flow
  is directed into the liquid volume.  Equivalently: outward unit normal is
  n = -∇f/|∇f|, so max(+∇f·v, 0) ≡ max(-n·v, 0).  This should be near-zero
  on outward-flowing side ridges and positive where capillary back-flow is
  pulling liquid into the trailing centerline groove.

Usage:
    python verify_trackb_weight.py [vtk_path]
        vtk_path  — default: output_phase2/line_scan_010000.vtk (t=800 μs)

Output:
    /tmp/trackb_verify.png  — 4-panel weight map
    printed tables          — zone sums, fixable/feedable fractions
"""
import sys
import os
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("ERROR: pyvista not installed.  Run: pip install pyvista", file=sys.stderr)
    sys.exit(2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── parameters ────────────────────────────────────────────────────────────────
VTK_FILE    = (sys.argv[1] if len(sys.argv) > 1
               else os.path.join(os.path.dirname(__file__),
                                 "../../output_phase2/line_scan_010000.vtk"))
VTK_FILE    = os.path.abspath(VTK_FILE)

LASER_START_UM  = 500.0    # μm — initial laser x
SCAN_VEL_MPS    = 0.80     # m/s
DT_NS           = 80.0     # ns per step (Phase-2 app)

# Zone thresholds (μm)
CENTERLINE_HALF = 15.0     # |y - y_mid| < this  → trailing centerline
SIDE_RIDGE_MIN  = 30.0     # |y - y_mid| > this  → side ridge
TRAILING_MARGIN = 100.0    # x < laser_x - this  → trailing zone
SPLASH_GUARD    = 200.0    # x > LASER_START + this  → exclude scan-start splash

# Interface / gradient thresholds
F_LO            = 0.01     # lower bound for interface cell (f > F_LO)
F_HI            = 0.99     # upper bound for interface cell (f < F_HI)
GRAD_THRESH_M   = 1e-3     # |∇f| threshold (m⁻¹) to compute unit normal
                            # typical interface: 5e5 m⁻¹ — threshold is near-zero guard

# Slice positions for visualisation
XSLICE_UM       = 940.0    # yz cross-section to plot (μm)
OUT_PNG         = "/tmp/trackb_verify_v2.png"

# ── load VTK ──────────────────────────────────────────────────────────────────
if not os.path.exists(VTK_FILE):
    print(f"ERROR: VTK file not found: {VTK_FILE}", file=sys.stderr)
    sys.exit(1)

print(f"Reading {VTK_FILE} ...")
m = pv.read(VTK_FILE)
nx, ny, nz = m.dimensions
dx, dy, dz = m.spacing
ox, oy, oz = m.origin

# Reshape fields to (nx, ny, nz) Fortran order (pyvista uses column-major)
f   = np.asarray(m.point_data["fill_level"]).reshape((nx, ny, nz), order="F")
vel = np.asarray(m.point_data["velocity"]).reshape((nx, ny, nz, 3), order="F")

print(f"Grid: {nx}×{ny}×{nz}, dx={dx*1e6:.1f} μm")
print(f"fill_level: [{f.min():.3f}, {f.max():.3f}]")
print(f"velocity (m/s): vx[{vel[...,0].min():.4f}, {vel[...,0].max():.4f}]  "
      f"vy[{vel[...,1].min():.4f}, {vel[...,1].max():.4f}]  "
      f"vz[{vel[...,2].min():.4f}, {vel[...,2].max():.4f}]")

# ── coordinate axes (μm) ─────────────────────────────────────────────────────
x_um = (ox + np.arange(nx) * dx) * 1e6
y_um = (oy + np.arange(ny) * dy) * 1e6
z_um = (oz + np.arange(nz) * dz) * 1e6
mid_j = ny // 2

# Infer laser x from filename step number
try:
    step    = int(os.path.basename(VTK_FILE).rsplit("_", 1)[-1].split(".")[0])
    t_us    = step * DT_NS * 1e-3
    laser_x_um = LASER_START_UM + SCAN_VEL_MPS * t_us
    print(f"Step {step}, t={t_us:.0f} μs → laser_x={laser_x_um:.0f} μm")
except (ValueError, IndexError):
    laser_x_um = float("nan")
    print("WARN: could not parse step from filename; laser_x_um=nan")

# ── interface normal via central-difference gradient of f ─────────────────────
# np.gradient with spacing in metres → gradient in m⁻¹
gx = np.gradient(f, dx, axis=0)   # ∂f/∂x  [m⁻¹]
gy = np.gradient(f, dy, axis=1)   # ∂f/∂y  [m⁻¹]
gz = np.gradient(f, dz, axis=2)   # ∂f/∂z  [m⁻¹]
g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

# Unit normal n = ∇f / |∇f|  (defined only where |∇f| is large enough)
# NOTE: ∇f points TOWARD liquid (f=1 in liquid, f=0 in gas), so the outward
# normal (liquid→gas) is -∇f/|∇f|.  Track-B uses max(+∇f·v, 0) ≡ max(-n·v, 0).
has_grad = g_mag > GRAD_THRESH_M
n_x = np.where(has_grad, gx / (g_mag + 1e-30), 0.0)
n_y = np.where(has_grad, gy / (g_mag + 1e-30), 0.0)
n_z = np.where(has_grad, gz / (g_mag + 1e-30), 0.0)

# ── compute weights ───────────────────────────────────────────────────────────
vx, vy, vz = vel[..., 0], vel[..., 1], vel[..., 2]

# Track-A: upward velocity (original implementation)
w_A      = np.maximum(vz, 0.0)

# Track-B (unnormalized): ∇f·v > 0 means flow is directed into the liquid.
# Sign convention: ∇f points toward liquid (high f), so +∇f·v > 0 = inflow.
# Equivalent to max(-n·v, 0) where n is the outward (liquid→gas) normal.
g_dot_v  = gx * vx + gy * vy + gz * vz   # [m⁻¹ · m/s]
w_B_raw  = np.maximum(+g_dot_v, 0.0)      # CORRECTED: was -g_dot_v (sign error)

# Track-B (normalized): same but with unit normal — unit is [m/s]
n_dot_v  = n_x * vx + n_y * vy + n_z * vz
w_B_norm = np.maximum(+n_dot_v, 0.0)      # CORRECTED: was -n_dot_v (sign error)

# ── zone masks ────────────────────────────────────────────────────────────────
y_dist  = np.abs(y_um[None, :, None] - y_um[mid_j])   # broadcast (nx,ny,nz)
iface   = (f > F_LO) & (f < F_HI) & has_grad

# Trailing zone: behind laser, past the scan-start splash
trailing = (x_um[:, None, None] > LASER_START_UM + SPLASH_GUARD) & \
           (x_um[:, None, None] < laser_x_um - TRAILING_MARGIN)

side_mask  = (y_dist > SIDE_RIDGE_MIN)  & trailing & iface
trail_mask = (y_dist < CENTERLINE_HALF) & trailing & iface

print(f"\nZone cell counts (interface, trailing, past splash-guard):")
print(f"  side-ridge    : {side_mask.sum():>8d}")
print(f"  trail-center  : {trail_mask.sum():>8d}")

# ── quantitative report ───────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("Σw comparison (interface cells, trailing zone)")
print("=" * 64)

def zone_report(name, mask):
    n = int(mask.sum())
    if n == 0:
        print(f"  {name}: 0 cells — skip")
        return {k: float("nan") for k in ("n", "wA", "wBn", "wBr")}
    wA  = float(w_A[mask].sum())
    wBn = float(w_B_norm[mask].sum())
    wBr = float(w_B_raw[mask].sum())
    fA  = float((w_A[mask]      > 1e-6).mean())
    fBn = float((w_B_norm[mask] > 1e-6).mean())
    print(f"  {name} (n={n})")
    print(f"    Σ w_A      = {wA:10.3f}   frac>0: {fA:.1%}")
    print(f"    Σ w_B_norm = {wBn:10.3f}   frac>0: {fBn:.1%}")
    print(f"    Σ w_B_raw  = {wBr:10.1f}")
    return dict(n=n, wA=wA, wBn=wBn, wBr=wBr)

sr = zone_report("side-ridge",     side_mask)
tc = zone_report("trail-center",   trail_mask)

print("\nSide / center weight ratios (lower = better redistribution to center):")
for label, s, c in [("w_A     ", sr["wA"],  tc["wA"]),
                     ("w_B_norm", sr["wBn"], tc["wBn"]),
                     ("w_B_raw ", sr["wBr"], tc["wBr"])]:
    ratio = s / c if (c and np.isfinite(c) and c > 0) else float("nan")
    print(f"  {label}: side={s:.3f}  center={c:.3f}  ratio={ratio:.3f}")

# ── fixable / feedable cell fractions ─────────────────────────────────────────
print("\nDiagnostic fractions:")
if side_mask.sum() > 0:
    # Fixable: Track-A fires but Track-B does not (w_A > 0 but w_B ≈ 0)
    fixable = side_mask & (w_A > 1e-6) & (w_B_norm < 1e-3)
    frac_fix = fixable.sum() / side_mask.sum()
    print(f"  Side-ridge fixable (w_A>0 & w_B~0): "
          f"{fixable.sum()} / {side_mask.sum()} = {frac_fix:.1%}")
if trail_mask.sum() > 0:
    # Feedable: Track-B has positive weight on trailing centerline
    feedable = trail_mask & (w_B_norm > 1e-6)
    frac_feed = feedable.sum() / trail_mask.sum()
    print(f"  Trail-center feedable (w_B>0)     : "
          f"{feedable.sum()} / {trail_mask.sum()} = {frac_feed:.1%}")

# ── per-cell average weight and histograms ────────────────────────────────────
print("\nPer-cell average weight (all interface cells in zone, including zeros):")
for wname, warr in [("w_A", w_A), ("w_B_norm", w_B_norm), ("w_B_raw", w_B_raw)]:
    sr_mean = float(warr[side_mask].mean()) if side_mask.sum() > 0 else float("nan")
    tc_mean = float(warr[trail_mask].mean()) if trail_mask.sum() > 0 else float("nan")
    ratio   = tc_mean / sr_mean if (sr_mean > 0 and np.isfinite(sr_mean)) else float("nan")
    print(f"  {wname:10s}  side-ridge mean={sr_mean:.5f}  trail-center mean={tc_mean:.5f}  "
          f"center/side ratio={ratio:.2f}x")

print("\nHistograms (side-ridge vs trail-center, non-zero cells only):")
for wname, warr in [("w_A", w_A), ("w_B_norm", w_B_norm)]:
    sr_vals  = warr[side_mask & (warr > 1e-6)]
    tc_vals  = warr[trail_mask & (warr > 1e-6)]
    sr_med   = float(np.median(sr_vals))  if sr_vals.size  > 0 else float("nan")
    tc_med   = float(np.median(tc_vals))  if tc_vals.size  > 0 else float("nan")
    sr_p95   = float(np.percentile(sr_vals, 95))  if sr_vals.size  > 0 else float("nan")
    tc_p95   = float(np.percentile(tc_vals, 95))  if tc_vals.size  > 0 else float("nan")
    print(f"  {wname}  side-ridge   median={sr_med:.5f}  p95={sr_p95:.5f}  n={sr_vals.size}")
    print(f"  {wname}  trail-center median={tc_med:.5f}  p95={tc_p95:.5f}  n={tc_vals.size}")

# ── visualisation: 4-panel weight map ────────────────────────────────────────
# Find slice indices
i_xsl = int(np.argmin(np.abs(x_um - XSLICE_UM)))
j_mid = mid_j

print(f"\nGenerating 4-panel figure → {OUT_PNG}")
print(f"  xz-slice: j={j_mid} (y={y_um[j_mid]:.0f} μm centerline)")
print(f"  yz-slice: i={i_xsl} (x={x_um[i_xsl]:.0f} μm)")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(
    f"Track-A vs Track-B weight maps  |  Phase-2 t={t_us:.0f} μs  |  "
    f"laser_x={laser_x_um:.0f} μm",
    fontsize=11)

def _mask_iface_2d(f_slice):
    """Return alpha mask: 1 for interface cells, 0 otherwise."""
    return ((f_slice > F_LO) & (f_slice < F_HI)).astype(float)

def plot_xz(ax, w_xz, f_xz, x_arr, z_arr, title, cmap="hot"):
    """XZ slice (y=mid): x horizontal, z vertical."""
    alpha = _mask_iface_2d(f_xz)
    # clip to trailing zone for clarity
    trail_xi = (x_arr > LASER_START_UM + SPLASH_GUARD) & (x_arr < laser_x_um + 50)
    w_plot = w_xz[trail_xi, :]
    f_plot = f_xz[trail_xi, :]
    alpha_plot = _mask_iface_2d(f_plot)
    x_plot = x_arr[trail_xi]
    vmax = float(np.percentile(w_plot[alpha_plot > 0.5], 99)) if alpha_plot.any() else 1.0
    vmax = max(vmax, 1e-9)
    im = ax.pcolormesh(x_plot, z_arr, (w_plot * alpha_plot).T,
                       cmap=cmap, vmin=0, vmax=vmax, shading="auto")
    # free-surface contour
    ax.contour(x_plot, z_arr, f_plot.T, levels=[0.5], colors="white", linewidths=0.7)
    if np.isfinite(laser_x_um):
        ax.axvline(laser_x_um,           color="cyan",   lw=1.2, ls="--", label="laser")
        ax.axvline(laser_x_um - TRAILING_MARGIN, color="lime", lw=0.8, ls=":")
    ax.set_xlabel("x (μm)"); ax.set_ylabel("z (μm)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="weight (m/s)")

def plot_yz(ax, w_yz, f_yz, y_arr, z_arr, title, cmap="hot"):
    """YZ slice (x=xslice): y horizontal, z vertical."""
    alpha = _mask_iface_2d(f_yz)
    vmax = float(np.percentile(w_yz[alpha > 0.5], 99)) if alpha.any() else 1.0
    vmax = max(vmax, 1e-9)
    im = ax.pcolormesh(y_arr, z_arr, (w_yz * alpha).T,
                       cmap=cmap, vmin=0, vmax=vmax, shading="auto")
    ax.contour(y_arr, z_arr, f_yz.T, levels=[0.5], colors="white", linewidths=0.7)
    ax.axvline(y_arr[j_mid],                 color="cyan", lw=1.2, ls="--", label="centerline")
    ax.axvline(y_arr[j_mid] - SIDE_RIDGE_MIN, color="magenta", lw=0.8, ls=":", label="ridge thresh")
    ax.axvline(y_arr[j_mid] + SIDE_RIDGE_MIN, color="magenta", lw=0.8, ls=":")
    ax.set_xlabel("y (μm)"); ax.set_ylabel("z (μm)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="weight (m/s)")

# XZ slices (y = centerline)
wA_xz  = w_A[:, j_mid, :]
wBr_xz = w_B_raw[:, j_mid, :]
f_xz   = f[:, j_mid, :]

plot_xz(axes[0, 0], wA_xz,  f_xz, x_um, z_um,
        "Panel A — Track-A w_A=max(v_z,0)  [xz, y=center]")
plot_xz(axes[0, 1], wBr_xz, f_xz, x_um, z_um,
        "Panel B — Track-B w_B_raw=max(+∇f·v,0)  [xz, y=center]", cmap="plasma")

# YZ slices (x = XSLICE_UM, trailing zone)
wA_yz  = w_A[i_xsl, :, :]
wBr_yz = w_B_raw[i_xsl, :, :]
f_yz   = f[i_xsl, :, :]

plot_yz(axes[1, 0], wA_yz,  f_yz, y_um, z_um,
        f"Panel C — Track-A w_A  [yz, x={x_um[i_xsl]:.0f} μm]")
plot_yz(axes[1, 1], wBr_yz, f_yz, y_um, z_um,
        f"Panel D — Track-B w_B_raw=max(+∇f·v,0)  [yz, x={x_um[i_xsl]:.0f} μm]", cmap="plasma")

for ax in axes.flat:
    ax.legend(fontsize=6, loc="upper right")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close(fig)
print(f"Saved: {OUT_PNG}")

print("\nDone.")
