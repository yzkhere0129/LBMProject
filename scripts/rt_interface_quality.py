#!/usr/bin/env python3
"""
RT benchmark interface quality analysis.

Reads fill_level from VTK files at late-stage steps and reports:
  - Number of disconnected interface blobs (connected components)
  - Interface thickness per column (mean / max)
  - Fragment count (isolated liquid or gas pockets)

VTK storage order: x varies fastest (ix + NX*(iy + NY*iz)).
Mid z-slice (kz = NZ//2 = 2) is extracted, giving a 2D array shaped (NY, NX).
"""

import numpy as np
from scipy import ndimage
import os

# === PARAMETERS ===
VTK_DIR   = "/home/yzk/LBMProject/build/output_rt_benchmark"
STEPS     = [14000, 16000, 18000, 20000]
NX, NY, NZ = 256, 1024, 4
KZ_MID    = NZ // 2          # = 2

# Interface band thresholds
F_LO = 0.01
F_HI = 0.99

# Fragment detection: isolated region threshold
F_LIQUID  = 0.5              # above = liquid, below = gas


def vtk_path(step):
    return os.path.join(VTK_DIR, f"rt_benchmark_step{step:06d}.vtk")


def load_fill_level(step):
    """Read fill_level from ASCII VTK and return full 3D array (NZ, NY, NX)."""
    path = vtk_path(step)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    # Skip 10-line header, then read NX*NY*NZ values (one per line)
    data = np.loadtxt(path, skiprows=10, max_rows=NX * NY * NZ, dtype=np.float32)

    # VTK STRUCTURED_POINTS: x fastest → reshape to (NZ, NY, NX)
    data = data.reshape(NZ, NY, NX)
    return data


def mid_slice(data):
    """Return 2D fill_level at the mid z-slice, shape (NY, NX)."""
    return data[KZ_MID]          # shape (NY, NX)


def count_interface_blobs(f2d):
    """
    Connected components of interface cells (F_LO < f < F_HI).
    Uses 8-connectivity (diagonal neighbours included) to avoid splitting
    thin diagonal streaks into spurious blobs.
    """
    mask = (f2d > F_LO) & (f2d < F_HI)
    struct = ndimage.generate_binary_structure(2, 2)   # 8-connectivity
    labeled, n_blobs = ndimage.label(mask, structure=struct)
    return n_blobs, labeled, mask


def interface_thickness(f2d):
    """
    For each x column, count how many y cells fall in the interface band.
    Returns (mean_thickness, max_thickness) across all NX columns.
    """
    mask = (f2d > F_LO) & (f2d < F_HI)     # shape (NY, NX)
    col_thickness = mask.sum(axis=0)         # shape (NX,)
    return float(col_thickness.mean()), int(col_thickness.max())


def count_fragments(f2d):
    """
    Detect isolated regions of one phase surrounded entirely by the other.

    Liquid fragments: connected components of (f > F_LIQUID) that are
    completely enclosed by gas (f < F_LIQUID) — i.e. do not touch any
    domain boundary.

    Gas fragments: symmetric — enclosed gas bubbles in liquid.

    Returns (n_liquid_frags, n_gas_frags).
    """
    struct = ndimage.generate_binary_structure(2, 2)   # 8-connectivity

    # --- liquid fragments (droplets in gas) ---
    liq_mask = f2d > F_LIQUID
    liq_labeled, n_liq = ndimage.label(liq_mask, structure=struct)
    n_liq_frags = 0
    for comp in range(1, n_liq + 1):
        region = liq_labeled == comp
        # touches boundary?
        if (region[0, :].any() or region[-1, :].any() or
                region[:, 0].any() or region[:, -1].any()):
            continue
        n_liq_frags += 1

    # --- gas fragments (bubbles in liquid) ---
    gas_mask = f2d < F_LIQUID
    gas_labeled, n_gas = ndimage.label(gas_mask, structure=struct)
    n_gas_frags = 0
    for comp in range(1, n_gas + 1):
        region = gas_labeled == comp
        if (region[0, :].any() or region[-1, :].any() or
                region[:, 0].any() or region[:, -1].any()):
            continue
        n_gas_frags += 1

    return n_liq_frags, n_gas_frags


def blob_size_stats(labeled, n_blobs):
    """Return (min_size, max_size, mean_size) in cells for interface blobs."""
    if n_blobs == 0:
        return 0, 0, 0.0
    sizes = ndimage.sum(np.ones_like(labeled, dtype=np.int32),
                        labeled, range(1, n_blobs + 1))
    sizes = np.array(sizes, dtype=int)
    return int(sizes.min()), int(sizes.max()), float(sizes.mean())


# === MAIN ===
header = (
    f"\n{'Step':>8s} | {'Blobs':>6s} | {'ThkMean':>8s} | {'ThkMax':>7s} | "
    f"{'LiqFrag':>8s} | {'GasFrag':>8s} | {'BlobMin':>8s} | {'BlobMax':>8s} | {'BlobMean':>9s}"
)
sep = "-" * len(header)

print("RT Benchmark — Interface Quality Analysis")
print(f"Grid: {NX}x{NY}x{NZ}, mid z-slice kz={KZ_MID}")
print(f"Interface band: {F_LO} < f < {F_HI}")
print(sep)
print(header)
print(sep)

results = []
for step in STEPS:
    path = vtk_path(step)
    if not os.path.exists(path):
        print(f"  step {step:6d}  ->  FILE NOT FOUND: {path}")
        continue

    data   = load_fill_level(step)
    f2d    = mid_slice(data)

    # Sanity: data range
    fmin, fmax, fmean = f2d.min(), f2d.max(), f2d.mean()

    # Metrics
    n_blobs, labeled, iface_mask = count_interface_blobs(f2d)
    thk_mean, thk_max            = interface_thickness(f2d)
    n_liq_frags, n_gas_frags     = count_fragments(f2d)
    b_min, b_max, b_mean         = blob_size_stats(labeled, n_blobs)

    # Interface cell fraction of total domain
    iface_frac = iface_mask.sum() / (NX * NY) * 100.0

    results.append({
        "step": step, "n_blobs": n_blobs,
        "thk_mean": thk_mean, "thk_max": thk_max,
        "n_liq_frags": n_liq_frags, "n_gas_frags": n_gas_frags,
        "b_min": b_min, "b_max": b_max, "b_mean": b_mean,
        "fmin": fmin, "fmax": fmax, "fmean": fmean,
        "iface_frac": iface_frac,
    })

    print(
        f"{step:>8d} | {n_blobs:>6d} | {thk_mean:>8.2f} | {thk_max:>7d} | "
        f"{n_liq_frags:>8d} | {n_gas_frags:>8d} | {b_min:>8d} | {b_max:>8d} | {b_mean:>9.1f}"
    )

print(sep)
print("\nPer-step supplementary info (data range, interface cell fraction):")
for r in results:
    print(
        f"  step {r['step']:6d}: fill in [{r['fmin']:.4f}, {r['fmax']:.4f}], "
        f"mean={r['fmean']:.4f}, interface cells={r['iface_frac']:.2f}% of slice"
    )

# Fragmentation verdict
print("\n--- Fragmentation Assessment ---")
for r in results:
    frags_total = r["n_liq_frags"] + r["n_gas_frags"]
    verdict = "CLEAN" if r["n_blobs"] <= 2 and frags_total == 0 else (
              "MILD"  if r["n_blobs"] <= 5 and frags_total <= 3 else
              "FRAGMENTED")
    print(
        f"  step {r['step']:6d}: {verdict}  "
        f"(blobs={r['n_blobs']}, frags={frags_total}, "
        f"thk_mean={r['thk_mean']:.1f}, thk_max={r['thk_max']})"
    )
