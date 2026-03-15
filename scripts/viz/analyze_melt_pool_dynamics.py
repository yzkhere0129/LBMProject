#!/usr/bin/env python3
"""
Melt Pool Dynamics Analysis for LPBF Line Scan Simulation

Reads all line_scan_*.vtk files, extracts melt pool metrics,
and generates publication-quality time-series plots.

Melt pool definition: fill_level > 0.5 AND liquid_fraction > 0.5
(metal cells that are above solidus temperature)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys

# ============================================================
# Configuration
# ============================================================
VTK_DIR = Path(__file__).parent.parent.parent / "output_line_scan"
OUT_FIG = Path(__file__).parent / "melt_pool_dynamics.png"

NX, NY, NZ = 250, 75, 50
DX = 2.0e-6       # m
DT_SIM = 1.0e-7   # s per step
LU_TO_MS = DX / DT_SIM  # lattice velocity → m/s

# Free surface z-position (cells)
Z_SURFACE = 40  # 80% of NZ=50 → z=40 cells = 80 μm

# Steady-state window for averaging (last N μs)
SS_WINDOW_US = 100.0

# ============================================================
# VTK Parser (lightweight, no external dependencies)
# ============================================================
def parse_vtk(filepath):
    """Parse ASCII VTK STRUCTURED_POINTS file. Returns dict of fields."""
    with open(filepath) as f:
        lines = f.readlines()

    fields = {}
    n_total = NX * NY * NZ
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("VECTORS"):
            name = line.split()[1]
            vals = []
            i += 1
            while i < len(lines) and len(vals) < n_total * 3:
                vals.extend(float(x) for x in lines[i].strip().split())
                i += 1
            fields[name] = np.array(vals, dtype=np.float32).reshape(NZ, NY, NX, 3)
        elif line.startswith("SCALARS"):
            name = line.split()[1]
            i += 2  # skip LOOKUP_TABLE line
            vals = []
            while i < len(lines) and len(vals) < n_total:
                try:
                    vals.append(float(lines[i].strip()))
                    i += 1
                except ValueError:
                    break
            fields[name] = np.array(vals, dtype=np.float32).reshape(NZ, NY, NX)
        else:
            i += 1
    return fields


def extract_step_number(filename):
    """Extract step number from filename like line_scan_001234.vtk"""
    m = re.search(r"_(\d+)\.vtk$", filename.name)
    return int(m.group(1)) if m else -1


# ============================================================
# Metrics Extraction
# ============================================================
def compute_metrics(fields):
    """Compute melt pool metrics from a single VTK snapshot."""
    T = fields["temperature"]
    fl = fields["fill_level"]
    lf = fields["liquid_fraction"]
    vel = fields["velocity"]

    # T_max, V_max
    T_max = float(T.max())
    vmag = np.sqrt((vel ** 2).sum(axis=-1)) * LU_TO_MS
    V_max = float(vmag.max())

    # Melt pool mask: metal (fill > 0.5) AND liquid (lf > 0.5)
    melt_mask = (fl > 0.5) & (lf > 0.5)

    if not melt_mask.any():
        return {
            "T_max": T_max, "V_max": V_max,
            "depth_um": 0.0, "width_um": 0.0, "length_um": 0.0,
            "mass": float(fl.sum()),
        }

    # Find bounding box of melt pool
    kk, jj, ii = np.where(melt_mask)

    length_um = (ii.max() - ii.min() + 1) * DX * 1e6
    width_um = (jj.max() - jj.min() + 1) * DX * 1e6
    depth_um = (Z_SURFACE - kk.min()) * DX * 1e6
    if depth_um < 0:
        depth_um = 0.0

    return {
        "T_max": T_max,
        "V_max": V_max,
        "depth_um": depth_um,
        "width_um": width_um,
        "length_um": length_um,
        "mass": float(fl.sum()),
    }


# ============================================================
# Main
# ============================================================
def main():
    # Find and sort VTK files
    vtk_files = sorted(VTK_DIR.glob("line_scan_*.vtk"), key=extract_step_number)
    if not vtk_files:
        print(f"No VTK files found in {VTK_DIR}")
        sys.exit(1)

    print(f"Found {len(vtk_files)} VTK files in {VTK_DIR}")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e6:.0f} μm")
    print(f"Surface at z={Z_SURFACE} cells ({Z_SURFACE*DX*1e6:.0f} μm)\n")

    # Process each file
    times_us = []
    metrics_list = []

    print(f"{'File':35s}  {'t[μs]':>7s}  {'T_max':>7s}  {'V_max':>7s}  "
          f"{'Depth':>7s}  {'Width':>7s}  {'Length':>7s}  {'Mass':>10s}")
    print("-" * 100)

    for vtk_path in vtk_files:
        step = extract_step_number(vtk_path)
        t_us = step * DT_SIM * 1e6

        fields = parse_vtk(vtk_path)
        m = compute_metrics(fields)

        times_us.append(t_us)
        metrics_list.append(m)

        print(f"{vtk_path.name:35s}  {t_us:7.1f}  {m['T_max']:7.0f}  {m['V_max']:7.3f}  "
              f"{m['depth_um']:7.1f}  {m['width_um']:7.1f}  {m['length_um']:7.1f}  "
              f"{m['mass']:10.0f}")

    times = np.array(times_us)
    T_max = np.array([m["T_max"] for m in metrics_list])
    V_max = np.array([m["V_max"] for m in metrics_list])
    depth = np.array([m["depth_um"] for m in metrics_list])
    width = np.array([m["width_um"] for m in metrics_list])
    length = np.array([m["length_um"] for m in metrics_list])
    mass = np.array([m["mass"] for m in metrics_list])

    # ============================================================
    # Steady-state analysis (last SS_WINDOW_US μs)
    # ============================================================
    t_max = times.max()
    ss_mask = times >= (t_max - SS_WINDOW_US)
    if ss_mask.sum() >= 2:
        ss_depth = depth[ss_mask].mean()
        ss_width = width[ss_mask].mean()
        ss_length = length[ss_mask].mean()
        ss_Tmax = T_max[ss_mask].mean()
        ss_Vmax = V_max[ss_mask].mean()

        print(f"\n{'='*60}")
        print(f"  Steady-State Averages (t > {t_max - SS_WINDOW_US:.0f} μs)")
        print(f"{'='*60}")
        print(f"  Melt Pool Depth:  {ss_depth:6.1f} μm")
        print(f"  Melt Pool Width:  {ss_width:6.1f} μm")
        print(f"  Melt Pool Length: {ss_length:6.1f} μm")
        print(f"  T_max:            {ss_Tmax:6.0f} K")
        print(f"  V_max:            {ss_Vmax:6.3f} m/s")
        print(f"  Mass change:      {(mass[-1]-mass[0])/mass[0]*100:+.3f}%")
        print(f"{'='*60}\n")

    # ============================================================
    # Plotting
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: Melt pool dimensions vs time
    ax1.plot(times, length, "o-", color="#e41a1c", label="Length (X)", markersize=4, linewidth=1.5)
    ax1.plot(times, width, "s-", color="#377eb8", label="Width (Y)", markersize=4, linewidth=1.5)
    ax1.plot(times, depth, "^-", color="#4daf4a", label="Depth (Z)", markersize=4, linewidth=1.5)
    ax1.set_ylabel("Dimension [μm]")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_title("Melt Pool Dimensions vs. Time", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Annotate steady-state region
    if ss_mask.sum() >= 2:
        ax1.axvspan(t_max - SS_WINDOW_US, t_max, alpha=0.08, color="gray",
                    label=f"Steady-state ({SS_WINDOW_US:.0f} μs)")
        ax1.axhline(ss_depth, color="#4daf4a", linestyle="--", alpha=0.5, linewidth=0.8)
        ax1.axhline(ss_width, color="#377eb8", linestyle="--", alpha=0.5, linewidth=0.8)

    # Bottom: T_max and V_max vs time
    color_T = "#d62728"
    ax2.plot(times, T_max, "o-", color=color_T, label="T_max", markersize=4, linewidth=1.5)
    ax2.set_ylabel("Temperature [K]", color=color_T)
    ax2.tick_params(axis="y", labelcolor=color_T)
    ax2.set_ylim(0, max(T_max) * 1.15)
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    color_V = "#1f77b4"
    ax2b.plot(times, V_max, "s-", color=color_V, label="V_max", markersize=4, linewidth=1.5)
    ax2b.set_ylabel("Velocity [m/s]", color=color_V)
    ax2b.tick_params(axis="y", labelcolor=color_V)
    ax2b.set_ylim(bottom=0)

    ax2.set_xlabel("Time [μs]")
    ax2.set_title("Peak Temperature & Velocity vs. Time", fontsize=12, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9)

    fig.suptitle(
        "LPBF Single-Track Line Scan: 316L Stainless Steel\n"
        r"P = 150 W, $r_0$ = 50 μm, v = 800 mm/s, dx = 2 μm",
        fontsize=13, fontweight="bold", y=1.0,
    )

    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
