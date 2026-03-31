#!/usr/bin/env python3
"""
Post-processing for Pure Conduction + Phase Change Benchmark (316L).

Reads CSV output from benchmark_conduction_316L and produces:
  1. T=1650K (solidus) isotherm maximum depth at t = 25, 50, 60, 75 us
  2. Centerline temperature profiles at each snapshot
  3. Probe temperature time series
  4. T_max vs time

Usage:
    python3 scripts/viz/plot_conduction_benchmark.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================================
# Parameters (must match benchmark_conduction.json)
# ============================================================================
T_SOL = 1650.0   # K (solidus)
T_LIQ = 1700.0   # K (liquidus)
T_INIT = 300.0    # K
DX = 2.0e-6       # m
NZ = 50

# ============================================================================
# Load data
# ============================================================================
probe_file = Path("benchmark_conduction_probes.csv")
profile_file = Path("benchmark_conduction_profiles.csv")

if not probe_file.exists() or not profile_file.exists():
    print("ERROR: Output files not found. Run benchmark_conduction_316L first.")
    sys.exit(1)

# Read probes (skip comment lines starting with #)
probes = pd.read_csv(probe_file, comment='#')

# Read profiles
profiles = pd.read_csv(profile_file, comment='#')

# ============================================================================
# 1. T=1650K isotherm depth at requested times
# ============================================================================
print("=" * 60)
print("  T = 1650 K (Solidus) Isotherm — Maximum Depth")
print("=" * 60)
print(f"{'Snapshot':>10s}  {'Depth [um]':>12s}  {'Status':>10s}")
print("-" * 40)

snapshot_tags = profiles['snapshot'].unique()
depth_data = {}

for tag in snapshot_tags:
    snap = profiles[profiles['snapshot'] == tag].sort_values('z_idx')

    # Scan from surface (z=NZ-1) downward to find deepest cell with T >= T_SOL
    max_depth = 0.0
    for _, row in snap.iterrows():
        if row['T_K'] >= T_SOL:
            d = row['depth_um']
            if d > max_depth:
                max_depth = d

    depth_data[tag] = max_depth
    status = "MELTED" if max_depth > 0 else "SOLID"
    print(f"{tag:>10s}  {max_depth:>10.1f} um  {status:>10s}")

print()

# Also report T=1700K (liquidus) depth
print("=" * 60)
print("  T = 1700 K (Liquidus) Isotherm — Maximum Depth")
print("=" * 60)
print(f"{'Snapshot':>10s}  {'Depth [um]':>12s}")
print("-" * 30)

for tag in snapshot_tags:
    snap = profiles[profiles['snapshot'] == tag]
    max_depth_liq = 0.0
    for _, row in snap.iterrows():
        if row['T_K'] >= T_LIQ:
            d = row['depth_um']
            if d > max_depth_liq:
                max_depth_liq = d
    print(f"{tag:>10s}  {max_depth_liq:>10.1f} um")

print()

# ============================================================================
# 2. Centerline temperature profiles
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2a. T vs depth
ax = axes[0]
for tag in snapshot_tags:
    snap = profiles[profiles['snapshot'] == tag].sort_values('depth_um')
    ax.plot(snap['depth_um'], snap['T_K'], '-o', markersize=2, label=tag)

ax.axhline(y=T_SOL, color='r', linestyle='--', alpha=0.5, label=f'T_sol = {T_SOL} K')
ax.axhline(y=T_LIQ, color='orange', linestyle='--', alpha=0.5, label=f'T_liq = {T_LIQ} K')
ax.set_xlabel('Depth from surface [um]')
ax.set_ylabel('Temperature [K]')
ax.set_title('Centerline Temperature Profile (x=0, y=0)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)

# 2b. Liquid fraction vs depth
ax = axes[1]
for tag in snapshot_tags:
    snap = profiles[profiles['snapshot'] == tag].sort_values('depth_um')
    ax.plot(snap['depth_um'], snap['fl'], '-o', markersize=2, label=tag)

ax.set_xlabel('Depth from surface [um]')
ax.set_ylabel('Liquid fraction')
ax.set_title('Centerline Liquid Fraction (x=0, y=0)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('benchmark_conduction_profiles.png', dpi=150)
print("Saved: benchmark_conduction_profiles.png")

# ============================================================================
# 3. Probe temperature time series
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 3a. All probes
ax = axes[0]
probe_cols = [c for c in probes.columns if c.startswith('T_')]
for col in probe_cols:
    label = col.replace('T_', '')
    ax.plot(probes['time_us'], probes[col], label=label)

ax.axhline(y=T_SOL, color='r', linestyle='--', alpha=0.4, label='T_sol')
ax.axhline(y=T_LIQ, color='orange', linestyle='--', alpha=0.4, label='T_liq')
ax.axvline(x=50.0, color='gray', linestyle=':', alpha=0.5, label='Laser OFF')
ax.set_xlabel('Time [us]')
ax.set_ylabel('Temperature [K]')
ax.set_title('Probe Temperature Time Series')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# 3b. T_max vs time
ax = axes[1]
ax.plot(probes['time_us'], probes['T_max_K'], 'k-', linewidth=1.5, label='T_max')
ax.axhline(y=T_SOL, color='r', linestyle='--', alpha=0.4)
ax.axhline(y=T_LIQ, color='orange', linestyle='--', alpha=0.4)
ax.axvline(x=50.0, color='gray', linestyle=':', alpha=0.5, label='Laser OFF')
ax.set_xlabel('Time [us]')
ax.set_ylabel('T_max [K]')
ax.set_title('Maximum Temperature in Domain vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_conduction_timeseries.png', dpi=150)
print("Saved: benchmark_conduction_timeseries.png")

# ============================================================================
# 4. Summary table for cross-platform comparison
# ============================================================================
print()
print("=" * 60)
print("  CROSS-PLATFORM COMPARISON TABLE (LBM side)")
print("  Paste this into your verification spreadsheet")
print("=" * 60)
print()
print(f"{'Metric':<40s}  {'Value':>12s}  {'Unit':>8s}")
print("-" * 65)

# T_max at key times
for tag in ['25us', '50us', '75us', '100us']:
    if tag in probes.columns:
        continue
    # Find closest step
    t_target = float(tag.replace('us', ''))
    row = probes.iloc[(probes['time_us'] - t_target).abs().argsort()[:1]]
    tmax = row['T_max_K'].values[0]
    print(f"T_max at t={tag:<8s}                      {tmax:>10.1f}    K")

# Solidus depth
for tag in ['25us', '50us', '60us', '75us']:
    if tag in depth_data:
        print(f"Solidus depth (T>=1650K) at t={tag:<8s}  {depth_data[tag]:>10.1f}    um")

# Solidification completion
last_row = probes.iloc[-1]
t_solid_complete = None
for i in range(len(probes) - 1, -1, -1):
    if probes.iloc[i]['T_max_K'] >= T_SOL:
        t_solid_complete = probes.iloc[i]['time_us']
        break
if t_solid_complete:
    print(f"Last time T_max >= T_sol                {t_solid_complete:>10.2f}    us")

print()
plt.show()
