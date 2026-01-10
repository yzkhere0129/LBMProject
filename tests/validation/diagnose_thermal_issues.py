#!/usr/bin/env python3
"""
Diagnostic script to identify thermal solver issues in Case 5.
Analyzes VTK data to pinpoint where numerical instability begins.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

METRICS_FILE = "/home/yzk/LBMProject/tests/validation/analysis_case5/timeseries_metrics.json"
OUTPUT_DIR = "/home/yzk/LBMProject/tests/validation/analysis_case5"

# Physical limits
T_AMBIENT = 300.0
T_MELTING = 1923.0
T_BOILING = 3560.0
T_MAX_REASONABLE = 4000.0  # Conservative upper bound

def load_data():
    """Load timeseries metrics."""
    with open(METRICS_FILE, 'r') as f:
        return json.load(f)

def detect_instability_onset(metrics):
    """Find when numerical instability begins."""
    timesteps = []
    peak_temps = []

    for m in metrics:
        timesteps.append(m.get('timestep', 0))
        peak_temps.append(m['peak_temperature']['value'])

    timesteps = np.array(timesteps)
    peak_temps = np.array(peak_temps)

    # Find first excursion above boiling
    unstable_mask = peak_temps > T_BOILING

    if unstable_mask.any():
        first_unstable = np.argmax(unstable_mask)
        return first_unstable, timesteps, peak_temps

    return None, timesteps, peak_temps

def analyze_temperature_gradient(peak_temps):
    """Analyze rate of temperature change."""
    dt_temps = np.diff(peak_temps)

    # Find largest temperature jumps
    largest_jumps_idx = np.argsort(np.abs(dt_temps))[-10:]

    return dt_temps, largest_jumps_idx

def check_energy_balance(metrics):
    """Estimate energy balance from melt pool data."""
    total_liquid_volumes = []
    liquid_fractions = []

    for m in metrics:
        mp = m['melt_pool']
        # Estimate volume from dimensions (rough)
        width = mp['width']
        depth = mp['depth']
        height = mp['height']

        # Approximate as ellipsoid
        volume = (4/3) * np.pi * (width/2) * (depth/2) * (height/2)
        total_liquid_volumes.append(volume)
        liquid_fractions.append(mp['liquid_fraction'])

    return np.array(total_liquid_volumes), np.array(liquid_fractions)

def main():
    print("=" * 80)
    print("THERMAL SOLVER DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    metrics = load_data()

    # Detect instability onset
    onset_idx, timesteps, peak_temps = detect_instability_onset(metrics)

    print("\n" + "-" * 80)
    print("INSTABILITY DETECTION")
    print("-" * 80)

    if onset_idx is not None:
        print(f"First unstable timestep: {timesteps[onset_idx]}")
        print(f"Temperature at onset: {peak_temps[onset_idx]:.2f} K")

        if onset_idx > 0:
            print(f"Previous timestep: {timesteps[onset_idx-1]}")
            print(f"Previous temperature: {peak_temps[onset_idx-1]:.2f} K")
            print(f"Temperature jump: {peak_temps[onset_idx] - peak_temps[onset_idx-1]:.2f} K")
    else:
        print("No instability detected (all temperatures < boiling point)")

    # Analyze temperature gradients
    dt_temps, jump_indices = analyze_temperature_gradient(peak_temps)

    print("\n" + "-" * 80)
    print("LARGEST TEMPERATURE JUMPS (Top 10)")
    print("-" * 80)

    for i, idx in enumerate(reversed(jump_indices), 1):
        if idx < len(timesteps) - 1:
            print(f"{i:2d}. Timestep {timesteps[idx]:5.0f} -> {timesteps[idx+1]:5.0f}: "
                  f"{peak_temps[idx]:8.2f} K -> {peak_temps[idx+1]:8.2f} K "
                  f"(ΔT = {dt_temps[idx]:+8.2f} K)")

    # Energy balance check
    volumes, liquid_fracs = check_energy_balance(metrics)

    print("\n" + "-" * 80)
    print("MELT POOL EVOLUTION")
    print("-" * 80)
    print(f"Maximum liquid fraction: {liquid_fracs.max()*100:.3f}%")
    print(f"Maximum melt volume: {volumes.max()*1e15:.2f} μm³")

    if liquid_fracs.max() < 0.05:
        print("\nWARNING: Very low liquid fraction suggests:")
        print("  1. Latent heat may not be properly absorbed")
        print("  2. VOF-thermal coupling may be incorrect")
        print("  3. Phase change implementation may have issues")

    # Temperature statistics by regime
    print("\n" + "-" * 80)
    print("TEMPERATURE REGIME STATISTICS")
    print("-" * 80)

    below_melting = np.sum(peak_temps < T_MELTING)
    mushy_zone = np.sum((peak_temps >= T_MELTING) & (peak_temps < T_BOILING))
    above_boiling = np.sum(peak_temps >= T_BOILING)
    extreme = np.sum(peak_temps > 5000.0)

    print(f"Below melting point (<{T_MELTING} K):     {below_melting:3d} timesteps ({below_melting/len(metrics)*100:5.1f}%)")
    print(f"Melting-Boiling ({T_MELTING}-{T_BOILING} K): {mushy_zone:3d} timesteps ({mushy_zone/len(metrics)*100:5.1f}%)")
    print(f"Above boiling point (>{T_BOILING} K):     {above_boiling:3d} timesteps ({above_boiling/len(metrics)*100:5.1f}%)")
    print(f"Extreme temperatures (>5000 K):        {extreme:3d} timesteps ({extreme/len(metrics)*100:5.1f}%)")

    # Generate diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Temperature evolution with regime coloring
    ax = axes[0, 0]

    # Estimate time (assuming dt = 1e-8 s)
    dt_estimate = 1e-8
    time_us = timesteps * dt_estimate * 1e6

    # Color by regime
    for i in range(len(timesteps)):
        if peak_temps[i] > T_BOILING:
            color = 'red'
            alpha = 0.8
        elif peak_temps[i] > T_MELTING:
            color = 'orange'
            alpha = 0.6
        else:
            color = 'blue'
            alpha = 0.4

        if i < len(timesteps) - 1:
            ax.plot(time_us[i:i+2], peak_temps[i:i+2], color=color, alpha=alpha, linewidth=2)

    ax.axhline(y=T_MELTING, color='green', linestyle='--', label='Melting')
    ax.axhline(y=T_BOILING, color='red', linestyle='--', label='Boiling')
    ax.set_xlabel('Time (μs)', fontsize=11)
    ax.set_ylabel('Peak Temperature (K)', fontsize=11)
    ax.set_title('Temperature Evolution (Color = Regime)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature rate of change
    ax = axes[0, 1]
    ax.plot(time_us[:-1], dt_temps, 'b-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (μs)', fontsize=11)
    ax.set_ylabel('dT/dt (K/step)', fontsize=11)
    ax.set_title('Rate of Temperature Change', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Liquid fraction vs temperature
    ax = axes[1, 0]
    ax.scatter(peak_temps, liquid_fracs*100, c=timesteps, cmap='viridis', s=30, alpha=0.6)
    ax.axvline(x=T_MELTING, color='g', linestyle='--', alpha=0.5)
    ax.axvline(x=T_BOILING, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Peak Temperature (K)', fontsize=11)
    ax.set_ylabel('Liquid Fraction (%)', fontsize=11)
    ax.set_title('Liquid Fraction vs Temperature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Timestep', fontsize=10)

    # Histogram of temperatures
    ax = axes[1, 1]
    ax.hist(peak_temps, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=T_MELTING, color='green', linestyle='--', linewidth=2, label='Melting')
    ax.axvline(x=T_BOILING, color='red', linestyle='--', linewidth=2, label='Boiling')
    ax.set_xlabel('Temperature (K)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Temperature Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_file = f"{OUTPUT_DIR}/thermal_diagnostics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to: {output_file}")

    # Recommendations
    print("\n" + "=" * 80)
    print("DIAGNOSTIC RECOMMENDATIONS")
    print("=" * 80)

    if onset_idx is not None and onset_idx < 100:
        print("\n1. EARLY INSTABILITY ONSET")
        print(f"   Issue: Instability begins at timestep {timesteps[onset_idx]}")
        print("   Likely cause: CFL condition violated or source term too strong")
        print("   Action: Reduce timestep or laser power")

    if liquid_fracs.max() < 0.05:
        print("\n2. LOW LIQUID FRACTION")
        print(f"   Issue: Maximum liquid fraction only {liquid_fracs.max()*100:.3f}%")
        print("   Likely cause: Latent heat not properly absorbed")
        print("   Action: Check VOF-thermal coupling and phase change implementation")

    if above_boiling > 0.5 * len(metrics):
        print("\n3. WIDESPREAD OVERHEATING")
        print(f"   Issue: {above_boiling/len(metrics)*100:.0f}% of timesteps exceed boiling")
        print("   Likely cause: Energy accumulation without proper dissipation")
        print("   Action: Verify thermal conductivity, boundary conditions, and energy conservation")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
