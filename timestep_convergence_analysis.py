#!/usr/bin/env python3
"""
Timestep Convergence Analysis for LPBF Simulation
Extracts temperature evolution and computes temporal convergence metrics
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_simulation_log(log_file):
    """Extract timestep and temperature data from simulation log"""
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            # Match lines like: "  1000        100.00        1536.2          60.175"
            match = re.match(r'\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+', line)
            if match:
                step = int(match.group(1))
                time_us = float(match.group(2))
                temp_max = float(match.group(3))
                data.append((step, time_us, temp_max))
    return np.array(data)

def interpolate_at_time(data, target_times):
    """Interpolate temperature at specific times"""
    times = data[:, 1]  # Column 1 is time in μs
    temps = data[:, 2]  # Column 2 is T_max

    interpolated = []
    for t in target_times:
        # Find closest data point or interpolate
        if t <= times.min():
            interpolated.append(temps[0])
        elif t >= times.max():
            interpolated.append(temps[-1])
        else:
            idx = np.searchsorted(times, t)
            if idx > 0 and idx < len(times):
                # Linear interpolation
                t0, t1 = times[idx-1], times[idx]
                T0, T1 = temps[idx-1], temps[idx]
                T_interp = T0 + (T1 - T0) * (t - t0) / (t1 - t0)
                interpolated.append(T_interp)
            else:
                interpolated.append(temps[idx])

    return np.array(interpolated)

def calculate_cfl_thermal(dt, dx, alpha):
    """Calculate thermal diffusion CFL number"""
    # CFL_thermal = alpha * dt / dx^2
    return alpha * dt / (dx**2)

def main():
    # Parse all three simulation logs
    print("Parsing simulation logs...")
    data_020us = parse_simulation_log('/tmp/timestep_020us.log')
    data_010us = parse_simulation_log('/tmp/timestep_010us.log')
    data_005us = parse_simulation_log('/tmp/timestep_005us.log')

    print(f"Coarse (dt=0.2μs): {len(data_020us)} data points, time range: {data_020us[0,1]:.1f}-{data_020us[-1,1]:.1f} μs")
    print(f"Baseline (dt=0.1μs): {len(data_010us)} data points, time range: {data_010us[0,1]:.1f}-{data_010us[-1,1]:.1f} μs")
    print(f"Fine (dt=0.05μs): {len(data_005us)} data points, time range: {data_005us[0,1]:.1f}-{data_005us[-1,1]:.1f} μs")

    # Find common time range
    max_common_time = min(data_020us[-1,1], data_010us[-1,1], data_005us[-1,1])
    print(f"\nCommon time range: 0 - {max_common_time:.1f} μs")

    # Target times for comparison (evenly spaced within laser duration)
    target_times = np.array([50, 100, 150, 200, 250, 300])
    target_times = target_times[target_times <= max_common_time]

    print(f"\nInterpolating temperatures at times: {target_times} μs")

    # Interpolate temperatures at target times
    temps_020us = interpolate_at_time(data_020us, target_times)
    temps_010us = interpolate_at_time(data_010us, target_times)
    temps_005us = interpolate_at_time(data_005us, target_times)

    # Print comparison table
    print("\n" + "="*80)
    print("TEMPERATURE EVOLUTION COMPARISON")
    print("="*80)
    print(f"{'Time [μs]':<12} {'dt=0.2μs [K]':<15} {'dt=0.1μs [K]':<15} {'dt=0.05μs [K]':<15}")
    print("-"*80)
    for i, t in enumerate(target_times):
        print(f"{t:<12.1f} {temps_020us[i]:<15.1f} {temps_010us[i]:<15.1f} {temps_005us[i]:<15.1f}")
    print("="*80)

    # Calculate convergence metrics
    print("\n" + "="*80)
    print("TEMPORAL CONVERGENCE ANALYSIS")
    print("="*80)

    # Compare coarse vs baseline
    diff_coarse_baseline = np.abs(temps_020us - temps_010us)
    rel_error_coarse_baseline = diff_coarse_baseline / temps_010us * 100

    # Compare baseline vs fine
    diff_baseline_fine = np.abs(temps_010us - temps_005us)
    rel_error_baseline_fine = diff_baseline_fine / temps_005us * 100

    # Compare coarse vs fine (full range)
    diff_coarse_fine = np.abs(temps_020us - temps_005us)
    rel_error_coarse_fine = diff_coarse_fine / temps_005us * 100

    print(f"\n1. Coarse (0.2μs) vs Baseline (0.1μs):")
    print(f"   Max absolute difference: {diff_coarse_baseline.max():.2f} K")
    print(f"   Max relative error: {rel_error_coarse_baseline.max():.2f}%")
    print(f"   Average relative error: {rel_error_coarse_baseline.mean():.2f}%")

    print(f"\n2. Baseline (0.1μs) vs Fine (0.05μs):")
    print(f"   Max absolute difference: {diff_baseline_fine.max():.2f} K")
    print(f"   Max relative error: {rel_error_baseline_fine.max():.2f}%")
    print(f"   Average relative error: {rel_error_baseline_fine.mean():.2f}%")

    print(f"\n3. Coarse (0.2μs) vs Fine (0.05μs):")
    print(f"   Max absolute difference: {diff_coarse_fine.max():.2f} K")
    print(f"   Max relative error: {rel_error_coarse_fine.max():.2f}%")
    print(f"   Average relative error: {rel_error_coarse_fine.mean():.2f}%")

    # CFL analysis
    print("\n" + "="*80)
    print("CFL NUMBER ANALYSIS")
    print("="*80)

    # Ti6Al4V thermal properties
    alpha = 5.8e-6  # m^2/s (thermal diffusivity)
    dx = 2.0e-6     # m (cell size)

    dt_values = [0.2e-6, 0.1e-6, 0.05e-6]  # seconds
    dt_labels = ['0.2μs', '0.1μs', '0.05μs']

    print(f"\nThermal diffusivity (Ti6Al4V): α = {alpha:.2e} m²/s")
    print(f"Cell size: dx = {dx:.2e} m")
    print(f"Stability limit (CFL < 0.5): dt_max = {0.5 * dx**2 / alpha:.2e} s = {0.5 * dx**2 / alpha * 1e6:.3f} μs")
    print()

    for dt, label in zip(dt_values, dt_labels):
        cfl = calculate_cfl_thermal(dt, dx, alpha)
        stability = "STABLE" if cfl < 0.5 else "UNSTABLE"
        print(f"dt = {label:8s}: CFL = {cfl:.4f}  [{stability}]")

    # Convergence assessment
    print("\n" + "="*80)
    print("CONVERGENCE ASSESSMENT")
    print("="*80)

    tolerance = 5.0  # 5% tolerance

    if rel_error_baseline_fine.max() < tolerance:
        convergence_status = "PASS"
        print(f"\n✓ PASS: Maximum deviation between baseline and fine timesteps is {rel_error_baseline_fine.max():.2f}%")
        print(f"  (within {tolerance}% tolerance)")
    else:
        convergence_status = "FAIL"
        print(f"\n✗ FAIL: Maximum deviation between baseline and fine timesteps is {rel_error_baseline_fine.max():.2f}%")
        print(f"  (exceeds {tolerance}% tolerance)")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if convergence_status == "PASS":
        print("\nOptimal timestep: dt = 0.1 μs (baseline)")
        print("Rationale:")
        print("  - Achieves temporal convergence within 5% tolerance")
        print("  - CFL number well within stability limit")
        print("  - 2x faster than fine timestep with negligible accuracy loss")
        print("  - Recommended for production runs")
    else:
        print("\nOptimal timestep: dt = 0.05 μs (fine)")
        print("Rationale:")
        print("  - Required for temporal convergence")
        print("  - Baseline timestep shows insufficient accuracy")
        print("  - Accept 2x computational cost for accuracy")

    # Plot temperature evolution
    print("\nGenerating convergence plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Temperature evolution
    ax1.plot(data_020us[:, 1], data_020us[:, 2], 'b-', linewidth=2, label='dt = 0.2 μs (coarse)', alpha=0.7)
    ax1.plot(data_010us[:, 1], data_010us[:, 2], 'g-', linewidth=2, label='dt = 0.1 μs (baseline)', alpha=0.7)
    ax1.plot(data_005us[:, 1], data_005us[:, 2], 'r-', linewidth=2, label='dt = 0.05 μs (fine)', alpha=0.7)
    ax1.scatter(target_times, temps_020us, c='blue', s=60, zorder=5)
    ax1.scatter(target_times, temps_010us, c='green', s=60, zorder=5)
    ax1.scatter(target_times, temps_005us, c='red', s=60, zorder=5)
    ax1.set_xlabel('Time [μs]', fontsize=12)
    ax1.set_ylabel('Maximum Temperature [K]', fontsize=12)
    ax1.set_title('Timestep Convergence Study: Temperature Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_common_time)

    # Plot 2: Relative error
    ax2.plot(target_times, rel_error_coarse_baseline, 'b-o', linewidth=2, label='Coarse vs Baseline', markersize=8)
    ax2.plot(target_times, rel_error_baseline_fine, 'g-s', linewidth=2, label='Baseline vs Fine', markersize=8)
    ax2.plot(target_times, rel_error_coarse_fine, 'r-^', linewidth=2, label='Coarse vs Fine', markersize=8)
    ax2.axhline(y=tolerance, color='k', linestyle='--', linewidth=1.5, label=f'{tolerance}% tolerance')
    ax2.set_xlabel('Time [μs]', fontsize=12)
    ax2.set_ylabel('Relative Error [%]', fontsize=12)
    ax2.set_title('Temporal Discretization Error', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_common_time)
    ax2.set_ylim(0, max(rel_error_coarse_fine.max() * 1.2, tolerance * 2))

    plt.tight_layout()
    output_path = '/home/yzk/LBMProject/timestep_convergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
