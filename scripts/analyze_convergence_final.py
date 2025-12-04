#!/usr/bin/env python3
"""
Final convergence analysis after Fluid LBM fix

This script analyzes temperature convergence across different timesteps
(dt = 0.05, 0.08, 0.10 μs) at steady state to verify that the Fluid LBM
fix has achieved temporal convergence.

Usage:
    python3 analyze_convergence_final.py

Expected input files:
    - convergence_fixed_dt005us.log (dt = 0.05 μs)
    - convergence_fixed_dt008us.log (dt = 0.08 μs)
    - steady_state_verification.log (dt = 0.10 μs)

Decision criteria:
    - PASS: Variation < 10% (FULL GO for Week 3)
    - MARGINAL: Variation 10-15% (CONDITIONAL GO for Week 3)
    - FAIL: Variation > 15% (NO GO - investigate further)
"""

import numpy as np
import sys
import os
from pathlib import Path

def parse_temperature_log(filepath, time_window=(13000, 15000)):
    """
    Parse temperature values from log file within specified time window

    Args:
        filepath: Path to log file
        time_window: Tuple (t_start, t_end) in microseconds

    Returns:
        List of temperature values
    """
    temps = []

    if not os.path.exists(filepath):
        print(f"WARNING: File not found: {filepath}")
        return temps

    with open(filepath, 'r') as f:
        for line in f:
            # Look for lines with temperature data
            # Expected format: "t = 14950.00 μs: T_max = 2450.5 K"
            if 'T_max' in line and 't = ' in line:
                try:
                    # Extract time
                    t_str = line.split('t = ')[1].split()[0]
                    t = float(t_str)

                    # Check if in time window
                    if time_window[0] <= t <= time_window[1]:
                        # Extract temperature
                        T_str = line.split('T_max')[1].split('=')[1].split()[0]
                        T = float(T_str)
                        temps.append(T)
                except (ValueError, IndexError) as e:
                    continue  # Skip malformed lines

    return temps

def compute_statistics(temps):
    """Compute statistical measures of temperature data"""
    if not temps:
        return None

    temps_array = np.array(temps)
    return {
        'mean': np.mean(temps_array),
        'std': np.std(temps_array),
        'min': np.min(temps_array),
        'max': np.max(temps_array),
        'count': len(temps_array)
    }

def analyze_convergence(log_files):
    """
    Analyze temperature convergence across timesteps

    Args:
        log_files: Dictionary {dt_value: filepath}

    Returns:
        Boolean indicating pass/fail
    """
    print("=" * 70)
    print("CONVERGENCE ANALYSIS RESULTS (Post-Fluid-LBM-Fix)")
    print("=" * 70)
    print()

    # Parse all log files
    results = {}
    for dt, filepath in log_files.items():
        print(f"Parsing {filepath}...")
        temps = parse_temperature_log(filepath)

        if temps:
            stats = compute_statistics(temps)
            results[dt] = stats
            print(f"  Found {stats['count']} data points")
        else:
            print(f"  WARNING: No temperature data found!")

    print()

    if len(results) < 2:
        print("ERROR: Need at least 2 successful runs for convergence analysis")
        print("Run the convergence simulations first!")
        return False

    # Display results
    print("Steady-State Temperatures (last 2000 μs):")
    print("-" * 70)
    print(f"{'dt (μs)':<10} {'T_mean (K)':<15} {'T_std (K)':<15} {'Range (K)':<15}")
    print("-" * 70)

    T_means = []
    for dt in sorted(results.keys()):
        stats = results[dt]
        T_means.append(stats['mean'])
        T_range = stats['max'] - stats['min']
        print(f"{dt:<10.2f} {stats['mean']:<15.1f} {stats['std']:<15.1f} {T_range:<15.1f}")

    print("-" * 70)
    print()

    # Convergence metrics
    T_avg = np.mean(T_means)
    T_range = max(T_means) - min(T_means)
    variation_pct = (T_range / T_avg) * 100.0 if T_avg > 0 else 100.0

    print("Convergence Metrics:")
    print(f"  Average T_max across timesteps: {T_avg:.1f} K")
    print(f"  Temperature range: {T_range:.1f} K")
    print(f"  Variation: {variation_pct:.2f}%")
    print()

    # Verdict
    if variation_pct < 10.0:
        verdict = "PASS"
        symbol = "✓"
        go_status = "FULL GO for Week 3"
        explanation = "Temporal convergence achieved!"
    elif variation_pct < 15.0:
        verdict = "MARGINAL"
        symbol = "⚠"
        go_status = "CONDITIONAL GO for Week 3"
        explanation = "Borderline convergence - use baseline dt=0.10 μs only"
    else:
        verdict = "FAIL"
        symbol = "✗"
        go_status = "NO GO - Further investigation needed"
        explanation = "Poor convergence - timestep dependence still present"

    print(f"{symbol} {verdict} - {explanation}")
    print(f"Week 3 Status: {go_status}")
    print("=" * 70)
    print()

    # Detailed comparison
    if len(results) >= 2:
        print("Pairwise Comparisons:")
        print("-" * 70)
        dt_list = sorted(results.keys())
        for i in range(len(dt_list) - 1):
            dt1, dt2 = dt_list[i], dt_list[i+1]
            T1 = results[dt1]['mean']
            T2 = results[dt2]['mean']
            diff = abs(T1 - T2)
            diff_pct = (diff / T_avg) * 100.0
            print(f"  dt={dt1:.2f} vs dt={dt2:.2f}: ΔT = {diff:.1f} K ({diff_pct:.2f}%)")
        print("-" * 70)
        print()

    return variation_pct < 10.0

def main():
    # Define log files to analyze
    build_dir = Path("/home/yzk/LBMProject/build")

    log_files = {
        0.05: build_dir / "convergence_fixed_dt005us.log",
        0.08: build_dir / "convergence_fixed_dt008us.log",
        0.10: build_dir / "steady_state_verification.log"
    }

    # Convert to string paths
    log_files = {dt: str(path) for dt, path in log_files.items()}

    # Run analysis
    success = analyze_convergence(log_files)

    # Write summary to file
    summary_path = build_dir / "convergence_analysis_summary.txt"
    print(f"Writing summary to {summary_path}")
    with open(summary_path, 'w') as f:
        f.write("CONVERGENCE ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Result: {'PASS' if success else 'FAIL'}\n")
        f.write(f"Date: {np.datetime64('now')}\n")

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
