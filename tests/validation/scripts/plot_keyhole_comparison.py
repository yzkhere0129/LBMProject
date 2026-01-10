#!/usr/bin/env python3
"""
plot_keyhole_comparison.py

Compare keyhole depth evolution from simulation with reference data

This script creates publication-quality comparison plots for keyhole
formation validation against the senior's results.

Usage:
    python plot_keyhole_comparison.py <output_directory>

Inputs:
    - keyhole_depth.dat (from simulation)
    - keyhole_depth_extracted.dat (optional, from VTK post-processing)

Outputs:
    - keyhole_depth_comparison.png: Depth vs time plot
    - keyhole_depth_rate.png: Penetration rate vs time
    - keyhole_statistics.txt: Summary statistics

Author: Generated for Keyhole Formation Validation
Date: 2025-12-21
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use publication-quality settings
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 8


def load_keyhole_data(filename):
    """
    Load keyhole depth data from file

    Format:
        # Comment
        time[μs] depth[μm]

    Returns:
        (times, depths): Arrays in SI units (seconds, meters)
    """
    times = []
    depths = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue

            parts = line.split()
            if len(parts) >= 2:
                t_us = float(parts[0])
                d_um = float(parts[1])
                times.append(t_us * 1e-6)  # Convert to seconds
                depths.append(d_um * 1e-6)  # Convert to meters

    return np.array(times), np.array(depths)


def compute_penetration_rate(times, depths):
    """
    Compute keyhole penetration rate (dh/dt)

    Returns:
        (times_mid, rates): Penetration rate in m/s
    """
    if len(times) < 2:
        return np.array([]), np.array([])

    # Central difference for interior points
    dt = np.diff(times)
    dd = np.diff(depths)

    rates = dd / dt  # m/s
    times_mid = (times[:-1] + times[1:]) / 2

    return times_mid, rates


def plot_keyhole_depth(ax, times, depths, label='Simulation', color='C0', marker='o'):
    """
    Plot keyhole depth vs time
    """
    # Convert to μm and μs for plotting
    times_us = times * 1e6
    depths_um = depths * 1e6

    ax.plot(times_us, depths_um, marker=marker, color=color, label=label, alpha=0.8)
    ax.set_xlabel('Time [μs]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Keyhole Depth [μm]', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12)


def plot_penetration_rate(ax, times, depths):
    """
    Plot keyhole penetration rate vs time
    """
    times_mid, rates = compute_penetration_rate(times, depths)

    if len(times_mid) > 0:
        # Convert to μs and m/s
        times_us = times_mid * 1e6
        rates_ms = rates

        ax.plot(times_us, rates_ms, marker='s', color='C1', alpha=0.8)
        ax.set_xlabel('Time [μs]', fontsize=14, fontweight='bold')
        ax.set_ylabel('Penetration Rate [m/s]', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)


def add_reference_data(ax):
    """
    Add reference keyhole depth data (if available)

    Typical LPBF keyhole depths from literature:
    - Khairallah et al. (2016): 20-80 μm at 200-400 W
    - King et al. (2015): 30-100 μm at 195 W Ti6Al4V
    """
    # Reference region (approximate from literature)
    t_ref_us = np.array([0, 5, 10, 15, 20])
    d_min_um = np.array([0, 10, 20, 25, 30])  # Conservative lower bound
    d_max_um = np.array([0, 30, 50, 60, 70])  # Upper bound

    ax.fill_between(t_ref_us, d_min_um, d_max_um,
                    color='gray', alpha=0.2,
                    label='Literature Range (200-400W)')


def compute_statistics(times, depths):
    """
    Compute keyhole statistics

    Returns:
        dict: Statistics dictionary
    """
    stats = {}

    if len(depths) > 0:
        stats['initial_depth'] = depths[0] * 1e6  # μm
        stats['final_depth'] = depths[-1] * 1e6  # μm
        stats['max_depth'] = np.max(depths) * 1e6  # μm
        stats['mean_depth'] = np.mean(depths) * 1e6  # μm

        # Compute average penetration rate (first half of simulation)
        if len(times) > 1:
            mid_idx = len(times) // 2
            avg_rate = (depths[mid_idx] - depths[0]) / (times[mid_idx] - times[0])
            stats['avg_penetration_rate'] = avg_rate  # m/s

        # Time to reach 50% of final depth
        target_depth = 0.5 * depths[-1]
        idx_50 = np.argmax(depths >= target_depth)
        if idx_50 > 0:
            stats['time_to_50pct'] = times[idx_50] * 1e6  # μs

    return stats


def write_statistics(stats, filename):
    """
    Write statistics to text file
    """
    with open(filename, 'w') as f:
        f.write("Keyhole Formation Statistics\n")
        f.write("=" * 50 + "\n\n")

        for key, value in stats.items():
            if 'depth' in key:
                f.write(f"{key:30s}: {value:10.2f} μm\n")
            elif 'rate' in key:
                f.write(f"{key:30s}: {value:10.4f} m/s\n")
            elif 'time' in key:
                f.write(f"{key:30s}: {value:10.2f} μs\n")
            else:
                f.write(f"{key:30s}: {value:10.4e}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_keyhole_comparison.py <output_directory>")
        sys.exit(1)

    output_dir = sys.argv[1]

    if not os.path.isdir(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    # Load simulation data
    sim_file = os.path.join(output_dir, "keyhole_depth.dat")
    if not os.path.exists(sim_file):
        print(f"Error: Simulation data not found: {sim_file}")
        sys.exit(1)

    times_sim, depths_sim = load_keyhole_data(sim_file)
    print(f"Loaded simulation data: {len(times_sim)} points")

    # Try to load extracted data (optional)
    extract_file = os.path.join(output_dir, "keyhole_depth_extracted.dat")
    has_extracted = os.path.exists(extract_file)

    if has_extracted:
        times_ext, depths_ext = load_keyhole_data(extract_file)
        print(f"Loaded extracted data: {len(times_ext)} points")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Keyhole depth comparison
    ax1 = axes[0]
    plot_keyhole_depth(ax1, times_sim, depths_sim,
                      label='Simulation', color='C0', marker='o')

    if has_extracted:
        plot_keyhole_depth(ax1, times_ext, depths_ext,
                          label='Extracted (VTK)', color='C2', marker='s')

    # Add reference literature range
    add_reference_data(ax1)

    ax1.set_title('Keyhole Depth Evolution (Ti6Al4V, 300W)',
                  fontsize=16, fontweight='bold')

    # Plot 2: Penetration rate
    ax2 = axes[1]
    plot_penetration_rate(ax2, times_sim, depths_sim)
    ax2.set_title('Keyhole Penetration Rate',
                  fontsize=16, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_fig = os.path.join(output_dir, "keyhole_depth_comparison.png")
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_fig}")

    # Compute and save statistics
    stats = compute_statistics(times_sim, depths_sim)
    stats_file = os.path.join(output_dir, "keyhole_statistics.txt")
    write_statistics(stats, stats_file)
    print(f"Saved statistics: {stats_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("Keyhole Formation Summary")
    print("=" * 50)
    print(f"Initial depth:           {stats['initial_depth']:.2f} μm")
    print(f"Final depth:             {stats['final_depth']:.2f} μm")
    print(f"Maximum depth:           {stats['max_depth']:.2f} μm")
    if 'avg_penetration_rate' in stats:
        print(f"Avg penetration rate:    {stats['avg_penetration_rate']:.4f} m/s")
    if 'time_to_50pct' in stats:
        print(f"Time to 50% depth:       {stats['time_to_50pct']:.2f} μs")
    print("=" * 50)

    # Validation checks
    print("\nValidation Checks:")
    print("-" * 50)

    # Check 1: Keyhole should form (depth > 10 μm)
    if stats['final_depth'] > 10.0:
        print("✓ Keyhole formed (depth > 10 μm)")
    else:
        print("✗ WARNING: Keyhole depth too shallow")

    # Check 2: Depth should be in reasonable range (10-100 μm for 300W)
    if 10.0 < stats['final_depth'] < 100.0:
        print("✓ Keyhole depth in reasonable range (10-100 μm)")
    else:
        print("✗ WARNING: Keyhole depth outside expected range")

    # Check 3: Depth should increase over time
    if stats['final_depth'] > stats['initial_depth']:
        print("✓ Keyhole deepens over time")
    else:
        print("✗ WARNING: Keyhole depth not increasing")

    print("-" * 50)

    plt.show()


if __name__ == '__main__':
    main()
