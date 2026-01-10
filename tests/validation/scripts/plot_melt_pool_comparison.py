#!/usr/bin/env python3
"""
Melt Pool Comparison Plotting Script

This script generates comparison plots for the laser melting validation test,
comparing simulation results with expected behavior.

Usage:
    python3 plot_melt_pool_comparison.py <csv_file>

Output:
    - PNG plots showing melt pool depth vs time
    - Temperature evolution
    - Velocity evolution
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def load_data(csv_file):
    """
    Load melt pool data from CSV file

    Args:
        csv_file: Path to CSV file (from test or extraction script)

    Returns:
        pandas DataFrame
    """
    df = pd.read_csv(csv_file)
    return df


def plot_melt_pool_depth(df, output_dir, laser_shutoff_time=50.0):
    """
    Plot melt pool depth vs time

    Args:
        df: DataFrame with columns [time_us, depth_um, ...]
        output_dir: Directory to save plots
        laser_shutoff_time: Time when laser turns off [μs]
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot depth
    ax.plot(df['time_us'], df['depth_um'], 'b-', linewidth=2, label='Melt Pool Depth')

    # Mark laser shutoff
    ax.axvline(laser_shutoff_time, color='r', linestyle='--', linewidth=1.5,
               label=f'Laser Shutoff ({laser_shutoff_time} μs)')

    # Highlight key times
    key_times = [25, 50, 60, 75]
    for t_key in key_times:
        idx = (np.abs(df['time_us'] - t_key)).idxmin()
        depth_at_t = df.loc[idx, 'depth_um']
        ax.plot(t_key, depth_at_t, 'ro', markersize=8)
        ax.annotate(f'{depth_at_t:.1f} μm',
                    xy=(t_key, depth_at_t),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax.set_xlabel('Time [μs]', fontsize=12)
    ax.set_ylabel('Melt Pool Depth [μm]', fontsize=12)
    ax.set_title('Laser Melting Validation: Melt Pool Depth vs Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'melt_pool_depth_vs_time.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_temperature_evolution(df, output_dir, laser_shutoff_time=50.0):
    """
    Plot maximum temperature vs time

    Args:
        df: DataFrame with columns [time_us, max_temp_K, ...]
        output_dir: Directory to save plots
        laser_shutoff_time: Time when laser turns off [μs]
    """
    if 'max_temp_K' not in df.columns:
        print("Warning: 'max_temp_K' column not found, skipping temperature plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot temperature
    ax.plot(df['time_us'], df['max_temp_K'], 'r-', linewidth=2, label='Max Temperature')

    # Mark melting point
    T_liquidus = 1923.0  # Ti6Al4V
    ax.axhline(T_liquidus, color='orange', linestyle='--', linewidth=1.5,
               label=f'Liquidus Temp ({T_liquidus} K)')

    # Mark laser shutoff
    ax.axvline(laser_shutoff_time, color='gray', linestyle='--', linewidth=1.5,
               label=f'Laser Shutoff ({laser_shutoff_time} μs)')

    ax.set_xlabel('Time [μs]', fontsize=12)
    ax.set_ylabel('Temperature [K]', fontsize=12)
    ax.set_title('Laser Melting Validation: Temperature Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'temperature_evolution.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_velocity_evolution(df, output_dir, laser_shutoff_time=50.0):
    """
    Plot maximum velocity (Marangoni flow) vs time

    Args:
        df: DataFrame with columns [time_us, max_velocity_m_s, ...]
        output_dir: Directory to save plots
        laser_shutoff_time: Time when laser turns off [μs]
    """
    if 'max_velocity_m_s' not in df.columns:
        print("Warning: 'max_velocity_m_s' column not found, skipping velocity plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot velocity
    ax.plot(df['time_us'], df['max_velocity_m_s'], 'g-', linewidth=2, label='Max Velocity (Marangoni)')

    # Mark realistic range (literature values)
    ax.axhspan(0.5, 2.0, alpha=0.2, color='green', label='Literature Range (0.5-2 m/s)')

    # Mark laser shutoff
    ax.axvline(laser_shutoff_time, color='gray', linestyle='--', linewidth=1.5,
               label=f'Laser Shutoff ({laser_shutoff_time} μs)')

    ax.set_xlabel('Time [μs]', fontsize=12)
    ax.set_ylabel('Velocity [m/s]', fontsize=12)
    ax.set_title('Laser Melting Validation: Marangoni Velocity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'velocity_evolution.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_combined_view(df, output_dir, laser_shutoff_time=50.0):
    """
    Create a combined 3-panel plot

    Args:
        df: DataFrame with all columns
        output_dir: Directory to save plots
        laser_shutoff_time: Time when laser turns off [μs]
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Panel 1: Melt pool depth
    ax1 = axes[0]
    ax1.plot(df['time_us'], df['depth_um'], 'b-', linewidth=2)
    ax1.axvline(laser_shutoff_time, color='r', linestyle='--', linewidth=1.5,
                label=f'Laser Shutoff ({laser_shutoff_time} μs)')
    ax1.set_ylabel('Melt Pool Depth [μm]', fontsize=11)
    ax1.set_title('Laser Melting Validation: Multiphysics Response', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Panel 2: Temperature
    if 'max_temp_K' in df.columns:
        ax2 = axes[1]
        ax2.plot(df['time_us'], df['max_temp_K'], 'r-', linewidth=2)
        ax2.axhline(1923.0, color='orange', linestyle='--', linewidth=1.5, label='Liquidus (1923 K)')
        ax2.axvline(laser_shutoff_time, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.set_ylabel('Max Temperature [K]', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'Temperature data not available',
                     ha='center', va='center', transform=axes[1].transAxes)

    # Panel 3: Velocity
    if 'max_velocity_m_s' in df.columns:
        ax3 = axes[2]
        ax3.plot(df['time_us'], df['max_velocity_m_s'], 'g-', linewidth=2)
        ax3.axhspan(0.5, 2.0, alpha=0.2, color='green', label='Literature Range (0.5-2 m/s)')
        ax3.axvline(laser_shutoff_time, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.set_xlabel('Time [μs]', fontsize=11)
        ax3.set_ylabel('Max Velocity [m/s]', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
    else:
        axes[2].text(0.5, 0.5, 'Velocity data not available',
                     ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'combined_view.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def print_validation_summary(df, laser_shutoff_time=50.0):
    """
    Print validation summary statistics

    Args:
        df: DataFrame with all columns
        laser_shutoff_time: Time when laser turns off [μs]
    """
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    # Peak depth
    max_depth = df['depth_um'].max()
    max_depth_time = df.loc[df['depth_um'].idxmax(), 'time_us']
    print(f"Peak melt pool depth: {max_depth:.2f} μm at t = {max_depth_time:.2f} μs")

    # Depth at key times
    print("\nMelt pool depth at key times:")
    for t_key in [25, 50, 60, 75]:
        idx = (np.abs(df['time_us'] - t_key)).idxmin()
        depth = df.loc[idx, 'depth_um']
        print(f"  t = {t_key:5.1f} μs: {depth:7.2f} μm")

    # Temperature
    if 'max_temp_K' in df.columns:
        max_T = df['max_temp_K'].max()
        print(f"\nMaximum temperature: {max_T:.1f} K")

    # Velocity
    if 'max_velocity_m_s' in df.columns:
        max_v = df['max_velocity_m_s'].max()
        print(f"Maximum Marangoni velocity: {max_v:.3f} m/s")

    # Solidification check
    idx_shutoff = (np.abs(df['time_us'] - laser_shutoff_time)).idxmin()
    idx_end = len(df) - 1
    depth_at_shutoff = df.loc[idx_shutoff, 'depth_um']
    depth_at_end = df.loc[idx_end, 'depth_um']

    print(f"\nSolidification after laser shutoff:")
    print(f"  Depth at shutoff ({laser_shutoff_time} μs): {depth_at_shutoff:.2f} μm")
    print(f"  Depth at end ({df.loc[idx_end, 'time_us']:.1f} μs): {depth_at_end:.2f} μm")

    if depth_at_end < depth_at_shutoff:
        print("  [PASS] Melt pool shrank after laser shutoff")
    else:
        print("  [FAIL] Melt pool did not shrink")

    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    checks_passed = 0
    total_checks = 0

    # Check 1: Melt pool formed
    total_checks += 1
    if max_depth > 0:
        print("[PASS] Melt pool formed (depth > 0)")
        checks_passed += 1
    else:
        print("[FAIL] Melt pool did not form")

    # Check 2: Depth in reasonable range
    total_checks += 1
    if 5.0 <= max_depth <= 100.0:
        print("[PASS] Melt pool depth in reasonable range (5-100 μm)")
        checks_passed += 1
    else:
        print(f"[FAIL] Melt pool depth out of range ({max_depth:.2f} μm)")

    # Check 3: Marangoni velocity realistic
    if 'max_velocity_m_s' in df.columns:
        total_checks += 1
        max_v = df['max_velocity_m_s'].max()
        if 0.1 <= max_v <= 10.0:
            print("[PASS] Marangoni velocity in realistic range (0.1-10 m/s)")
            checks_passed += 1
        else:
            print(f"[FAIL] Marangoni velocity out of range ({max_v:.3f} m/s)")

    # Check 4: Solidification occurred
    total_checks += 1
    if depth_at_end < depth_at_shutoff:
        print("[PASS] Melt pool shrank after laser shutoff")
        checks_passed += 1
    else:
        print("[FAIL] Melt pool did not shrink")

    print("\n" + "=" * 80)
    print(f"VALIDATION RESULT: {checks_passed}/{total_checks} checks passed")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_melt_pool_comparison.py <csv_file>")
        print("Example: python3 plot_melt_pool_comparison.py output_laser_melting_senior/melt_pool_depth.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not os.path.isfile(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    print(f"Loading data from: {csv_file}")

    # Load data
    df = load_data(csv_file)
    print(f"Loaded {len(df)} data points")

    # Determine output directory
    output_dir = os.path.dirname(csv_file)
    if not output_dir:
        output_dir = '.'

    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Generate plots
    print("\nGenerating plots...")
    plot_melt_pool_depth(df, output_dir)
    plot_temperature_evolution(df, output_dir)
    plot_velocity_evolution(df, output_dir)
    plot_combined_view(df, output_dir)

    # Print validation summary
    print_validation_summary(df)

    print("\n" + "=" * 80)
    print("Plotting complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
