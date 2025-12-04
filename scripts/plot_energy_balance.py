#!/usr/bin/env python3
"""
Energy Balance Visualization Script

This script reads the energy_balance.dat file produced by MultiphysicsSolver
and generates comprehensive energy balance plots for validation.

Week 3 P1: Essential for debugging energy conservation issues

Usage:
    python plot_energy_balance.py energy_balance.dat
    python plot_energy_balance.py energy_balance.dat --output plots/
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

def load_energy_data(filename):
    """
    Load energy balance data from ASCII file

    Returns:
        dict: Dictionary with arrays for each column
    """
    # Read data (skip header lines starting with #)
    data = np.loadtxt(filename)

    return {
        'time': data[:, 0],          # Time [s]
        'step': data[:, 1].astype(int),  # Step number
        'E_thermal': data[:, 2],     # Thermal energy [J]
        'E_kinetic': data[:, 3],     # Kinetic energy [J]
        'E_latent': data[:, 4],      # Latent energy [J]
        'E_total': data[:, 5],       # Total energy [J]
        'P_laser': data[:, 6],       # Laser power [W]
        'P_evap': data[:, 7],        # Evaporation power [W]
        'P_rad': data[:, 8],         # Radiation power [W]
        'P_substrate': data[:, 9],   # Substrate power [W]
        'dE_dt_computed': data[:, 10],  # dE/dt from state [W]
        'dE_dt_balance': data[:, 11],   # dE/dt from balance [W]
        'error_percent': data[:, 12]    # Error [%]
    }

def plot_energy_evolution(data, output_dir=None):
    """
    Plot energy component evolution over time
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Convert time to microseconds for readability
    t_us = data['time'] * 1e6

    # Top panel: Total energy and components
    ax1.plot(t_us, data['E_total'], 'k-', linewidth=2, label='E_total')
    ax1.plot(t_us, data['E_thermal'], 'r-', label='E_thermal (sensible)')
    ax1.plot(t_us, data['E_kinetic'], 'b-', label='E_kinetic (fluid)')
    ax1.plot(t_us, data['E_latent'], 'g-', label='E_latent (phase change)')

    ax1.set_ylabel('Energy [J]', fontsize=12)
    ax1.set_title('Energy Balance Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Relative fractions
    E_total_safe = np.maximum(data['E_total'], 1e-10)  # Avoid division by zero
    frac_thermal = 100.0 * data['E_thermal'] / E_total_safe
    frac_kinetic = 100.0 * data['E_kinetic'] / E_total_safe
    frac_latent = 100.0 * data['E_latent'] / E_total_safe

    ax2.plot(t_us, frac_thermal, 'r-', label='Thermal')
    ax2.plot(t_us, frac_kinetic, 'b-', label='Kinetic')
    ax2.plot(t_us, frac_latent, 'g-', label='Latent')

    ax2.set_xlabel('Time [μs]', fontsize=12)
    ax2.set_ylabel('Energy Fraction [%]', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'energy_evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.savefig('energy_evolution.png', dpi=300, bbox_inches='tight')
        print("Saved: energy_evolution.png")

    plt.close()

def plot_power_balance(data, output_dir=None):
    """
    Plot power terms and energy balance
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    t_us = data['time'] * 1e6

    # Top panel: Power terms
    ax1.plot(t_us, data['P_laser'], 'r-', linewidth=2, label='P_laser (input)')
    ax1.plot(t_us, -data['P_evap'], 'b-', label='P_evap (output)')
    ax1.plot(t_us, -data['P_rad'], 'g-', label='P_rad (output)')
    ax1.plot(t_us, -data['P_substrate'], 'm-', label='P_substrate (output)')
    ax1.plot(t_us, data['dE_dt_computed'], 'k--', linewidth=2, label='dE/dt (storage)')

    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Power [W]', fontsize=12)
    ax1.set_title('Power Balance Terms', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: dE/dt comparison
    ax2.plot(t_us, data['dE_dt_computed'], 'b-', linewidth=2, label='dE/dt (from state)')
    ax2.plot(t_us, data['dE_dt_balance'], 'r--', linewidth=2, label='dE/dt (from balance)')

    ax2.set_xlabel('Time [μs]', fontsize=12)
    ax2.set_ylabel('Energy Rate [W]', fontsize=12)
    ax2.set_title('Energy Conservation Check', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'power_balance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.savefig('power_balance.png', dpi=300, bbox_inches='tight')
        print("Saved: power_balance.png")

    plt.close()

def plot_error_analysis(data, output_dir=None):
    """
    Plot energy balance error over time
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    t_us = data['time'] * 1e6

    # Plot error
    ax.plot(t_us, data['error_percent'], 'k-', linewidth=2, label='Error')

    # Add threshold lines
    ax.axhline(5.0, color='orange', linestyle='--', linewidth=1.5, label='5% Target')
    ax.axhline(10.0, color='red', linestyle='--', linewidth=1.5, label='10% Limit')
    ax.axhline(-5.0, color='orange', linestyle='--', linewidth=1.5)
    ax.axhline(-10.0, color='red', linestyle='--', linewidth=1.5)

    # Shade acceptable region
    ax.fill_between(t_us, -5, 5, alpha=0.1, color='green', label='Acceptable (<5%)')

    ax.set_xlabel('Time [μs]', fontsize=12)
    ax.set_ylabel('Energy Balance Error [%]', fontsize=12)
    ax.set_title('Energy Conservation Error', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    mean_error = np.mean(np.abs(data['error_percent']))
    max_error = np.max(np.abs(data['error_percent']))
    rms_error = np.sqrt(np.mean(data['error_percent']**2))

    stats_text = f'Statistics:\n'
    stats_text += f'  Mean |error|: {mean_error:.2f}%\n'
    stats_text += f'  Max |error|:  {max_error:.2f}%\n'
    stats_text += f'  RMS error:    {rms_error:.2f}%'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'error_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: error_analysis.png")

    plt.close()

def print_summary(data):
    """
    Print summary statistics to console
    """
    print("\n" + "="*70)
    print("ENERGY BALANCE SUMMARY")
    print("="*70)

    # Time range
    print(f"\nSimulation time: {data['time'][0]*1e6:.2f} - {data['time'][-1]*1e6:.2f} μs")
    print(f"Number of snapshots: {len(data['time'])}")

    # Energy statistics
    print("\nEnergy Statistics:")
    print(f"  Initial E_total:  {data['E_total'][0]:.3e} J")
    print(f"  Final E_total:    {data['E_total'][-1]:.3e} J")
    print(f"  Energy change:    {data['E_total'][-1] - data['E_total'][0]:.3e} J")

    # Power statistics
    print("\nPower Statistics (time-averaged):")
    print(f"  <P_laser>:      {np.mean(data['P_laser']):.2f} W")
    print(f"  <P_evap>:       {np.mean(data['P_evap']):.2f} W")
    print(f"  <P_rad>:        {np.mean(data['P_rad']):.2f} W")
    print(f"  <P_substrate>:  {np.mean(data['P_substrate']):.2f} W")

    # Error statistics
    print("\nError Statistics:")
    mean_error = np.mean(np.abs(data['error_percent']))
    max_error = np.max(np.abs(data['error_percent']))
    rms_error = np.sqrt(np.mean(data['error_percent']**2))

    print(f"  Mean |error|: {mean_error:.2f}%")
    print(f"  Max |error|:  {max_error:.2f}%")
    print(f"  RMS error:    {rms_error:.2f}%")

    # Validation
    print("\nValidation:")
    if mean_error < 5.0:
        print("  ✓ PASS: Mean error < 5%")
    elif mean_error < 10.0:
        print("  ⚠ WARNING: Mean error 5-10% (target < 5%)")
    else:
        print("  ✗ FAIL: Mean error > 10%")

    if max_error < 10.0:
        print("  ✓ PASS: Max error < 10%")
    else:
        print("  ✗ FAIL: Max error > 10%")

    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize energy balance data')
    parser.add_argument('filename', help='Energy balance data file (energy_balance.dat)')
    parser.add_argument('--output', '-o', default=None, help='Output directory for plots')
    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.filename):
        print(f"Error: File not found: {args.filename}")
        return 1

    # Create output directory if specified
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {args.filename}")
    data = load_energy_data(args.filename)

    # Print summary
    print_summary(data)

    # Generate plots
    print("Generating plots...")
    plot_energy_evolution(data, args.output)
    plot_power_balance(data, args.output)
    plot_error_analysis(data, args.output)

    print("\nDone!")
    return 0

if __name__ == '__main__':
    exit(main())
