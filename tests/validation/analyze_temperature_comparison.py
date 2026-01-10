#!/usr/bin/env python3
"""
Temperature Comparison Analysis for Laser Melting Validation

This script compares the temperature evolution between:
1. Our simulation (before parameter changes)
2. Our simulation (after parameter changes)
3. walberla reference (target values)

Author: Temperature Validation Analysis
Date: 2025-12-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_temperature_evolution():
    """Analyze and compare temperature evolution"""

    # Load current results
    output_dir = Path("/home/yzk/LBMProject/tests/validation/output_laser_melting_senior")
    csv_file = output_dir / "melt_pool_depth.csv"

    if not csv_file.exists():
        print(f"ERROR: CSV file not found at {csv_file}")
        return

    df = pd.read_csv(csv_file)

    # Extract key metrics
    print("=" * 80)
    print("LASER MELTING VALIDATION - TEMPERATURE COMPARISON")
    print("=" * 80)
    print()

    # Peak temperature analysis
    peak_temp = df['max_temp_K'].max()
    peak_time = df.loc[df['max_temp_K'].idxmax(), 'time_us']

    print("CURRENT SIMULATION RESULTS:")
    print("-" * 80)
    print(f"Peak Temperature:        {peak_temp:8.1f} K")
    print(f"Peak Time:               {peak_time:8.1f} μs")
    print()

    # Temperature at key times
    print("Temperature at key times:")
    key_times = [25, 50, 60, 75, 100]
    for t in key_times:
        row = df[df['time_us'] == t]
        if not row.empty:
            temp = row['max_temp_K'].values[0]
            depth = row['depth_um'].values[0]
            velocity = row['max_velocity_m_s'].values[0]
            print(f"  t = {t:3d} μs:  T = {temp:7.1f} K  |  Depth = {depth:6.2f} μm  |  v_max = {velocity:.3f} m/s")
    print()

    # Comparison with walberla reference
    print("=" * 80)
    print("COMPARISON WITH REFERENCE VALUES")
    print("=" * 80)
    print()

    # Reference values from walberla simulation
    walberla_peak = 17500.0  # K (from senior's simulation)
    previous_peak = 11848.0  # K (our previous result before changes)

    print("Peak Temperature Comparison:")
    print("-" * 80)
    print(f"  walberla (target):       {walberla_peak:8.1f} K  (100.0%)")
    print(f"  Previous (before):       {previous_peak:8.1f} K  ({100*previous_peak/walberla_peak:5.1f}%)")
    print(f"  Current (after):         {peak_temp:8.1f} K  ({100*peak_temp/walberla_peak:5.1f}%)")
    print()

    # Calculate percentage differences
    diff_walberla = abs(peak_temp - walberla_peak) / walberla_peak * 100
    diff_previous = abs(peak_temp - previous_peak) / previous_peak * 100
    improvement = abs(walberla_peak - peak_temp) / abs(walberla_peak - previous_peak) * 100

    print("Percentage Differences:")
    print("-" * 80)
    print(f"  Current vs walberla:     {diff_walberla:6.1f}% difference")
    print(f"  Current vs Previous:     {diff_previous:6.1f}% change")
    print(f"  Gap closure:             {100-improvement:6.1f}% (remaining gap)")
    print()

    # Identify potential issues
    print("=" * 80)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    print()

    if peak_temp < 0.5 * walberla_peak:
        print("ISSUE: Temperature is much lower than expected (< 50% of target)")
        print()
        print("Possible causes:")
        print("  1. Laser power too low or absorption too low")
        print("  2. Thermal diffusivity too high (heat spreading too fast)")
        print("  3. Specific heat too high (requires more energy to heat)")
        print("  4. Time step too large (numerical diffusion)")
        print("  5. Grid resolution too coarse")
        print()
    elif peak_temp < 0.8 * walberla_peak:
        print("ISSUE: Temperature moderately lower than expected (50-80% of target)")
        print()
        print("Possible causes:")
        print("  1. Thermal properties not fully matched")
        print("  2. Boundary condition differences")
        print("  3. Numerical scheme differences")
        print()
    elif peak_temp > 1.2 * walberla_peak:
        print("ISSUE: Temperature higher than expected (> 120% of target)")
        print()
        print("Possible causes:")
        print("  1. Laser power or absorption too high")
        print("  2. Heat loss mechanisms not properly implemented")
        print("  3. Thermal diffusivity too low")
        print()
    else:
        print("STATUS: Temperature is reasonably close to target (within 20%)")
        print()

    # Thermal physics check
    print("Thermal Physics Parameters (from simulation output):")
    print("-" * 80)
    print("These should be verified against walberla:")
    print("  - Thermal diffusivity (alpha)")
    print("  - Lattice tau/omega")
    print("  - CFL number")
    print("  - Grid resolution and time step")
    print()

    # Energy balance check
    laser_power = 200.0  # W
    spot_radius = 50e-6  # m
    absorptivity = 0.35
    absorbed_power = laser_power * absorptivity  # W

    print("Energy Balance:")
    print("-" * 80)
    print(f"  Laser power:             {laser_power:8.1f} W")
    print(f"  Absorptivity:            {absorptivity:8.2f}")
    print(f"  Absorbed power:          {absorbed_power:8.1f} W")
    print()

    # Estimate energy input
    laser_on_time = 50e-6  # s
    total_energy = absorbed_power * laser_on_time  # J

    print(f"  Laser on time:           {laser_on_time*1e6:8.1f} μs")
    print(f"  Total energy input:      {total_energy:8.4f} J")
    print()

    # Estimate required energy for temperature rise
    # Domain: 150×300×300 μm³ = 40×80×80 cells
    dx = 3.75e-6  # m
    volume = (40 * dx) * (80 * dx) * (80 * dx)  # m³
    rho = 4420  # kg/m³ (solid)
    cp = 610  # J/(kg·K)
    T_initial = 300  # K
    mass = rho * volume  # kg

    delta_T = peak_temp - T_initial
    energy_required = mass * cp * delta_T  # J

    print("Estimated Energy for Temperature Rise:")
    print("-" * 80)
    print(f"  Domain volume:           {volume*1e9:8.2e} mm³")
    print(f"  Mass:                    {mass*1e6:8.2f} mg")
    print(f"  Temperature rise:        {delta_T:8.1f} K")
    print(f"  Energy required:         {energy_required:8.4f} J")
    print(f"  Efficiency:              {energy_required/total_energy*100:6.1f}%")
    print()

    if energy_required > total_energy * 2:
        print("WARNING: Energy required is much higher than energy input!")
        print("  This suggests energy is being conserved but temperature is limited")
        print("  by thermal diffusion or other loss mechanisms.")
        print()

    return df, peak_temp, walberla_peak, previous_peak


def create_temperature_plots(df):
    """Create diagnostic plots for temperature evolution"""

    output_dir = Path("/home/yzk/LBMProject/tests/validation/output_laser_melting_senior")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Temperature vs time
    ax = axes[0, 0]
    ax.plot(df['time_us'], df['max_temp_K'], 'b-', linewidth=2, label='Current simulation')
    ax.axvline(50, color='r', linestyle='--', alpha=0.5, label='Laser shutoff')
    ax.axhline(1923, color='g', linestyle='--', alpha=0.5, label='T_liquidus (Ti6Al4V)')
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Maximum Temperature (K)', fontsize=12)
    ax.set_title('Temperature Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Melt pool depth vs time
    ax = axes[0, 1]
    ax.plot(df['time_us'], df['depth_um'], 'r-', linewidth=2)
    ax.axvline(50, color='r', linestyle='--', alpha=0.5, label='Laser shutoff')
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Melt Pool Depth (μm)', fontsize=12)
    ax.set_title('Melt Pool Depth Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Maximum velocity vs time
    ax = axes[1, 0]
    ax.plot(df['time_us'], df['max_velocity_m_s'], 'g-', linewidth=2)
    ax.axvline(50, color='r', linestyle='--', alpha=0.5, label='Laser shutoff')
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Maximum Velocity (m/s)', fontsize=12)
    ax.set_title('Marangoni Flow Velocity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Heating and cooling rates
    ax = axes[1, 1]
    time = df['time_us'].values
    temp = df['max_temp_K'].values

    # Compute temperature rate (K/μs)
    dt = np.diff(time)
    dT = np.diff(temp)
    rate = dT / dt
    time_mid = (time[:-1] + time[1:]) / 2

    ax.plot(time_mid, rate, 'purple', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(50, color='r', linestyle='--', alpha=0.5, label='Laser shutoff')
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Heating Rate (K/μs)', fontsize=12)
    ax.set_title('Temperature Rate of Change', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save figure
    plot_file = output_dir / "temperature_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    plt.close()


def create_diagnostic_report(df, peak_temp, walberla_peak, previous_peak):
    """Create a detailed diagnostic report"""

    output_dir = Path("/home/yzk/LBMProject/tests/validation/output_laser_melting_senior")
    report_file = output_dir / "diagnostic_report.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LASER MELTING VALIDATION - DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: 2025-12-22\n")
        f.write(f"Configuration: Senior's Fe setup adapted to Ti6Al4V\n")
        f.write("\n")

        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Current peak temperature:    {peak_temp:8.1f} K\n")
        f.write(f"Target (walberla):           {walberla_peak:8.1f} K\n")
        f.write(f"Previous (before changes):   {previous_peak:8.1f} K\n")
        f.write(f"Achievement:                 {peak_temp/walberla_peak*100:6.1f}% of target\n")
        f.write("\n")

        gap_remaining = abs(walberla_peak - peak_temp)
        gap_initial = abs(walberla_peak - previous_peak)
        gap_closed = gap_initial - gap_remaining

        f.write(f"Initial gap:                 {gap_initial:8.1f} K\n")
        f.write(f"Gap closed:                  {gap_closed:8.1f} K ({gap_closed/gap_initial*100:5.1f}%)\n")
        f.write(f"Remaining gap:               {gap_remaining:8.1f} K ({gap_remaining/gap_initial*100:5.1f}%)\n")
        f.write("\n")

        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n")

        if peak_temp < 0.5 * walberla_peak:
            f.write("STATUS: CRITICAL - Temperature far below target\n\n")
            f.write("The simulation is producing temperatures much lower than expected.\n")
            f.write("This indicates a fundamental discrepancy in:\n")
            f.write("  - Thermal parameters (diffusivity, conductivity, specific heat)\n")
            f.write("  - Laser energy deposition\n")
            f.write("  - Numerical scheme stability\n")
        elif peak_temp < 0.8 * walberla_peak:
            f.write("STATUS: MODERATE DISCREPANCY - Temperature moderately below target\n\n")
            f.write("The simulation captures the general physics but with quantitative differences.\n")
            f.write("This suggests:\n")
            f.write("  - Thermal parameters are close but need fine-tuning\n")
            f.write("  - Possible differences in boundary conditions\n")
            f.write("  - Numerical scheme effects\n")
        else:
            f.write("STATUS: GOOD AGREEMENT - Temperature close to target\n\n")
            f.write("The simulation is producing reasonable results.\n")

        f.write("\n")

        f.write("RECOMMENDED NEXT STEPS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Compare thermal parameters with walberla line-by-line:\n")
        f.write("   - Thermal diffusivity (alpha)\n")
        f.write("   - Thermal conductivity (k)\n")
        f.write("   - Specific heat (cp)\n")
        f.write("   - Lattice relaxation time (tau)\n")
        f.write("\n")
        f.write("2. Verify laser energy deposition:\n")
        f.write("   - Power distribution\n")
        f.write("   - Absorption model\n")
        f.write("   - Spatial distribution (Gaussian profile)\n")
        f.write("\n")
        f.write("3. Check numerical parameters:\n")
        f.write("   - Time step (dt)\n")
        f.write("   - Grid spacing (dx)\n")
        f.write("   - CFL condition\n")
        f.write("   - Collision operator\n")
        f.write("\n")
        f.write("4. Examine boundary conditions:\n")
        f.write("   - Radiation boundary\n")
        f.write("   - Substrate cooling\n")
        f.write("   - Side wall conditions\n")
        f.write("\n")

        # Detailed data table
        f.write("\n")
        f.write("DETAILED TIME SERIES DATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Time (μs)':>10} {'Temp (K)':>10} {'Depth (μm)':>12} {'Vel (m/s)':>12}\n")
        f.write("-" * 80 + "\n")

        for _, row in df.iterrows():
            f.write(f"{row['time_us']:10.1f} {row['max_temp_K']:10.1f} {row['depth_um']:12.2f} {row['max_velocity_m_s']:12.4f}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\nDiagnostic report saved to: {report_file}")


def main():
    """Main analysis function"""
    print("\n" + "=" * 80)
    print("LASER MELTING VALIDATION - TEMPERATURE ANALYSIS")
    print("=" * 80 + "\n")

    # Run analysis
    df, peak_temp, walberla_peak, previous_peak = analyze_temperature_evolution()

    # Create plots
    print("\nGenerating diagnostic plots...")
    create_temperature_plots(df)

    # Create detailed report
    print("\nGenerating diagnostic report...")
    create_diagnostic_report(df, peak_temp, walberla_peak, previous_peak)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: /home/yzk/LBMProject/tests/validation/output_laser_melting_senior/")
    print("\nFiles created:")
    print("  - melt_pool_depth.csv           (time series data)")
    print("  - temperature_analysis.png      (diagnostic plots)")
    print("  - diagnostic_report.txt         (detailed report)")
    print("\n")


if __name__ == "__main__":
    main()
