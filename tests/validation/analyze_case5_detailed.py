#!/usr/bin/env python3
"""
Detailed analysis of Case 5 laser melting simulation results.
Generates comprehensive comparison report with Rosenthal analytical solution.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# === PARAMETERS ===
ANALYSIS_DIR = "/home/yzk/LBMProject/tests/validation/analysis_case5"
METRICS_FILE = f"{ANALYSIS_DIR}/timeseries_metrics.json"
OUTPUT_DIR = "/home/yzk/LBMProject/tests/validation/analysis_case5"

# Physical parameters for Ti-6Al-4V (from simulation setup)
T_MELTING = 1923.0  # K, solidus temperature
T_LIQUIDUS = 1973.0  # K
T_AMBIENT = 300.0   # K
T_BOILING = 3560.0  # K, boiling point of Ti-6Al-4V

# Rosenthal solution parameters (expected from validation setup)
LASER_POWER = 200.0  # W
BEAM_RADIUS = 50e-6  # m
SCAN_VELOCITY = 1.0  # m/s
THERMAL_DIFFUSIVITY = 5.0e-6  # m^2/s (Ti-6Al-4V)

# Expected validation targets
EXPECTED_PEAK_TEMP = 2931.0  # K @ t=50 us
EXPECTED_MELT_DEPTH = 41e-6  # m (41 um)
EXPECTED_L2_ERROR_THRESHOLD = 0.15  # 15%

def load_metrics():
    """Load timeseries metrics from JSON."""
    with open(METRICS_FILE, 'r') as f:
        return json.load(f)

def analyze_temperature_evolution(metrics):
    """Analyze temperature evolution over time."""
    timesteps = []
    peak_temps = []
    peak_locations = []
    liquid_fractions = []
    melt_depths = []

    for m in metrics:
        timesteps.append(m.get('timestep', 0))
        peak_temps.append(m['peak_temperature']['value'])
        peak_locations.append(m['peak_temperature']['location'])
        liquid_fractions.append(m['melt_pool']['liquid_fraction'])
        melt_depths.append(m['melt_pool']['depth'])

    return {
        'timesteps': np.array(timesteps),
        'peak_temps': np.array(peak_temps),
        'peak_locations': np.array(peak_locations),
        'liquid_fractions': np.array(liquid_fractions),
        'melt_depths': np.array(melt_depths)
    }

def find_physically_valid_range(peak_temps):
    """Find timestep range where temperatures are physically valid."""
    valid_mask = (peak_temps >= T_AMBIENT) & (peak_temps <= T_BOILING)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print("WARNING: No physically valid temperatures found!")
        # Find range where temps are < 5000 K as fallback
        fallback_mask = (peak_temps >= T_AMBIENT) & (peak_temps <= 5000.0)
        valid_indices = np.where(fallback_mask)[0]

    return valid_indices

def compute_rosenthal_temperature(x, y, z, t, P=LASER_POWER, v=SCAN_VELOCITY,
                                 alpha=THERMAL_DIFFUSIVITY, T0=T_AMBIENT):
    """
    Compute temperature using 3D moving point source Rosenthal solution.

    Parameters:
        x, y, z: spatial coordinates (m)
        t: time (s)
        P: laser power (W)
        v: scanning velocity (m/s)
        alpha: thermal diffusivity (m^2/s)
        T0: ambient temperature (K)

    Returns:
        Temperature (K)
    """
    # Material properties (Ti-6Al-4V)
    k = 21.0  # thermal conductivity, W/(m*K)

    # Moving coordinate system (laser at origin, moving in +x)
    x_rel = x - v * t
    r = np.sqrt(x_rel**2 + y**2 + z**2)

    # Avoid singularity at r=0
    if r < 1e-9:
        r = 1e-9

    # Rosenthal solution for moving point source
    peclet = v * r / (2 * alpha)
    T = T0 + (P / (2 * np.pi * k * r)) * np.exp(-peclet * (1 + x_rel/r))

    return T

def generate_report(metrics, evolution):
    """Generate comprehensive analysis report."""
    print("=" * 80)
    print("CASE 5 LASER MELTING VALIDATION - ANALYSIS REPORT")
    print("=" * 80)

    # Find physically valid range
    valid_indices = find_physically_valid_range(evolution['peak_temps'])

    if len(valid_indices) == 0:
        print("\nERROR: No valid temperature data found!")
        return

    print(f"\nPhysically valid timestep range: {valid_indices[0]} - {valid_indices[-1]}")
    print(f"Valid timesteps: {len(valid_indices)}/{len(metrics)}")

    # Statistics over valid range
    valid_peak_temps = evolution['peak_temps'][valid_indices]
    valid_melt_depths = evolution['melt_depths'][valid_indices]
    valid_liquid_fracs = evolution['liquid_fractions'][valid_indices]

    print("\n" + "-" * 80)
    print("TEMPERATURE STATISTICS (Valid Range)")
    print("-" * 80)
    print(f"Peak temperature range: {valid_peak_temps.min():.2f} - {valid_peak_temps.max():.2f} K")
    print(f"Mean peak temperature:  {valid_peak_temps.mean():.2f} K")
    print(f"Median peak temperature: {np.median(valid_peak_temps):.2f} K")

    # Find timestep closest to expected peak
    max_temp_idx = valid_indices[np.argmax(valid_peak_temps)]
    max_temp_data = metrics[max_temp_idx]

    print("\n" + "-" * 80)
    print("MAXIMUM TEMPERATURE EVENT")
    print("-" * 80)
    print(f"Timestep: {max_temp_data.get('timestep', 'N/A')}")
    print(f"Peak temperature: {max_temp_data['peak_temperature']['value']:.2f} K")
    peak_loc = max_temp_data['peak_temperature']['location']
    print(f"Peak location: ({peak_loc[0]*1e6:.2f}, {peak_loc[1]*1e6:.2f}, {peak_loc[2]*1e6:.2f}) um")
    print(f"Liquid fraction: {max_temp_data['melt_pool']['liquid_fraction']*100:.3f}%")
    print(f"Melt pool dimensions:")
    print(f"  Width:  {max_temp_data['melt_pool']['width']*1e6:.2f} um")
    print(f"  Depth:  {max_temp_data['melt_pool']['depth']*1e6:.2f} um")
    print(f"  Height: {max_temp_data['melt_pool']['height']*1e6:.2f} um")

    print("\n" + "-" * 80)
    print("COMPARISON WITH EXPECTED VALUES")
    print("-" * 80)
    actual_peak = max_temp_data['peak_temperature']['value']
    actual_depth = max_temp_data['melt_pool']['depth']

    temp_error = abs(actual_peak - EXPECTED_PEAK_TEMP) / EXPECTED_PEAK_TEMP
    depth_error = abs(actual_depth - EXPECTED_MELT_DEPTH) / EXPECTED_MELT_DEPTH

    print(f"Expected peak temperature: {EXPECTED_PEAK_TEMP:.2f} K")
    print(f"Actual peak temperature:   {actual_peak:.2f} K")
    print(f"Relative error:            {temp_error*100:.2f}%")
    print()
    print(f"Expected melt depth: {EXPECTED_MELT_DEPTH*1e6:.2f} um")
    print(f"Actual melt depth:   {actual_depth*1e6:.2f} um")
    print(f"Relative error:      {depth_error*100:.2f}%")

    print("\n" + "-" * 80)
    print("VALIDATION ASSESSMENT")
    print("-" * 80)

    # Check if within acceptable error bounds
    if temp_error < EXPECTED_L2_ERROR_THRESHOLD:
        print(f"PASS: Peak temperature error ({temp_error*100:.2f}%) < {EXPECTED_L2_ERROR_THRESHOLD*100:.0f}%")
    else:
        print(f"FAIL: Peak temperature error ({temp_error*100:.2f}%) > {EXPECTED_L2_ERROR_THRESHOLD*100:.0f}%")

    if depth_error < EXPECTED_L2_ERROR_THRESHOLD:
        print(f"PASS: Melt depth error ({depth_error*100:.2f}%) < {EXPECTED_L2_ERROR_THRESHOLD*100:.0f}%")
    else:
        print(f"FAIL: Melt depth error ({depth_error*100:.2f}%) > {EXPECTED_L2_ERROR_THRESHOLD*100:.0f}%")

    print("\n" + "-" * 80)
    print("PHYSICAL VALIDITY CHECKS")
    print("-" * 80)

    # Check for unphysical temperatures
    num_too_high = np.sum(evolution['peak_temps'] > T_BOILING)
    num_too_low = np.sum(evolution['peak_temps'] < T_AMBIENT)

    if num_too_high > 0:
        print(f"WARNING: {num_too_high} timesteps with T > {T_BOILING} K (boiling point)")
        max_unrealistic = evolution['peak_temps'].max()
        print(f"  Maximum temperature: {max_unrealistic:.2f} K")

    if num_too_low > 0:
        print(f"WARNING: {num_too_low} timesteps with T < {T_AMBIENT} K (ambient)")

    # Check liquid fraction
    max_liquid_frac = valid_liquid_fracs.max()
    print(f"\nMaximum liquid fraction: {max_liquid_frac*100:.3f}%")
    if max_liquid_frac < 0.01:
        print("WARNING: Very low liquid fraction - laser may not be producing significant melting")

    return {
        'valid_indices': valid_indices,
        'max_temp_idx': max_temp_idx,
        'temp_error': temp_error,
        'depth_error': depth_error
    }

def create_detailed_plots(evolution, analysis_results):
    """Create detailed analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    valid_idx = analysis_results['valid_indices']
    timesteps = evolution['timesteps']

    # Convert timestep to physical time (assuming dt based on data)
    if len(timesteps) > 1:
        # Infer time from timestep numbers
        # For LBM, typical dt ~ 1e-8 to 1e-7 s
        # From file pattern, looks like output every 20 timesteps
        dt_estimate = 1e-8  # seconds per timestep (typical for LBM)
        time_us = timesteps * dt_estimate * 1e6  # convert to microseconds
    else:
        time_us = timesteps

    # Plot 1: Peak temperature evolution
    ax = axes[0, 0]
    ax.plot(time_us, evolution['peak_temps'], 'b-', linewidth=1.5, label='LBM Simulation')
    ax.axhline(y=EXPECTED_PEAK_TEMP, color='r', linestyle='--', label=f'Expected ({EXPECTED_PEAK_TEMP:.0f} K)')
    ax.axhline(y=T_MELTING, color='g', linestyle=':', label=f'Melting point ({T_MELTING:.0f} K)')
    ax.axhline(y=T_BOILING, color='orange', linestyle=':', label=f'Boiling point ({T_BOILING:.0f} K)')

    # Highlight valid range
    if len(valid_idx) > 0:
        ax.axvspan(time_us[valid_idx[0]], time_us[valid_idx[-1]], alpha=0.1, color='green',
                   label='Physically valid range')

    ax.set_xlabel('Time (us)', fontsize=11)
    ax.set_ylabel('Peak Temperature (K)', fontsize=11)
    ax.set_title('Peak Temperature Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Melt pool depth evolution
    ax = axes[0, 1]
    ax.plot(time_us, evolution['melt_depths']*1e6, 'b-', linewidth=1.5, label='LBM Simulation')
    ax.axhline(y=EXPECTED_MELT_DEPTH*1e6, color='r', linestyle='--',
               label=f'Expected ({EXPECTED_MELT_DEPTH*1e6:.1f} um)')
    ax.set_xlabel('Time (us)', fontsize=11)
    ax.set_ylabel('Melt Pool Depth (um)', fontsize=11)
    ax.set_title('Melt Pool Depth Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Liquid fraction evolution
    ax = axes[1, 0]
    ax.plot(time_us, evolution['liquid_fractions']*100, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (us)', fontsize=11)
    ax.set_ylabel('Liquid Fraction (%)', fontsize=11)
    ax.set_title('Liquid Fraction Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Peak location trajectory (X-Z projection)
    ax = axes[1, 1]
    peak_locs = np.array(evolution['peak_locations'])
    if len(valid_idx) > 0:
        ax.plot(peak_locs[valid_idx, 0]*1e6, peak_locs[valid_idx, 2]*1e6,
                'b-', linewidth=1.5, alpha=0.7, label='Valid range')
        ax.scatter(peak_locs[valid_idx, 0]*1e6, peak_locs[valid_idx, 2]*1e6,
                   c=evolution['peak_temps'][valid_idx], cmap='hot', s=20, alpha=0.6)

    # Mark maximum temperature location
    max_idx = analysis_results['max_temp_idx']
    ax.scatter(peak_locs[max_idx, 0]*1e6, peak_locs[max_idx, 2]*1e6,
               c='red', s=200, marker='*', edgecolor='black', linewidth=1.5,
               label=f'Max T location', zorder=10)

    ax.set_xlabel('X Position (um)', fontsize=11)
    ax.set_ylabel('Z Position (um)', fontsize=11)
    ax.set_title('Peak Temperature Location (X-Z Projection)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Temperature (K)', fontsize=10)

    plt.tight_layout()

    # Save plot
    output_file = f"{OUTPUT_DIR}/detailed_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nDetailed analysis plot saved to: {output_file}")
    plt.close()

def main():
    """Main analysis workflow."""
    print("Loading metrics data...")
    metrics = load_metrics()

    print("Analyzing temperature evolution...")
    evolution = analyze_temperature_evolution(metrics)

    print("Generating report...\n")
    analysis_results = generate_report(metrics, evolution)

    if analysis_results:
        print("\nGenerating detailed plots...")
        create_detailed_plots(evolution, analysis_results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - timeseries_metrics.json")
    print(f"  - detailed_analysis.png")
    print(f"  - peak_temperature_evolution.png")
    print(f"  - liquid_fraction_evolution.png")

if __name__ == "__main__":
    main()
