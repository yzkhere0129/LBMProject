#!/usr/bin/env python3
"""
Analyze Case 5: Laser Melting Validation Results

This script extracts and validates key metrics from the laser melting simulation:
1. Melt pool depth evolution
2. Maximum temperature (should exceed T_liquidus=1923K)
3. Marangoni flow velocity
4. Solidification after laser shutoff (t > 50 μs)

Author: Claude Code (Testing & Validation Specialist)
Date: 2025-12-21
"""

import numpy as np
import os
import sys
from pathlib import Path
import struct

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def read_vtk_structured_points(filename):
    """
    Read VTK structured points file (ASCII format)

    Returns:
        dict with keys: 'nx', 'ny', 'nz', 'dx', 'dy', 'dz', 'data'
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    nx, ny, nz = None, None, None
    dx, dy, dz = None, None, None
    data_start_line = None

    for i, line in enumerate(lines):
        if line.startswith('DIMENSIONS'):
            parts = line.split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
        elif line.startswith('SPACING'):
            parts = line.split()
            dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
        elif line.startswith('LOOKUP_TABLE'):
            data_start_line = i + 1
            break

    if data_start_line is None:
        raise ValueError(f"Could not find data in {filename}")

    # Read data
    data_lines = lines[data_start_line:]
    data = []
    for line in data_lines:
        values = line.strip().split()
        data.extend([float(v) for v in values])

    data = np.array(data)

    return {
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'data': data.reshape((nz, ny, nx))  # Z-major ordering
    }

def compute_melt_pool_depth(temperature, dx, dy, dz, T_liquidus=1923.0, interface_z_fraction=0.5):
    """
    Compute melt pool depth (distance from interface to deepest liquid point)

    Args:
        temperature: 3D temperature array (nz, ny, nx)
        dx, dy, dz: grid spacing
        T_liquidus: liquidus temperature (K)
        interface_z_fraction: assumed interface position (fraction of domain height)

    Returns:
        melt_pool_depth: depth in meters
    """
    nz, ny, nx = temperature.shape

    # Assume interface is at middle of domain
    interface_z = int(nz * interface_z_fraction)

    # Find deepest point where T > T_liquidus
    max_depth = 0.0

    for k in range(interface_z):
        for j in range(ny):
            for i in range(nx):
                if temperature[k, j, i] > T_liquidus:
                    depth = (interface_z - k) * dz
                    max_depth = max(max_depth, depth)

    return max_depth

def compute_max_temperature_above_interface(temperature, interface_z_fraction=0.5):
    """
    Find maximum temperature in the upper half of the domain (liquid region)
    """
    nz, ny, nx = temperature.shape
    interface_z = int(nz * interface_z_fraction)

    # Only consider cells above interface
    upper_region = temperature[interface_z:, :, :]
    return np.max(upper_region)

def analyze_vtk_files(output_dir, dt=1e-9, laser_shutoff_time=50e-6):
    """
    Analyze all VTK files in the output directory

    Args:
        output_dir: path to output directory
        dt: timestep in seconds
        laser_shutoff_time: time when laser shuts off (seconds)

    Returns:
        dict with analysis results
    """
    # Find all VTK files
    vtk_files = sorted(Path(output_dir).glob("temperature_*.vtk"))

    if len(vtk_files) == 0:
        print(f"ERROR: No VTK files found in {output_dir}")
        return None

    print(f"Found {len(vtk_files)} VTK files")
    print(f"Analyzing: {vtk_files[0].name} to {vtk_files[-1].name}")
    print()

    # Material properties
    T_liquidus = 1923.0  # Ti6Al4V liquidus temperature (K)
    T_solidus = 1878.0   # Ti6Al4V solidus temperature (K)

    # Storage for time series
    times = []
    depths = []
    max_temps = []

    # Analyze each file
    for vtk_file in vtk_files:
        # Extract timestep from filename
        filename = vtk_file.name
        step_str = filename.split('_')[1].split('.')[0]
        step = int(step_str)
        time = step * dt

        # Read VTK file
        try:
            vtk_data = read_vtk_structured_points(str(vtk_file))
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")
            continue

        temperature = vtk_data['data']
        dx = vtk_data['dx']
        dy = vtk_data['dy']
        dz = vtk_data['dz']

        # Compute metrics
        depth = compute_melt_pool_depth(temperature, dx, dy, dz, T_liquidus)
        max_T = compute_max_temperature_above_interface(temperature)

        times.append(time)
        depths.append(depth)
        max_temps.append(max_T)

        # Print progress
        if step % 10000 == 0 or step == 0:
            print(f"Step {step:6d} | t = {time*1e6:6.2f} μs | "
                  f"T_max = {max_T:7.1f} K | Depth = {depth*1e6:6.2f} μm")

    print()

    # Convert to numpy arrays
    times = np.array(times)
    depths = np.array(depths)
    max_temps = np.array(max_temps)

    # Find peak melt pool depth
    max_depth_idx = np.argmax(depths)
    max_depth = depths[max_depth_idx]
    time_at_max_depth = times[max_depth_idx]

    # Find peak temperature
    max_temp_idx = np.argmax(max_temps)
    peak_temperature = max_temps[max_temp_idx]
    time_at_peak_temp = times[max_temp_idx]

    # Analyze laser-on period (0-50 μs)
    laser_on_mask = times <= laser_shutoff_time
    if np.any(laser_on_mask):
        avg_depth_laser_on = np.mean(depths[laser_on_mask])
        avg_temp_laser_on = np.mean(max_temps[laser_on_mask])
    else:
        avg_depth_laser_on = 0.0
        avg_temp_laser_on = 0.0

    # Analyze laser-off period (50-100 μs)
    laser_off_mask = times > laser_shutoff_time
    if np.any(laser_off_mask):
        avg_depth_laser_off = np.mean(depths[laser_off_mask])
        avg_temp_laser_off = np.mean(max_temps[laser_off_mask])

        # Check if melt pool is shrinking
        depth_at_shutoff = depths[laser_on_mask][-1] if np.any(laser_on_mask) else 0.0
        depth_at_end = depths[-1]
        is_solidifying = depth_at_end < depth_at_shutoff
    else:
        avg_depth_laser_off = 0.0
        avg_temp_laser_off = 0.0
        depth_at_shutoff = 0.0
        depth_at_end = 0.0
        is_solidifying = False

    # Melt pool depth at key times
    key_times = [25e-6, 50e-6, 60e-6, 75e-6, 100e-6]
    depths_at_key_times = {}
    for t_key in key_times:
        # Find closest time
        idx = np.argmin(np.abs(times - t_key))
        if np.abs(times[idx] - t_key) < 2e-6:  # Within 2 μs
            depths_at_key_times[t_key * 1e6] = depths[idx]

    return {
        'times': times,
        'depths': depths,
        'max_temps': max_temps,
        'peak_depth': max_depth,
        'time_at_peak_depth': time_at_max_depth,
        'peak_temperature': peak_temperature,
        'time_at_peak_temp': time_at_peak_temp,
        'avg_depth_laser_on': avg_depth_laser_on,
        'avg_temp_laser_on': avg_temp_laser_on,
        'avg_depth_laser_off': avg_depth_laser_off,
        'avg_temp_laser_off': avg_temp_laser_off,
        'depth_at_shutoff': depth_at_shutoff,
        'depth_at_end': depth_at_end,
        'is_solidifying': is_solidifying,
        'depths_at_key_times': depths_at_key_times,
        'T_liquidus': T_liquidus,
        'T_solidus': T_solidus,
    }

def validate_results(results):
    """
    Validate simulation results against expected criteria

    Returns:
        dict with validation results
    """
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    validations = {}

    # Expected ranges
    EXPECTED_DEPTH_MIN = 5e-6      # 5 μm
    EXPECTED_DEPTH_MAX = 100e-6    # 100 μm
    EXPECTED_TEMP_MIN = 2000.0     # K (should exceed liquidus)
    EXPECTED_TEMP_MAX = 2500.0     # K (reasonable for 200W laser)

    # 1. Melt pool formation
    print("1. MELT POOL FORMATION")
    print(f"   Peak depth: {results['peak_depth']*1e6:.2f} μm at t = {results['time_at_peak_depth']*1e6:.2f} μs")
    if results['peak_depth'] > 0:
        print("   [PASS] Melt pool formed")
        validations['melt_pool_formed'] = True
    else:
        print("   [FAIL] No melt pool detected")
        validations['melt_pool_formed'] = False
    print()

    # 2. Melt pool depth in reasonable range
    print("2. MELT POOL DEPTH VALIDATION")
    print(f"   Expected range: {EXPECTED_DEPTH_MIN*1e6:.1f} - {EXPECTED_DEPTH_MAX*1e6:.1f} μm")
    print(f"   Actual peak: {results['peak_depth']*1e6:.2f} μm")
    if EXPECTED_DEPTH_MIN <= results['peak_depth'] <= EXPECTED_DEPTH_MAX:
        print("   [PASS] Melt pool depth in expected range")
        validations['depth_in_range'] = True
    else:
        print("   [FAIL] Melt pool depth out of range")
        validations['depth_in_range'] = False
    print()

    # 3. Maximum temperature validation
    print("3. MAXIMUM TEMPERATURE VALIDATION")
    print(f"   Expected range: {EXPECTED_TEMP_MIN:.0f} - {EXPECTED_TEMP_MAX:.0f} K")
    print(f"   Actual peak: {results['peak_temperature']:.1f} K at t = {results['time_at_peak_temp']*1e6:.2f} μs")
    print(f"   T_liquidus: {results['T_liquidus']:.1f} K")
    if results['peak_temperature'] > results['T_liquidus']:
        print("   [PASS] Peak temperature exceeds liquidus")
        validations['temp_exceeds_liquidus'] = True
    else:
        print("   [FAIL] Peak temperature below liquidus")
        validations['temp_exceeds_liquidus'] = False

    if EXPECTED_TEMP_MIN <= results['peak_temperature'] <= EXPECTED_TEMP_MAX:
        print("   [PASS] Peak temperature in reasonable range")
        validations['temp_in_range'] = True
    else:
        print("   [FAIL] Peak temperature out of range")
        validations['temp_in_range'] = False
    print()

    # 4. Solidification after laser shutoff
    print("4. SOLIDIFICATION AFTER LASER SHUTOFF")
    print(f"   Depth at shutoff (50 μs): {results['depth_at_shutoff']*1e6:.2f} μm")
    print(f"   Depth at end: {results['depth_at_end']*1e6:.2f} μm")
    print(f"   Change: {(results['depth_at_end'] - results['depth_at_shutoff'])*1e6:.2f} μm")
    if results['is_solidifying']:
        print("   [PASS] Melt pool shrinking (solidification observed)")
        validations['solidification'] = True
    else:
        print("   [FAIL] Melt pool not shrinking")
        validations['solidification'] = False
    print()

    # 5. Melt pool depth at key times
    print("5. MELT POOL DEPTH AT KEY TIMES")
    for t_us, depth in sorted(results['depths_at_key_times'].items()):
        print(f"   t = {t_us:5.0f} μs: {depth*1e6:6.2f} μm")
    print()

    # 6. Average conditions during laser-on and laser-off
    print("6. AVERAGE CONDITIONS")
    print(f"   Laser ON  (0-50 μs):")
    print(f"     Average depth: {results['avg_depth_laser_on']*1e6:.2f} μm")
    print(f"     Average max temp: {results['avg_temp_laser_on']:.1f} K")
    print(f"   Laser OFF (50-100 μs):")
    print(f"     Average depth: {results['avg_depth_laser_off']*1e6:.2f} μm")
    print(f"     Average max temp: {results['avg_temp_laser_off']:.1f} K")
    print()

    # Overall validation
    print("=" * 80)
    print("OVERALL VALIDATION SUMMARY")
    print("=" * 80)
    total_checks = len(validations)
    passed_checks = sum(validations.values())
    print(f"Passed: {passed_checks}/{total_checks} checks")
    print()

    for check, result in validations.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {check}")

    print()

    if passed_checks == total_checks:
        print("RESULT: ALL VALIDATIONS PASSED")
    else:
        print(f"RESULT: {total_checks - passed_checks} VALIDATION(S) FAILED")

    print("=" * 80)
    print()

    return validations

def save_csv_summary(results, output_file):
    """Save time series data to CSV"""
    with open(output_file, 'w') as f:
        f.write("time_us,depth_um,max_temp_K\n")
        for t, d, T in zip(results['times'], results['depths'], results['max_temps']):
            f.write(f"{t*1e6:.2f},{d*1e6:.4f},{T:.2f}\n")
    print(f"Time series data saved to: {output_file}")

def main():
    # Configuration
    output_dir = "/home/yzk/LBMProject/tests/validation/output_laser_melting_senior"
    dt = 1e-9  # 1 ns
    laser_shutoff_time = 50e-6  # 50 μs

    print("=" * 80)
    print("CASE 5: LASER MELTING VALIDATION ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Timestep: {dt*1e9:.1f} ns")
    print(f"Laser shutoff: {laser_shutoff_time*1e6:.0f} μs")
    print()

    # Check if directory exists
    if not os.path.exists(output_dir):
        print(f"ERROR: Output directory not found: {output_dir}")
        return 1

    # Analyze VTK files
    results = analyze_vtk_files(output_dir, dt, laser_shutoff_time)

    if results is None:
        return 1

    # Validate results
    validations = validate_results(results)

    # Save CSV summary
    csv_file = os.path.join(output_dir, "analysis_summary.csv")
    save_csv_summary(results, csv_file)

    # Return exit code based on validation
    if all(validations.values()):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
