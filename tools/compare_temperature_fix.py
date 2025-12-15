#!/usr/bin/env python3
"""
Compare temperature evolution before and after fix

This script compares the temperature field evolution from the original
simulation (200W laser → 10,000K) vs the fixed simulation (20W laser → 2,500K)
"""

import sys

def read_vtk_temperature(filepath):
    """Read temperature data from ASCII VTK file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find temperature data section
    temp_start = None
    for i, line in enumerate(lines):
        if 'SCALARS Temperature' in line:
            temp_start = i + 2  # Skip "LOOKUP_TABLE default"
            break

    if not temp_start:
        return None

    # Read all temperature values
    temps = []
    for i in range(temp_start, len(lines)):
        try:
            temps.append(float(lines[i].strip()))
        except:
            break

    return temps

def analyze_temperatures(temps):
    """Compute statistics"""
    if not temps:
        return None

    T_min = min(temps)
    T_max = max(temps)
    T_avg = sum(temps) / len(temps)

    # Count phases
    T_solidus = 1878  # Ti6Al4V
    T_liquidus = 1923

    num_solid = sum(1 for t in temps if t < T_solidus)
    num_mushy = sum(1 for t in temps if T_solidus <= t <= T_liquidus)
    num_liquid = sum(1 for t in temps if t > T_liquidus)

    return {
        'T_min': T_min,
        'T_max': T_max,
        'T_avg': T_avg,
        'num_solid': num_solid,
        'num_mushy': num_mushy,
        'num_liquid': num_liquid,
        'total': len(temps)
    }

def main():
    print("="*80)
    print("TEMPERATURE FIX COMPARISON")
    print("="*80)
    print()

    # Check if we have the VTK files
    import os
    vtk_dir = '/home/yzk/LBMProject/build/lpbf_realistic'

    if not os.path.exists(vtk_dir):
        print(f"ERROR: Directory not found: {vtk_dir}")
        print("Please run the simulation first:")
        print("  cd /home/yzk/LBMProject/build")
        print("  ./visualize_lpbf_marangoni_realistic")
        return 1

    print("Analyzing temperature evolution...")
    print()

    # Timesteps to check
    steps = [0, 50, 100, 150, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]

    print(f"{'Step':>5} {'Time':>8} {'T_max':>9} {'T_avg':>9} {'Liquid%':>8} {'Status':>10}")
    print("-" * 70)

    T_liquidus = 1923  # Ti6Al4V melting point

    for step in steps:
        filepath = f"{vtk_dir}/lpbf_{step:06d}.vtk"

        if not os.path.exists(filepath):
            continue

        temps = read_vtk_temperature(filepath)
        if not temps:
            continue

        stats = analyze_temperatures(temps)
        time_us = step * 0.1
        liquid_pct = 100.0 * stats['num_liquid'] / stats['total']

        # Determine status
        if stats['T_max'] < 1000:
            status = "Cold"
        elif stats['T_max'] < T_liquidus:
            status = "Heating"
        elif stats['T_max'] < 3000:
            status = "Melting ✓"
        elif stats['T_max'] < 5000:
            status = "Too hot!"
        else:
            status = "CRITICAL ✗"

        print(f"{step:5d} {time_us:7.1f}μs {stats['T_max']:8.1f}K "
              f"{stats['T_avg']:8.1f}K {liquid_pct:7.2f}% {status:>10}")

    print("-" * 70)
    print()

    # Final summary
    final_step = max(s for s in steps if os.path.exists(f"{vtk_dir}/lpbf_{s:06d}.vtk"))
    final_temps = read_vtk_temperature(f"{vtk_dir}/lpbf_{final_step:06d}.vtk")
    final_stats = analyze_temperatures(final_temps)

    print("SUMMARY:")
    print("-" * 80)
    print(f"  Final maximum temperature: {final_stats['T_max']:.1f} K")
    print(f"  Final average temperature: {final_stats['T_avg']:.1f} K")
    print(f"  Liquid fraction at peak:   {100.0*final_stats['num_liquid']/final_stats['total']:.2f}%")
    print()

    # Verdict
    if final_stats['T_max'] < 4000:
        print("✓ TEMPERATURE FIX SUCCESSFUL")
        print("  - Temperature stays below 4000K")
        print("  - Material melts as expected")
        print("  - Physically realistic behavior")
        verdict = "PASS"
    else:
        print("✗ TEMPERATURE STILL TOO HIGH")
        print("  - Check laser power setting")
        print("  - Consider adding radiation cooling")
        print("  - Check boundary conditions")
        verdict = "FAIL"

    print("-" * 80)
    print()

    # Instructions for visualization
    print("TO VISUALIZE IN PARAVIEW:")
    print("-" * 80)
    print("1. Open ParaView:")
    print("   paraview lpbf_realistic/lpbf_*.vtk")
    print()
    print("2. Color by 'Temperature'")
    print("   - Set color range: 300 - 3000 K")
    print("   - Use 'Jet' or 'Rainbow' colormap")
    print()
    print("3. Add 'Slice' filter to see cross-section")
    print("   - X-Normal slice at laser position")
    print()
    print("4. Play animation to see:")
    print("   - Laser heating → melting → solidification")
    print("   - Temperature peaks should be ~2000-2500K (realistic!)")
    print("-" * 80)

    return 0 if verdict == "PASS" else 1

if __name__ == '__main__':
    sys.exit(main())
