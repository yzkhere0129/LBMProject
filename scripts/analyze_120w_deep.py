#!/usr/bin/env python3
"""
Analyze 120W Deep Penetration LPBF Simulation
Compare with 100W baseline to verify:
- Depth significantly increased
- Width approximately unchanged

Strategy: Smaller spot size (35μm vs 50μm) + higher power (120W vs 100W)
→ Higher energy density → deeper penetration without widening
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# Configuration
VTK_DIR_120W = '/home/yzk/LBMProject/build/lpbf_120w_deep'
VTK_DIR_100W = '/home/yzk/LBMProject/build/lpbf_100w'
OUTPUT_DIR = '/home/yzk/LBMProject/build/validation_120w'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation parameters
DX = 2.0e-6  # m
DT = 1.0e-7  # s
T_melt = 1933  # K (Ti6Al4V solidus)
T_liquid = 1993  # K (Ti6Al4V liquidus)

def read_vtk_field(filename, field_name):
    """Read a scalar field from ASCII VTK file"""
    with open(filename, 'r') as f:
        content = f.read()

    # Get dimensions
    dim_match = re.search(r'DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)', content)
    if dim_match:
        nx, ny, nz = int(dim_match.group(1)), int(dim_match.group(2)), int(dim_match.group(3))
    else:
        return None, None

    # Find the field
    lines = content.split('\n')
    data_start = None

    for i, line in enumerate(lines):
        if f'SCALARS {field_name}' in line:
            data_start = i + 2
            break

    if data_start is None:
        return None, (nx, ny, nz)

    num_points = nx * ny * nz
    data = []

    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('SCALARS') or line.startswith('VECTORS') or line.startswith('LOOKUP'):
            if len(data) >= num_points:
                break
            continue
        try:
            values = [float(x) for x in line.split()]
            data.extend(values)
        except ValueError:
            break
        if len(data) >= num_points:
            break

    data = np.array(data[:num_points]).reshape((nz, ny, nx))
    return data, (nx, ny, nz)


def analyze_melt_pool(vtk_dir, step):
    """Analyze melt pool dimensions at a specific timestep"""
    vtk_file = os.path.join(vtk_dir, f'lpbf_{step:06d}.vtk')

    if not os.path.exists(vtk_file):
        return None

    temp, dims = read_vtk_field(vtk_file, 'Temperature')
    if temp is None:
        return None

    nx, ny, nz = dims
    results = {
        'step': step,
        'time_us': step * DT * 1e6,
    }

    # Find melt pool (T > T_melt)
    melt_mask = temp > T_melt

    if not np.any(melt_mask):
        results['width_um'] = 0
        results['depth_um'] = 0
        results['T_max'] = float(np.max(temp))
        return results

    melt_coords = np.where(melt_mask)
    z_indices, y_indices, x_indices = melt_coords

    # Width (Y direction)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    results['width_um'] = (y_max - y_min + 1) * DX * 1e6

    # Depth (Z direction from surface)
    z_surface = 70  # surface level
    z_min = np.min(z_indices)
    results['depth_um'] = (z_surface - z_min) * DX * 1e6

    results['T_max'] = float(np.max(temp))

    return results


def compare_simulations():
    """Compare 100W and 120W simulations"""

    # Key timesteps to analyze (during steady-state laser scanning)
    timesteps = [3000, 4000, 5000, 6000, 7000]

    results_100w = []
    results_120w = []

    for step in timesteps:
        r100 = analyze_melt_pool(VTK_DIR_100W, step)
        r120 = analyze_melt_pool(VTK_DIR_120W, step)

        if r100:
            results_100w.append(r100)
        if r120:
            results_120w.append(r120)

    return results_100w, results_120w


def create_comparison_plot(results_100w, results_120w):
    """Create comparison plot between 100W and 120W simulations"""

    if not results_100w or not results_120w:
        print("Insufficient data for comparison")
        return

    # Calculate averages
    avg_width_100w = np.mean([r['width_um'] for r in results_100w if r['width_um'] > 0])
    avg_depth_100w = np.mean([r['depth_um'] for r in results_100w if r['depth_um'] > 0])
    avg_width_120w = np.mean([r['width_um'] for r in results_120w if r['width_um'] > 0])
    avg_depth_120w = np.mean([r['depth_um'] for r in results_120w if r['depth_um'] > 0])

    print("\n" + "="*70)
    print("MELT POOL COMPARISON: 100W vs 120W (Deep Penetration)")
    print("="*70)
    print(f"\n100W Baseline (50μm spot):")
    print(f"  Width:  {avg_width_100w:.0f} μm")
    print(f"  Depth:  {avg_depth_100w:.0f} μm")
    print(f"  Aspect ratio (d/w): {avg_depth_100w/avg_width_100w:.2f}")

    print(f"\n120W Deep (35μm spot):")
    print(f"  Width:  {avg_width_120w:.0f} μm")
    print(f"  Depth:  {avg_depth_120w:.0f} μm")
    print(f"  Aspect ratio (d/w): {avg_depth_120w/avg_width_120w:.2f}")

    print(f"\nChange:")
    print(f"  Width change: {(avg_width_120w - avg_width_100w)/avg_width_100w*100:+.1f}%")
    print(f"  Depth change: {(avg_depth_120w - avg_depth_100w)/avg_depth_100w*100:+.1f}%")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Width comparison
    ax1 = axes[0]
    x = [0, 1]
    heights = [avg_width_100w, avg_width_120w]
    colors = ['#2196F3', '#FF5722']
    labels = ['100W\n(50μm spot)', '120W Deep\n(35μm spot)']

    bars1 = ax1.bar(x, heights, color=colors, width=0.6, edgecolor='black', linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('Melt Pool Width (μm)', fontsize=12)
    ax1.set_title('(a) Width Comparison', fontsize=14, fontweight='bold')

    for bar, h in zip(bars1, heights):
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.0f} μm',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    width_change = (avg_width_120w - avg_width_100w) / avg_width_100w * 100
    color = 'green' if abs(width_change) < 15 else 'red'
    ax1.text(0.5, 0.95, f'Change: {width_change:+.1f}%', transform=ax1.transAxes,
            ha='center', fontsize=11, color=color)
    ax1.set_ylim(0, max(heights) * 1.3)

    # Depth comparison
    ax2 = axes[1]
    heights2 = [avg_depth_100w, avg_depth_120w]

    bars2 = ax2.bar(x, heights2, color=colors, width=0.6, edgecolor='black', linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel('Melt Pool Depth (μm)', fontsize=12)
    ax2.set_title('(b) Depth Comparison', fontsize=14, fontweight='bold')

    for bar, h in zip(bars2, heights2):
        ax2.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.0f} μm',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    depth_change = (avg_depth_120w - avg_depth_100w) / avg_depth_100w * 100
    color = 'green' if depth_change > 30 else 'orange'
    ax2.text(0.5, 0.95, f'Change: {depth_change:+.1f}%', transform=ax2.transAxes,
            ha='center', fontsize=11, color=color)
    ax2.set_ylim(0, max(heights2) * 1.3)

    plt.suptitle('120W Deep Penetration vs 100W Baseline\n'
                 'Strategy: Smaller spot (35μm) + Higher power (120W)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_100w_vs_120w_deep.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR}/comparison_100w_vs_120w_deep.png")

    # Validation summary figure
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Create table
    table_data = [
        ['Parameter', '100W Baseline', '120W Deep', 'Change', 'Goal Met?'],
        ['Laser Power', '100 W', '120 W', '+20%', '-'],
        ['Spot Radius', '50 μm', '35 μm', '-30%', '-'],
        ['Energy Density', '1.0x (ref)', '~2.45x', '+145%', '-'],
        ['Melt Pool Width', f'{avg_width_100w:.0f} μm', f'{avg_width_120w:.0f} μm',
         f'{width_change:+.1f}%', 'YES' if abs(width_change) < 15 else 'NO'],
        ['Melt Pool Depth', f'{avg_depth_100w:.0f} μm', f'{avg_depth_120w:.0f} μm',
         f'{depth_change:+.1f}%', 'YES' if depth_change > 30 else 'NO'],
        ['Aspect Ratio (d/w)', f'{avg_depth_100w/avg_width_100w:.2f}',
         f'{avg_depth_120w/avg_width_120w:.2f}',
         f'{((avg_depth_120w/avg_width_120w)/(avg_depth_100w/avg_width_100w)-1)*100:+.0f}%', '-'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor('#1565C0')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color goal column
    for i in [4, 5]:  # Width and Depth rows
        goal_met = table_data[i][4]
        if goal_met == 'YES':
            table[(i, 4)].set_facecolor('#C8E6C9')
        else:
            table[(i, 4)].set_facecolor('#FFCDD2')

    ax.set_title('120W Deep Penetration Configuration Validation\n'
                 'Goal: Increase depth significantly while keeping width unchanged',
                 fontsize=14, fontweight='bold', y=1.02)

    # Success status
    width_ok = abs(width_change) < 15
    depth_ok = depth_change > 30
    if width_ok and depth_ok:
        status = "GOAL ACHIEVED: Deeper penetration with similar width"
        color = 'green'
    elif depth_ok:
        status = "PARTIAL SUCCESS: Depth increased, width also changed"
        color = 'orange'
    else:
        status = "GOAL NOT MET: Need to adjust parameters"
        color = 'red'

    ax.text(0.5, -0.08, status, transform=ax.transAxes, ha='center',
           fontsize=14, fontweight='bold', color=color,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'validation_120w_deep_summary.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/validation_120w_deep_summary.png")


def check_simulation_status():
    """Check if simulations have enough data"""
    vtk_files_120w = glob.glob(os.path.join(VTK_DIR_120W, 'lpbf_*.vtk'))
    vtk_files_100w = glob.glob(os.path.join(VTK_DIR_100W, 'lpbf_*.vtk'))

    print(f"100W simulation: {len(vtk_files_100w)} VTK files")
    print(f"120W simulation: {len(vtk_files_120w)} VTK files")

    return len(vtk_files_120w) >= 50 and len(vtk_files_100w) >= 50


if __name__ == "__main__":
    print("="*70)
    print("120W Deep Penetration LPBF Analysis")
    print("="*70)

    if check_simulation_status():
        results_100w, results_120w = compare_simulations()
        create_comparison_plot(results_100w, results_120w)
    else:
        print("\nWaiting for simulations to complete...")
        print("Run this script again after simulations finish.")
