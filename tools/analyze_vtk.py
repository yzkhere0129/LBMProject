#!/usr/bin/env python3
"""
VTK Data Analysis Tool for LPBF Simulation

This script extracts and analyzes key physical quantities from VTK output files
for comparison with literature data and verification of energy conservation.

Usage:
    python analyze_vtk.py <vtk_file> [--plot] [--output report.txt]

Author: Claude Code
Date: 2025-11-19
"""

import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Physical constants
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)

# Material properties (Ti6Al4V)
MATERIAL = {
    'name': 'Ti6Al4V',
    'T_solidus': 1878,      # K (solidus temperature)
    'T_liquidus': 1923,     # K (liquidus temperature)
    'T_vaporization': 3560, # K (boiling point)
    'rho_liquid': 4110,     # kg/m³ (liquid density)
    'rho_solid': 4430,      # kg/m³ (solid density)
    'cp_liquid': 831,       # J/(kg·K) (liquid specific heat)
    'cp_solid': 546,        # J/(kg·K) (solid specific heat)
    'L_fusion': 286e3,      # J/kg (latent heat of fusion)
    'L_vaporization': 9.83e6, # J/kg (latent heat of vaporization)
}

def read_vtk_structured_points(filename):
    """
    Read VTK STRUCTURED_POINTS file and extract all fields.

    Args:
        filename: Path to VTK file

    Returns:
        dict with keys:
            - 'dimensions': (nx, ny, nz)
            - 'origin': (x0, y0, z0)
            - 'spacing': (dx, dy, dz)
            - 'velocity': (nx, ny, nz, 3) array
            - 'temperature': (nx, ny, nz) array (if present)
            - 'fill_level': (nx, ny, nz) array (if present)
            - 'liquid_fraction': (nx, ny, nz) array (if present)
    """
    print(f"Reading VTK file: {filename}")

    data = {}
    current_field = None
    field_data = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('DIMENSIONS'):
            parts = line.split()
            data['dimensions'] = (int(parts[1]), int(parts[2]), int(parts[3]))
            print(f"  Dimensions: {data['dimensions']}")

        elif line.startswith('ORIGIN'):
            parts = line.split()
            data['origin'] = (float(parts[1]), float(parts[2]), float(parts[3]))

        elif line.startswith('SPACING'):
            parts = line.split()
            data['spacing'] = (float(parts[1]), float(parts[2]), float(parts[3]))
            print(f"  Spacing: {data['spacing']} m")

        elif line.startswith('POINT_DATA'):
            parts = line.split()
            num_points = int(parts[1])
            print(f"  Total points: {num_points}")

        elif line.startswith('VECTORS'):
            parts = line.split()
            field_name = parts[1]
            current_field = field_name
            field_data = []
            print(f"  Reading VECTORS: {field_name}")

        elif line.startswith('SCALARS'):
            parts = line.split()
            field_name = parts[1]
            current_field = field_name
            field_data = []
            print(f"  Reading SCALARS: {field_name}")
            i += 1  # Skip LOOKUP_TABLE line

        elif current_field is not None:
            # Parse numerical data
            parts = line.split()
            if parts:
                try:
                    values = [float(x) for x in parts]
                    field_data.extend(values)
                except ValueError:
                    pass

        i += 1

    # Reshape data into 3D arrays
    nx, ny, nz = data['dimensions']
    num_cells = nx * ny * nz

    if 'velocity' in locals() or len(field_data) == num_cells * 3:
        # Velocity field (3 components)
        vel = np.array(field_data).reshape((nz, ny, nx, 3))
        data['velocity'] = np.transpose(vel, (2, 1, 0, 3))  # (nx, ny, nz, 3)

    print(f"Successfully loaded VTK file")
    return data


def read_vtk_with_vtk_library(filename):
    """
    Read VTK file using vtk library (more robust).
    Falls back to manual parsing if vtk not available.
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        print(f"Reading VTK file with vtk library: {filename}")

        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(filename)
        reader.Update()

        vtk_data = reader.GetOutput()

        # Get dimensions
        dims = vtk_data.GetDimensions()
        nx, ny, nz = dims[0], dims[1], dims[2]

        # Get spacing
        spacing = vtk_data.GetSpacing()
        dx, dy, dz = spacing[0], spacing[1], spacing[2]

        # Get origin
        origin = vtk_data.GetOrigin()

        data = {
            'dimensions': (nx, ny, nz),
            'spacing': (dx, dy, dz),
            'origin': origin,
        }

        print(f"  Dimensions: {nx} x {ny} x {nz}")
        print(f"  Spacing: {dx*1e6:.2f} x {dy*1e6:.2f} x {dz*1e6:.2f} μm")

        # Extract velocity
        point_data = vtk_data.GetPointData()
        if point_data.GetArray('Velocity'):
            vel_vtk = point_data.GetArray('Velocity')
            vel = vtk_to_numpy(vel_vtk).reshape((nz, ny, nx, 3))
            data['velocity'] = np.transpose(vel, (2, 1, 0, 3))  # (nx, ny, nz, 3)
            print(f"  Loaded: Velocity")

        # Extract temperature
        if point_data.GetArray('Temperature'):
            temp_vtk = point_data.GetArray('Temperature')
            temp = vtk_to_numpy(temp_vtk).reshape((nz, ny, nx))
            data['temperature'] = np.transpose(temp, (2, 1, 0))  # (nx, ny, nz)
            print(f"  Loaded: Temperature")

        # Extract fill level
        if point_data.GetArray('FillLevel'):
            fill_vtk = point_data.GetArray('FillLevel')
            fill = vtk_to_numpy(fill_vtk).reshape((nz, ny, nx))
            data['fill_level'] = np.transpose(fill, (2, 1, 0))  # (nx, ny, nz)
            print(f"  Loaded: FillLevel")

        # Extract liquid fraction
        if point_data.GetArray('LiquidFraction'):
            lf_vtk = point_data.GetArray('LiquidFraction')
            lf = vtk_to_numpy(lf_vtk).reshape((nz, ny, nx))
            data['liquid_fraction'] = np.transpose(lf, (2, 1, 0))  # (nx, ny, nz)
            print(f"  Loaded: LiquidFraction")

        return data

    except ImportError:
        print("WARNING: vtk library not available, falling back to manual parsing")
        return read_vtk_structured_points(filename)


def analyze_melt_pool(data, T_liquidus=None):
    """
    Analyze melt pool geometry from temperature field.

    Args:
        data: VTK data dict
        T_liquidus: Liquidus temperature [K] (default: material property)

    Returns:
        dict with melt pool properties:
            - length, width, depth [μm]
            - volume [μm³]
            - max_temperature [K]
            - surface_temperature [K]
    """
    if 'temperature' not in data:
        print("ERROR: Temperature field not found in VTK file")
        return None

    if T_liquidus is None:
        T_liquidus = MATERIAL['T_liquidus']

    temp = data['temperature']
    dx, dy, dz = data['spacing']
    nx, ny, nz = data['dimensions']

    # Find melt pool (T > T_liquidus)
    melt_mask = temp > T_liquidus

    if not np.any(melt_mask):
        print("WARNING: No melt pool found (no cells above T_liquidus)")
        return None

    # Find bounding box of melt pool
    melt_indices = np.argwhere(melt_mask)

    x_min, y_min, z_min = melt_indices.min(axis=0)
    x_max, y_max, z_max = melt_indices.max(axis=0)

    # Compute dimensions
    length = (x_max - x_min + 1) * dx * 1e6  # μm
    width = (y_max - y_min + 1) * dy * 1e6   # μm
    depth = (z_max - z_min + 1) * dz * 1e6   # μm

    # Compute volume
    volume = np.sum(melt_mask) * dx * dy * dz * 1e18  # μm³

    # Find maximum temperature
    T_max = np.max(temp)
    T_max_idx = np.unravel_index(np.argmax(temp), temp.shape)

    # Surface temperature (top layer, z_max)
    surface_temp = temp[:, :, -1]
    T_surface_max = np.max(surface_temp)

    results = {
        'length_um': length,
        'width_um': width,
        'depth_um': depth,
        'volume_um3': volume,
        'T_max_K': T_max,
        'T_max_location': T_max_idx,
        'T_surface_max_K': T_surface_max,
        'num_liquid_cells': np.sum(melt_mask),
    }

    print(f"\n=== MELT POOL ANALYSIS ===")
    print(f"  Liquidus temperature: {T_liquidus} K")
    print(f"  Melt pool dimensions:")
    print(f"    Length: {length:.1f} μm")
    print(f"    Width:  {width:.1f} μm")
    print(f"    Depth:  {depth:.1f} μm")
    print(f"    Volume: {volume:.1f} μm³")
    print(f"  Liquid cells: {results['num_liquid_cells']}")
    print(f"  Maximum temperature: {T_max:.1f} K")
    print(f"    Location: ({T_max_idx[0]}, {T_max_idx[1]}, {T_max_idx[2]})")
    print(f"  Surface max temperature: {T_surface_max:.1f} K")

    return results


def analyze_velocity_field(data):
    """
    Analyze velocity field statistics.

    Returns:
        dict with velocity properties:
            - v_max [m/s]
            - v_mean [m/s]
            - v_max_location [indices]
    """
    if 'velocity' not in data:
        print("ERROR: Velocity field not found in VTK file")
        return None

    vel = data['velocity']

    # Compute velocity magnitude
    v_mag = np.linalg.norm(vel, axis=3)

    v_max = np.max(v_mag)
    v_mean = np.mean(v_mag)
    v_max_idx = np.unravel_index(np.argmax(v_mag), v_mag.shape)

    results = {
        'v_max_m_s': v_max,
        'v_mean_m_s': v_mean,
        'v_max_location': v_max_idx,
    }

    print(f"\n=== VELOCITY FIELD ANALYSIS ===")
    print(f"  Maximum velocity: {v_max*1000:.3f} mm/s")
    print(f"    Location: ({v_max_idx[0]}, {v_max_idx[1]}, {v_max_idx[2]})")
    print(f"  Mean velocity: {v_mean*1000:.3f} mm/s")

    return results


def analyze_temperature_profile(data, axis='z'):
    """
    Extract temperature profile along specified axis.

    Args:
        data: VTK data dict
        axis: 'x', 'y', or 'z'

    Returns:
        (positions, temperatures) arrays
    """
    if 'temperature' not in data:
        return None, None

    temp = data['temperature']
    dx, dy, dz = data['spacing']

    # Find location of maximum temperature
    T_max_idx = np.unravel_index(np.argmax(temp), temp.shape)
    ix, iy, iz = T_max_idx

    if axis == 'z':
        # Vertical profile through hottest point
        profile = temp[ix, iy, :]
        positions = np.arange(len(profile)) * dz * 1e6  # μm
    elif axis == 'x':
        profile = temp[:, iy, iz]
        positions = np.arange(len(profile)) * dx * 1e6
    elif axis == 'y':
        profile = temp[ix, :, iz]
        positions = np.arange(len(profile)) * dy * 1e6
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return positions, profile


def compare_with_literature(results):
    """
    Compare simulation results with literature data.

    Literature reference:
    - Mohr et al. 2020 (ISS microgravity experiments)
        - Laser power: 100-200 W
        - Material: Ti6Al4V
        - Peak temperature: 2,400-2,800 K
        - Melt pool length: 150-300 μm
    """
    print(f"\n=== LITERATURE COMPARISON ===")
    print(f"Reference: Mohr et al. 2020 (ISS microgravity)")
    print(f"  Material: Ti6Al4V")
    print(f"  Laser power: 195 W")
    print()

    # Compare temperature
    lit_T_min, lit_T_max = 2400, 2800
    sim_T_max = results['melt_pool']['T_max_K']

    print(f"Peak Temperature:")
    print(f"  Literature: {lit_T_min}-{lit_T_max} K")
    print(f"  Simulation: {sim_T_max:.1f} K")

    if lit_T_min <= sim_T_max <= lit_T_max:
        print(f"  Status: WITHIN RANGE ✓")
    elif sim_T_max < lit_T_min:
        print(f"  Status: TOO LOW ({(lit_T_min - sim_T_max):.1f} K below range)")
    else:
        print(f"  Status: TOO HIGH ({(sim_T_max - lit_T_max):.1f} K above range)")

    print()

    # Compare melt pool length
    lit_L_min, lit_L_max = 150, 300  # μm
    sim_L = results['melt_pool']['length_um']

    print(f"Melt Pool Length:")
    print(f"  Literature: {lit_L_min}-{lit_L_max} μm")
    print(f"  Simulation: {sim_L:.1f} μm")

    if lit_L_min <= sim_L <= lit_L_max:
        print(f"  Status: WITHIN RANGE ✓")
    elif sim_L < lit_L_min:
        print(f"  Status: TOO SHORT ({(lit_L_min - sim_L):.1f} μm below range)")
    else:
        print(f"  Status: TOO LONG ({(sim_L - lit_L_max):.1f} μm above range)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze LPBF simulation VTK output files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python analyze_vtk.py build/lpbf_test_H_full_physics/lpbf_005000.vtk

  # Generate plots
  python analyze_vtk.py lpbf_005000.vtk --plot

  # Save report to file
  python analyze_vtk.py lpbf_005000.vtk --output report.txt
        """
    )

    parser.add_argument('vtk_file', help='Path to VTK file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', help='Save report to file')

    args = parser.parse_args()

    # Check file exists
    if not os.path.exists(args.vtk_file):
        print(f"ERROR: File not found: {args.vtk_file}")
        sys.exit(1)

    # Read VTK file
    data = read_vtk_with_vtk_library(args.vtk_file)

    # Analyze
    results = {}

    melt_pool = analyze_melt_pool(data)
    if melt_pool:
        results['melt_pool'] = melt_pool

    velocity = analyze_velocity_field(data)
    if velocity:
        results['velocity'] = velocity

    # Compare with literature
    if 'melt_pool' in results:
        compare_with_literature(results)

    # Plot if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            print("\nGenerating plots...")
            plot_results(data, results)
        except ImportError:
            print("ERROR: matplotlib not available, cannot generate plots")

    # Save report if requested
    if args.output:
        save_report(args.output, results)
        print(f"\nReport saved to: {args.output}")

    print("\nAnalysis complete.")


def plot_results(data, results):
    """Generate diagnostic plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Temperature distribution (top view)
    if 'temperature' in data:
        temp = data['temperature']
        ax = axes[0, 0]
        im = ax.imshow(temp[:, :, -1].T, origin='lower', cmap='hot')
        ax.set_title('Surface Temperature (top layer)')
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
        plt.colorbar(im, ax=ax, label='Temperature [K]')

    # Velocity magnitude (top view)
    if 'velocity' in data:
        vel = data['velocity']
        v_mag = np.linalg.norm(vel, axis=3)
        ax = axes[0, 1]
        im = ax.imshow(v_mag[:, :, -1].T, origin='lower', cmap='viridis')
        ax.set_title('Surface Velocity Magnitude')
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
        plt.colorbar(im, ax=ax, label='Velocity [m/s]')

    # Temperature profile (depth)
    if 'temperature' in data:
        positions, profile = analyze_temperature_profile(data, axis='z')
        if profile is not None:
            ax = axes[1, 0]
            ax.plot(profile, positions)
            ax.axvline(MATERIAL['T_liquidus'], color='r', linestyle='--', label='T_liquidus')
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Depth [μm]')
            ax.set_title('Temperature Profile (vertical)')
            ax.legend()
            ax.grid(True)

    # Velocity distribution histogram
    if 'velocity' in data:
        vel = data['velocity']
        v_mag = np.linalg.norm(vel, axis=3).flatten()
        ax = axes[1, 1]
        ax.hist(v_mag * 1000, bins=50, edgecolor='black')
        ax.set_xlabel('Velocity [mm/s]')
        ax.set_ylabel('Frequency')
        ax.set_title('Velocity Distribution')
        ax.set_yscale('log')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('vtk_analysis.png', dpi=150)
    print("Plot saved to: vtk_analysis.png")


def save_report(filename, results):
    """Save analysis results to text file."""
    with open(filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("LPBF SIMULATION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        if 'melt_pool' in results:
            f.write("MELT POOL GEOMETRY:\n")
            mp = results['melt_pool']
            f.write(f"  Length: {mp['length_um']:.1f} μm\n")
            f.write(f"  Width:  {mp['width_um']:.1f} μm\n")
            f.write(f"  Depth:  {mp['depth_um']:.1f} μm\n")
            f.write(f"  Volume: {mp['volume_um3']:.1f} μm³\n")
            f.write(f"  Max temperature: {mp['T_max_K']:.1f} K\n")
            f.write(f"  Surface max temperature: {mp['T_surface_max_K']:.1f} K\n")
            f.write("\n")

        if 'velocity' in results:
            f.write("VELOCITY FIELD:\n")
            v = results['velocity']
            f.write(f"  Max velocity: {v['v_max_m_s']*1000:.3f} mm/s\n")
            f.write(f"  Mean velocity: {v['v_mean_m_s']*1000:.3f} mm/s\n")
            f.write("\n")


if __name__ == '__main__':
    main()
