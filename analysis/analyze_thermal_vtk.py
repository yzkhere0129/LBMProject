#!/usr/bin/env python3
"""
Thermal Physics Analysis Script for LBM CFD Simulation VTK Output

Analyzes temperature fields, phase change phenomena, and energy distribution
from VTK files produced by the LBM thermal solver.

Physical context: Metal additive manufacturing (LPBF), laser melting simulations
Material: Ti6Al4V (Tm = 1923 K)
"""

import numpy as np
import glob
import os
import sys

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: pyvista not available, using basic VTK parsing")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

# === PHYSICAL PARAMETERS (Ti6Al4V) ===
T_MELTING = 1923.0      # K, melting point
T_AMBIENT = 300.0       # K, ambient/initial temperature
T_SOLIDUS = 1878.0      # K, solidus temperature
T_LIQUIDUS = 1928.0     # K, liquidus temperature
DENSITY = 4420.0        # kg/m³
CP = 610.0              # J/(kg·K)
LATENT_HEAT = 286000.0  # J/kg

# === ANALYSIS PARAMETERS ===
VTK_DIRECTORY = "/home/yzk/LBMProject/build/visualization_output"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/thermal_analysis"
REGION_OF_INTEREST = None  # None = analyze full domain, or {'x': (min, max), ...}


def parse_vtk_ascii(filename):
    """Parse ASCII VTK file manually (fallback if pyvista unavailable)."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    dims = None
    origin = None
    spacing = None

    for i, line in enumerate(lines):
        if line.startswith('DIMENSIONS'):
            dims = [int(x) for x in line.split()[1:4]]
        elif line.startswith('ORIGIN'):
            origin = [float(x) for x in line.split()[1:4]]
        elif line.startswith('SPACING'):
            spacing = [float(x) for x in line.split()[1:4]]
        elif line.startswith('POINT_DATA'):
            n_points = int(line.split()[1])
            data_start = i + 1
            break

    if dims is None:
        raise ValueError(f"Could not parse dimensions from {filename}")

    # Parse data arrays
    data = {}
    current_field = None
    current_data = []

    for line in lines[data_start:]:
        line = line.strip()
        if line.startswith('VECTORS'):
            if current_field and current_data:
                data[current_field] = np.array(current_data)
            current_field = line.split()[1]
            current_data = []
        elif line.startswith('SCALARS'):
            if current_field and current_data:
                data[current_field] = np.array(current_data)
            current_field = line.split()[1]
            current_data = []
        elif line.startswith('LOOKUP_TABLE'):
            continue
        elif line and not line.startswith('#'):
            values = [float(x) for x in line.split()]
            current_data.extend(values)

    if current_field and current_data:
        data[current_field] = np.array(current_data)

    # Reshape scalar fields
    for key in data:
        if key == 'Velocity':
            data[key] = data[key].reshape(-1, 3)
        else:
            data[key] = data[key].reshape(-1)

    return {
        'dimensions': dims,
        'origin': origin,
        'spacing': spacing,
        'data': data
    }


def load_vtk_file(filename):
    """Load VTK file using pyvista or fallback parser."""
    if HAS_PYVISTA:
        mesh = pv.read(filename)
        return {
            'mesh': mesh,
            'dimensions': mesh.dimensions,
            'origin': mesh.origin,
            'spacing': mesh.spacing,
            'data': {name: mesh[name] for name in mesh.array_names}
        }
    else:
        return parse_vtk_ascii(filename)


def compute_temperature_statistics(temp_field):
    """Compute comprehensive temperature statistics."""
    valid_temps = temp_field[~np.isnan(temp_field)]

    if len(valid_temps) == 0:
        return None

    stats = {
        'min': np.min(valid_temps),
        'max': np.max(valid_temps),
        'mean': np.mean(valid_temps),
        'median': np.median(valid_temps),
        'std': np.std(valid_temps),
        'q25': np.percentile(valid_temps, 25),
        'q75': np.percentile(valid_temps, 75),
        'n_valid': len(valid_temps),
        'n_nan': np.sum(np.isnan(temp_field))
    }

    # Phase-specific statistics
    stats['n_molten'] = np.sum(valid_temps >= T_SOLIDUS)
    stats['n_mushy'] = np.sum((valid_temps >= T_SOLIDUS) & (valid_temps <= T_LIQUIDUS))
    stats['n_superheated'] = np.sum(valid_temps > T_LIQUIDUS)
    stats['fraction_molten'] = stats['n_molten'] / len(valid_temps)

    # Maximum superheat
    stats['max_superheat'] = stats['max'] - T_LIQUIDUS

    return stats


def compute_temperature_gradients(temp_field, spacing, dimensions):
    """Compute temperature gradient magnitudes."""
    # Reshape to 3D grid
    temp_3d = temp_field.reshape(dimensions[::-1])  # VTK uses reverse order (Z,Y,X)

    # Compute gradients using central differences
    dx, dy, dz = spacing

    # Gradient in each direction (converting to physical units: K/m)
    grad_x = np.gradient(temp_3d, dx, axis=2)
    grad_y = np.gradient(temp_3d, dy, axis=1)
    grad_z = np.gradient(temp_3d, dz, axis=0)

    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    # Convert to K/mm for easier interpretation
    grad_mag_mm = grad_mag / 1000.0

    return {
        'max_gradient': np.max(grad_mag_mm),
        'mean_gradient': np.mean(grad_mag_mm),
        'grad_field': grad_mag_mm.flatten()
    }


def compute_thermal_energy(temp_field, spacing, dimensions):
    """Compute total thermal energy in the domain."""
    # Cell volume
    dx, dy, dz = spacing
    cell_volume = dx * dy * dz  # m³

    # Mass per cell
    cell_mass = DENSITY * cell_volume  # kg

    # Temperature above ambient
    delta_T = temp_field - T_AMBIENT

    # Sensible heat per cell
    sensible_heat = cell_mass * CP * delta_T  # J

    # Total energy
    total_energy = np.sum(sensible_heat[~np.isnan(sensible_heat)])

    return {
        'total_energy_J': total_energy,
        'total_energy_mJ': total_energy * 1000.0,
        'cell_volume_um3': cell_volume * 1e18,  # µm³
        'cell_mass_ng': cell_mass * 1e12  # ng
    }


def analyze_melt_pool_geometry(temp_field, liquid_fraction, spacing, dimensions):
    """Extract melt pool geometry and characteristics."""
    # Identify molten region
    molten_mask = temp_field >= T_SOLIDUS

    if np.sum(molten_mask) == 0:
        return {'exists': False}

    # Reshape to 3D
    molten_3d = molten_mask.reshape(dimensions[::-1])
    temp_3d = temp_field.reshape(dimensions[::-1])

    # Find melt pool boundaries
    molten_indices = np.argwhere(molten_3d)

    if len(molten_indices) == 0:
        return {'exists': False}

    # Melt pool extent (in grid indices)
    z_min, y_min, x_min = molten_indices.min(axis=0)
    z_max, y_max, x_max = molten_indices.max(axis=0)

    # Convert to physical dimensions (µm)
    dx, dy, dz = spacing

    width_x = (x_max - x_min + 1) * dx * 1e6  # µm
    width_y = (y_max - y_min + 1) * dy * 1e6  # µm
    depth_z = (z_max - z_min + 1) * dz * 1e6  # µm

    # Melt pool volume
    n_molten_cells = np.sum(molten_mask)
    cell_volume = dx * dy * dz  # m³
    melt_volume_um3 = n_molten_cells * cell_volume * 1e18

    # Peak temperature in melt pool
    molten_temps = temp_3d[molten_3d]

    return {
        'exists': True,
        'width_x_um': width_x,
        'width_y_um': width_y,
        'depth_z_um': depth_z,
        'volume_um3': melt_volume_um3,
        'n_cells': n_molten_cells,
        'peak_temp_K': np.max(molten_temps),
        'mean_temp_K': np.mean(molten_temps),
        'aspect_ratio': depth_z / max(width_x, width_y) if max(width_x, width_y) > 0 else 0
    }


def extract_centerline_profile(temp_field, dimensions, spacing):
    """Extract temperature profile along centerline (z-axis at domain center)."""
    temp_3d = temp_field.reshape(dimensions[::-1])

    # Center of domain in x-y
    nx, ny, nz = dimensions
    center_x = nx // 2
    center_y = ny // 2

    # Extract centerline (varying z)
    centerline_temps = temp_3d[:, center_y, center_x]

    # Z-coordinates (physical units: µm)
    z_coords = np.arange(nz) * spacing[2] * 1e6

    return z_coords, centerline_temps


def analyze_thermal_vtk_file(filename, verbose=True):
    """Complete thermal analysis of a single VTK file."""
    if verbose:
        print(f"\nAnalyzing: {os.path.basename(filename)}")
        print("=" * 60)

    # Load data
    vtk_data = load_vtk_file(filename)

    # Extract fields
    temp_field = vtk_data['data'].get('Temperature')
    liquid_frac = vtk_data['data'].get('LiquidFraction')
    velocity = vtk_data['data'].get('Velocity')

    if temp_field is None:
        print("ERROR: No temperature field found in VTK file")
        return None

    dims = vtk_data['dimensions']
    spacing = vtk_data['spacing']

    results = {
        'filename': filename,
        'dimensions': dims,
        'spacing_um': [s * 1e6 for s in spacing],
        'domain_size_um': [d * s * 1e6 for d, s in zip(dims, spacing)]
    }

    # Temperature statistics
    temp_stats = compute_temperature_statistics(temp_field)
    results['temperature'] = temp_stats

    if verbose and temp_stats:
        print(f"\nTemperature Statistics:")
        print(f"  Min:     {temp_stats['min']:.2f} K")
        print(f"  Max:     {temp_stats['max']:.2f} K")
        print(f"  Mean:    {temp_stats['mean']:.2f} K")
        print(f"  Std:     {temp_stats['std']:.2f} K")
        print(f"  Molten fraction: {temp_stats['fraction_molten']*100:.2f}%")
        print(f"  Max superheat:   {temp_stats['max_superheat']:.2f} K")

    # Temperature gradients
    grad_results = compute_temperature_gradients(temp_field, spacing, dims)
    results['gradients'] = {
        'max_K_per_mm': grad_results['max_gradient'],
        'mean_K_per_mm': grad_results['mean_gradient']
    }

    if verbose:
        print(f"\nTemperature Gradients:")
        print(f"  Max:  {grad_results['max_gradient']:.2e} K/mm")
        print(f"  Mean: {grad_results['mean_gradient']:.2e} K/mm")

    # Thermal energy
    energy = compute_thermal_energy(temp_field, spacing, dims)
    results['energy'] = energy

    if verbose:
        print(f"\nThermal Energy:")
        print(f"  Total: {energy['total_energy_mJ']:.4f} mJ")

    # Melt pool geometry
    if liquid_frac is not None:
        melt_pool = analyze_melt_pool_geometry(temp_field, liquid_frac, spacing, dims)
        results['melt_pool'] = melt_pool

        if verbose and melt_pool['exists']:
            print(f"\nMelt Pool Geometry:")
            print(f"  Width (X):  {melt_pool['width_x_um']:.2f} µm")
            print(f"  Width (Y):  {melt_pool['width_y_um']:.2f} µm")
            print(f"  Depth (Z):  {melt_pool['depth_z_um']:.2f} µm")
            print(f"  Volume:     {melt_pool['volume_um3']:.2f} µm³")
            print(f"  Peak temp:  {melt_pool['peak_temp_K']:.2f} K")
            print(f"  Aspect ratio (D/W): {melt_pool['aspect_ratio']:.3f}")

    # Data quality checks
    n_nan = np.sum(np.isnan(temp_field))
    n_negative = np.sum(temp_field < 0)
    n_extreme = np.sum(temp_field > 5000)  # Unrealistically high

    results['quality'] = {
        'n_nan': n_nan,
        'n_negative': n_negative,
        'n_extreme_high': n_extreme
    }

    if verbose and (n_nan > 0 or n_negative > 0 or n_extreme > 0):
        print(f"\nData Quality Issues:")
        if n_nan > 0:
            print(f"  WARNING: {n_nan} NaN values detected")
        if n_negative > 0:
            print(f"  WARNING: {n_negative} negative temperatures")
        if n_extreme > 0:
            print(f"  WARNING: {n_extreme} extreme temperatures (>5000K)")

    return results


def analyze_time_series(vtk_directory, pattern="*.vtk"):
    """Analyze time series of VTK files."""
    vtk_files = sorted(glob.glob(os.path.join(vtk_directory, pattern)))

    if len(vtk_files) == 0:
        print(f"ERROR: No VTK files found in {vtk_directory}")
        return None

    print(f"Found {len(vtk_files)} VTK files")
    print(f"First: {os.path.basename(vtk_files[0])}")
    print(f"Last:  {os.path.basename(vtk_files[-1])}")

    results = []

    # Analyze subset if too many files
    if len(vtk_files) > 50:
        print(f"\nAnalyzing subset of files (every {len(vtk_files)//50} files)...")
        vtk_files = vtk_files[::len(vtk_files)//50]

    for i, vtk_file in enumerate(vtk_files):
        print(f"\nProgress: {i+1}/{len(vtk_files)}")
        result = analyze_thermal_vtk_file(vtk_file, verbose=(i == 0 or i == len(vtk_files)-1))
        if result:
            results.append(result)

    return results


def plot_time_evolution(results):
    """Plot temperature evolution over time."""
    if not HAS_MATPLOTLIB:
        print("Cannot create plots without matplotlib")
        return

    # Extract time series data
    max_temps = [r['temperature']['max'] for r in results]
    mean_temps = [r['temperature']['mean'] for r in results]
    molten_fracs = [r['temperature']['fraction_molten'] * 100 for r in results]
    energies = [r['energy']['total_energy_mJ'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Temperature evolution
    axes[0, 0].plot(max_temps, 'r-', linewidth=2, label='Max')
    axes[0, 0].plot(mean_temps, 'b-', linewidth=2, label='Mean')
    axes[0, 0].axhline(T_MELTING, color='k', linestyle='--', label='Melting point')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Temperature (K)')
    axes[0, 0].set_title('Temperature Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Molten fraction
    axes[0, 1].plot(molten_fracs, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Molten fraction (%)')
    axes[0, 1].set_title('Melt Pool Evolution')
    axes[0, 1].grid(True, alpha=0.3)

    # Thermal energy
    axes[1, 0].plot(energies, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Total energy (mJ)')
    axes[1, 0].set_title('Thermal Energy Conservation')
    axes[1, 0].grid(True, alpha=0.3)

    # Melt pool volume (if available)
    melt_volumes = []
    for r in results:
        if 'melt_pool' in r and r['melt_pool'].get('exists', False):
            melt_volumes.append(r['melt_pool']['volume_um3'])
        else:
            melt_volumes.append(0)

    axes[1, 1].plot(melt_volumes, 'c-', linewidth=2)
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('Melt pool volume (µm³)')
    axes[1, 1].set_title('Melt Pool Volume')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(OUTPUT_DIR, 'thermal_evolution.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    return fig


def generate_summary_report(results, output_file=None):
    """Generate comprehensive summary report."""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, 'thermal_analysis_report.txt')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THERMAL PHYSICS VTK ANALYSIS REPORT\n")
        f.write("LBM CFD Simulation - Metal Additive Manufacturing\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Analysis date: {os.popen('date').read()}\n")
        f.write(f"Number of files analyzed: {len(results)}\n")
        f.write(f"Material: Ti6Al4V (Tm = {T_MELTING} K)\n\n")

        # Overall statistics
        all_max_temps = [r['temperature']['max'] for r in results]
        all_energies = [r['energy']['total_energy_mJ'] for r in results]

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Peak temperature reached:     {max(all_max_temps):.2f} K\n")
        f.write(f"Maximum superheat:            {max(all_max_temps) - T_MELTING:.2f} K\n")
        f.write(f"Energy range:                 {min(all_energies):.4f} - {max(all_energies):.4f} mJ\n")
        f.write(f"Energy drift:                 {(all_energies[-1] - all_energies[0]):.4e} mJ\n")
        f.write(f"Relative energy drift:        {100*(all_energies[-1]/all_energies[0] - 1):.4f}%\n\n")

        # First and last snapshots
        f.write("INITIAL STATE (t=0)\n")
        f.write("-" * 80 + "\n")
        r0 = results[0]
        f.write(f"Temperature: {r0['temperature']['min']:.2f} - {r0['temperature']['max']:.2f} K\n")
        f.write(f"Mean temperature: {r0['temperature']['mean']:.2f} K\n")
        f.write(f"Thermal energy: {r0['energy']['total_energy_mJ']:.4f} mJ\n\n")

        f.write("FINAL STATE\n")
        f.write("-" * 80 + "\n")
        rf = results[-1]
        f.write(f"Temperature: {rf['temperature']['min']:.2f} - {rf['temperature']['max']:.2f} K\n")
        f.write(f"Mean temperature: {rf['temperature']['mean']:.2f} K\n")
        f.write(f"Molten fraction: {rf['temperature']['fraction_molten']*100:.2f}%\n")
        f.write(f"Thermal energy: {rf['energy']['total_energy_mJ']:.4f} mJ\n")

        if 'melt_pool' in rf and rf['melt_pool'].get('exists', False):
            mp = rf['melt_pool']
            f.write(f"\nMelt Pool Geometry:\n")
            f.write(f"  Dimensions: {mp['width_x_um']:.1f} x {mp['width_y_um']:.1f} x {mp['depth_z_um']:.1f} µm\n")
            f.write(f"  Volume: {mp['volume_um3']:.2f} µm³\n")
            f.write(f"  Peak temperature: {mp['peak_temp_K']:.2f} K\n")

        # Data quality summary
        f.write("\n\nDATA QUALITY ASSESSMENT\n")
        f.write("-" * 80 + "\n")

        total_nan = sum(r['quality']['n_nan'] for r in results)
        total_negative = sum(r['quality']['n_negative'] for r in results)
        total_extreme = sum(r['quality']['n_extreme_high'] for r in results)

        f.write(f"Total NaN values:         {total_nan}\n")
        f.write(f"Negative temperatures:    {total_negative}\n")
        f.write(f"Extreme values (>5000K):  {total_extreme}\n")

        if total_nan + total_negative + total_extreme == 0:
            f.write("\nSTATUS: All data quality checks PASSED\n")
        else:
            f.write("\nWARNING: Data quality issues detected - review required\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")

    print(f"\nSummary report saved to: {output_file}")

    # Print to console as well
    with open(output_file, 'r') as f:
        print("\n" + f.read())


def main():
    """Main analysis workflow."""
    global VTK_DIRECTORY

    print("=" * 80)
    print("LBM THERMAL PHYSICS VTK ANALYSIS")
    print("=" * 80)
    print(f"\nVTK directory: {VTK_DIRECTORY}")

    # Check if directory exists
    if not os.path.isdir(VTK_DIRECTORY):
        print(f"ERROR: Directory not found: {VTK_DIRECTORY}")
        print("\nSearching for alternative VTK output directories...")

        # Search for VTK files
        search_dirs = [
            "/home/yzk/LBMProject/build/tests/integration/test_output",
            "/home/yzk/LBMProject/build/tests/validation",
            "/home/yzk/LBMProject/build/test_output"
        ]

        for search_dir in search_dirs:
            if os.path.isdir(search_dir):
                vtk_files = glob.glob(os.path.join(search_dir, "*.vtk"))
                if vtk_files:
                    print(f"Found {len(vtk_files)} VTK files in: {search_dir}")
                    response = input(f"Analyze this directory? (y/n): ")
                    if response.lower() == 'y':
                        VTK_DIRECTORY = search_dir
                        break
        else:
            print("ERROR: No VTK files found in any standard location")
            return 1

    # Analyze time series
    results = analyze_time_series(VTK_DIRECTORY)

    if not results:
        print("ERROR: No results generated")
        return 1

    # Generate plots
    if HAS_MATPLOTLIB:
        plot_time_evolution(results)

    # Generate report
    generate_summary_report(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
