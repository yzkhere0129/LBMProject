#!/usr/bin/env python3
"""
Analyze mass loss pattern in Rayleigh-Taylor simulation using pure VTK.

This script investigates where and when mass is being lost by examining
the fill level (volume fraction) field across multiple timesteps.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# === PARAMETERS ===
VTK_FILES = [
    "/home/yzk/LBMProject/build/output_rt_mushroom/rt_mushroom_t0.00.vtk",
    "/home/yzk/LBMProject/build/output_rt_mushroom/rt_mushroom_t0.20.vtk",
    "/home/yzk/LBMProject/build/output_rt_mushroom/rt_mushroom_t0.70.vtk",
    "/home/yzk/LBMProject/build/output_rt_mushroom/rt_mushroom_t1.00.vtk",
]

TIMES = [0.00, 0.20, 0.70, 1.00]  # Physical times in seconds

# Thresholds for categorizing regions
INTERFACE_THRESHOLD = (0.1, 0.9)  # f values considered interface
VELOCITY_THRESHOLD_LOW = 0.05    # m/s - below this is "low velocity"
VELOCITY_THRESHOLD_HIGH = 0.15   # m/s - above this is "high velocity"

OUTPUT_DIR = Path("/home/yzk/LBMProject/build/output_rt_mushroom")
OUTPUT_PREFIX = "mass_loss_analysis"

# === HELPER FUNCTIONS ===

def load_vtk_data(filepath: str):
    """Load VTK structured grid file."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"VTK file not found: {filepath}")

    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filepath)
    reader.Update()

    output = reader.GetOutput()

    # Extract metadata
    dims = output.GetDimensions()
    spacing = output.GetSpacing()
    origin = output.GetOrigin()
    point_data = output.GetPointData()

    print(f"Loaded: {Path(filepath).name}")
    print(f"  Dimensions: {dims}")
    print(f"  Spacing: {spacing}")
    print(f"  Origin: {origin}")
    print(f"  Number of arrays: {point_data.GetNumberOfArrays()}")

    # List available arrays
    array_names = []
    for i in range(point_data.GetNumberOfArrays()):
        array_names.append(point_data.GetArrayName(i))
    print(f"  Available arrays: {array_names}")

    return output, dims, spacing, origin, array_names


def get_array(vtk_data, array_name: str):
    """Extract numpy array from VTK data."""
    point_data = vtk_data.GetPointData()
    vtk_array = point_data.GetArray(array_name)

    if vtk_array is None:
        return None

    return vtk_to_numpy(vtk_array)


def compute_total_mass(vtk_data, spacing, fill_array_name: str = "fill") -> float:
    """
    Compute total mass as sum of fill level.

    For VOF, total "mass" is the sum of volume fractions times cell volume.
    """
    fill = get_array(vtk_data, fill_array_name)

    if fill is None:
        print(f"Warning: Array '{fill_array_name}' not found")
        return np.nan

    # Cell volume (assuming uniform grid)
    cell_volume = spacing[0] * spacing[1] * spacing[2]

    total_mass = np.sum(fill) * cell_volume

    return total_mass


def categorize_cells(vtk_data, dims, fill_array_name: str = "fill",
                     velocity_array_name: str = "velocity") -> Dict[str, np.ndarray]:
    """
    Categorize cells into different regions for analysis.

    Returns masks for:
    - interface: cells with fill in intermediate range
    - bulk_fluid: cells with fill > 0.9
    - bulk_empty: cells with fill < 0.1
    - boundary: cells on domain boundaries
    - high_velocity: cells with |v| > threshold
    """
    fill = get_array(vtk_data, fill_array_name)

    if fill is None:
        raise ValueError(f"Fill array '{fill_array_name}' not found")

    # Interface cells
    interface_mask = (fill >= INTERFACE_THRESHOLD[0]) & (fill <= INTERFACE_THRESHOLD[1])

    # Bulk regions
    bulk_fluid_mask = fill > 0.9
    bulk_empty_mask = fill < 0.1

    # Boundary cells (first and last in each direction)
    nx, ny, nz = dims
    boundary_mask = np.zeros_like(fill, dtype=bool)

    # Convert to 3D index space
    # VTK structured points are ordered: x varies fastest, then y, then z
    indices = np.arange(len(fill))
    iz = indices // (nx * ny)
    iy = (indices % (nx * ny)) // nx
    ix = indices % nx

    boundary_mask = (ix == 0) | (ix == nx-1) | \
                   (iy == 0) | (iy == ny-1) | \
                   (iz == 0) | (iz == nz-1)

    # High velocity cells
    high_velocity_mask = np.zeros_like(fill, dtype=bool)
    velocity = get_array(vtk_data, velocity_array_name)

    if velocity is not None:
        if velocity.ndim == 2:  # Vector field
            v_mag = np.linalg.norm(velocity, axis=1)
        else:
            v_mag = np.abs(velocity)
        high_velocity_mask = v_mag > VELOCITY_THRESHOLD_HIGH

    return {
        'interface': interface_mask,
        'bulk_fluid': bulk_fluid_mask,
        'bulk_empty': bulk_empty_mask,
        'boundary': boundary_mask,
        'high_velocity': high_velocity_mask,
    }


def compute_mass_in_regions(vtk_data, spacing, masks: Dict[str, np.ndarray],
                            fill_array_name: str = "fill") -> Dict[str, float]:
    """Compute total mass in each categorized region."""
    fill = get_array(vtk_data, fill_array_name)
    cell_volume = spacing[0] * spacing[1] * spacing[2]

    masses = {}
    for region_name, mask in masks.items():
        masses[region_name] = np.sum(fill[mask]) * cell_volume

    return masses


def get_2d_slice_middle(vtk_data, dims, spacing, origin, array_name: str):
    """Extract a 2D slice through the middle of the domain (constant x)."""
    nx, ny, nz = dims

    # Middle index in x direction
    mid_x = nx // 2

    # Extract data for this slice
    data = get_array(vtk_data, array_name)

    if data is None:
        return None, None, None

    # Reshape to 3D grid
    if data.ndim == 1:
        data_3d = data.reshape((nz, ny, nx))
    else:
        # Vector field - take magnitude
        v_mag = np.linalg.norm(data, axis=1)
        data_3d = v_mag.reshape((nz, ny, nx))

    # Extract slice at mid_x
    slice_data = data_3d[:, :, mid_x]

    # Create coordinate arrays for the slice
    y_coords = origin[1] + np.arange(ny) * spacing[1]
    z_coords = origin[2] + np.arange(nz) * spacing[2]

    # Create meshgrid
    Y, Z = np.meshgrid(y_coords, z_coords)

    return Y, Z, slice_data


# === MAIN ANALYSIS ===

def main():
    print("="*70)
    print("Rayleigh-Taylor Mass Loss Analysis")
    print("="*70)

    # Load all timesteps
    vtk_datasets = []
    dims_list = []
    spacing_list = []
    origin_list = []
    array_names_list = []

    for filepath in VTK_FILES:
        try:
            vtk_data, dims, spacing, origin, array_names = load_vtk_data(filepath)
            vtk_datasets.append(vtk_data)
            dims_list.append(dims)
            spacing_list.append(spacing)
            origin_list.append(origin)
            array_names_list.append(array_names)
            print()
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            sys.exit(1)

    # Determine fill array name
    fill_array_name = None
    for name in ['fill', 'fill_level', 'f', 'vof']:
        if name in array_names_list[0]:
            fill_array_name = name
            break

    if fill_array_name is None:
        print(f"Error: No fill level array found. Available: {array_names_list[0]}")
        sys.exit(1)

    print(f"\nUsing fill array: '{fill_array_name}'")

    # Determine velocity array name
    velocity_array_name = None
    for name in ['velocity', 'u', 'vel']:
        if name in array_names_list[0]:
            velocity_array_name = name
            break

    if velocity_array_name:
        print(f"Using velocity array: '{velocity_array_name}'")
    else:
        print("Warning: No velocity array found")

    # === 1. TOTAL MASS VS TIME ===
    print("\n" + "="*70)
    print("1. TOTAL MASS VS TIME")
    print("="*70)

    total_masses = []
    for i, (vtk_data, spacing, time) in enumerate(zip(vtk_datasets, spacing_list, TIMES)):
        mass = compute_total_mass(vtk_data, spacing, fill_array_name)
        total_masses.append(mass)

        if i == 0:
            mass_loss_pct = 0.0
        else:
            mass_loss_pct = 100.0 * (mass - total_masses[0]) / total_masses[0]

        print(f"t = {time:.2f}s: Mass = {mass:.6e}, Loss = {mass_loss_pct:+.4f}%")

    # Plot mass vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(TIMES, total_masses, 'o-', linewidth=2, markersize=8, color='navy')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Total Mass (arbitrary units)', fontsize=12)
    ax.set_title('Mass Conservation in Rayleigh-Taylor Simulation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add percentage loss annotations
    for i, (t, m) in enumerate(zip(TIMES, total_masses)):
        if i > 0:
            pct_loss = 100.0 * (m - total_masses[0]) / total_masses[0]
            ax.annotate(f'{pct_loss:+.3f}%',
                       xy=(t, m),
                       xytext=(8, 8),
                       textcoords='offset points',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{OUTPUT_PREFIX}_mass_vs_time.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved: {output_file}")
    plt.close()

    # === 2. REGIONAL MASS DISTRIBUTION ===
    print("\n" + "="*70)
    print("2. MASS DISTRIBUTION BY REGION")
    print("="*70)

    regional_masses = {region: [] for region in ['interface', 'bulk_fluid', 'boundary', 'high_velocity']}

    for i, (vtk_data, dims, spacing, time) in enumerate(zip(vtk_datasets, dims_list, spacing_list, TIMES)):
        print(f"\nt = {time:.2f}s:")
        masks = categorize_cells(vtk_data, dims, fill_array_name, velocity_array_name)
        masses = compute_mass_in_regions(vtk_data, spacing, masks, fill_array_name)

        for region, mass in masses.items():
            if region in regional_masses:
                regional_masses[region].append(mass)
                if i == 0:
                    pct_change = 0.0
                else:
                    pct_change = 100.0 * (mass - regional_masses[region][0]) / (regional_masses[region][0] + 1e-10)
                print(f"  {region:15s}: {mass:.6e} ({pct_change:+.4f}%)")

    # Plot regional mass evolution
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'interface': 'orange', 'bulk_fluid': 'blue', 'boundary': 'red', 'high_velocity': 'green'}
    for region, masses in regional_masses.items():
        # Normalize to initial value
        if masses[0] > 1e-10:
            normalized = 100.0 * np.array(masses) / masses[0]
            ax.plot(TIMES, normalized, 'o-', label=region, linewidth=2, markersize=6,
                   color=colors.get(region, 'gray'))

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Mass (% of initial)', fontsize=12)
    ax.set_title('Regional Mass Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='k', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{OUTPUT_PREFIX}_regional_mass.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved: {output_file}")
    plt.close()

    # === 3. SPATIAL DISTRIBUTION OF MASS CHANGE ===
    print("\n" + "="*70)
    print("3. SPATIAL DISTRIBUTION OF MASS CHANGE")
    print("="*70)

    # Get fill at initial and final times
    fill_init = get_array(vtk_datasets[0], fill_array_name)
    fill_final = get_array(vtk_datasets[-1], fill_array_name)

    # Per-cell change
    delta_fill = fill_final - fill_init

    # Statistics
    stats = {
        'total_loss': np.sum(delta_fill[delta_fill < 0]),
        'total_gain': np.sum(delta_fill[delta_fill > 0]),
        'net_change': np.sum(delta_fill),
        'max_loss_location': np.unravel_index(np.argmin(delta_fill), dims_list[0]),
        'max_loss_value': np.min(delta_fill),
        'num_cells_losing': np.sum(delta_fill < -0.01),
        'num_cells_gaining': np.sum(delta_fill > 0.01),
    }

    print(f"\nTotal mass loss: {stats['total_loss']:.6e}")
    print(f"Total mass gain: {stats['total_gain']:.6e}")
    print(f"Net change: {stats['net_change']:.6e}")
    print(f"Max loss at location {stats['max_loss_location']}: {stats['max_loss_value']:.6e}")
    print(f"Cells with significant loss (>1%): {stats['num_cells_losing']}")
    print(f"Cells with significant gain (>1%): {stats['num_cells_gaining']}")

    # Get 2D slices for visualization
    dims = dims_list[0]
    spacing = spacing_list[0]
    origin = origin_list[0]

    Y_init, Z_init, f_init_slice = get_2d_slice_middle(vtk_datasets[0], dims, spacing, origin, fill_array_name)
    Y_final, Z_final, f_final_slice = get_2d_slice_middle(vtk_datasets[-1], dims, spacing, origin, fill_array_name)

    # Compute delta for slice
    delta_fill_3d = delta_fill.reshape((dims[2], dims[1], dims[0]))
    mid_x = dims[0] // 2
    delta_slice = delta_fill_3d[:, :, mid_x]

    # Plot spatial distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Initial fill level
    ax = axes[0, 0]
    im = ax.contourf(Y_init, Z_init, f_init_slice, levels=20, cmap='RdBu_r', vmin=0, vmax=1)
    ax.set_xlabel('y (m)', fontsize=10)
    ax.set_ylabel('z (m)', fontsize=10)
    ax.set_title('(a) Initial Fill Level (t=0)', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Fill Level')

    # (b) Final fill level
    ax = axes[0, 1]
    im = ax.contourf(Y_final, Z_final, f_final_slice, levels=20, cmap='RdBu_r', vmin=0, vmax=1)
    ax.set_xlabel('y (m)', fontsize=10)
    ax.set_ylabel('z (m)', fontsize=10)
    ax.set_title('(b) Final Fill Level (t=1.0)', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Fill Level')

    # (c) Change in fill level
    ax = axes[1, 0]
    vmax = max(abs(delta_slice.min()), abs(delta_slice.max()))
    if vmax > 0:
        im = ax.contourf(Y_init, Z_init, delta_slice, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    else:
        im = ax.contourf(Y_init, Z_init, delta_slice, levels=20, cmap='RdBu_r')
    ax.set_xlabel('y (m)', fontsize=10)
    ax.set_ylabel('z (m)', fontsize=10)
    ax.set_title('(c) Fill Change (final - initial)', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Δf')

    # (d) Absolute change magnitude
    ax = axes[1, 1]
    abs_change = np.abs(delta_slice)
    im = ax.contourf(Y_init, Z_init, abs_change, levels=20, cmap='hot_r')
    ax.set_xlabel('y (m)', fontsize=10)
    ax.set_ylabel('z (m)', fontsize=10)
    ax.set_title('(d) Absolute Change Magnitude', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|Δf|')

    plt.suptitle('Spatial Distribution of Mass Change (center slice, x=const)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{OUTPUT_PREFIX}_spatial_distribution.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved: {output_file}")
    plt.close()

    # === 4. CORRELATION WITH VELOCITY ===
    if velocity_array_name:
        print("\n" + "="*70)
        print("4. CORRELATION WITH VELOCITY FIELD")
        print("="*70)

        velocity = get_array(vtk_datasets[-1], velocity_array_name)

        if velocity is not None:
            if velocity.ndim == 2:
                v_mag = np.linalg.norm(velocity, axis=1)
            else:
                v_mag = np.abs(velocity)

            # Bin by velocity magnitude
            v_bins = np.linspace(0, max(v_mag.max(), 1e-10), 20)

            binned_loss = []
            bin_centers = []

            for i in range(len(v_bins) - 1):
                mask = (v_mag >= v_bins[i]) & (v_mag < v_bins[i+1])
                if np.sum(mask) > 0:
                    avg_loss = np.mean(delta_fill[mask])
                    binned_loss.append(avg_loss)
                    bin_centers.append(0.5 * (v_bins[i] + v_bins[i+1]))

            # Plot correlation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Scatter plot (sample for performance)
            sample_size = min(10000, len(v_mag))
            sample_idx = np.random.choice(len(v_mag), sample_size, replace=False)

            ax1.scatter(v_mag[sample_idx], delta_fill[sample_idx],
                       alpha=0.1, s=1, c='blue')
            ax1.set_xlabel('Velocity Magnitude (m/s)', fontsize=11)
            ax1.set_ylabel('Fill Change Δf', fontsize=11)
            ax1.set_title('(a) Velocity vs Mass Change (scatter)', fontsize=12, fontweight='bold')
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3)

            # Binned average
            if len(bin_centers) > 0:
                ax2.plot(bin_centers, binned_loss, 'o-', linewidth=2, markersize=6, color='darkgreen')
                ax2.set_xlabel('Velocity Magnitude (m/s)', fontsize=11)
                ax2.set_ylabel('Average Fill Change Δf', fontsize=11)
                ax2.set_title('(b) Binned Average', fontsize=12, fontweight='bold')
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = OUTPUT_DIR / f"{OUTPUT_PREFIX}_velocity_correlation.png"
            plt.savefig(output_file, dpi=150)
            print(f"\nSaved: {output_file}")
            plt.close()

            # Compute correlation coefficient
            corr = np.corrcoef(v_mag, delta_fill)[0, 1]
            print(f"\nPearson correlation coefficient: {corr:.4f}")

            # High velocity region analysis
            high_v_mask = v_mag > VELOCITY_THRESHOLD_HIGH
            low_v_mask = v_mag < VELOCITY_THRESHOLD_LOW

            high_v_loss = np.mean(delta_fill[high_v_mask]) if np.sum(high_v_mask) > 0 else 0
            low_v_loss = np.mean(delta_fill[low_v_mask]) if np.sum(low_v_mask) > 0 else 0

            print(f"Average Δf in high velocity regions (|v| > {VELOCITY_THRESHOLD_HIGH}): {high_v_loss:.6e}")
            print(f"Average Δf in low velocity regions (|v| < {VELOCITY_THRESHOLD_LOW}): {low_v_loss:.6e}")

    # === 5. INTERFACE VS BULK ANALYSIS ===
    print("\n" + "="*70)
    print("5. INTERFACE VS BULK ANALYSIS")
    print("="*70)

    masks_init = categorize_cells(vtk_datasets[0], dims_list[0], fill_array_name, velocity_array_name)

    interface_loss = np.mean(delta_fill[masks_init['interface']]) if np.sum(masks_init['interface']) > 0 else 0
    bulk_fluid_loss = np.mean(delta_fill[masks_init['bulk_fluid']]) if np.sum(masks_init['bulk_fluid']) > 0 else 0
    bulk_empty_loss = np.mean(delta_fill[masks_init['bulk_empty']]) if np.sum(masks_init['bulk_empty']) > 0 else 0
    boundary_loss = np.mean(delta_fill[masks_init['boundary']]) if np.sum(masks_init['boundary']) > 0 else 0

    print(f"Average Δf at interface: {interface_loss:.6e}")
    print(f"Average Δf in bulk fluid: {bulk_fluid_loss:.6e}")
    print(f"Average Δf in bulk empty: {bulk_empty_loss:.6e}")
    print(f"Average Δf at boundaries: {boundary_loss:.6e}")

    # Compute total mass loss from each region
    spacing = spacing_list[0]
    cell_volume = spacing[0] * spacing[1] * spacing[2]

    interface_total_loss = np.sum(delta_fill[masks_init['interface']]) * cell_volume
    bulk_fluid_total_loss = np.sum(delta_fill[masks_init['bulk_fluid']]) * cell_volume
    bulk_empty_total_loss = np.sum(delta_fill[masks_init['bulk_empty']]) * cell_volume
    boundary_total_loss = np.sum(delta_fill[masks_init['boundary']]) * cell_volume

    print(f"\nTotal mass loss from interface: {interface_total_loss:.6e}")
    print(f"Total mass loss from bulk fluid: {bulk_fluid_total_loss:.6e}")
    print(f"Total mass loss from bulk empty: {bulk_empty_total_loss:.6e}")
    print(f"Total mass loss from boundaries: {boundary_total_loss:.6e}")

    # Bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    regions = ['Interface', 'Bulk Fluid', 'Bulk Empty', 'Boundary']
    avg_losses = [interface_loss, bulk_fluid_loss, bulk_empty_loss, boundary_loss]
    total_losses = [interface_total_loss, bulk_fluid_total_loss, bulk_empty_total_loss, boundary_total_loss]
    colors = ['orange', 'blue', 'lightblue', 'red']

    # Average change
    bars = ax1.bar(regions, avg_losses, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Average Fill Change Δf', fontsize=12)
    ax1.set_title('(a) Average Mass Change by Region', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, loss in zip(bars, avg_losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.2e}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9)

    # Total change
    bars = ax2.bar(regions, total_losses, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Total Mass Change', fontsize=12)
    ax2.set_title('(b) Total Mass Change by Region', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, loss in zip(bars, total_losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.2e}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{OUTPUT_PREFIX}_region_comparison.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved: {output_file}")
    plt.close()

    # === SUMMARY ===
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_mass_loss_pct = 100.0 * (total_masses[-1] - total_masses[0]) / total_masses[0]
    print(f"\n1. Total mass loss: {total_mass_loss_pct:+.4f}%")
    print(f"2. Cells with significant loss: {stats['num_cells_losing']} ({100.0*stats['num_cells_losing']/len(delta_fill):.2f}%)")
    print(f"3. Maximum local loss: {stats['max_loss_value']:.6e} at {stats['max_loss_location']}")

    # Identify dominant loss region
    loss_magnitudes = [abs(interface_total_loss), abs(bulk_fluid_total_loss),
                      abs(bulk_empty_total_loss), abs(boundary_total_loss)]
    dominant_region = regions[np.argmax(loss_magnitudes)]
    print(f"4. Dominant loss region: {dominant_region}")

    if velocity_array_name and velocity is not None:
        print(f"5. Velocity correlation: {corr:.4f}")
        if abs(corr) > 0.3:
            print("   → Moderate to strong correlation with velocity")
        elif abs(corr) > 0.1:
            print("   → Weak correlation with velocity")
        else:
            print("   → Negligible correlation with velocity")

    print("\n" + "="*70)
    print("Analysis complete. Check output directory for plots:")
    print(f"  {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
