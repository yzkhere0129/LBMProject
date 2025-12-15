#!/usr/bin/env python3
"""
VTK File Comparison Tool for LBMProject and WalBerla
=====================================================

This script provides comprehensive comparison capabilities for VTK files
from LBMProject and WalBerla simulations, including:
- Loading and parsing VTK structured points data
- Extracting velocity, temperature, and scalar fields
- Computing difference metrics (L2 norm, max error, RMSE)
- Generating comparison plots and visualizations
- Supporting multiple comparison modes

Author: LBMProject Team
Date: 2025-12-04
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from typing import Dict, Tuple, Optional, List
import re

class VTKData:
    """Container for VTK structured points data."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.dimensions = None
        self.origin = None
        self.spacing = None
        self.point_data = {}
        self.metadata = {}
        self._load_vtk()

    def _load_vtk(self):
        """Load VTK file and parse structured points data."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # Parse header
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Parse dimensions
            if line.startswith('DIMENSIONS'):
                parts = line.split()
                self.dimensions = tuple(map(int, parts[1:4]))
                self.metadata['dimensions'] = self.dimensions

            # Parse origin
            elif line.startswith('ORIGIN'):
                parts = line.split()
                self.origin = tuple(map(float, parts[1:4]))
                self.metadata['origin'] = self.origin

            # Parse spacing
            elif line.startswith('SPACING'):
                parts = line.split()
                self.spacing = tuple(map(float, parts[1:4]))
                self.metadata['spacing'] = self.spacing

            # Parse point data
            elif line.startswith('POINT_DATA'):
                npoints = int(line.split()[1])
                self.metadata['npoints'] = npoints

            # Parse vectors
            elif line.startswith('VECTORS'):
                parts = line.split()
                var_name = parts[1]
                dtype = parts[2]
                i += 1
                data = self._read_vector_data(lines, i, self.metadata['npoints'])
                self.point_data[var_name] = data
                i += self.metadata['npoints'] - 1

            # Parse scalars
            elif line.startswith('SCALARS'):
                parts = line.split()
                var_name = parts[1]
                dtype = parts[2]
                # Skip LOOKUP_TABLE line
                i += 2
                data = self._read_scalar_data(lines, i, self.metadata['npoints'])
                self.point_data[var_name] = data
                i += self.metadata['npoints'] - 1

            i += 1

    def _read_vector_data(self, lines: List[str], start_idx: int, npoints: int) -> np.ndarray:
        """Read vector data from VTK file."""
        data = []
        for i in range(start_idx, start_idx + npoints):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(data)

    def _read_scalar_data(self, lines: List[str], start_idx: int, npoints: int) -> np.ndarray:
        """Read scalar data from VTK file."""
        data = []
        for i in range(start_idx, start_idx + npoints):
            if i >= len(lines):
                break
            val = lines[i].strip()
            if val:
                data.append(float(val))
        return np.array(data)

    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get field data by name (case-insensitive matching)."""
        # Try exact match first
        if field_name in self.point_data:
            return self.point_data[field_name]

        # Try case-insensitive match
        for key in self.point_data.keys():
            if key.lower() == field_name.lower():
                return self.point_data[key]

        return None

    def get_velocity_magnitude(self) -> Optional[np.ndarray]:
        """Compute velocity magnitude from velocity vectors."""
        vel = self.get_field('Velocity') or self.get_field('u')
        if vel is not None and len(vel.shape) == 2 and vel.shape[1] == 3:
            return np.linalg.norm(vel, axis=1)
        return None

    def reshape_to_grid(self, data: np.ndarray) -> np.ndarray:
        """Reshape 1D data to 3D grid."""
        if self.dimensions is None:
            raise ValueError("Dimensions not available")
        nx, ny, nz = self.dimensions
        if len(data.shape) == 1:
            return data.reshape((nz, ny, nx))
        elif len(data.shape) == 2:  # Vector data
            return data.reshape((nz, ny, nx, data.shape[1]))
        return data


class VTKComparator:
    """Compare VTK files and compute metrics."""

    def __init__(self, file1: str, file2: str):
        self.vtk1 = VTKData(file1)
        self.vtk2 = VTKData(file2)
        self.file1 = file1
        self.file2 = file2

    def compute_l2_error(self, field_name: str) -> Optional[float]:
        """Compute L2 error between two fields."""
        data1 = self.vtk1.get_field(field_name)
        data2 = self.vtk2.get_field(field_name)

        if data1 is None or data2 is None:
            return None

        if data1.shape != data2.shape:
            print(f"Warning: Shape mismatch for {field_name}: {data1.shape} vs {data2.shape}")
            return None

        diff = data1 - data2
        if len(diff.shape) == 2:  # Vector data
            diff = np.linalg.norm(diff, axis=1)

        l2_error = np.linalg.norm(diff) / np.sqrt(len(diff))
        return l2_error

    def compute_max_error(self, field_name: str) -> Optional[float]:
        """Compute maximum absolute error."""
        data1 = self.vtk1.get_field(field_name)
        data2 = self.vtk2.get_field(field_name)

        if data1 is None or data2 is None:
            return None

        if data1.shape != data2.shape:
            return None

        diff = data1 - data2
        if len(diff.shape) == 2:  # Vector data
            diff = np.linalg.norm(diff, axis=1)

        return np.max(np.abs(diff))

    def compute_rmse(self, field_name: str) -> Optional[float]:
        """Compute root mean square error."""
        data1 = self.vtk1.get_field(field_name)
        data2 = self.vtk2.get_field(field_name)

        if data1 is None or data2 is None:
            return None

        if data1.shape != data2.shape:
            return None

        diff = data1 - data2
        if len(diff.shape) == 2:  # Vector data
            diff = np.linalg.norm(diff, axis=1)

        return np.sqrt(np.mean(diff**2))

    def compute_relative_error(self, field_name: str) -> Optional[float]:
        """Compute relative L2 error."""
        data1 = self.vtk1.get_field(field_name)
        data2 = self.vtk2.get_field(field_name)

        if data1 is None or data2 is None:
            return None

        if data1.shape != data2.shape:
            return None

        diff = data1 - data2
        if len(diff.shape) == 2:  # Vector data
            diff_norm = np.linalg.norm(diff, axis=1)
            ref_norm = np.linalg.norm(data1, axis=1)
        else:
            diff_norm = np.abs(diff)
            ref_norm = np.abs(data1)

        # Avoid division by zero
        mask = ref_norm > 1e-10
        if not np.any(mask):
            return 0.0

        rel_error = np.mean(diff_norm[mask] / ref_norm[mask])
        return rel_error

    def print_comparison_report(self, field_names: List[str]):
        """Print comprehensive comparison report."""
        print("=" * 80)
        print("VTK FILE COMPARISON REPORT")
        print("=" * 80)
        print(f"\nFile 1: {self.file1}")
        print(f"File 2: {self.file2}")
        print()

        # Compare metadata
        print("METADATA COMPARISON:")
        print(f"  Dimensions: {self.vtk1.dimensions} vs {self.vtk2.dimensions}")
        print(f"  Origin:     {self.vtk1.origin} vs {self.vtk2.origin}")
        print(f"  Spacing:    {self.vtk1.spacing} vs {self.vtk2.spacing}")
        print()

        # Compare fields
        print("FIELD COMPARISON:")
        print(f"  File 1 fields: {list(self.vtk1.point_data.keys())}")
        print(f"  File 2 fields: {list(self.vtk2.point_data.keys())}")
        print()

        # Compute metrics
        print("ERROR METRICS:")
        print(f"{'Field':<20} {'L2 Error':<15} {'Max Error':<15} {'RMSE':<15} {'Rel Error':<15}")
        print("-" * 80)

        for field in field_names:
            l2 = self.compute_l2_error(field)
            max_err = self.compute_max_error(field)
            rmse = self.compute_rmse(field)
            rel = self.compute_relative_error(field)

            if l2 is not None:
                print(f"{field:<20} {l2:<15.6e} {max_err:<15.6e} {rmse:<15.6e} {rel:<15.6e}")
            else:
                print(f"{field:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

        print("=" * 80)


def plot_velocity_comparison(vtk1: VTKData, vtk2: VTKData, output_path: str):
    """Generate velocity comparison plots."""
    vel1 = vtk1.get_velocity_magnitude()
    vel2 = vtk2.get_velocity_magnitude()

    if vel1 is None or vel2 is None:
        print("Warning: Could not extract velocity data for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Velocity magnitude comparison (line plot)
    ax = axes[0, 0]
    ax.plot(vel1, 'b-', alpha=0.7, label='File 1', linewidth=0.5)
    ax.plot(vel2, 'r-', alpha=0.7, label='File 2', linewidth=0.5)
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Velocity Magnitude (m/s)')
    ax.set_title('Velocity Magnitude Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute difference
    ax = axes[0, 1]
    diff = np.abs(vel1 - vel2)
    ax.plot(diff, 'g-', linewidth=0.5)
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Absolute Difference (m/s)')
    ax.set_title('Absolute Velocity Difference')
    ax.grid(True, alpha=0.3)

    # Plot 3: Histogram comparison
    ax = axes[1, 0]
    ax.hist(vel1, bins=50, alpha=0.5, label='File 1', color='blue')
    ax.hist(vel2, bins=50, alpha=0.5, label='File 2', color='red')
    ax.set_xlabel('Velocity Magnitude (m/s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Velocity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Scatter plot
    ax = axes[1, 1]
    ax.scatter(vel1, vel2, alpha=0.3, s=1)

    # Add perfect agreement line
    max_val = max(np.max(vel1), np.max(vel2))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Agreement')

    ax.set_xlabel('File 1 Velocity (m/s)')
    ax.set_ylabel('File 2 Velocity (m/s)')
    ax.set_title('Velocity Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Velocity comparison plot saved to: {output_path}")
    plt.close()


def plot_temperature_comparison(vtk1: VTKData, vtk2: VTKData, output_path: str):
    """Generate temperature comparison plots."""
    temp1 = vtk1.get_field('Temperature') or vtk1.get_field('temperature')
    temp2 = vtk2.get_field('Temperature') or vtk2.get_field('temperature')

    if temp1 is None or temp2 is None:
        print("Warning: Could not extract temperature data for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Temperature comparison (line plot)
    ax = axes[0, 0]
    ax.plot(temp1, 'b-', alpha=0.7, label='File 1', linewidth=0.5)
    ax.plot(temp2, 'r-', alpha=0.7, label='File 2', linewidth=0.5)
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute difference
    ax = axes[0, 1]
    diff = np.abs(temp1 - temp2)
    ax.plot(diff, 'g-', linewidth=0.5)
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Absolute Difference (K)')
    ax.set_title('Absolute Temperature Difference')
    ax.grid(True, alpha=0.3)

    # Plot 3: Histogram comparison
    ax = axes[1, 0]
    ax.hist(temp1, bins=50, alpha=0.5, label='File 1', color='blue')
    ax.hist(temp2, bins=50, alpha=0.5, label='File 2', color='red')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Temperature Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Scatter plot
    ax = axes[1, 1]
    ax.scatter(temp1, temp2, alpha=0.3, s=1)

    # Add perfect agreement line
    min_val = min(np.min(temp1), np.min(temp2))
    max_val = max(np.max(temp1), np.max(temp2))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')

    ax.set_xlabel('File 1 Temperature (K)')
    ax.set_ylabel('File 2 Temperature (K)')
    ax.set_title('Temperature Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Temperature comparison plot saved to: {output_path}")
    plt.close()


def plot_slice_comparison(vtk1: VTKData, vtk2: VTKData, field_name: str,
                         slice_axis: int, slice_index: int, output_path: str):
    """Plot 2D slice comparison of a field."""
    data1 = vtk1.get_field(field_name)
    data2 = vtk2.get_field(field_name)

    if data1 is None or data2 is None:
        print(f"Warning: Could not extract {field_name} data for slice plotting")
        return

    # Convert to magnitude if vector data
    if len(data1.shape) == 2:
        data1 = np.linalg.norm(data1, axis=1)
        data2 = np.linalg.norm(data2, axis=1)

    # Reshape to grid
    try:
        grid1 = vtk1.reshape_to_grid(data1)
        grid2 = vtk2.reshape_to_grid(data2)
    except ValueError as e:
        print(f"Warning: Could not reshape data for slice plotting: {e}")
        return

    # Extract slice
    if slice_axis == 0:  # X-axis
        slice1 = grid1[:, :, slice_index]
        slice2 = grid2[:, :, slice_index]
    elif slice_axis == 1:  # Y-axis
        slice1 = grid1[:, slice_index, :]
        slice2 = grid2[:, slice_index, :]
    else:  # Z-axis
        slice1 = grid1[slice_index, :, :]
        slice2 = grid2[slice_index, :, :]

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin = min(np.min(slice1), np.min(slice2))
    vmax = max(np.max(slice1), np.max(slice2))

    # File 1
    im1 = axes[0].imshow(slice1, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(f'File 1: {field_name}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])

    # File 2
    im2 = axes[1].imshow(slice2, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(f'File 2: {field_name}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])

    # Difference
    diff = np.abs(slice1 - slice2)
    im3 = axes[2].imshow(diff, cmap='hot', origin='lower')
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])

    axis_names = ['X', 'Y', 'Z']
    fig.suptitle(f'{field_name} Comparison - {axis_names[slice_axis]}-slice at index {slice_index}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Slice comparison plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare VTK files from LBMProject and WalBerla simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python compare_vtk_files.py file1.vtk file2.vtk

  # With velocity and temperature plots
  python compare_vtk_files.py file1.vtk file2.vtk --plot-velocity --plot-temperature

  # Compare specific fields
  python compare_vtk_files.py file1.vtk file2.vtk --fields Velocity Temperature

  # Generate slice comparison
  python compare_vtk_files.py file1.vtk file2.vtk --slice-field Velocity --slice-axis 2 --slice-index 25
        """
    )

    parser.add_argument('file1', help='First VTK file path')
    parser.add_argument('file2', help='Second VTK file path')
    parser.add_argument('--fields', nargs='+', default=['Velocity', 'Temperature'],
                       help='Fields to compare (default: Velocity Temperature)')
    parser.add_argument('--plot-velocity', action='store_true',
                       help='Generate velocity comparison plots')
    parser.add_argument('--plot-temperature', action='store_true',
                       help='Generate temperature comparison plots')
    parser.add_argument('--slice-field', help='Field name for slice comparison')
    parser.add_argument('--slice-axis', type=int, default=2, choices=[0, 1, 2],
                       help='Axis for slice (0=X, 1=Y, 2=Z, default=2)')
    parser.add_argument('--slice-index', type=int,
                       help='Index along slice axis (default: middle)')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for plots (default: current directory)')

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}")
        sys.exit(1)
    if not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparator and print report
    comparator = VTKComparator(args.file1, args.file2)
    comparator.print_comparison_report(args.fields)

    # Generate plots
    if args.plot_velocity:
        output_path = output_dir / 'velocity_comparison.png'
        plot_velocity_comparison(comparator.vtk1, comparator.vtk2, str(output_path))

    if args.plot_temperature:
        output_path = output_dir / 'temperature_comparison.png'
        plot_temperature_comparison(comparator.vtk1, comparator.vtk2, str(output_path))

    if args.slice_field:
        slice_index = args.slice_index
        if slice_index is None:
            # Use middle slice by default
            dims = comparator.vtk1.dimensions
            slice_index = dims[args.slice_axis] // 2

        output_path = output_dir / f'{args.slice_field}_slice_comparison.png'
        plot_slice_comparison(comparator.vtk1, comparator.vtk2, args.slice_field,
                            args.slice_axis, slice_index, str(output_path))


if __name__ == '__main__':
    main()
