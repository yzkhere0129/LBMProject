#!/usr/bin/env python3
"""
extract_keyhole_depth.py

Extract keyhole depth from VTK output files

This script reads VTK structured point files containing fill level data
and computes the keyhole depth by finding the lowest point where the
liquid-gas interface (fill level = 0.5) exists.

Usage:
    python extract_keyhole_depth.py <vtk_directory>

Output:
    - keyhole_depth_extracted.dat: Time vs depth data
    - Console output with statistics

Author: Generated for Keyhole Formation Validation
Date: 2025-12-21
"""

import sys
import os
import glob
import re
import numpy as np
import struct
from pathlib import Path


def read_vtk_structured_points(filename):
    """
    Read VTK structured points file (binary format)

    Returns:
        dict: Contains 'dimensions', 'spacing', 'temperature', 'fill_level', etc.
    """
    data = {}

    with open(filename, 'rb') as f:
        # Read header
        header = f.readline().decode('ascii').strip()
        if not header.startswith('# vtk DataFile Version'):
            raise ValueError(f"Invalid VTK file: {filename}")

        # Description
        description = f.readline().decode('ascii').strip()

        # Format (BINARY or ASCII)
        format_line = f.readline().decode('ascii').strip()
        is_binary = (format_line == 'BINARY')

        # Dataset type
        dataset_line = f.readline().decode('ascii').strip()
        if not dataset_line.startswith('DATASET STRUCTURED_POINTS'):
            raise ValueError(f"Expected STRUCTURED_POINTS dataset")

        # Dimensions
        dims_line = f.readline().decode('ascii').strip()
        dims_match = re.match(r'DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)', dims_line)
        if dims_match:
            nx, ny, nz = map(int, dims_match.groups())
            data['dimensions'] = (nx, ny, nz)

        # Spacing
        spacing_line = f.readline().decode('ascii').strip()
        spacing_match = re.match(r'SPACING\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)', spacing_line)
        if spacing_match:
            dx, dy, dz = map(float, spacing_match.groups())
            data['spacing'] = (dx, dy, dz)

        # Origin
        origin_line = f.readline().decode('ascii').strip()

        # Point data
        point_data_line = f.readline().decode('ascii').strip()

        # Read scalar fields
        num_points = nx * ny * nz

        while True:
            line = f.readline()
            if not line:
                break

            line_str = line.decode('ascii').strip()

            # Check for SCALARS keyword
            if line_str.startswith('SCALARS'):
                parts = line_str.split()
                field_name = parts[1]

                # Read LOOKUP_TABLE line
                f.readline()

                # Read binary data
                if is_binary:
                    field_data = np.fromfile(f, dtype='>f4', count=num_points)
                else:
                    # ASCII mode (not commonly used)
                    field_data = np.fromfile(f, dtype=np.float32, count=num_points, sep=' ')

                data[field_name] = field_data.reshape((nz, ny, nx))

    return data


def compute_keyhole_depth_from_vtk(vtk_data):
    """
    Compute keyhole depth from VTK data

    Args:
        vtk_data: Dictionary from read_vtk_structured_points()

    Returns:
        float: Keyhole depth in meters
    """
    fill_level = vtk_data['fill_level']
    nx, ny, nz = vtk_data['dimensions']
    dx, dy, dz = vtk_data['spacing']

    # Find center column (x=nx/2, y=ny/2)
    ix = nx // 2
    iy = ny // 2

    # Initial surface at z = nz/2
    initial_surface_z = nz // 2

    # Scan from top to bottom to find liquid surface (fill > 0.5)
    deepest_z = nz - 1

    for iz in range(nz - 1, -1, -1):
        if fill_level[iz, iy, ix] > 0.5:
            deepest_z = iz
            break

    # Keyhole depth in cells
    keyhole_cells = initial_surface_z - deepest_z

    # Convert to physical units (meters)
    depth_m = keyhole_cells * dz

    return depth_m


def extract_time_from_filename(filename):
    """
    Extract simulation time from VTK filename

    Expected format: keyhole_<time>us.vtk

    Returns:
        float: Time in seconds
    """
    basename = os.path.basename(filename)
    match = re.search(r'(\d+)us\.vtk', basename)
    if match:
        time_us = float(match.group(1))
        return time_us * 1e-6  # Convert to seconds
    else:
        raise ValueError(f"Cannot extract time from filename: {filename}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_keyhole_depth.py <vtk_directory>")
        sys.exit(1)

    vtk_dir = sys.argv[1]

    if not os.path.isdir(vtk_dir):
        print(f"Error: Directory not found: {vtk_dir}")
        sys.exit(1)

    # Find all VTK files
    vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "keyhole_*.vtk")))

    if len(vtk_files) == 0:
        print(f"Error: No VTK files found in {vtk_dir}")
        sys.exit(1)

    print(f"Found {len(vtk_files)} VTK files in {vtk_dir}")
    print()

    # Extract data
    times = []
    depths = []

    for vtk_file in vtk_files:
        try:
            # Read VTK file
            vtk_data = read_vtk_structured_points(vtk_file)

            # Extract time from filename
            time_s = extract_time_from_filename(vtk_file)

            # Compute keyhole depth
            depth_m = compute_keyhole_depth_from_vtk(vtk_data)

            times.append(time_s)
            depths.append(depth_m * 1e6)  # Convert to μm

            print(f"  {os.path.basename(vtk_file)}: t = {time_s*1e6:.1f} μs, depth = {depth_m*1e6:.2f} μm")

        except Exception as e:
            print(f"Warning: Failed to process {vtk_file}: {e}")

    # Write output file
    output_file = os.path.join(vtk_dir, "keyhole_depth_extracted.dat")

    with open(output_file, 'w') as f:
        f.write("# Keyhole Depth (Extracted from VTK)\n")
        f.write("# Time[μs] Depth[μm]\n")
        for t, d in zip(times, depths):
            f.write(f"{t*1e6:.6e} {d:.6e}\n")

    print()
    print(f"Written extracted data to: {output_file}")
    print()

    # Statistics
    if len(depths) > 0:
        print("Statistics:")
        print(f"  Initial depth: {depths[0]:.2f} μm")
        print(f"  Final depth: {depths[-1]:.2f} μm")
        print(f"  Maximum depth: {max(depths):.2f} μm")
        print(f"  Depth change: {depths[-1] - depths[0]:.2f} μm")
        print(f"  Time range: {times[0]*1e6:.1f} - {times[-1]*1e6:.1f} μs")


if __name__ == '__main__':
    main()
