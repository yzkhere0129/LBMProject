#!/usr/bin/env python3
"""
Quick VTK File Diagnostic

Rapidly check a single VTK file for common issues.
Useful for spot-checking during development.

Usage:
    python quick_vtk_check.py /path/to/file.vtk
"""

import sys
import numpy as np
from pathlib import Path


def read_vtk_quick(filename):
    """Quick read of VTK file - returns fields dictionary."""
    fields = {}
    dimensions = None
    spacing = None

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    n_cells = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith('DIMENSIONS'):
            parts = line.split()
            dimensions = tuple(map(int, parts[1:4]))
            n_cells = np.prod(dimensions)

        elif line.startswith('SPACING'):
            parts = line.split()
            spacing = tuple(map(float, parts[1:4]))

        elif line.startswith('SCALARS'):
            parts = line.split()
            field_name = parts[1]
            i += 2  # Skip LOOKUP_TABLE line

            field_data = []
            while i < len(lines) and len(field_data) < n_cells:
                line_data = lines[i].split()
                if not line_data or line_data[0].startswith(('SCALARS', 'VECTORS')):
                    i -= 1
                    break
                field_data.extend([float(x) for x in line_data])
                i += 1
            fields[field_name] = np.array(field_data[:n_cells])

        i += 1

    return fields, dimensions, spacing


def check_field(name, data):
    """Check a single field for issues."""
    issues = []
    warnings = []

    # NaN/Inf check
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))

    if nan_count > 0:
        issues.append(f"{nan_count} NaN values")
    if inf_count > 0:
        issues.append(f"{inf_count} Inf values")

    # Get stats on valid data
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        issues.append("All values are NaN/Inf")
        return issues, warnings

    valid_data = data[valid_mask]
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)

    # Field-specific checks
    if 'temp' in name.lower() or name.lower() == 't':
        if min_val < 0:
            issues.append(f"Temperature < 0 K: min={min_val:.2f}")
        if max_val > 50000:
            warnings.append(f"Very high temperature: max={max_val:.2f} K")
        if max_val > 100000:
            issues.append(f"Temperature > 100000 K: max={max_val:.2f}")

    elif 'fill' in name.lower() or 'vof' in name.lower() or name.lower() in ['f', 'alpha']:
        if min_val < -1e-6:
            issues.append(f"Fill level < 0: min={min_val:.6f}")
        if max_val > 1.0 + 1e-6:
            issues.append(f"Fill level > 1: max={max_val:.6f}")

    # Print stats
    print(f"\n  {name}:")
    print(f"    Range: [{min_val:.6e}, {max_val:.6e}]")
    print(f"    Mean:  {mean_val:.6e}")
    print(f"    Std:   {std_val:.6e}")

    if nan_count > 0 or inf_count > 0:
        print(f"    Invalid: {nan_count} NaN, {inf_count} Inf")

    return issues, warnings


def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_vtk_check.py <vtk_file>")
        sys.exit(1)

    vtk_file = Path(sys.argv[1])

    if not vtk_file.exists():
        print(f"Error: File not found: {vtk_file}")
        sys.exit(1)

    print("=" * 60)
    print(f"QUICK VTK CHECK: {vtk_file.name}")
    print("=" * 60)

    try:
        fields, dimensions, spacing = read_vtk_quick(str(vtk_file))

        if dimensions:
            print(f"\nGrid: {dimensions[0]} x {dimensions[1]} x {dimensions[2]} = {np.prod(dimensions)} cells")
        if spacing:
            print(f"Spacing: {spacing[0]:.6e} x {spacing[1]:.6e} x {spacing[2]:.6e}")

        print(f"\nFields found: {list(fields.keys())}")

        all_issues = []
        all_warnings = []

        for name, data in fields.items():
            issues, warnings = check_field(name, data)
            all_issues.extend([f"{name}: {issue}" for issue in issues])
            all_warnings.extend([f"{name}: {warning}" for warning in warnings])

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if all_issues:
            print(f"\nISSUES FOUND ({len(all_issues)}):")
            for issue in all_issues:
                print(f"  [FAIL] {issue}")

        if all_warnings:
            print(f"\nWARNINGS ({len(all_warnings)}):")
            for warning in all_warnings:
                print(f"  [WARN] {warning}")

        if not all_issues and not all_warnings:
            print("\n[PASS] No issues detected")

        print("=" * 60)

        sys.exit(1 if all_issues else 0)

    except Exception as e:
        print(f"\nError reading VTK file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
