#!/usr/bin/env python3
"""
Extract key metrics from simulation log files.

Metrics extracted:
- Velocity evolution (v_max per timestep)
- Temperature evolution (T_max, T_min)
- Mass conservation (error percentage)
- CFL numbers
- Phase distribution
- Completion status

Output: JSON file with all extracted metrics
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# Metric Extraction Functions
# ============================================================================

def extract_metrics(log_file: Path) -> Dict:
    """
    Extract all relevant metrics from simulation log file.

    Returns:
        Dictionary with comprehensive simulation metrics
    """

    if not log_file.exists():
        print(f"ERROR: Log file not found: {log_file}", file=sys.stderr)
        return None

    metrics = {
        # Time series data
        'v_max_history': [],
        'v_avg_history': [],
        'T_max_history': [],
        'T_min_history': [],
        'T_avg_history': [],
        'mass_error_history': [],
        'cfl_history': [],

        # Final values
        'final_v_max': None,
        'final_v_avg': None,
        'final_T_max': None,
        'final_T_min': None,
        'final_T_avg': None,
        'final_mass_error': None,
        'final_cfl': None,

        # Phase information
        'liquid_fraction': None,
        'gas_fraction': None,
        'solid_fraction': None,

        # Simulation status
        'completed': False,
        'num_steps': 0,
        'has_nan': False,
        'has_inf': False,
        'has_cuda_error': False,
        'timeout': False,

        # Output files
        'vtk_files': [],
        'vtk_count': 0,

        # Timing
        'wall_time_seconds': None,
        'steps_per_second': None,

        # Errors/warnings
        'errors': [],
        'warnings': [],
    }

    step_times = {}  # Track wall time per step

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # ================================================================
            # Error/Warning Detection
            # ================================================================

            if 'NaN' in line or 'nan' in line.lower():
                metrics['has_nan'] = True
                metrics['errors'].append(f"NaN detected: {line[:100]}")

            if 'inf' in line.lower() and 'infinite' in line.lower():
                metrics['has_inf'] = True
                metrics['errors'].append(f"Inf detected: {line[:100]}")

            if 'CUDA error' in line:
                metrics['has_cuda_error'] = True
                metrics['errors'].append(f"CUDA error: {line[:100]}")

            if 'TIMEOUT' in line:
                metrics['timeout'] = True
                metrics['errors'].append("Simulation timed out")

            if 'WARNING' in line.upper():
                metrics['warnings'].append(line[:100])

            # ================================================================
            # Velocity Metrics
            # ================================================================

            # Max velocity (example: "v_max = 234.5 mm/s")
            v_max_match = re.search(r'v_max\s*=\s*([\d.]+)\s*mm/s', line, re.IGNORECASE)
            if v_max_match:
                v_max = float(v_max_match.group(1))
                metrics['v_max_history'].append(v_max)
                metrics['final_v_max'] = v_max

            # Average velocity (example: "v_avg = 45.2 mm/s")
            v_avg_match = re.search(r'v_avg\s*=\s*([\d.]+)\s*mm/s', line, re.IGNORECASE)
            if v_avg_match:
                v_avg = float(v_avg_match.group(1))
                metrics['v_avg_history'].append(v_avg)
                metrics['final_v_avg'] = v_avg

            # ================================================================
            # Temperature Metrics
            # ================================================================

            # Max temperature (example: "T_max = 4305.2 K")
            T_max_match = re.search(r'T_max\s*=\s*([\d.]+)\s*K', line, re.IGNORECASE)
            if T_max_match:
                T_max = float(T_max_match.group(1))
                metrics['T_max_history'].append(T_max)
                metrics['final_T_max'] = T_max

            # Min temperature (example: "T_min = 65.4 K")
            T_min_match = re.search(r'T_min\s*=\s*([\d.]+)\s*K', line, re.IGNORECASE)
            if T_min_match:
                T_min = float(T_min_match.group(1))
                metrics['T_min_history'].append(T_min)
                metrics['final_T_min'] = T_min

            # Average temperature (example: "T_avg = 1234.5 K")
            T_avg_match = re.search(r'T_avg\s*=\s*([\d.]+)\s*K', line, re.IGNORECASE)
            if T_avg_match:
                T_avg = float(T_avg_match.group(1))
                metrics['T_avg_history'].append(T_avg)
                metrics['final_T_avg'] = T_avg

            # ================================================================
            # Conservation Metrics
            # ================================================================

            # Mass error (example: "Mass error: 0.8%", "Mass conservation: 99.2%")
            mass_error_match = re.search(r'[Mm]ass\s*error\s*:?\s*([\d.]+)\s*%', line)
            if mass_error_match:
                mass_err = float(mass_error_match.group(1))
                metrics['mass_error_history'].append(mass_err)
                metrics['final_mass_error'] = mass_err

            mass_cons_match = re.search(r'[Mm]ass\s*conservation\s*:?\s*([\d.]+)\s*%', line)
            if mass_cons_match:
                mass_cons = float(mass_cons_match.group(1))
                mass_err = 100.0 - mass_cons
                metrics['mass_error_history'].append(mass_err)
                metrics['final_mass_error'] = mass_err

            # ================================================================
            # CFL Number
            # ================================================================

            # CFL number (example: "CFL = 0.00035", "max CFL: 0.12")
            cfl_match = re.search(r'CFL\s*=?\s*:?\s*([\d.]+(?:e[+-]?\d+)?)', line, re.IGNORECASE)
            if cfl_match:
                cfl = float(cfl_match.group(1))
                metrics['cfl_history'].append(cfl)
                metrics['final_cfl'] = cfl

            # ================================================================
            # Phase Distribution
            # ================================================================

            # Liquid fraction (example: "Liquid: 45.2%")
            liquid_match = re.search(r'[Ll]iquid\s*:?\s*([\d.]+)\s*%', line)
            if liquid_match:
                metrics['liquid_fraction'] = float(liquid_match.group(1))

            # Gas fraction (example: "Gas: 54.3%")
            gas_match = re.search(r'[Gg]as\s*:?\s*([\d.]+)\s*%', line)
            if gas_match:
                metrics['gas_fraction'] = float(gas_match.group(1))

            # Solid fraction (example: "Solid: 0.5%")
            solid_match = re.search(r'[Ss]olid\s*:?\s*([\d.]+)\s*%', line)
            if solid_match:
                metrics['solid_fraction'] = float(solid_match.group(1))

            # ================================================================
            # Timestep Tracking
            # ================================================================

            # Step number (example: "Step 100", "Step 1000/6000")
            step_match = re.search(r'[Ss]tep\s*(\d+)', line)
            if step_match:
                step_num = int(step_match.group(1))
                metrics['num_steps'] = max(metrics['num_steps'], step_num)

            # ================================================================
            # VTK Output Files
            # ================================================================

            # VTK file output (example: "Writing VTK: output_0100.vtk")
            vtk_match = re.search(r'([a-zA-Z0-9_]+\.vtk)', line, re.IGNORECASE)
            if vtk_match:
                vtk_file = vtk_match.group(1)
                if vtk_file not in metrics['vtk_files']:
                    metrics['vtk_files'].append(vtk_file)
                    metrics['vtk_count'] = len(metrics['vtk_files'])

            # ================================================================
            # Timing Information
            # ================================================================

            # Wall time (example: "Wall time: 123.45 seconds")
            wall_time_match = re.search(r'[Ww]all\s*time\s*:?\s*([\d.]+)\s*(?:s|seconds?)', line)
            if wall_time_match:
                metrics['wall_time_seconds'] = float(wall_time_match.group(1))

            # Performance (example: "Performance: 12.5 steps/s")
            perf_match = re.search(r'([\d.]+)\s*steps?/s(?:ec)?', line, re.IGNORECASE)
            if perf_match:
                metrics['steps_per_second'] = float(perf_match.group(1))

            # ================================================================
            # Completion Status
            # ================================================================

            # Completion markers
            if any(marker in line.lower() for marker in ['simulation completed', 'finished', 'done']):
                metrics['completed'] = True

    # ================================================================
    # Post-processing
    # ================================================================

    # If we have step data and no errors, assume completion
    if metrics['num_steps'] > 0:
        if not metrics['has_nan'] and not metrics['has_cuda_error'] and not metrics['timeout']:
            metrics['completed'] = True

    # Compute derived metrics
    if metrics['wall_time_seconds'] and metrics['num_steps'] > 0:
        if not metrics['steps_per_second']:
            metrics['steps_per_second'] = metrics['num_steps'] / metrics['wall_time_seconds']

    # Compute statistics on time series
    if metrics['v_max_history']:
        metrics['v_max_mean'] = sum(metrics['v_max_history']) / len(metrics['v_max_history'])
        metrics['v_max_max'] = max(metrics['v_max_history'])
        metrics['v_max_min'] = min(metrics['v_max_history'])

    if metrics['T_max_history']:
        metrics['T_max_mean'] = sum(metrics['T_max_history']) / len(metrics['T_max_history'])
        metrics['T_max_max'] = max(metrics['T_max_history'])

    if metrics['mass_error_history']:
        metrics['mass_error_mean'] = sum(metrics['mass_error_history']) / len(metrics['mass_error_history'])
        metrics['mass_error_max'] = max(metrics['mass_error_history'])

    return metrics

# ============================================================================
# Pretty Printing
# ============================================================================

def print_metrics_summary(metrics: Dict):
    """Print human-readable summary of metrics."""

    print("=" * 72)
    print("SIMULATION METRICS SUMMARY")
    print("=" * 72)
    print()

    # Status
    print("STATUS:")
    status = "COMPLETED" if metrics['completed'] else "INCOMPLETE/FAILED"
    print(f"  Simulation: {status}")
    print(f"  Steps: {metrics['num_steps']}")
    if metrics['wall_time_seconds']:
        print(f"  Wall time: {metrics['wall_time_seconds']:.1f} s")
    if metrics['steps_per_second']:
        print(f"  Performance: {metrics['steps_per_second']:.2f} steps/s")
    print()

    # Velocity
    if metrics['final_v_max'] is not None:
        print("VELOCITY:")
        print(f"  Final v_max: {metrics['final_v_max']:.2f} mm/s")
        if 'v_max_mean' in metrics:
            print(f"  Mean v_max: {metrics['v_max_mean']:.2f} mm/s")
            print(f"  Range: {metrics['v_max_min']:.2f} - {metrics['v_max_max']:.2f} mm/s")
        print()

    # Temperature
    if metrics['final_T_max'] is not None:
        print("TEMPERATURE:")
        print(f"  Final T_max: {metrics['final_T_max']:.1f} K")
        if metrics['final_T_min'] is not None:
            print(f"  Final T_min: {metrics['final_T_min']:.1f} K")
        if 'T_max_mean' in metrics:
            print(f"  Mean T_max: {metrics['T_max_mean']:.1f} K")
        print()

    # Conservation
    if metrics['final_mass_error'] is not None:
        print("CONSERVATION:")
        print(f"  Final mass error: {metrics['final_mass_error']:.3f}%")
        if 'mass_error_max' in metrics:
            print(f"  Max mass error: {metrics['mass_error_max']:.3f}%")
        print()

    # CFL
    if metrics['final_cfl'] is not None:
        print("CFL NUMBER:")
        print(f"  Final CFL: {metrics['final_cfl']:.6f}")
        if metrics['cfl_history']:
            print(f"  Max CFL: {max(metrics['cfl_history']):.6f}")
        print()

    # Output
    if metrics['vtk_count'] > 0:
        print("OUTPUT:")
        print(f"  VTK files: {metrics['vtk_count']}")
        print()

    # Errors/Warnings
    if metrics['has_nan'] or metrics['has_cuda_error'] or metrics['timeout']:
        print("ERRORS:")
        if metrics['has_nan']:
            print("  - NaN detected")
        if metrics['has_inf']:
            print("  - Inf detected")
        if metrics['has_cuda_error']:
            print("  - CUDA error")
        if metrics['timeout']:
            print("  - Timeout")
        print()

    if metrics['warnings']:
        print(f"WARNINGS: {len(metrics['warnings'])}")
        for warn in metrics['warnings'][:5]:  # Show first 5
            print(f"  - {warn}")
        if len(metrics['warnings']) > 5:
            print(f"  ... and {len(metrics['warnings']) - 5} more")
        print()

    print("=" * 72)

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract metrics from simulation log file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('log_file', type=str,
                        help='Path to simulation log file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file (default: print to stdout)')
    parser.add_argument('--pretty', '-p', action='store_true',
                        help='Print human-readable summary instead of JSON')

    args = parser.parse_args()

    # Extract metrics
    log_file = Path(args.log_file)
    metrics = extract_metrics(log_file)

    if metrics is None:
        sys.exit(2)

    # Output
    if args.pretty:
        print_metrics_summary(metrics)
    else:
        output_json = json.dumps(metrics, indent=2)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(output_json)
            print(f"Metrics written to: {output_path}")
        else:
            print(output_json)

    sys.exit(0)


if __name__ == '__main__':
    main()
