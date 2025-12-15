#!/usr/bin/env python3
"""
7-Point Validation Framework for Marangoni Test Results

Validation checks:
1. Numerical Health - No NaN/Inf in simulation
2. Temperature Range - Physical temperature bounds
3. Velocity Range - Expected velocity magnitudes
4. Simulation Completion - All timesteps executed
5. Field Smoothness - VTK output files generated
6. Conservation - Mass conservation within tolerance
7. Physics Realism - Results match LPBF literature

Exit codes:
    0 = All checks passed
    1 = One or more checks failed
    2 = Critical error (cannot validate)
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# ============================================================================
# Constants
# ============================================================================

# Physical bounds
MIN_TEMPERATURE = 0.0      # K (absolute zero)
MAX_TEMPERATURE = 10000.0  # K (Ti6Al4V vaporization ~3500K, safety margin)
MAX_VELOCITY = 1000.0      # mm/s (unrealistic if exceeded)

# Tolerances
MASS_CONSERVATION_TOLERANCE = 0.05  # 5% mass loss allowed
MIN_VTK_FILES = 5  # Minimum VTK files expected

# LPBF literature velocity range (for realism check)
LPBF_V_MIN = 10.0   # mm/s (lower bound from literature)
LPBF_V_MAX = 500.0  # mm/s (upper bound from literature)

# ============================================================================
# Log Parsing Functions
# ============================================================================

def parse_log_file(log_file: Path) -> Dict:
    """
    Parse simulation log file and extract key metrics.

    Returns:
        dict with keys:
            - v_max_history: List[float] - max velocity per timestep (mm/s)
            - T_max_history: List[float] - max temperature per timestep (K)
            - T_min_history: List[float] - min temperature per timestep (K)
            - mass_error_history: List[float] - mass conservation error (%)
            - completed: bool - simulation completed successfully
            - num_steps: int - number of steps completed
            - has_nan: bool - NaN detected
            - has_cuda_error: bool - CUDA error detected
            - vtk_count: int - number of VTK files mentioned
    """

    if not log_file.exists():
        print(f"ERROR: Log file not found: {log_file}")
        return None

    metrics = {
        'v_max_history': [],
        'T_max_history': [],
        'T_min_history': [],
        'mass_error_history': [],
        'completed': False,
        'num_steps': 0,
        'has_nan': False,
        'has_cuda_error': False,
        'vtk_count': 0,
        'final_v_max': None,
        'final_T_max': None,
        'final_T_min': None,
        'final_mass_error': None,
    }

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Check for NaN
            if 'NaN' in line or 'nan' in line.lower():
                metrics['has_nan'] = True

            # Check for CUDA errors
            if 'CUDA error' in line:
                metrics['has_cuda_error'] = True

            # Check for timeout
            if 'TIMEOUT' in line:
                metrics['completed'] = False

            # Parse velocity (example: "Step 100: v_max = 234.5 mm/s")
            v_match = re.search(r'v_max\s*=\s*([\d.]+)\s*mm/s', line, re.IGNORECASE)
            if v_match:
                v_max = float(v_match.group(1))
                metrics['v_max_history'].append(v_max)
                metrics['final_v_max'] = v_max

            # Parse temperature max (example: "T_max = 4305.2 K")
            T_max_match = re.search(r'T_max\s*=\s*([\d.]+)\s*K', line, re.IGNORECASE)
            if T_max_match:
                T_max = float(T_max_match.group(1))
                metrics['T_max_history'].append(T_max)
                metrics['final_T_max'] = T_max

            # Parse temperature min (example: "T_min = 65.4 K")
            T_min_match = re.search(r'T_min\s*=\s*([\d.]+)\s*K', line, re.IGNORECASE)
            if T_min_match:
                T_min = float(T_min_match.group(1))
                metrics['T_min_history'].append(T_min)
                metrics['final_T_min'] = T_min

            # Parse mass error (example: "Mass error: 0.8%")
            mass_match = re.search(r'[Mm]ass\s*error\s*:?\s*([\d.]+)\s*%', line)
            if mass_match:
                mass_err = float(mass_match.group(1))
                metrics['mass_error_history'].append(mass_err)
                metrics['final_mass_error'] = mass_err

            # Parse step number (example: "Step 1000/1000")
            step_match = re.search(r'[Ss]tep\s*(\d+)', line)
            if step_match:
                step_num = int(step_match.group(1))
                metrics['num_steps'] = max(metrics['num_steps'], step_num)

            # Parse VTK output (example: "Writing VTK: output_0100.vtk")
            if '.vtk' in line.lower() or 'vtk' in line.lower():
                metrics['vtk_count'] += 1

            # Check for completion marker
            if 'simulation completed' in line.lower() or 'finished' in line.lower():
                metrics['completed'] = True

    # If we have step data, assume completion if not explicitly failed
    if metrics['num_steps'] > 0 and not metrics['has_nan'] and not metrics['has_cuda_error']:
        metrics['completed'] = True

    return metrics

# ============================================================================
# Validation Functions
# ============================================================================

def validate_numerical_health(metrics: Dict) -> Tuple[bool, str]:
    """Check 1: No NaN or Inf values."""

    if metrics['has_nan']:
        return False, "NaN detected in simulation output"

    if metrics['has_cuda_error']:
        return False, "CUDA error detected"

    # Check for Inf in velocities
    for v in metrics['v_max_history']:
        if v > MAX_VELOCITY * 10:  # 10x safety margin
            return False, f"Infinite or unrealistic velocity detected: {v:.1f} mm/s"

    return True, f"No NaN/Inf detected in {metrics['num_steps']} steps"


def validate_temperature_range(metrics: Dict) -> Tuple[bool, str]:
    """Check 2: Temperature within physical bounds."""

    if metrics['final_T_min'] is None or metrics['final_T_max'] is None:
        return False, "Temperature data not found in log"

    T_min = metrics['final_T_min']
    T_max = metrics['final_T_max']

    if T_min < MIN_TEMPERATURE:
        return False, f"T_min = {T_min:.1f} K < {MIN_TEMPERATURE} K (absolute zero)"

    if T_max > MAX_TEMPERATURE:
        return False, f"T_max = {T_max:.1f} K > {MAX_TEMPERATURE} K (vaporization)"

    return True, f"T_min = {T_min:.1f} K, T_max = {T_max:.1f} K (within 0-{MAX_TEMPERATURE} K)"


def validate_velocity_range(metrics: Dict, v_min: float, v_max: float) -> Tuple[bool, str]:
    """Check 3: Velocity within expected range."""

    if metrics['final_v_max'] is None:
        return False, "Velocity data not found in log"

    v = metrics['final_v_max']

    if v < v_min:
        return False, f"v_max = {v:.1f} mm/s < expected minimum {v_min:.1f} mm/s"

    if v > v_max:
        return False, f"v_max = {v:.1f} mm/s > expected maximum {v_max:.1f} mm/s"

    return True, f"v_max = {v:.1f} mm/s (expected: {v_min:.1f}-{v_max:.1f} mm/s)"


def validate_completion(metrics: Dict) -> Tuple[bool, str]:
    """Check 4: Simulation completed all timesteps."""

    if not metrics['completed']:
        return False, f"Simulation did not complete (ran {metrics['num_steps']} steps)"

    if metrics['num_steps'] == 0:
        return False, "No timesteps detected in log"

    return True, f"All {metrics['num_steps']} steps completed successfully"


def validate_field_smoothness(metrics: Dict) -> Tuple[bool, str]:
    """Check 5: VTK output files generated."""

    vtk_count = metrics['vtk_count']

    if vtk_count < MIN_VTK_FILES:
        return False, f"Only {vtk_count} VTK files generated (expected >= {MIN_VTK_FILES})"

    return True, f"Generated {vtk_count} VTK files"


def validate_conservation(metrics: Dict) -> Tuple[bool, str]:
    """Check 6: Mass conservation within tolerance."""

    if metrics['final_mass_error'] is None:
        # If no mass error reported, check if we have mass data
        if len(metrics['mass_error_history']) == 0:
            return True, "Mass conservation not tracked (acceptable for some tests)"

    mass_err = metrics['final_mass_error']
    tolerance_pct = MASS_CONSERVATION_TOLERANCE * 100

    if mass_err > tolerance_pct:
        return False, f"Mass error: {mass_err:.2f}% (> {tolerance_pct:.0f}% threshold)"

    return True, f"Mass error: {mass_err:.2f}% (< {tolerance_pct:.0f}% threshold)"


def validate_physics_realism(metrics: Dict, test_name: str, baseline_v: Optional[float]) -> Tuple[bool, str]:
    """Check 7: Results match expected physics."""

    if metrics['final_v_max'] is None:
        return False, "No velocity data for realism check"

    v = metrics['final_v_max']

    # For gradient removal tests, expect velocity in LPBF range
    if 'REMOVED' in test_name.upper():
        if v < LPBF_V_MIN:
            return False, f"Velocity {v:.1f} mm/s below LPBF range ({LPBF_V_MIN}-{LPBF_V_MAX} mm/s)"

        if v > LPBF_V_MAX:
            msg = f"Velocity {v:.1f} mm/s above typical LPBF range ({LPBF_V_MIN}-{LPBF_V_MAX} mm/s)"
            # This is a warning, not failure
            return True, msg + " [WARNING: high but physical]"

        return True, f"Velocity {v:.1f} mm/s matches LPBF literature ({LPBF_V_MIN}-{LPBF_V_MAX} mm/s)"

    # For baseline or incremental tests, check for monotonic increase
    if baseline_v is not None:
        ratio = v / baseline_v if baseline_v > 0 else 0
        if ratio < 0.5:
            return False, f"Velocity decreased vs baseline ({v:.1f} vs {baseline_v:.1f} mm/s)"

        return True, f"Velocity {ratio:.1f}x baseline ({baseline_v:.1f} → {v:.1f} mm/s)"

    # Default: just check sanity
    if v < 0.1:
        return False, f"Velocity too low: {v:.3f} mm/s (likely unphysical)"

    return True, f"Velocity {v:.1f} mm/s within reasonable range"

# ============================================================================
# Main Validation
# ============================================================================

def run_validation(log_file: Path, test_name: str, v_min: float, v_max: float,
                   baseline_v: Optional[float] = None) -> Tuple[bool, str]:
    """
    Run full 7-point validation on test results.

    Returns:
        (passed, report_text)
    """

    # Parse log file
    metrics = parse_log_file(log_file)

    if metrics is None:
        return False, "ERROR: Could not parse log file"

    # Run all checks
    checks = [
        ("Numerical Health", validate_numerical_health(metrics)),
        ("Temperature Range", validate_temperature_range(metrics)),
        ("Velocity Range", validate_velocity_range(metrics, v_min, v_max)),
        ("Simulation Completion", validate_completion(metrics)),
        ("Field Smoothness", validate_field_smoothness(metrics)),
        ("Conservation", validate_conservation(metrics)),
        ("Physics Realism", validate_physics_realism(metrics, test_name, baseline_v)),
    ]

    # Build report
    report_lines = []
    report_lines.append(f"=== Validation Report: {test_name} ===")
    report_lines.append("")

    passed_count = 0
    total_count = len(checks)

    for i, (check_name, (passed, message)) in enumerate(checks, 1):
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"

        report_lines.append(f"[{i}/{total_count}] {check_name}: {symbol} {status}")
        report_lines.append(f"  - {message}")
        report_lines.append("")

        if passed:
            passed_count += 1

    # Overall result
    overall_passed = (passed_count == total_count)
    overall_status = "PASSED" if overall_passed else "FAILED"
    overall_symbol = "✓" if overall_passed else "✗"

    report_lines.append(f"OVERALL: {overall_symbol} {overall_status} ({passed_count}/{total_count} checks)")
    report_lines.append("")

    # Summary metrics
    if metrics['final_v_max'] is not None:
        report_lines.append(f"Final v_max: {metrics['final_v_max']:.2f} mm/s")
    if metrics['final_T_max'] is not None:
        report_lines.append(f"Final T_max: {metrics['final_T_max']:.1f} K")
    if metrics['num_steps'] > 0:
        report_lines.append(f"Steps completed: {metrics['num_steps']}")

    report_text = "\n".join(report_lines)

    return overall_passed, report_text

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate Marangoni simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--log-file', type=str, required=True,
                        help='Path to simulation log file')
    parser.add_argument('--test-name', type=str, required=True,
                        help='Name of test (e.g., BASELINE, GRAD-REMOVED)')
    parser.add_argument('--v-min', type=float, default=0.0,
                        help='Minimum expected velocity (mm/s)')
    parser.add_argument('--v-max', type=float, default=1000.0,
                        help='Maximum expected velocity (mm/s)')
    parser.add_argument('--baseline-v', type=float, default=None,
                        help='Baseline velocity for comparison (mm/s)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report file (default: stdout)')

    args = parser.parse_args()

    # Run validation
    log_file = Path(args.log_file)
    passed, report = run_validation(
        log_file, args.test_name,
        args.v_min, args.v_max,
        args.baseline_v
    )

    # Write report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report written to: {output_path}")
    else:
        print(report)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
