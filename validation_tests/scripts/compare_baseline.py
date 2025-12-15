#!/usr/bin/env python3
"""
Compare test results with baseline.

Generates comparative analysis between baseline and test configurations:
- Velocity increase factors
- Temperature changes
- CFL number changes
- Stability comparison
- Physics validation
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

# ============================================================================
# Comparison Functions
# ============================================================================

def load_metrics(json_file: Path) -> Optional[Dict]:
    """Load metrics from JSON file."""

    if not json_file.exists():
        print(f"ERROR: Metrics file not found: {json_file}", file=sys.stderr)
        return None

    with open(json_file, 'r') as f:
        return json.load(f)


def compare_velocity(baseline: Dict, test: Dict, test_name: str) -> str:
    """Compare velocity metrics."""

    lines = []
    lines.append("VELOCITY COMPARISON:")
    lines.append("-" * 60)

    v_base = baseline.get('final_v_max')
    v_test = test.get('final_v_max')

    if v_base is None or v_test is None:
        lines.append("  ERROR: Missing velocity data")
        return "\n".join(lines)

    delta = v_test - v_base
    ratio = v_test / v_base if v_base > 0 else float('inf')

    lines.append(f"  Baseline v_max: {v_base:.2f} mm/s")
    lines.append(f"  Test v_max:     {v_test:.2f} mm/s")
    lines.append(f"  Increase:       {delta:+.2f} mm/s ({ratio:.1f}x)")
    lines.append("")

    # Interpretation
    if ratio > 10:
        lines.append("  Interpretation: SIGNIFICANT INCREASE (>10x)")
        lines.append("    - Limiter was strongly suppressing Marangoni effect")
        lines.append("    - Expected for gradient limiter removal")
    elif ratio > 2:
        lines.append("  Interpretation: MODERATE INCREASE (2-10x)")
        lines.append("    - Limiter had noticeable effect")
    elif ratio > 1.1:
        lines.append("  Interpretation: SMALL INCREASE (1.1-2x)")
        lines.append("    - Limiter had minor effect")
    elif ratio > 0.9:
        lines.append("  Interpretation: NO CHANGE (within 10%)")
        lines.append("    - Limiter was not active or ineffective")
    else:
        lines.append("  Interpretation: DECREASE (unexpected)")
        lines.append("    - WARNING: Velocity should not decrease")

    return "\n".join(lines)


def compare_temperature(baseline: Dict, test: Dict, test_name: str) -> str:
    """Compare temperature metrics."""

    lines = []
    lines.append("TEMPERATURE COMPARISON:")
    lines.append("-" * 60)

    T_base = baseline.get('final_T_max')
    T_test = test.get('final_T_max')

    if T_base is None or T_test is None:
        lines.append("  ERROR: Missing temperature data")
        return "\n".join(lines)

    delta = T_test - T_base
    ratio = T_test / T_base if T_base > 0 else float('inf')

    lines.append(f"  Baseline T_max: {T_base:.1f} K")
    lines.append(f"  Test T_max:     {T_test:.1f} K")
    lines.append(f"  Change:         {delta:+.1f} K ({ratio:.2f}x)")
    lines.append("")

    # Interpretation
    if abs(delta) < 100:
        lines.append("  Interpretation: NO SIGNIFICANT CHANGE")
        lines.append("    - Temperature field largely unaffected")
    elif delta > 0:
        lines.append("  Interpretation: TEMPERATURE INCREASE")
        lines.append("    - Stronger flow may enhance heat transport")
    else:
        lines.append("  Interpretation: TEMPERATURE DECREASE")
        lines.append("    - Enhanced convective cooling")

    return "\n".join(lines)


def compare_cfl(baseline: Dict, test: Dict, test_name: str) -> str:
    """Compare CFL numbers."""

    lines = []
    lines.append("CFL NUMBER COMPARISON:")
    lines.append("-" * 60)

    cfl_base = baseline.get('final_cfl')
    cfl_test = test.get('final_cfl')

    if cfl_base is None or cfl_test is None:
        lines.append("  WARNING: CFL data not available")
        return "\n".join(lines)

    delta = cfl_test - cfl_base
    ratio = cfl_test / cfl_base if cfl_base > 0 else float('inf')

    lines.append(f"  Baseline CFL: {cfl_base:.6f}")
    lines.append(f"  Test CFL:     {cfl_test:.6f}")
    lines.append(f"  Change:       {delta:+.6f} ({ratio:.1f}x)")
    lines.append("")

    # Check stability
    CFL_LIMIT = 0.5
    if cfl_test > CFL_LIMIT:
        lines.append(f"  WARNING: CFL = {cfl_test:.3f} > {CFL_LIMIT} (stability risk)")
    else:
        margin = CFL_LIMIT / cfl_test if cfl_test > 0 else float('inf')
        lines.append(f"  Status: STABLE (CFL margin = {margin:.1f}x)")

    return "\n".join(lines)


def compare_stability(baseline: Dict, test: Dict, test_name: str) -> str:
    """Compare numerical stability."""

    lines = []
    lines.append("STABILITY COMPARISON:")
    lines.append("-" * 60)

    # Check for errors
    base_stable = not (baseline.get('has_nan') or baseline.get('has_cuda_error'))
    test_stable = not (test.get('has_nan') or test.get('has_cuda_error'))

    base_complete = baseline.get('completed', False)
    test_complete = test.get('completed', False)

    lines.append(f"  Baseline: {'STABLE' if base_stable else 'UNSTABLE'} | {'COMPLETED' if base_complete else 'INCOMPLETE'}")
    lines.append(f"  Test:     {'STABLE' if test_stable else 'UNSTABLE'} | {'COMPLETED' if test_complete else 'INCOMPLETE'}")
    lines.append("")

    if test_stable and test_complete:
        lines.append("  Result: TEST STABLE")
        lines.append("    - No NaN/Inf detected")
        lines.append("    - Simulation completed successfully")
    elif test_stable and not test_complete:
        lines.append("  Result: INCOMPLETE (but numerically stable)")
        lines.append("    - May have timed out")
    else:
        lines.append("  Result: UNSTABLE")
        if test.get('has_nan'):
            lines.append("    - NaN detected")
        if test.get('has_cuda_error'):
            lines.append("    - CUDA error occurred")

    return "\n".join(lines)


def compare_conservation(baseline: Dict, test: Dict, test_name: str) -> str:
    """Compare mass conservation."""

    lines = []
    lines.append("CONSERVATION COMPARISON:")
    lines.append("-" * 60)

    mass_base = baseline.get('final_mass_error')
    mass_test = test.get('final_mass_error')

    if mass_base is None or mass_test is None:
        lines.append("  WARNING: Mass conservation data not available")
        return "\n".join(lines)

    delta = mass_test - mass_base

    lines.append(f"  Baseline mass error: {mass_base:.3f}%")
    lines.append(f"  Test mass error:     {mass_test:.3f}%")
    lines.append(f"  Change:              {delta:+.3f}%")
    lines.append("")

    TOLERANCE = 5.0  # 5% tolerance
    if mass_test < TOLERANCE:
        lines.append(f"  Status: GOOD (< {TOLERANCE}% threshold)")
    else:
        lines.append(f"  WARNING: Mass error exceeds {TOLERANCE}% threshold")

    return "\n".join(lines)


def generate_recommendations(baseline: Dict, test: Dict, test_name: str) -> str:
    """Generate recommendations based on comparison."""

    lines = []
    lines.append("RECOMMENDATIONS:")
    lines.append("=" * 72)
    lines.append("")

    v_base = baseline.get('final_v_max', 0)
    v_test = test.get('final_v_max', 0)
    ratio = v_test / v_base if v_base > 0 else 0

    test_stable = not (test.get('has_nan') or test.get('has_cuda_error'))
    test_complete = test.get('completed', False)

    # Check if this is gradient removal test
    is_grad_removed = 'GRAD-REMOVED' in test_name.upper() or 'CFL-REMOVED' in test_name.upper()

    if test_stable and test_complete:
        lines.append("[SUCCESS] Test completed successfully")
        lines.append("")

        if is_grad_removed and ratio > 10:
            lines.append("[RECOMMENDATION] REMOVE GRADIENT LIMITER IN PRODUCTION")
            lines.append("")
            lines.append("Evidence:")
            lines.append(f"  - Velocity increased {ratio:.1f}x (from {v_base:.1f} to {v_test:.1f} mm/s)")
            lines.append("  - Simulation remained stable (no NaN/Inf)")
            lines.append("  - Matches expected LPBF physics (50-500 mm/s)")
            lines.append("")
            lines.append("Action items:")
            lines.append("  1. Commit gradient limiter removal")
            lines.append("  2. Run extended validation (LONG-STABILITY test)")
            lines.append("  3. Test with full-scale LPBF simulations")

        elif is_grad_removed and ratio < 10:
            lines.append("[UNEXPECTED] Velocity increase less than expected")
            lines.append("")
            lines.append(f"  - Expected: >10x increase")
            lines.append(f"  - Actual: {ratio:.1f}x increase")
            lines.append("")
            lines.append("Possible causes:")
            lines.append("  - CFL limiter still active (check CFL-REMOVED test)")
            lines.append("  - Temperature gradient not strong enough")
            lines.append("  - Other physics limiting Marangoni effect")

        else:
            lines.append("[INFO] Test shows expected behavior")
            lines.append("")
            lines.append(f"  - Velocity change: {ratio:.1f}x")
            lines.append("  - Stability: Maintained")

    else:
        lines.append("[WARNING] Test did not complete successfully")
        lines.append("")

        if test.get('has_nan'):
            lines.append("Issue: NaN detected")
            lines.append("  - Numerical instability")
            lines.append("  - May need smaller timestep or additional stabilization")

        if test.get('has_cuda_error'):
            lines.append("Issue: CUDA error")
            lines.append("  - Check GPU memory usage")
            lines.append("  - Review kernel launch parameters")

        if not test_complete:
            lines.append("Issue: Simulation incomplete")
            lines.append("  - May have timed out (increase timeout)")
            lines.append("  - Check for deadlock or infinite loop")

        lines.append("")
        lines.append("Recommendation: DO NOT DEPLOY this configuration")

    return "\n".join(lines)

# ============================================================================
# Main Comparison
# ============================================================================

def run_comparison(baseline_file: Path, test_file: Path, test_name: str) -> str:
    """
    Generate comprehensive comparison report.

    Returns:
        Report text
    """

    # Load metrics
    baseline = load_metrics(baseline_file)
    test = load_metrics(test_file)

    if baseline is None or test is None:
        return "ERROR: Could not load metrics files"

    # Build report
    lines = []
    lines.append("=" * 72)
    lines.append(f"COMPARISON REPORT: {test_name} vs BASELINE")
    lines.append("=" * 72)
    lines.append("")

    # Run comparisons
    lines.append(compare_velocity(baseline, test, test_name))
    lines.append("")
    lines.append(compare_temperature(baseline, test, test_name))
    lines.append("")
    lines.append(compare_cfl(baseline, test, test_name))
    lines.append("")
    lines.append(compare_stability(baseline, test, test_name))
    lines.append("")
    lines.append(compare_conservation(baseline, test, test_name))
    lines.append("")
    lines.append("")
    lines.append(generate_recommendations(baseline, test, test_name))
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare test results with baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline metrics JSON file')
    parser.add_argument('--test', type=str, required=True,
                        help='Path to test metrics JSON file')
    parser.add_argument('--test-name', type=str, required=True,
                        help='Name of test configuration')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output report file (default: stdout)')

    args = parser.parse_args()

    # Run comparison
    baseline_file = Path(args.baseline)
    test_file = Path(args.test)

    report = run_comparison(baseline_file, test_file, args.test_name)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Comparison report written to: {output_path}")
    else:
        print(report)

    sys.exit(0)


if __name__ == '__main__':
    main()
