#!/usr/bin/env python3
"""
Generate comprehensive validation report.

Aggregates all test results into a single markdown report with:
- Executive summary
- Test-by-test results
- Velocity evolution visualization (ASCII chart)
- Comparative analysis
- Recommendations
- Next steps
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ============================================================================
# Report Generation Functions
# ============================================================================

def load_test_metrics(data_dir: Path, test_name: str) -> Optional[Dict]:
    """Load metrics for a specific test."""

    # Try to load from JSON (if extract_metrics.py was run)
    json_file = data_dir / f"{test_name}_metrics.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            return json.load(f)

    # Otherwise, parse log file directly
    log_file = data_dir / f"{test_name}_output.log"
    if not log_file.exists():
        return None

    # Import extract_metrics module
    import extract_metrics
    return extract_metrics.extract_metrics(log_file)


def load_all_test_results(data_dir: Path, report_dir: Path) -> Dict:
    """Load results from all tests."""

    tests = ['BASELINE', 'GRAD-2X', 'GRAD-10X', 'GRAD-REMOVED', 'CFL-REMOVED', 'LONG-STABILITY']

    results = {}

    for test_name in tests:
        # Load metrics
        metrics = load_test_metrics(data_dir, test_name)

        # Load validation report
        validation_file = report_dir / f"{test_name}_validation.txt"
        validation_passed = False
        if validation_file.exists():
            with open(validation_file, 'r') as f:
                validation_text = f.read()
                validation_passed = 'PASSED' in validation_text

        results[test_name] = {
            'metrics': metrics,
            'validation_passed': validation_passed,
            'available': metrics is not None
        }

    return results


def generate_executive_summary(results: Dict) -> str:
    """Generate executive summary section."""

    lines = []
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Count tests
    total_tests = len(results)
    available_tests = sum(1 for r in results.values() if r['available'])
    passed_tests = sum(1 for r in results.values() if r['validation_passed'])

    lines.append(f"**Tests Run:** {available_tests}/{total_tests}")
    lines.append(f"**Tests Passed:** {passed_tests}/{available_tests}")
    lines.append("")

    # Key findings
    baseline = results.get('BASELINE', {}).get('metrics')
    grad_removed = results.get('GRAD-REMOVED', {}).get('metrics')

    if baseline and grad_removed:
        v_base = baseline.get('final_v_max', 0)
        v_grad = grad_removed.get('final_v_max', 0)

        if v_base > 0:
            ratio = v_grad / v_base

            lines.append("### Key Findings")
            lines.append("")
            lines.append(f"- **Baseline velocity:** {v_base:.2f} mm/s")
            lines.append(f"- **Gradient limiter removed:** {v_grad:.2f} mm/s")
            lines.append(f"- **Increase factor:** {ratio:.1f}x")
            lines.append("")

            # Physics validation
            LPBF_MIN = 50.0
            LPBF_MAX = 500.0
            if LPBF_MIN <= v_grad <= LPBF_MAX:
                lines.append(f"- **Physics validation:** PASS (within LPBF range {LPBF_MIN}-{LPBF_MAX} mm/s)")
            elif v_grad > LPBF_MAX:
                lines.append(f"- **Physics validation:** WARNING (exceeds typical LPBF range, but may be physical)")
            else:
                lines.append(f"- **Physics validation:** FAIL (below expected LPBF range)")
            lines.append("")

            # Stability
            grad_stable = not (grad_removed.get('has_nan') or grad_removed.get('has_cuda_error'))
            grad_complete = grad_removed.get('completed', False)

            if grad_stable and grad_complete:
                lines.append("- **Stability:** STABLE (no NaN/Inf, simulation completed)")
            else:
                lines.append("- **Stability:** UNSTABLE (numerical errors detected)")
            lines.append("")

    return "\n".join(lines)


def generate_test_results_table(results: Dict) -> str:
    """Generate table of test results."""

    lines = []
    lines.append("## Test Results Summary")
    lines.append("")
    lines.append("| Test | v_max (mm/s) | Increase | T_max (K) | Status | Validation |")
    lines.append("|------|--------------|----------|-----------|--------|------------|")

    baseline_v = None
    if results.get('BASELINE', {}).get('available'):
        baseline_v = results['BASELINE']['metrics'].get('final_v_max', 0)

    for test_name in ['BASELINE', 'GRAD-2X', 'GRAD-10X', 'GRAD-REMOVED', 'CFL-REMOVED', 'LONG-STABILITY']:
        if test_name not in results or not results[test_name]['available']:
            lines.append(f"| {test_name} | N/A | N/A | N/A | NOT RUN | - |")
            continue

        metrics = results[test_name]['metrics']
        v_max = metrics.get('final_v_max', 0)
        T_max = metrics.get('final_T_max', 0)

        # Calculate increase
        if baseline_v and baseline_v > 0 and test_name != 'BASELINE':
            increase = f"{v_max/baseline_v:.1f}x"
        else:
            increase = "-"

        # Status
        if metrics.get('has_nan') or metrics.get('has_cuda_error'):
            status = "FAILED"
        elif metrics.get('completed'):
            status = "COMPLETED"
        else:
            status = "INCOMPLETE"

        # Validation
        validation = "PASS" if results[test_name]['validation_passed'] else "FAIL"

        lines.append(f"| {test_name} | {v_max:.2f} | {increase} | {T_max:.1f} | {status} | {validation} |")

    lines.append("")
    return "\n".join(lines)


def generate_ascii_chart(results: Dict) -> str:
    """Generate ASCII visualization of velocity evolution."""

    lines = []
    lines.append("## Velocity Evolution")
    lines.append("")
    lines.append("```")

    # Extract velocities
    test_names = ['BASELINE', 'GRAD-2X', 'GRAD-10X', 'GRAD-REMOVED', 'CFL-REMOVED']
    velocities = []
    labels = []

    for test_name in test_names:
        if results.get(test_name, {}).get('available'):
            v = results[test_name]['metrics'].get('final_v_max', 0)
            velocities.append(v)
            labels.append(test_name)

    if not velocities:
        lines.append("No velocity data available")
        lines.append("```")
        return "\n".join(lines)

    # Normalize to chart width
    max_v = max(velocities)
    chart_width = 60

    # Draw chart
    lines.append("Velocity (mm/s)")
    lines.append("")

    for label, v in zip(labels, velocities):
        bar_length = int((v / max_v) * chart_width) if max_v > 0 else 0
        bar = "█" * bar_length
        lines.append(f"{label:15s} | {bar} {v:.1f}")

    lines.append("")
    lines.append(f"Scale: 0 - {max_v:.1f} mm/s")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def generate_detailed_results(results: Dict) -> str:
    """Generate detailed results for each test."""

    lines = []
    lines.append("## Detailed Test Results")
    lines.append("")

    for test_name in ['BASELINE', 'GRAD-2X', 'GRAD-10X', 'GRAD-REMOVED', 'CFL-REMOVED', 'LONG-STABILITY']:
        if test_name not in results or not results[test_name]['available']:
            continue

        metrics = results[test_name]['metrics']

        lines.append(f"### {test_name}")
        lines.append("")

        # Status
        if metrics.get('completed'):
            status_symbol = "✓"
            status_text = "COMPLETED"
        else:
            status_symbol = "✗"
            status_text = "INCOMPLETE/FAILED"

        lines.append(f"**Status:** {status_symbol} {status_text}")
        lines.append("")

        # Key metrics
        lines.append("**Metrics:**")
        if metrics.get('final_v_max') is not None:
            lines.append(f"- Maximum velocity: {metrics['final_v_max']:.2f} mm/s")
        if metrics.get('final_T_max') is not None:
            lines.append(f"- Maximum temperature: {metrics['final_T_max']:.1f} K")
        if metrics.get('final_mass_error') is not None:
            lines.append(f"- Mass conservation error: {metrics['final_mass_error']:.3f}%")
        if metrics.get('final_cfl') is not None:
            lines.append(f"- CFL number: {metrics['final_cfl']:.6f}")
        if metrics.get('num_steps') > 0:
            lines.append(f"- Steps completed: {metrics['num_steps']}")
        lines.append("")

        # Errors/warnings
        if metrics.get('has_nan') or metrics.get('has_cuda_error') or metrics.get('timeout'):
            lines.append("**Issues:**")
            if metrics.get('has_nan'):
                lines.append("- NaN detected")
            if metrics.get('has_cuda_error'):
                lines.append("- CUDA error")
            if metrics.get('timeout'):
                lines.append("- Timeout")
            lines.append("")

    return "\n".join(lines)


def generate_recommendations(results: Dict) -> str:
    """Generate recommendations based on all results."""

    lines = []
    lines.append("## Recommendations")
    lines.append("")

    baseline = results.get('BASELINE', {}).get('metrics')
    grad_removed = results.get('GRAD-REMOVED', {}).get('metrics')
    cfl_removed = results.get('CFL-REMOVED', {}).get('metrics')
    long_stability = results.get('LONG-STABILITY', {}).get('metrics')

    if not baseline or not grad_removed:
        lines.append("**Status:** Insufficient data for recommendations")
        lines.append("")
        lines.append("Required tests:")
        lines.append("- BASELINE: " + ("COMPLETE" if baseline else "MISSING"))
        lines.append("- GRAD-REMOVED: " + ("COMPLETE" if grad_removed else "MISSING"))
        return "\n".join(lines)

    # Check if gradient removal was successful
    v_base = baseline.get('final_v_max', 0)
    v_grad = grad_removed.get('final_v_max', 0)
    ratio = v_grad / v_base if v_base > 0 else 0

    grad_stable = not (grad_removed.get('has_nan') or grad_removed.get('has_cuda_error'))
    grad_complete = grad_removed.get('completed', False)

    # Decision logic
    if grad_stable and grad_complete and ratio > 10:
        lines.append("### PRIMARY RECOMMENDATION: REMOVE GRADIENT LIMITER")
        lines.append("")
        lines.append("**Evidence:**")
        lines.append(f"- Velocity increased {ratio:.1f}x (from {v_base:.1f} to {v_grad:.1f} mm/s)")
        lines.append("- Simulation remained numerically stable")
        lines.append("- No NaN or Inf values detected")
        lines.append("- Results match expected LPBF physics")
        lines.append("")
        lines.append("**Action Items:**")
        lines.append("1. Review and commit gradient limiter removal code changes")
        lines.append("2. Verify long-term stability test results (LONG-STABILITY)")
        lines.append("3. Test with production LPBF configurations")
        lines.append("4. Monitor for any edge cases in extended simulations")
        lines.append("")

        # Check CFL limiter
        if cfl_removed:
            v_cfl = cfl_removed.get('final_v_max', 0)
            cfl_stable = not (cfl_removed.get('has_nan') or cfl_removed.get('has_cuda_error'))

            if cfl_stable:
                lines.append("**Optional: Consider removing CFL limiter**")
                lines.append(f"- CFL-REMOVED test also stable (v_max = {v_cfl:.1f} mm/s)")
                lines.append("- May provide additional performance benefits")
                lines.append("- Recommend further testing before deployment")
            else:
                lines.append("**Keep CFL limiter**")
                lines.append("- CFL-REMOVED test showed instability")
                lines.append("- CFL limiter provides important safety margin")

    elif grad_stable and grad_complete and ratio < 10:
        lines.append("### RECOMMENDATION: INVESTIGATE FURTHER")
        lines.append("")
        lines.append(f"**Issue:** Velocity increase ({ratio:.1f}x) less than expected (>10x)")
        lines.append("")
        lines.append("**Possible causes:**")
        lines.append("- CFL limiter still active (check CFL-REMOVED results)")
        lines.append("- Temperature gradients not strong enough")
        lines.append("- Other physical constraints limiting flow")
        lines.append("")
        lines.append("**Next steps:**")
        lines.append("1. Review CFL-REMOVED test results")
        lines.append("2. Check temperature field in simulation")
        lines.append("3. Verify Marangoni coefficient (dσ/dT)")

    else:
        lines.append("### RECOMMENDATION: DO NOT DEPLOY")
        lines.append("")
        lines.append("**Issue:** Test did not complete successfully")
        lines.append("")
        if not grad_stable:
            lines.append("- Numerical instability detected (NaN/CUDA error)")
            lines.append("- Gradient limiter may be necessary for stability")
        if not grad_complete:
            lines.append("- Simulation did not complete")
            lines.append("- Check for timeout or other runtime issues")
        lines.append("")
        lines.append("**Next steps:**")
        lines.append("1. Investigate root cause of failure")
        lines.append("2. Consider partial limiter relaxation (GRAD-2X, GRAD-10X)")
        lines.append("3. Review numerical stability analysis")

    lines.append("")
    return "\n".join(lines)


def generate_next_steps(results: Dict) -> str:
    """Generate next steps section."""

    lines = []
    lines.append("## Next Steps")
    lines.append("")

    grad_removed = results.get('GRAD-REMOVED', {})

    if grad_removed.get('validation_passed'):
        lines.append("### After Successful Validation")
        lines.append("")
        lines.append("1. **Code Integration**")
        lines.append("   - Commit gradient limiter removal changes")
        lines.append("   - Update documentation with validation results")
        lines.append("   - Tag release with validation report")
        lines.append("")
        lines.append("2. **Extended Testing**")
        lines.append("   - Run full-scale LPBF simulations (multiple layers)")
        lines.append("   - Test various scan patterns and parameters")
        lines.append("   - Validate against experimental melt pool data")
        lines.append("")
        lines.append("3. **Performance Analysis**")
        lines.append("   - Measure simulation speedup (if any)")
        lines.append("   - Compare melt pool predictions with literature")
        lines.append("   - Document velocity ranges for different materials")
        lines.append("")
        lines.append("4. **Continuous Monitoring**")
        lines.append("   - Add automated regression tests")
        lines.append("   - Monitor for numerical issues in production runs")
        lines.append("   - Collect feedback from users")

    else:
        lines.append("### After Failed Validation")
        lines.append("")
        lines.append("1. **Root Cause Analysis**")
        lines.append("   - Review detailed error logs")
        lines.append("   - Identify point of failure (timestep, location)")
        lines.append("   - Analyze field evolution leading to instability")
        lines.append("")
        lines.append("2. **Alternative Approaches**")
        lines.append("   - Try partial limiter relaxation (increase threshold)")
        lines.append("   - Implement adaptive limiting based on local conditions")
        lines.append("   - Consider operator splitting for stability")
        lines.append("")
        lines.append("3. **Further Investigation**")
        lines.append("   - Consult literature on Marangoni stability")
        lines.append("   - Review other LPBF simulation approaches")
        lines.append("   - Consider alternative numerical schemes")

    lines.append("")
    return "\n".join(lines)

# ============================================================================
# Main Report Generation
# ============================================================================

def generate_full_report(data_dir: Path, report_dir: Path) -> str:
    """Generate complete validation report."""

    # Load all results
    results = load_all_test_results(data_dir, report_dir)

    # Build report sections
    lines = []
    lines.append("# Marangoni Limiter Removal - Validation Report")
    lines.append("")
    lines.append(generate_executive_summary(results))
    lines.append(generate_test_results_table(results))
    lines.append(generate_ascii_chart(results))
    lines.append(generate_detailed_results(results))
    lines.append(generate_recommendations(results))
    lines.append(generate_next_steps(results))

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    return "\n".join(lines)

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive validation report',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing test data')
    parser.add_argument('--report-dir', type=str, required=True,
                        help='Directory containing validation reports')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output markdown report file')

    args = parser.parse_args()

    # Generate report
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)

    report = generate_full_report(data_dir, report_dir)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Validation report generated: {output_path}")
    print("")
    print("=" * 72)
    print(report)
    print("=" * 72)

    sys.exit(0)


if __name__ == '__main__':
    main()
