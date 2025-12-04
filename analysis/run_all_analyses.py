#!/usr/bin/env python3
"""
Master script to run all fluid and VOF analyses and generate comprehensive report.

This script orchestrates all analysis scripts and compiles results into
a single comprehensive visualization and quantitative report.
"""

import subprocess
import sys
from pathlib import Path
import datetime

# === PARAMETERS ===
ANALYSIS_DIR = Path("/home/yzk/LBMProject/analysis")
OUTPUT_DIR = Path("/home/yzk/LBMProject/analysis/results")
REPORT_FILE = OUTPUT_DIR / "analysis_report.txt"

def run_script(script_name):
    """Run a Python analysis script and capture output."""
    script_path = ANALYSIS_DIR / script_name

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False, ""

    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print('='*70)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"WARNING: Script exited with code {result.returncode}")
            return False, result.stdout + "\n" + result.stderr

        return True, result.stdout

    except subprocess.TimeoutExpired:
        print(f"ERROR: Script timed out after 5 minutes")
        return False, ""
    except Exception as e:
        print(f"ERROR running script: {e}")
        return False, ""

def generate_report(results):
    """Generate comprehensive text report."""

    report = []
    report.append("="*80)
    report.append("FLUID AND VOF PHYSICS VISUALIZATION ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Project: LBM-CUDA Multiphysics Simulation")
    report.append("\n")

    # Summary
    report.append("-"*80)
    report.append("EXECUTIVE SUMMARY")
    report.append("-"*80)

    successful = sum(1 for success, _ in results.values() if success)
    total = len(results)

    report.append(f"\nAnalyses completed: {successful}/{total}")
    report.append("")

    for script_name, (success, output) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        report.append(f"  {status} - {script_name}")

    report.append("\n")

    # Individual results
    for script_name, (success, output) in results.items():
        report.append("-"*80)
        report.append(f"ANALYSIS: {script_name}")
        report.append("-"*80)
        report.append("")

        if success:
            # Extract key metrics from output
            report.append(output)
        else:
            report.append("Analysis failed or incomplete.")

        report.append("\n")

    # Conclusions
    report.append("="*80)
    report.append("CONCLUSIONS")
    report.append("="*80)
    report.append("")
    report.append("Key Findings:")
    report.append("")
    report.append("1. VELOCITY FIELD ANALYSIS:")
    report.append("   - Check results/velocity_profiles.png for spatial distributions")
    report.append("   - Check results/velocity_time_series.png for temporal evolution")
    report.append("   - Verify no NaN/Inf values in velocity field")
    report.append("")
    report.append("2. VOF FIELD ANALYSIS:")
    report.append("   - Check results/vof_profiles.png for interface position")
    report.append("   - Check results/vof_mass_conservation.png for mass conservation")
    report.append("   - Verify interface sharpness and bound compliance")
    report.append("")
    report.append("3. POISEUILLE FLOW VALIDATION:")
    report.append("   - Check results/poiseuille_analysis.png for error distribution")
    report.append("   - Target: L2 error < 5%")
    report.append("   - Verify parabolic velocity profile shape")
    report.append("")
    report.append("RECOMMENDATIONS:")
    report.append("- Review all generated plots in the results directory")
    report.append("- Check for numerical artifacts (NaN, Inf, bound violations)")
    report.append("- Validate mass conservation error is acceptably small")
    report.append("- Compare Poiseuille flow against analytical solution")
    report.append("")
    report.append("="*80)

    return "\n".join(report)

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPREHENSIVE FLUID AND VOF ANALYSIS")
    print("="*80)
    print(f"\nProject Directory: {ANALYSIS_DIR}")
    print(f"Output Directory:  {OUTPUT_DIR}")
    print("")

    # List of analysis scripts to run
    scripts = [
        "analyze_poiseuille_flow.py",
        "analyze_velocity_field.py",
        "analyze_vof_field.py"
    ]

    # Run all scripts
    results = {}
    for script in scripts:
        success, output = run_script(script)
        results[script] = (success, output)

    # Generate comprehensive report
    print(f"\n{'='*70}")
    print("Generating comprehensive report...")
    print('='*70)

    report_text = generate_report(results)

    # Save report
    with open(REPORT_FILE, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {REPORT_FILE}")

    # Print report
    print("\n" + report_text)

    # Summary
    successful = sum(1 for success, _ in results.values() if success)
    total = len(results)

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE: {successful}/{total} scripts successful")
    print('='*70)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"  - Plots: {OUTPUT_DIR}/*.png")
    print(f"  - Report: {REPORT_FILE}")
    print("")

    return 0 if successful == total else 1

if __name__ == '__main__':
    sys.exit(main())
