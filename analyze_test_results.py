#!/usr/bin/env python3
"""
Comprehensive Test Results Analysis
Analyzes VTK outputs and log files to verify fixes
"""

import glob
import os
import sys
import numpy as np
from pathlib import Path

class TestAnalyzer:
    def __init__(self):
        self.results = {}

    def analyze_log_file(self, log_path):
        """Extract key metrics from log files"""
        if not os.path.exists(log_path):
            return None

        metrics = {
            'max_temperature': 0.0,
            'max_velocity': 0.0,
            'has_nan': False,
            'completed': False,
            'warnings': []
        }

        with open(log_path, 'r') as f:
            for line in f:
                # Check for completion
                if 'complete' in line.lower() or 'PASS' in line:
                    metrics['completed'] = True

                # Check for NaN
                if 'NaN' in line or 'nan' in line:
                    metrics['has_nan'] = True

                # Extract temperature
                if 'T_max' in line or 'Temperature' in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'T_max' in part or part == 'K':
                                if i > 0:
                                    temp = float(parts[i-1].replace('K', '').replace(',', ''))
                                    metrics['max_temperature'] = max(metrics['max_temperature'], temp)
                    except:
                        pass

                # Extract velocity
                if 'v_max' in line or 'velocity' in line.lower():
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'v_max' in part or 'm/s' in part:
                                if i > 0:
                                    vel = float(parts[i-1].replace('m/s', '').replace(',', ''))
                                    metrics['max_velocity'] = max(metrics['max_velocity'], vel)
                    except:
                        pass

                # Collect warnings
                if 'WARNING' in line or 'FAIL' in line:
                    metrics['warnings'].append(line.strip())

        return metrics

    def check_vtk_outputs(self, directory):
        """Check if VTK files were generated"""
        vtk_files = glob.glob(os.path.join(directory, '*.vtk'))
        return {
            'count': len(vtk_files),
            'exists': len(vtk_files) > 0,
            'files': sorted(vtk_files)
        }

    def analyze_test_suite(self, build_dir):
        """Analyze all test results"""
        print("=" * 70)
        print("  COMPREHENSIVE TEST RESULTS ANALYSIS")
        print("=" * 70)
        print()

        # Test 1: Newton Bisection
        print("TEST 1: Newton-Raphson Bisection Fallback")
        print("-" * 70)
        vtk_data = self.check_vtk_outputs(os.path.join(build_dir, 'test_newton_stress'))
        print(f"  VTK outputs: {vtk_data['count']} files")

        if os.path.exists(os.path.join(build_dir, 'test_1_1.log')):
            metrics = self.analyze_log_file(os.path.join(build_dir, 'test_1_1.log'))
            if metrics:
                print(f"  Max temperature: {metrics['max_temperature']:.1f} K")
                print(f"  Completed: {metrics['completed']}")
                print(f"  NaN detected: {metrics['has_nan']}")
        print()

        # Test 2: Laser Shutoff
        print("TEST 2: Laser Shutoff Configuration")
        print("-" * 70)
        print("  Checking laser behavior at different shutoff times...")

        # Check for test logs
        for test_log in ['test_2_2.log']:
            if os.path.exists(os.path.join(build_dir, test_log)):
                metrics = self.analyze_log_file(os.path.join(build_dir, test_log))
                if metrics:
                    print(f"  {test_log}: T_max={metrics['max_temperature']:.1f}K, "
                          f"Completed={metrics['completed']}")
        print()

        # Test 3: Marangoni Gradient Limiter
        print("TEST 3: Marangoni Gradient Limiter (CRITICAL)")
        print("-" * 70)
        vtk_data = self.check_vtk_outputs(os.path.join(build_dir, 'test_marangoni_limiter'))
        print(f"  VTK outputs: {vtk_data['count']} files")
        print(f"  Expected gradient limiter: 5e8 K/m")

        if os.path.exists(os.path.join(build_dir, 'test_3_1_marangoni.log')):
            metrics = self.analyze_log_file(os.path.join(build_dir, 'test_3_1_marangoni.log'))
            if metrics:
                print(f"  Max temperature: {metrics['max_temperature']:.1f} K")
                print(f"  Max velocity: {metrics['max_velocity']:.4f} m/s")
                print(f"  Completed: {metrics['completed']}")
                print(f"  NaN detected: {metrics['has_nan']}")

                # Critical checks
                if metrics['max_velocity'] > 10.0:
                    print("  *** CRITICAL: Velocity exceeds physical limit! ***")
                if metrics['max_temperature'] > 3533:
                    print(f"  *** WARNING: Temperature exceeds boiling point (3533K) ***")
        print()

        # Test 4: Realistic LPBF
        print("TEST 4: Realistic LPBF with All Fixes")
        print("-" * 70)
        vtk_data = self.check_vtk_outputs(os.path.join(build_dir, 'lpbf_realistic'))
        print(f"  VTK outputs: {vtk_data['count']} files")

        if os.path.exists(os.path.join(build_dir, 'lpbf_realistic_test.log')):
            metrics = self.analyze_log_file(os.path.join(build_dir, 'lpbf_realistic_test.log'))
            if metrics:
                print(f"  Max temperature: {metrics['max_temperature']:.1f} K")
                print(f"  Max velocity: {metrics['max_velocity']:.4f} m/s")
                print(f"  Completed: {metrics['completed']}")
                print(f"  NaN detected: {metrics['has_nan']}")

                if metrics['warnings']:
                    print(f"  Warnings: {len(metrics['warnings'])}")
        print()

        # Summary
        print("=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print()

        # Count tests
        total_tests = 4
        passed_tests = 0

        # Simple pass/fail logic based on file existence and basic checks
        test_dirs = ['test_newton_stress', 'test_marangoni_limiter', 'lpbf_realistic']
        for test_dir in test_dirs:
            vtk_data = self.check_vtk_outputs(os.path.join(build_dir, test_dir))
            if vtk_data['count'] > 0:
                passed_tests += 1

        print(f"Tests passed: {passed_tests} / {total_tests}")
        print()

        if passed_tests == total_tests:
            print("✓ All tests generated outputs successfully")
            print("✓ System appears stable with new fixes")
        else:
            print("✗ Some tests did not complete successfully")
            print("  Review individual test outputs for details")

        return passed_tests == total_tests

if __name__ == '__main__':
    build_dir = '/home/yzk/LBMProject/build'

    analyzer = TestAnalyzer()
    success = analyzer.analyze_test_suite(build_dir)

    sys.exit(0 if success else 1)
