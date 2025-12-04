#!/usr/bin/env python3
"""
Analyze Poiseuille flow simulation and compare with analytical solution.

Validates velocity profile against parabolic analytical solution,
computes error metrics, and checks for numerical artifacts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# === PARAMETERS (modify these) ===
DATA_FILE = "/home/yzk/LBMProject/build/tests/integration/poiseuille_profile_fluidlbm.txt"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/results"

def load_poiseuille_data(filepath):
    """Load Poiseuille flow profile data."""
    try:
        # Skip header lines
        data = np.loadtxt(filepath, skiprows=2)
        print(f"Loaded Poiseuille data from: {filepath}")
        print(f"  Data shape: {data.shape}")

        y_positions = data[:, 0]
        u_numerical = data[:, 1]
        u_analytical = data[:, 2]
        error = data[:, 3]

        return {
            'y': y_positions,
            'u_num': u_numerical,
            'u_ana': u_analytical,
            'error': error
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_poiseuille_profile(data):
    """Compute error metrics and statistics for Poiseuille flow."""

    y = data['y']
    u_num = data['u_num']
    u_ana = data['u_ana']
    error = data['error']

    print(f"\n=== Poiseuille Flow Analysis ===")

    # Basic statistics
    print(f"\nVelocity Statistics:")
    print(f"  Numerical - Min: {u_num.min():.6e}, Max: {u_num.max():.6e}, Mean: {u_num.mean():.6e}")
    print(f"  Analytical - Min: {u_ana.min():.6e}, Max: {u_ana.max():.6e}, Mean: {u_ana.mean():.6e}")

    # Error metrics
    abs_error = np.abs(error)
    rel_error = np.abs(error / (u_ana + 1e-10)) * 100  # Avoid division by zero

    # L2 norm error
    l2_error = np.sqrt(np.sum(error**2) / np.sum(u_ana**2)) * 100

    # L-infinity (max) error
    linf_error = np.max(abs_error)

    print(f"\nError Metrics:")
    print(f"  L2 relative error: {l2_error:.4f}%")
    print(f"  L∞ (max) error:    {linf_error:.6e}")
    print(f"  Mean absolute error: {abs_error.mean():.6e}")
    print(f"  Std of error:        {error.std():.6e}")

    # Find location of maximum error
    max_error_idx = np.argmax(abs_error)
    print(f"\nMaximum error location:")
    print(f"  y-position: {y[max_error_idx]}")
    print(f"  Numerical:  {u_num[max_error_idx]:.6e}")
    print(f"  Analytical: {u_ana[max_error_idx]:.6e}")
    print(f"  Error:      {error[max_error_idx]:.6e}")

    # Check parabolic shape preservation
    # For Poiseuille flow: u(y) = u_max * (1 - (y/h)^2)
    # Maximum should be at center
    center_idx = len(y) // 2
    u_max_idx = np.argmax(np.abs(u_num))

    print(f"\nParabolic Profile Check:")
    print(f"  Center index: {center_idx}")
    print(f"  Max velocity index: {u_max_idx}")
    if u_max_idx == center_idx or u_max_idx == center_idx + 1:
        print(f"  ✓ Maximum velocity at center (correct)")
    else:
        print(f"  ✗ WARNING: Maximum velocity NOT at center!")

    # Check symmetry
    n = len(y)
    first_half = u_num[:n//2]
    second_half = u_num[n//2:][::-1]  # Reverse second half
    symmetry_error = np.mean(np.abs(first_half - second_half))

    print(f"\nSymmetry Check:")
    print(f"  Symmetry error: {symmetry_error:.6e}")
    if symmetry_error < 1e-6:
        print(f"  ✓ Profile is symmetric")
    else:
        print(f"  ✗ WARNING: Profile asymmetry detected!")

    # Check boundary conditions (should be zero at walls)
    print(f"\nBoundary Conditions:")
    print(f"  u at y=0:  {u_num[0]:.6e} (should be ≈ 0)")
    print(f"  u at y=-1: {u_num[-1]:.6e} (should be ≈ 0)")

    if np.abs(u_num[0]) < 1e-6 and np.abs(u_num[-1]) < 1e-6:
        print(f"  ✓ No-slip boundary conditions satisfied")
    else:
        print(f"  ✗ WARNING: Boundary conditions violated!")

    return {
        'l2_error': l2_error,
        'linf_error': linf_error,
        'mean_abs_error': abs_error.mean(),
        'symmetry_error': symmetry_error
    }

def plot_poiseuille_comparison(data, output_dir):
    """Plot velocity profile comparison."""

    y = data['y']
    u_num = data['u_num']
    u_ana = data['u_ana']
    error = data['error']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Velocity profiles comparison
    axes[0, 0].plot(y, u_num, 'bo-', label='Numerical', markersize=4)
    axes[0, 0].plot(y, u_ana, 'r--', label='Analytical', linewidth=2)
    axes[0, 0].set_xlabel('y-position (cell index)')
    axes[0, 0].set_ylabel('Velocity (lattice units)')
    axes[0, 0].set_title('Poiseuille Flow: Velocity Profile Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Error distribution
    axes[0, 1].plot(y, error, 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('y-position (cell index)')
    axes[0, 1].set_ylabel('Error (u_num - u_ana)')
    axes[0, 1].set_title('Point-wise Error Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Relative error
    rel_error = np.abs(error / (np.abs(u_ana) + 1e-10)) * 100
    axes[1, 0].plot(y, rel_error, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('y-position (cell index)')
    axes[1, 0].set_ylabel('Relative Error (%)')
    axes[1, 0].set_title('Relative Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Log-scale comparison (to see boundary layer detail)
    axes[1, 1].semilogy(y, np.abs(u_num), 'bo-', label='|u_num|', markersize=3)
    axes[1, 1].semilogy(y, np.abs(u_ana), 'r--', label='|u_ana|', linewidth=2)
    axes[1, 1].set_xlabel('y-position (cell index)')
    axes[1, 1].set_ylabel('|Velocity| (log scale)')
    axes[1, 1].set_title('Velocity Profile (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'poiseuille_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved Poiseuille analysis plot to: {output_file}")
    plt.close()

def plot_parabolic_fit(data, output_dir):
    """Fit parabola to numerical data and compare with analytical."""

    y = data['y']
    u_num = data['u_num']
    u_ana = data['u_ana']

    # Normalize y to [-1, 1]
    y_norm = (y - y.mean()) / (y.max() - y.min()) * 2

    # Fit parabola: u = a + b*y + c*y^2
    coeffs = np.polyfit(y_norm, u_num, 2)
    u_fit = np.polyval(coeffs, y_norm)

    print(f"\nParabolic Fit:")
    print(f"  Coefficients: u = {coeffs[0]:.6e}*y² + {coeffs[1]:.6e}*y + {coeffs[2]:.6e}")
    print(f"  Fit R²: {1 - np.sum((u_num - u_fit)**2) / np.sum((u_num - u_num.mean())**2):.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(y, u_num, 'bo', label='Numerical', markersize=6, alpha=0.6)
    ax.plot(y, u_ana, 'r-', label='Analytical', linewidth=3, alpha=0.8)
    ax.plot(y, u_fit, 'g--', label='Parabolic Fit', linewidth=2)

    ax.set_xlabel('y-position (cell index)', fontsize=12)
    ax.set_ylabel('Velocity (lattice units)', fontsize=12)
    ax.set_title('Poiseuille Flow: Parabolic Fit Validation', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'poiseuille_parabolic_fit.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved parabolic fit plot to: {output_file}")
    plt.close()

def main():
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("Poiseuille Flow Analysis")
    print('='*60)

    # Load data
    data = load_poiseuille_data(DATA_FILE)
    if data is None:
        print("ERROR: Failed to load Poiseuille data")
        return 1

    # Analyze profile
    metrics = analyze_poiseuille_profile(data)

    # Generate plots
    plot_poiseuille_comparison(data, output_dir)
    plot_parabolic_fit(data, output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print('='*60)
    print(f"  L2 Error: {metrics['l2_error']:.4f}%")
    print(f"  Max Error: {metrics['linf_error']:.6e}")

    # Validation criteria (typical for LBM)
    if metrics['l2_error'] < 5.0:
        print(f"  ✓ PASS: L2 error < 5%")
    else:
        print(f"  ✗ FAIL: L2 error ≥ 5%")

    if metrics['symmetry_error'] < 1e-5:
        print(f"  ✓ PASS: Profile is symmetric")
    else:
        print(f"  ✗ FAIL: Profile asymmetry detected")

    print(f"\nResults saved to: {output_dir}")
    print('='*60)

    return 0

if __name__ == '__main__':
    sys.exit(main())
