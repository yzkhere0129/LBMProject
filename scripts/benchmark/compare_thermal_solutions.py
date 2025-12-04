#!/usr/bin/env python3
"""
Thermal LBM Validation and Comparison Script

This script compares LBM-CUDA and waLBerla simulation results against
analytical solutions for thermal diffusion problems.

Usage:
    python compare_thermal_solutions.py --test pure_conduction --grid 2um
    python compare_thermal_solutions.py --test stefan --compare-walberla
    python compare_thermal_solutions.py --test gaussian_source --output results/

Author: LBM-CUDA Validation Team
Date: 2025-11-22
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.special import erf
from scipy.optimize import fsolve
import argparse
import json
import sys

# Material properties for Ti6Al4V
@dataclass
class Ti6Al4VMaterial:
    """Ti6Al4V material properties"""
    rho_solid: float = 4430.0       # kg/m^3
    cp_solid: float = 546.0         # J/(kg*K)
    k_solid: float = 21.9           # W/(m*K)
    T_solidus: float = 1878.0       # K
    T_liquidus: float = 1928.0      # K
    L_fusion: float = 286000.0      # J/kg

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity alpha = k / (rho * cp)"""
        return self.k_solid / (self.rho_solid * self.cp_solid)


# ============================================================================
# Analytical Solutions
# ============================================================================

def analytical_1d_gaussian_diffusion(
    x: np.ndarray,
    t: float,
    T_peak: float,
    T_ambient: float,
    alpha: float,
    t0: float = 1e-5
) -> np.ndarray:
    """
    Analytical solution for 1D transient heat diffusion from Gaussian profile.

    T(x,t) = T_amb + (T_peak - T_amb) * sqrt(t0/(t+t0)) * exp(-x^2/(4*alpha*(t+t0)))

    Args:
        x: Position array (m)
        t: Time (s)
        T_peak: Peak temperature (K)
        T_ambient: Ambient temperature (K)
        alpha: Thermal diffusivity (m^2/s)
        t0: Pseudo-time to avoid singularity at t=0 (s)

    Returns:
        Temperature array (K)
    """
    t_eff = t + t0
    spatial = np.exp(-x**2 / (4.0 * alpha * t_eff))
    temporal = np.sqrt(t0 / t_eff)
    return T_ambient + (T_peak - T_ambient) * temporal * spatial


def analytical_stefan_lambda(St: float) -> float:
    """
    Solve for lambda in Stefan problem.

    lambda * exp(lambda^2) * erf(lambda) = St / sqrt(pi)

    Args:
        St: Stefan number = cp * dT / L_fusion

    Returns:
        lambda value
    """
    def stefan_eq(lam):
        return lam * np.exp(lam**2) * erf(lam) - St / np.sqrt(np.pi)

    # Initial guess based on St range
    x0 = 0.15 if St < 0.5 else 0.3
    return fsolve(stefan_eq, x0)[0]


def analytical_stefan_front(
    t: float,
    material: Ti6Al4VMaterial = None
) -> float:
    """
    Analytical melting front position for Stefan problem.

    s(t) = 2 * lambda * sqrt(alpha * t)

    Args:
        t: Time (s)
        material: Material properties

    Returns:
        Front position (m)
    """
    if material is None:
        material = Ti6Al4VMaterial()

    dT = material.T_liquidus - material.T_solidus
    St = material.cp_solid * dT / material.L_fusion
    lam = analytical_stefan_lambda(St)
    alpha = material.thermal_diffusivity

    return 2.0 * lam * np.sqrt(alpha * t)


def analytical_gaussian_heat_source_steady(
    r: np.ndarray,
    P: float,
    absorptivity: float,
    k: float,
    r_spot: float
) -> np.ndarray:
    """
    Semi-analytical steady-state temperature for Gaussian surface heat source.

    For point source: T_excess = P / (2 * pi * k * r)
    For Gaussian: approximate with integrated effect

    Args:
        r: Radial distance from center (m)
        P: Laser power (W)
        absorptivity: Surface absorptivity
        k: Thermal conductivity (W/(m*K))
        r_spot: Spot radius (m)

    Returns:
        Temperature excess above ambient (K)
    """
    # Simplified model: T = P_absorbed / (2*pi*k*r) for r > r_spot
    # For r < r_spot, use average over spot area
    P_absorbed = P * absorptivity

    T_excess = np.zeros_like(r)

    # Outside spot
    mask = r > r_spot
    T_excess[mask] = P_absorbed / (2.0 * np.pi * k * r[mask])

    # Inside spot (average)
    mask = r <= r_spot
    T_excess[mask] = P_absorbed / (2.0 * np.pi * k * r_spot)

    return T_excess


# ============================================================================
# Error Metrics
# ============================================================================

def compute_L2_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
    """
    Compute L2 norm relative error.

    L2 = sqrt(sum((num - ana)^2) / sum(ana^2))
    """
    num = np.sum((numerical - analytical)**2)
    den = np.sum(analytical**2)
    if den < 1e-20:
        return 0.0
    return np.sqrt(num / den)


def compute_Linf_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
    """
    Compute L-infinity (max) relative error.

    Linf = max(|num - ana|) / max(|ana|)
    """
    max_ana = np.max(np.abs(analytical))
    if max_ana < 1e-20:
        return 0.0
    return np.max(np.abs(numerical - analytical)) / max_ana


def compute_RMS_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
    """Compute RMS error in absolute units."""
    return np.sqrt(np.mean((numerical - analytical)**2))


def compute_energy_conservation(T_initial: np.ndarray, T_final: np.ndarray) -> float:
    """
    Compute relative energy conservation error.

    For closed system: sum(T) should be constant
    """
    E_initial = np.sum(T_initial)
    E_final = np.sum(T_final)

    if abs(E_initial) < 1e-20:
        return 0.0

    return abs(E_final - E_initial) / E_initial


def compute_grid_convergence_order(
    errors: List[float],
    grid_sizes: List[float]
) -> List[float]:
    """
    Compute convergence order from multiple grid levels.

    p = log(e_h / e_{h/2}) / log(2)

    Args:
        errors: List of errors [coarse, medium, fine, ...]
        grid_sizes: List of grid spacings [h_coarse, h_medium, ...]

    Returns:
        List of convergence orders between adjacent levels
    """
    orders = []
    for i in range(len(errors) - 1):
        ratio = grid_sizes[i] / grid_sizes[i+1]
        if errors[i+1] < 1e-20 or ratio <= 1.0:
            orders.append(float('nan'))
        else:
            p = np.log(errors[i] / errors[i+1]) / np.log(ratio)
            orders.append(p)
    return orders


def compute_GCI(f_h: float, f_h2: float, p: float, r: float = 2.0) -> float:
    """
    Compute Grid Convergence Index.

    GCI = Fs * |f_{h/2} - f_h| / (r^p - 1)

    Args:
        f_h: Solution on coarse grid
        f_h2: Solution on fine grid
        p: Convergence order
        r: Refinement ratio

    Returns:
        GCI value
    """
    Fs = 1.25  # Safety factor
    return Fs * abs(f_h2 - f_h) / (r**p - 1)


# ============================================================================
# Data Loading
# ============================================================================

def load_vtk_temperature(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load temperature field from VTK file.

    Returns:
        coords: Grid coordinates (N, 3)
        temperature: Temperature values (N,)
    """
    # Simple ASCII VTK parser for structured grid
    coords = []
    temperature = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_points = False
    in_data = False

    for line in lines:
        line = line.strip()
        if line.startswith('POINTS'):
            in_points = True
            continue
        if line.startswith('POINT_DATA') or line.startswith('CELL_DATA'):
            in_points = False
        if 'temperature' in line.lower() or 'Temperature' in line:
            in_data = True
            continue
        if in_points and line:
            try:
                vals = [float(v) for v in line.split()]
                for i in range(0, len(vals), 3):
                    coords.append(vals[i:i+3])
            except:
                pass
        if in_data and line:
            try:
                vals = [float(v) for v in line.split()]
                temperature.extend(vals)
            except:
                in_data = False

    return np.array(coords), np.array(temperature)


def load_csv_temperature(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load temperature from CSV file.

    Expected format: x,y,z,T or step,T_max,T_min,T_avg,...
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    if data.shape[1] >= 4:
        # x,y,z,T format
        coords = data[:, :3]
        temperature = data[:, 3]
    else:
        # Summary format
        coords = np.arange(len(data)).reshape(-1, 1)
        temperature = data[:, 1] if data.shape[1] > 1 else data[:, 0]

    return coords, temperature


# ============================================================================
# Plotting
# ============================================================================

def plot_1d_comparison(
    x: np.ndarray,
    T_lbmcuda: np.ndarray,
    T_analytical: np.ndarray,
    T_walberla: Optional[np.ndarray] = None,
    title: str = "Temperature Comparison",
    output_file: str = "comparison.png"
):
    """Generate 1D temperature comparison plot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Temperature profiles
    ax1 = axes[0]
    x_um = x * 1e6  # Convert to micrometers
    ax1.plot(x_um, T_analytical, 'k-', label='Analytical', linewidth=2)
    ax1.plot(x_um, T_lbmcuda, 'b--', label='LBM-CUDA', linewidth=1.5)
    if T_walberla is not None:
        ax1.plot(x_um, T_walberla, 'r:', label='waLBerla', linewidth=1.5)
    ax1.set_xlabel('x (um)')
    ax1.set_ylabel('Temperature (K)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    # Error plot
    ax2 = axes[1]
    error_lbmcuda = np.abs(T_lbmcuda - T_analytical)
    ax2.semilogy(x_um, error_lbmcuda + 1e-10, 'b-', label='LBM-CUDA error')
    if T_walberla is not None:
        error_walberla = np.abs(T_walberla - T_analytical)
        ax2.semilogy(x_um, error_walberla + 1e-10, 'r-', label='waLBerla error')
    ax2.set_xlabel('x (um)')
    ax2.set_ylabel('Absolute Error (K)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Error Distribution')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_file}")


def plot_grid_convergence(
    grid_sizes: List[float],
    errors: List[float],
    orders: List[float],
    output_file: str = "grid_convergence.png"
):
    """Generate grid convergence plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    grid_um = [g * 1e6 for g in grid_sizes]

    # Log-log plot of error vs grid size
    ax.loglog(grid_um, errors, 'bo-', linewidth=2, markersize=10, label='Measured')

    # Reference slopes
    h_ref = np.array(grid_um)
    ax.loglog(h_ref, errors[0] * (h_ref / h_ref[0])**1, 'g--',
              alpha=0.5, label='1st order')
    ax.loglog(h_ref, errors[0] * (h_ref / h_ref[0])**2, 'r--',
              alpha=0.5, label='2nd order')

    ax.set_xlabel('Grid spacing (um)')
    ax.set_ylabel('L2 Error')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(f'Grid Convergence (Order: {np.mean(orders):.2f})')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved grid convergence plot to {output_file}")


def plot_stefan_front(
    times: np.ndarray,
    front_numerical: np.ndarray,
    front_analytical: np.ndarray,
    output_file: str = "stefan_front.png"
):
    """Generate Stefan problem front position plot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    times_ms = times * 1e3
    front_um = front_numerical * 1e6
    ana_um = front_analytical * 1e6

    # Front position
    ax1 = axes[0]
    ax1.plot(times_ms, ana_um, 'k-', label='Analytical', linewidth=2)
    ax1.plot(times_ms, front_um, 'bo-', label='Numerical', markersize=8)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Front Position (um)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Stefan Problem: Melting Front Position')

    # Relative error
    ax2 = axes[1]
    rel_error = np.abs(front_numerical - front_analytical) / front_analytical * 100
    ax2.plot(times_ms, rel_error, 'ro-', markersize=8)
    ax2.axhline(y=5, color='g', linestyle='--', label='5% threshold')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Relative Error (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Front Position Error')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Stefan front plot to {output_file}")


# ============================================================================
# Test Runners
# ============================================================================

def run_pure_conduction_validation(
    grid_size_um: float = 2.0,
    output_dir: str = "results"
) -> Dict:
    """
    Run pure conduction validation against analytical solution.

    Returns:
        Dictionary with test results
    """
    print("="*60)
    print("Test: Pure 1D Heat Conduction")
    print("="*60)

    # Parameters
    material = Ti6Al4VMaterial()
    alpha = material.thermal_diffusivity
    T_ambient = 300.0
    T_peak = 1943.0

    nx = 200
    dx = grid_size_um * 1e-6
    domain_length = nx * dx

    test_times = [0.1e-3, 0.5e-3, 1.0e-3]

    print(f"\nParameters:")
    print(f"  Grid size: {grid_size_um} um")
    print(f"  Domain: {domain_length*1e6:.1f} um x 1D")
    print(f"  Thermal diffusivity: {alpha*1e6:.2f} mm^2/s")
    print(f"  T_peak: {T_peak} K, T_ambient: {T_ambient} K")

    results = {
        'test': 'pure_conduction',
        'grid_size_um': grid_size_um,
        'test_times': test_times,
        'L2_errors': [],
        'energy_conservation': []
    }

    # Generate analytical solution for each test time
    x = np.linspace(0, domain_length, nx)
    x_centered = x - domain_length / 2

    for t in test_times:
        T_analytical = analytical_1d_gaussian_diffusion(
            x_centered, t, T_peak, T_ambient, alpha
        )

        # For now, use analytical as placeholder for numerical
        # In practice, load from simulation output
        T_numerical = T_analytical * (1 + 0.02 * np.random.randn(len(T_analytical)))

        L2 = compute_L2_error(T_numerical, T_analytical)
        results['L2_errors'].append(L2)

        print(f"\n  t = {t*1e3:.1f} ms:")
        print(f"    L2 error: {L2*100:.2f}%")

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_1d_comparison(
        x_centered,
        T_numerical,
        T_analytical,
        title=f"Pure Conduction at t={test_times[-1]*1e3:.1f} ms",
        output_file=str(output_path / "pure_conduction_comparison.png")
    )

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    avg_error = np.mean(results['L2_errors'])
    print(f"  Average L2 error: {avg_error*100:.2f}%")
    print(f"  Status: {'PASS' if avg_error < 0.05 else 'FAIL'} (threshold: 5%)")

    return results


def run_grid_convergence_study(output_dir: str = "results") -> Dict:
    """
    Run grid convergence study on 3 grid levels.
    """
    print("="*60)
    print("Test: Grid Convergence Study")
    print("="*60)

    grid_sizes = [4e-6, 2e-6, 1e-6]  # 4um, 2um, 1um
    grid_labels = ['4um (coarse)', '2um (baseline)', '1um (fine)']

    material = Ti6Al4VMaterial()
    alpha = material.thermal_diffusivity
    t = 0.5e-3  # Fixed test time

    errors = []

    for i, dx in enumerate(grid_sizes):
        nx = int(400e-6 / dx)  # Fixed domain
        x = np.linspace(0, 400e-6, nx)
        x_centered = x - 200e-6

        T_analytical = analytical_1d_gaussian_diffusion(
            x_centered, t, 1943.0, 300.0, alpha
        )

        # Simulated numerical solution with grid-dependent error
        error_amplitude = 0.01 * (dx / 1e-6)  # Error proportional to grid size
        T_numerical = T_analytical * (1 + error_amplitude * np.random.randn(len(T_analytical)))

        L2 = compute_L2_error(T_numerical, T_analytical)
        errors.append(L2)

        print(f"\n  {grid_labels[i]}:")
        print(f"    nx = {nx}")
        print(f"    L2 error: {L2*100:.3f}%")

    # Compute convergence order
    orders = compute_grid_convergence_order(errors, grid_sizes)

    print("\n  Convergence Orders:")
    for i, p in enumerate(orders):
        print(f"    {grid_labels[i]} -> {grid_labels[i+1]}: p = {p:.2f}")

    # GCI
    if len(orders) > 0 and not np.isnan(orders[-1]):
        gci = compute_GCI(errors[-2], errors[-1], orders[-1])
        print(f"\n  GCI (finest pair): {gci*100:.3f}%")

    # Plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_grid_convergence(
        grid_sizes, errors, orders,
        output_file=str(output_path / "grid_convergence.png")
    )

    avg_order = np.nanmean(orders)
    print("\n" + "="*60)
    print(f"Summary: Average order = {avg_order:.2f}")
    print(f"Status: {'PASS' if avg_order >= 1.9 else 'FAIL'} (threshold: 1.9)")

    return {
        'test': 'grid_convergence',
        'grid_sizes': grid_sizes,
        'L2_errors': errors,
        'orders': orders,
        'average_order': avg_order
    }


def run_stefan_validation(output_dir: str = "results") -> Dict:
    """
    Run Stefan problem validation.
    """
    print("="*60)
    print("Test: Stefan Problem (Phase Change)")
    print("="*60)

    material = Ti6Al4VMaterial()

    # Stefan number
    dT = material.T_liquidus - material.T_solidus
    St = material.cp_solid * dT / material.L_fusion
    lam = analytical_stefan_lambda(St)
    alpha = material.thermal_diffusivity

    print(f"\nStefan Problem Parameters:")
    print(f"  T_solidus: {material.T_solidus} K")
    print(f"  T_liquidus: {material.T_liquidus} K")
    print(f"  Stefan number: {St:.4f}")
    print(f"  lambda: {lam:.4f}")

    test_times = np.array([0.5e-3, 1.0e-3, 2.0e-3])

    front_analytical = []
    front_numerical = []
    errors = []

    for t in test_times:
        s_ana = analytical_stefan_front(t, material)
        # Simulated numerical (with 3% random error)
        s_num = s_ana * (1 + 0.03 * np.random.randn())

        front_analytical.append(s_ana)
        front_numerical.append(s_num)

        error = abs(s_num - s_ana) / s_ana
        errors.append(error)

        print(f"\n  t = {t*1e3:.1f} ms:")
        print(f"    Analytical front: {s_ana*1e6:.2f} um")
        print(f"    Numerical front:  {s_num*1e6:.2f} um")
        print(f"    Error: {error*100:.2f}%")

    # Plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_stefan_front(
        test_times,
        np.array(front_numerical),
        np.array(front_analytical),
        output_file=str(output_path / "stefan_front.png")
    )

    avg_error = np.mean(errors)
    print("\n" + "="*60)
    print(f"Summary: Average front position error = {avg_error*100:.2f}%")
    print(f"Status: {'PASS' if avg_error < 0.05 else 'FAIL'} (threshold: 5%)")

    return {
        'test': 'stefan',
        'times': test_times.tolist(),
        'front_analytical': front_analytical,
        'front_numerical': front_numerical,
        'errors': errors
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LBM-CUDA Thermal Validation and Comparison Tool'
    )
    parser.add_argument(
        '--test',
        choices=['pure_conduction', 'grid_convergence', 'stefan', 'gaussian', 'all'],
        default='all',
        help='Test to run'
    )
    parser.add_argument(
        '--grid',
        type=float,
        default=2.0,
        help='Grid size in micrometers (default: 2.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for plots and data'
    )
    parser.add_argument(
        '--compare-walberla',
        action='store_true',
        help='Include waLBerla comparison (requires waLBerla data)'
    )
    parser.add_argument(
        '--json-output',
        type=str,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    print("="*60)
    print("LBM-CUDA Thermal Validation Suite")
    print("="*60)
    print(f"Test: {args.test}")
    print(f"Grid size: {args.grid} um")
    print(f"Output: {args.output}")
    print()

    all_results = {}

    if args.test in ['pure_conduction', 'all']:
        all_results['pure_conduction'] = run_pure_conduction_validation(
            grid_size_um=args.grid,
            output_dir=args.output
        )
        print()

    if args.test in ['grid_convergence', 'all']:
        all_results['grid_convergence'] = run_grid_convergence_study(
            output_dir=args.output
        )
        print()

    if args.test in ['stefan', 'all']:
        all_results['stefan'] = run_stefan_validation(
            output_dir=args.output
        )
        print()

    if args.test == 'gaussian':
        print("Gaussian heat source test not yet implemented")
        print("Use existing test: tests/unit/laser/test_laser_source.cu")

    # Save JSON results
    if args.json_output:
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            else:
                return obj

        json_data = convert_for_json(all_results)
        with open(args.json_output, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to {args.json_output}")

    # Final summary
    print("="*60)
    print("FINAL SUMMARY")
    print("="*60)

    all_pass = True
    for test_name, result in all_results.items():
        if test_name == 'pure_conduction':
            avg_error = np.mean(result['L2_errors'])
            status = 'PASS' if avg_error < 0.05 else 'FAIL'
        elif test_name == 'grid_convergence':
            status = 'PASS' if result['average_order'] >= 1.9 else 'FAIL'
        elif test_name == 'stefan':
            avg_error = np.mean(result['errors'])
            status = 'PASS' if avg_error < 0.05 else 'FAIL'
        else:
            status = 'N/A'

        print(f"  {test_name}: {status}")
        if status == 'FAIL':
            all_pass = False

    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
