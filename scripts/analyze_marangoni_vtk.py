#!/usr/bin/env python3
"""
Marangoni Benchmark VTK Analysis Script

Extracts and analyzes VTK output files from the Marangoni benchmark test
with extreme precision. Any deviation from expected physics is flagged.

Project: /home/yzk/LBMProject
VTK files: /home/yzk/LBMProject/build/marangoni_benchmark_output/

Author: VTK Analysis Specialist
Date: 2025-12-03
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# PHYSICAL PARAMETERS (from test configuration)
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration parameters for Marangoni benchmark"""
    # Domain
    nx: int = 200              # Grid cells in X
    ny: int = 100              # Grid cells in Y
    nz: int = 2                # Grid cells in Z (quasi-2D)
    dx: float = 2.0e-6         # Grid spacing [m]
    dt: float = 1.0e-9         # Timestep [s]

    # Temperature
    T_hot: float = 2000.0      # Left wall temperature [K]
    T_cold: float = 1800.0     # Right wall temperature [K]
    T_ref: float = 1900.0      # Reference temperature [K]

    # Material (Ti-6Al-4V)
    rho: float = 4420.0        # Density [kg/m³]
    mu: float = 4.0e-3         # Dynamic viscosity [Pa·s]
    dsigma_dT: float = -0.26e-3  # Surface tension coeff [N/(m·K)]

    # Interface
    interface_position: float = 0.7  # Fraction from bottom
    interface_width: float = 3.0     # Cells

    # Derived quantities
    @property
    def dT(self) -> float:
        """Temperature difference [K]"""
        return self.T_hot - self.T_cold

    @property
    def dT_dx_expected(self) -> float:
        """Expected temperature gradient [K/m]"""
        Lx = (self.nx - 1) * self.dx
        return self.dT / Lx

    @property
    def grad_f_expected(self) -> float:
        """Expected fill level gradient magnitude [1/m]"""
        # Interface width in physical units
        h = self.interface_width * self.dx
        # Characteristic gradient: ~1/h for tanh profile
        return 1.0 / h

    @property
    def force_expected(self) -> float:
        """Expected Marangoni force magnitude [N/m³]"""
        # F = |dσ/dT| × |∇T| × |∇f|
        return abs(self.dsigma_dT) * self.dT_dx_expected * self.grad_f_expected

    @property
    def interface_y_cell(self) -> float:
        """Expected interface Y position [cell index]"""
        return self.interface_position * self.ny

    @property
    def velocity_scale(self) -> float:
        """Characteristic Marangoni velocity [m/s]"""
        return abs(self.dsigma_dT) * self.dT / self.mu

config = BenchmarkConfig()

# ============================================================================
# VTK DATA LOADING
# ============================================================================

def load_vtk_file(filepath: Path) -> vtk.vtkStructuredPoints:
    """Load a single VTK file using legacy reader"""
    try:
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(str(filepath))
        reader.Update()
        return reader.GetOutput()
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return None

def get_vtk_files(directory: Path) -> List[Path]:
    """Get sorted list of VTK files"""
    files = sorted(directory.glob("marangoni_*.vtk"))
    return files

def extract_timestep(filepath: Path) -> int:
    """Extract timestep number from filename"""
    # marangoni_000000.vtk -> 0
    stem = filepath.stem
    return int(stem.split('_')[1])

# ============================================================================
# FIELD EXTRACTION
# ============================================================================

def extract_fields(mesh: vtk.vtkStructuredPoints) -> Tuple[Dict[str, np.ndarray], Tuple[int, int, int]]:
    """Extract all fields from VTK mesh and reshape to 3D arrays"""
    fields = {}

    # Get dimensions
    dims = mesh.GetDimensions()
    nx, ny, nz = dims

    # Get point data
    point_data = mesh.GetPointData()

    # Extract each array
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        vtk_array = point_data.GetArray(i)

        # Convert to numpy
        np_array = vtk_to_numpy(vtk_array)

        # Handle vector fields (3 components)
        if vtk_array.GetNumberOfComponents() == 3:
            # Reshape to (nx, ny, nz, 3)
            fields[array_name] = np_array.reshape((nx, ny, nz, 3), order='F')
        # Handle scalar fields
        else:
            # Reshape to (nx, ny, nz)
            fields[array_name] = np_array.reshape((nx, ny, nz), order='F')

    return fields, (nx, ny, nz)

# ============================================================================
# TEMPERATURE FIELD ANALYSIS
# ============================================================================

def analyze_temperature_field(T: np.ndarray, dims: Tuple[int, int, int]) -> Dict:
    """
    Analyze temperature field T(x, y, z)

    Expected: Linear gradient from T_hot (x=0) to T_cold (x=nx-1)
    """
    nx, ny, nz = dims
    results = {}

    # Global statistics
    results['T_min'] = np.min(T)
    results['T_max'] = np.max(T)
    results['T_mean'] = np.mean(T)
    results['T_std'] = np.std(T)

    # Boundary values
    results['T_left'] = np.mean(T[0, :, :])  # x=0
    results['T_right'] = np.mean(T[-1, :, :])  # x=nx-1

    # Expected values
    results['T_hot_expected'] = config.T_hot
    results['T_cold_expected'] = config.T_cold

    # Temperature gradient along X
    T_avg_x = np.mean(T, axis=(1, 2))  # Average over Y and Z
    x_cells = np.arange(nx)

    # Linear fit: T = a*x + b
    coeffs = np.polyfit(x_cells, T_avg_x, 1)
    dT_dx_measured = coeffs[0] * config.nx / ((config.nx - 1) * config.dx)  # [K/m]

    results['dT_dx_measured'] = dT_dx_measured
    results['dT_dx_expected'] = config.dT_dx_expected
    results['dT_dx_error'] = abs(dT_dx_measured - config.dT_dx_expected) / abs(config.dT_dx_expected)

    # Linearity check: R² value
    T_fit = np.polyval(coeffs, x_cells)
    ss_res = np.sum((T_avg_x - T_fit)**2)
    ss_tot = np.sum((T_avg_x - np.mean(T_avg_x))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    results['T_linearity_R2'] = r_squared

    # Temperature profile for plotting
    results['x_cells'] = x_cells
    results['T_profile'] = T_avg_x
    results['T_fit'] = T_fit

    return results

# ============================================================================
# INTERFACE ANALYSIS
# ============================================================================

def analyze_interface(f: np.ndarray, dims: Tuple[int, int, int]) -> Dict:
    """
    Analyze fill level field f(x, y, z)

    Expected: Interface at y ≈ 70% of domain height with tanh profile
    """
    nx, ny, nz = dims
    results = {}

    # Global statistics
    results['f_min'] = np.min(f)
    results['f_max'] = np.max(f)
    results['f_mean'] = np.mean(f)

    # Mass conservation: total fill level
    results['total_mass'] = np.sum(f)

    # Interface cells: 0.1 < f < 0.9
    interface_mask = (f > 0.1) & (f < 0.9)
    results['n_interface_cells'] = np.sum(interface_mask)

    # Interface position: average Y coordinate of interface cells
    if np.sum(interface_mask) > 0:
        # Create coordinate grids
        y_indices = np.arange(ny)
        y_grid = np.broadcast_to(y_indices[np.newaxis, :, np.newaxis], (nx, ny, nz))

        y_interface = y_grid[interface_mask]
        results['interface_y_mean'] = np.mean(y_interface)
        results['interface_y_std'] = np.std(y_interface)
        results['interface_y_expected'] = config.interface_y_cell
        results['interface_y_error'] = abs(results['interface_y_mean'] - results['interface_y_expected'])
    else:
        results['interface_y_mean'] = np.nan
        results['interface_y_std'] = np.nan
        results['interface_y_expected'] = config.interface_y_cell
        results['interface_y_error'] = np.nan

    # Interface profile along Y (averaged over X and Z)
    f_avg_y = np.mean(f, axis=(0, 2))
    results['y_cells'] = np.arange(ny)
    results['f_profile'] = f_avg_y

    # Interface thickness: measure width where 0.1 < f < 0.9
    interface_y_mask = (f_avg_y > 0.1) & (f_avg_y < 0.9)
    if np.sum(interface_y_mask) > 0:
        y_interface_cells = results['y_cells'][interface_y_mask]
        results['interface_thickness_cells'] = y_interface_cells.max() - y_interface_cells.min()
        results['interface_thickness_expected'] = config.interface_width
    else:
        results['interface_thickness_cells'] = np.nan
        results['interface_thickness_expected'] = config.interface_width

    return results

# ============================================================================
# VELOCITY FIELD ANALYSIS
# ============================================================================

def analyze_velocity_field(v: np.ndarray, f: np.ndarray, dims: Tuple[int, int, int]) -> Dict:
    """
    Analyze velocity field v(x, y, z) = [vx, vy, vz]

    Expected: Maximum velocity at interface, flow in +x direction
    """
    nx, ny, nz = dims
    results = {}

    # Extract components
    vx = v[:, :, :, 0]
    vy = v[:, :, :, 1]
    vz = v[:, :, :, 2]

    # Velocity magnitude
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Global statistics
    results['v_max'] = np.max(v_mag)
    results['v_mean'] = np.mean(v_mag)
    results['vx_max'] = np.max(np.abs(vx))
    results['vy_max'] = np.max(np.abs(vy))
    results['vz_max'] = np.max(np.abs(vz))

    # Interface velocity (where 0.1 < f < 0.9)
    interface_mask = (f > 0.1) & (f < 0.9)
    if np.sum(interface_mask) > 0:
        v_interface = v_mag[interface_mask]
        vx_interface = vx[interface_mask]

        results['v_interface_max'] = np.max(v_interface)
        results['v_interface_mean'] = np.mean(v_interface)
        results['vx_interface_mean'] = np.mean(vx_interface)

        # Flow direction check: vx should be positive (hot to cold)
        # For negative dσ/dT, flow is from hot (low σ) to cold (high σ)
        n_positive_vx = np.sum(vx_interface > 0)
        n_negative_vx = np.sum(vx_interface < 0)
        results['vx_direction_correct'] = (n_positive_vx > n_negative_vx)
        results['vx_positive_fraction'] = n_positive_vx / len(vx_interface) if len(vx_interface) > 0 else 0.0
    else:
        results['v_interface_max'] = np.nan
        results['v_interface_mean'] = np.nan
        results['vx_interface_mean'] = np.nan
        results['vx_direction_correct'] = False
        results['vx_positive_fraction'] = np.nan

    # Velocity profile along Y at center X
    x_center = nx // 2
    v_profile_y = np.mean(v_mag[x_center, :, :], axis=1)
    vx_profile_y = np.mean(vx[x_center, :, :], axis=1)

    results['y_cells'] = np.arange(ny)
    results['v_profile_y'] = v_profile_y
    results['vx_profile_y'] = vx_profile_y

    # Expected velocity scale
    results['v_scale_expected'] = config.velocity_scale

    return results

# ============================================================================
# MARANGONI FORCE ANALYSIS
# ============================================================================

def analyze_marangoni_force(F: np.ndarray, f: np.ndarray, dims: Tuple[int, int, int]) -> Dict:
    """
    Analyze Marangoni force field F(x, y, z) = [Fx, Fy, Fz]

    Expected: Force localized at interface, tangential direction (Fy ≈ 0)
    """
    nx, ny, nz = dims
    results = {}

    # Extract components
    Fx = F[:, :, :, 0]
    Fy = F[:, :, :, 1]
    Fz = F[:, :, :, 2]

    # Force magnitude
    F_mag = np.sqrt(Fx**2 + Fy**2 + Fz**2)

    # Global statistics
    results['F_max'] = np.max(F_mag)
    results['F_mean'] = np.mean(F_mag)
    results['Fx_max'] = np.max(np.abs(Fx))
    results['Fy_max'] = np.max(np.abs(Fy))
    results['Fz_max'] = np.max(np.abs(Fz))

    # Interface force (where 0.1 < f < 0.9)
    interface_mask = (f > 0.1) & (f < 0.9)
    non_interface_mask = (f <= 0.1) | (f >= 0.9)

    if np.sum(interface_mask) > 0:
        F_interface = F_mag[interface_mask]
        Fx_interface = Fx[interface_mask]
        Fy_interface = Fy[interface_mask]

        results['F_interface_max'] = np.max(F_interface)
        results['F_interface_mean'] = np.mean(F_interface)

        # Force should be tangential: Fy should be small compared to Fx
        results['Fx_interface_mean'] = np.mean(np.abs(Fx_interface))
        results['Fy_interface_mean'] = np.mean(np.abs(Fy_interface))
        results['Fy_Fx_ratio'] = results['Fy_interface_mean'] / (results['Fx_interface_mean'] + 1e-10)

        # Force localization: compare interface vs non-interface
        if np.sum(non_interface_mask) > 0:
            F_non_interface = F_mag[non_interface_mask]
            results['F_non_interface_mean'] = np.mean(F_non_interface)
            results['F_localization_ratio'] = results['F_interface_mean'] / (results['F_non_interface_mean'] + 1e-10)
        else:
            results['F_non_interface_mean'] = 0.0
            results['F_localization_ratio'] = np.inf
    else:
        results['F_interface_max'] = np.nan
        results['F_interface_mean'] = np.nan
        results['Fx_interface_mean'] = np.nan
        results['Fy_interface_mean'] = np.nan
        results['Fy_Fx_ratio'] = np.nan
        results['F_non_interface_mean'] = np.nan
        results['F_localization_ratio'] = np.nan

    # Expected force magnitude
    results['F_expected'] = config.force_expected

    return results

# ============================================================================
# CONSERVATION CHECKS
# ============================================================================

def check_conservation(all_results: List[Dict]) -> Dict:
    """Check conservation laws across all timesteps"""
    results = {}

    # Extract time series
    timesteps = [r['timestep'] for r in all_results]
    times = [r['time'] for r in all_results]
    total_mass = [r['interface']['total_mass'] for r in all_results]

    # Mass conservation
    mass_initial = total_mass[0]
    mass_final = total_mass[-1]
    mass_change = abs(mass_final - mass_initial)
    mass_change_percent = 100 * mass_change / mass_initial if mass_initial > 0 else 0.0

    results['mass_initial'] = mass_initial
    results['mass_final'] = mass_final
    results['mass_change'] = mass_change
    results['mass_change_percent'] = mass_change_percent
    results['mass_conserved'] = mass_change_percent < 1.0  # Less than 1% change

    # Time series
    results['timesteps'] = timesteps
    results['times'] = times
    results['total_mass'] = total_mass

    return results

# ============================================================================
# TIME EVOLUTION ANALYSIS
# ============================================================================

def analyze_time_evolution(all_results: List[Dict]) -> Dict:
    """Analyze how system evolves over time"""
    results = {}

    # Extract time series
    timesteps = [r['timestep'] for r in all_results]
    times = [r['time'] for r in all_results]
    v_max = [r['velocity']['v_max'] for r in all_results]
    v_interface_max = [r['velocity']['v_interface_max'] for r in all_results]

    results['timesteps'] = timesteps
    results['times'] = times
    results['v_max'] = v_max
    results['v_interface_max'] = v_interface_max

    # Steady-state check: has velocity converged?
    # Compare last 25% of simulation to see if stable
    n = len(v_max)
    n_quarter = n // 4
    if n_quarter > 0:
        v_final = v_max[-n_quarter:]
        v_final_mean = np.mean(v_final)
        v_final_std = np.std(v_final)
        v_final_cv = v_final_std / (v_final_mean + 1e-10)  # Coefficient of variation

        results['v_final_mean'] = v_final_mean
        results['v_final_std'] = v_final_std
        results['v_final_cv'] = v_final_cv
        results['steady_state_reached'] = v_final_cv < 0.1  # Less than 10% variation
    else:
        results['steady_state_reached'] = False

    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_analysis_plots(all_results: List[Dict], output_dir: Path):
    """Generate comprehensive analysis plots"""

    # Extract final timestep for spatial analysis
    final = all_results[-1]
    T_results = final['temperature']
    f_results = final['interface']
    v_results = final['velocity']
    F_results = final['force']

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # ========== Row 1: Temperature and Interface ==========

    # 1. Temperature profile along X
    ax1 = plt.subplot(3, 4, 1)
    x_mm = T_results['x_cells'] * config.dx * 1e6  # Convert to μm
    ax1.plot(x_mm, T_results['T_profile'], 'b-', linewidth=2, label='Measured')
    ax1.plot(x_mm, T_results['T_fit'], 'r--', linewidth=2, label='Linear fit')
    ax1.axhline(config.T_hot, color='orange', linestyle=':', label=f'T_hot = {config.T_hot}K')
    ax1.axhline(config.T_cold, color='cyan', linestyle=':', label=f'T_cold = {config.T_cold}K')
    ax1.set_xlabel('X position [μm]', fontsize=12)
    ax1.set_ylabel('Temperature [K]', fontsize=12)
    ax1.set_title(f'Temperature Profile (R² = {T_results["T_linearity_R2"]:.6f})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Interface profile along Y
    ax2 = plt.subplot(3, 4, 2)
    y_mm = f_results['y_cells'] * config.dx * 1e6
    ax2.plot(f_results['f_profile'], y_mm, 'g-', linewidth=2)
    ax2.axhline(config.interface_y_cell * config.dx * 1e6, color='red', linestyle='--',
                label=f'Expected y = {config.interface_y_cell:.1f} cells')
    ax2.set_xlabel('Fill Level f', fontsize=12)
    ax2.set_ylabel('Y position [μm]', fontsize=12)
    ax2.set_title(f'Interface Position (y = {f_results["interface_y_mean"]:.1f} cells)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Velocity profile along Y
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(v_results['v_profile_y'], y_mm, 'b-', linewidth=2, label='|v|')
    ax3.plot(v_results['vx_profile_y'], y_mm, 'r--', linewidth=2, label='vx')
    ax3.axhline(config.interface_y_cell * config.dx * 1e6, color='green', linestyle=':',
                label='Interface')
    ax3.set_xlabel('Velocity [m/s]', fontsize=12)
    ax3.set_ylabel('Y position [μm]', fontsize=12)
    ax3.set_title(f'Velocity Profile (v_max = {v_results["v_max"]:.3f} m/s)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Placeholder for 2D field
    ax4 = plt.subplot(3, 4, 4)
    ax4.text(0.5, 0.5, f'Interface Cells: {f_results["n_interface_cells"]}\nExpected y: {config.interface_y_cell:.1f}\nMeasured y: {f_results["interface_y_mean"]:.1f}',
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Interface Statistics', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # ========== Row 2: Time Evolution ==========

    time_results = analyze_time_evolution(all_results)
    conservation = check_conservation(all_results)

    # 5. Velocity vs time
    ax5 = plt.subplot(3, 4, 5)
    times_us = np.array(time_results['times']) * 1e6  # Convert to μs
    ax5.plot(times_us, time_results['v_max'], 'b-', linewidth=2, label='v_max (global)')
    v_int_clean = [v if not np.isnan(v) else 0 for v in time_results['v_interface_max']]
    ax5.plot(times_us, v_int_clean, 'r--', linewidth=2, label='v_max (interface)')
    ax5.axhline(config.velocity_scale, color='green', linestyle=':',
                label=f'Expected scale = {config.velocity_scale:.3f} m/s')
    ax5.set_xlabel('Time [μs]', fontsize=12)
    ax5.set_ylabel('Maximum Velocity [m/s]', fontsize=12)
    ax5.set_title('Velocity Evolution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Mass conservation
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(times_us, conservation['total_mass'], 'g-', linewidth=2)
    ax6.axhline(conservation['mass_initial'], color='black', linestyle='--',
                label=f'Initial = {conservation["mass_initial"]:.1f}')
    ax6.set_xlabel('Time [μs]', fontsize=12)
    ax6.set_ylabel('Total Fill Level (integrated)', fontsize=12)
    ax6.set_title(f'Mass Conservation (Δ = {conservation["mass_change_percent"]:.2f}%)',
                  fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Temperature gradient check
    ax7 = plt.subplot(3, 4, 7)
    dT_dx_series = [r['temperature']['dT_dx_measured'] for r in all_results]
    ax7.plot(times_us, np.array(dT_dx_series) * 1e-6, 'b-', linewidth=2, label='Measured')
    ax7.axhline(config.dT_dx_expected * 1e-6, color='red', linestyle='--',
                label=f'Expected = {config.dT_dx_expected*1e-6:.2f} K/μm')
    ax7.set_xlabel('Time [μs]', fontsize=12)
    ax7.set_ylabel('Temperature Gradient [K/μm]', fontsize=12)
    ax7.set_title('Temperature Gradient vs Time', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Interface position tracking
    ax8 = plt.subplot(3, 4, 8)
    interface_y_series = [r['interface']['interface_y_mean'] for r in all_results]
    interface_y_clean = [y if not np.isnan(y) else config.interface_y_cell for y in interface_y_series]
    ax8.plot(times_us, interface_y_clean, 'g-', linewidth=2)
    ax8.axhline(config.interface_y_cell, color='red', linestyle='--',
                label=f'Expected = {config.interface_y_cell:.1f} cells')
    ax8.set_xlabel('Time [μs]', fontsize=12)
    ax8.set_ylabel('Interface Y Position [cells]', fontsize=12)
    ax8.set_title('Interface Position Stability', fontsize=14, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # ========== Row 3: Validation Metrics ==========

    # 9. Temperature boundary validation
    ax9 = plt.subplot(3, 4, 9)
    T_left_series = [r['temperature']['T_left'] for r in all_results]
    T_right_series = [r['temperature']['T_right'] for r in all_results]
    ax9.plot(times_us, T_left_series, 'r-', linewidth=2, label='Left wall')
    ax9.plot(times_us, T_right_series, 'b-', linewidth=2, label='Right wall')
    ax9.axhline(config.T_hot, color='red', linestyle='--', alpha=0.5)
    ax9.axhline(config.T_cold, color='blue', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Time [μs]', fontsize=12)
    ax9.set_ylabel('Temperature [K]', fontsize=12)
    ax9.set_title('Boundary Conditions', fontsize=14, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 10. Force magnitude evolution
    ax10 = plt.subplot(3, 4, 10)
    F_max_series = [r['force']['F_max'] for r in all_results]
    F_interface_max_series = [r['force']['F_interface_max'] for r in all_results]
    F_int_clean = [f if not np.isnan(f) else 0 for f in F_interface_max_series]
    ax10.plot(times_us, np.array(F_max_series) * 1e-6, 'b-', linewidth=2, label='F_max (global)')
    ax10.plot(times_us, np.array(F_int_clean) * 1e-6, 'r--', linewidth=2, label='F_max (interface)')
    ax10.axhline(config.force_expected * 1e-6, color='green', linestyle=':',
                 label=f'Expected = {config.force_expected*1e-6:.1f} MN/m³')
    ax10.set_xlabel('Time [μs]', fontsize=12)
    ax10.set_ylabel('Marangoni Force [MN/m³]', fontsize=12)
    ax10.set_title('Marangoni Force Evolution', fontsize=14, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # 11. Force direction validation
    ax11 = plt.subplot(3, 4, 11)
    Fy_Fx_ratio_series = [r['force']['Fy_Fx_ratio'] if not np.isnan(r['force']['Fy_Fx_ratio']) else 1e-10
                           for r in all_results]
    ax11.plot(times_us, Fy_Fx_ratio_series, 'purple', linewidth=2)
    ax11.axhline(0.1, color='red', linestyle='--', label='Threshold = 0.1')
    ax11.set_xlabel('Time [μs]', fontsize=12)
    ax11.set_ylabel('|Fy| / |Fx| Ratio', fontsize=12)
    ax11.set_title('Force Tangentiality Check', fontsize=14, fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.set_yscale('log')

    # 12. Validation summary text
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    # Compile validation summary
    summary_text = "VALIDATION SUMMARY\n" + "="*40 + "\n\n"

    # Temperature
    T_error = T_results['dT_dx_error'] * 100
    T_status = "PASS" if T_error < 5.0 else "FAIL"
    summary_text += f"Temperature Gradient:\n"
    summary_text += f"  Status: {T_status}\n"
    summary_text += f"  Error: {T_error:.2f}%\n"
    summary_text += f"  R²: {T_results['T_linearity_R2']:.6f}\n\n"

    # Interface
    f_error = f_results['interface_y_error']
    f_status = "PASS" if not np.isnan(f_error) and f_error < 2.0 else "FAIL"
    summary_text += f"Interface Position:\n"
    summary_text += f"  Status: {f_status}\n"
    summary_text += f"  Position: {f_results['interface_y_mean']:.1f} cells\n"
    summary_text += f"  Expected: {config.interface_y_cell:.1f} cells\n"
    summary_text += f"  Error: {f_error:.2f} cells\n\n"

    # Velocity
    v_ratio = v_results['v_max'] / config.velocity_scale
    v_status = "PASS" if 0.01 <= v_ratio <= 10.0 else "FAIL"
    summary_text += f"Velocity Magnitude:\n"
    summary_text += f"  Status: {v_status}\n"
    summary_text += f"  Max: {v_results['v_max']:.4f} m/s\n"
    summary_text += f"  Expected scale: {config.velocity_scale:.4f} m/s\n"
    summary_text += f"  Ratio: {v_ratio:.2f}\n\n"

    # Force direction
    vx_status = "PASS" if v_results.get('vx_direction_correct', False) else "FAIL"
    summary_text += f"Flow Direction:\n"
    summary_text += f"  Status: {vx_status}\n"
    vx_frac = v_results.get('vx_positive_fraction', 0)
    summary_text += f"  vx > 0: {vx_frac*100:.1f}%\n\n" if not np.isnan(vx_frac) else "  vx > 0: N/A\n\n"

    # Mass conservation
    mass_status = "PASS" if conservation['mass_conserved'] else "FAIL"
    summary_text += f"Mass Conservation:\n"
    summary_text += f"  Status: {mass_status}\n"
    summary_text += f"  Change: {conservation['mass_change_percent']:.3f}%\n"

    ax12.text(0.1, 0.95, summary_text, transform=ax12.transAxes,
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved analysis plot: {output_dir / 'analysis_summary.png'}")
    plt.close()

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_all_vtk_files(vtk_dir: Path, output_dir: Path):
    """Main analysis pipeline"""

    # Get all VTK files
    vtk_files = get_vtk_files(vtk_dir)
    print(f"Found {len(vtk_files)} VTK files in {vtk_dir}")

    if len(vtk_files) == 0:
        print("ERROR: No VTK files found!")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each file
    all_results = []

    for i, filepath in enumerate(vtk_files):
        print(f"\nAnalyzing [{i+1}/{len(vtk_files)}]: {filepath.name}")

        # Load VTK file
        mesh = load_vtk_file(filepath)
        if mesh is None:
            continue

        # Extract fields
        fields, dims = extract_fields(mesh)

        # Check required fields
        required_fields = ['Temperature', 'FillLevel', 'Velocity', 'MarangoniForce']
        missing = [f for f in required_fields if f not in fields]
        if missing:
            print(f"  WARNING: Missing fields: {missing}")
            continue

        # Extract timestep
        timestep = extract_timestep(filepath)
        time = timestep * config.dt

        # Analyze each component
        T = fields['Temperature']
        f = fields['FillLevel']
        v = fields['Velocity']
        F = fields['MarangoniForce']

        T_results = analyze_temperature_field(T, dims)
        f_results = analyze_interface(f, dims)
        v_results = analyze_velocity_field(v, f, dims)
        F_results = analyze_marangoni_force(F, f, dims)

        # Store results
        result = {
            'timestep': timestep,
            'time': time,
            'temperature': T_results,
            'interface': f_results,
            'velocity': v_results,
            'force': F_results,
            'fields': fields,
            'dims': dims
        }
        all_results.append(result)

        # Print quick summary
        print(f"  T_grad: {T_results['dT_dx_measured']*1e-6:.2f} K/μm (expected: {config.dT_dx_expected*1e-6:.2f})")
        print(f"  Interface y: {f_results['interface_y_mean']:.1f} cells (expected: {config.interface_y_cell:.1f})")
        print(f"  v_max: {v_results['v_max']:.4f} m/s")
        print(f"  F_max: {F_results['F_max']*1e-6:.1f} MN/m³ (expected: {config.force_expected*1e-6:.1f})")

    if len(all_results) == 0:
        print("ERROR: No valid results to analyze!")
        return

    # Conservation and time evolution analysis
    print("\n" + "="*80)
    print("CONSERVATION AND TIME EVOLUTION ANALYSIS")
    print("="*80)

    conservation = check_conservation(all_results)
    time_evolution = analyze_time_evolution(all_results)

    print(f"\nMass Conservation:")
    print(f"  Initial mass: {conservation['mass_initial']:.2f}")
    print(f"  Final mass: {conservation['mass_final']:.2f}")
    print(f"  Change: {conservation['mass_change_percent']:.3f}%")
    print(f"  Status: {'PASS' if conservation['mass_conserved'] else 'FAIL'}")

    print(f"\nSteady State:")
    if time_evolution.get('steady_state_reached', False):
        print(f"  Status: REACHED")
        print(f"  Final velocity: {time_evolution['v_final_mean']:.4f} ± {time_evolution['v_final_std']:.4f} m/s")
        print(f"  Coefficient of variation: {time_evolution['v_final_cv']*100:.2f}%")
    else:
        print(f"  Status: NOT REACHED (still evolving)")

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING ANALYSIS PLOTS")
    print("="*80)
    create_analysis_plots(all_results, output_dir)

    # Save detailed results to CSV
    print("\nSaving detailed results to CSV...")
    save_results_to_csv(all_results, output_dir)

    # Print final validation summary
    print_validation_summary(all_results[-1], conservation, time_evolution)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")

def save_results_to_csv(all_results: List[Dict], output_dir: Path):
    """Save analysis results to CSV file"""

    # Prepare data for CSV
    data = []
    for r in all_results:
        row = {
            'timestep': r['timestep'],
            'time_us': r['time'] * 1e6,
            'T_min': r['temperature']['T_min'],
            'T_max': r['temperature']['T_max'],
            'T_mean': r['temperature']['T_mean'],
            'dT_dx_measured': r['temperature']['dT_dx_measured'],
            'T_linearity_R2': r['temperature']['T_linearity_R2'],
            'interface_y_mean': r['interface']['interface_y_mean'],
            'interface_y_std': r['interface']['interface_y_std'],
            'n_interface_cells': r['interface']['n_interface_cells'],
            'total_mass': r['interface']['total_mass'],
            'v_max': r['velocity']['v_max'],
            'v_interface_max': r['velocity']['v_interface_max'],
            'vx_positive_fraction': r['velocity'].get('vx_positive_fraction', np.nan),
            'F_max': r['force']['F_max'],
            'F_interface_max': r['force']['F_interface_max'],
            'Fy_Fx_ratio': r['force']['Fy_Fx_ratio'],
        }
        data.append(row)

    df = pd.DataFrame(data)
    csv_path = output_dir / 'analysis_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

def print_validation_summary(final_result: Dict, conservation: Dict, time_evolution: Dict):
    """Print comprehensive validation summary"""

    print("\n" + "="*80)
    print("FINAL VALIDATION SUMMARY")
    print("="*80)

    T = final_result['temperature']
    f = final_result['interface']
    v = final_result['velocity']
    F = final_result['force']

    # Temperature validation
    print("\n1. TEMPERATURE FIELD")
    print("-" * 80)
    T_error = T['dT_dx_error'] * 100
    T_status = "PASS" if T_error < 5.0 else "FAIL"
    print(f"   Gradient: {T['dT_dx_measured']*1e-6:.3f} K/μm (expected: {config.dT_dx_expected*1e-6:.3f} K/μm)")
    print(f"   Error: {T_error:.2f}%")
    print(f"   Linearity (R²): {T['T_linearity_R2']:.6f}")
    print(f"   Boundary (left): {T['T_left']:.2f} K (expected: {config.T_hot:.2f} K)")
    print(f"   Boundary (right): {T['T_right']:.2f} K (expected: {config.T_cold:.2f} K)")
    print(f"   STATUS: {T_status}")
    if T_status == "FAIL":
        print(f"   DEVIATION: Temperature gradient deviates by {T_error:.2f}%")

    # Interface validation
    print("\n2. INTERFACE POSITION AND PROFILE")
    print("-" * 80)
    f_error = f['interface_y_error']
    f_status = "PASS" if not np.isnan(f_error) and f_error < 2.0 else "FAIL"
    print(f"   Position: {f['interface_y_mean']:.2f} cells (expected: {config.interface_y_cell:.2f} cells)")
    print(f"   Error: {f_error:.2f} cells")
    f_thick = f.get('interface_thickness_cells', np.nan)
    print(f"   Thickness: {f_thick:.2f} cells (expected: ~{config.interface_width:.1f} cells)")
    print(f"   Interface cells: {f['n_interface_cells']}")
    print(f"   STATUS: {f_status}")
    if f_status == "FAIL":
        print(f"   DEVIATION: Interface position off by {f_error:.2f} cells")

    # Velocity validation
    print("\n3. VELOCITY FIELD")
    print("-" * 80)
    v_ratio = v['v_max'] / config.velocity_scale
    v_status = "PASS" if 0.01 <= v_ratio <= 10.0 else "FAIL"
    print(f"   Maximum: {v['v_max']:.4f} m/s")
    v_int = v.get('v_interface_max', np.nan)
    print(f"   Interface max: {v_int:.4f} m/s")
    print(f"   Expected scale: {config.velocity_scale:.4f} m/s")
    print(f"   Ratio (actual/expected): {v_ratio:.3f}")
    print(f"   Flow direction: {'CORRECT (+x)' if v.get('vx_direction_correct', False) else 'INCORRECT'}")
    vx_frac = v.get('vx_positive_fraction', np.nan)
    if not np.isnan(vx_frac):
        print(f"   Positive vx fraction: {vx_frac*100:.1f}%")
    print(f"   STATUS: {v_status}")
    if v_status == "FAIL":
        print(f"   DEVIATION: Velocity magnitude {v_ratio:.2f}x expected scale (should be 0.01-10x)")

    # Force validation
    print("\n4. MARANGONI FORCE")
    print("-" * 80)
    F_int_max = F.get('F_interface_max', 0)
    F_ratio = F_int_max / config.force_expected if config.force_expected > 0 else 0
    F_status = "PASS" if 0.1 <= F_ratio <= 10.0 else "FAIL"
    print(f"   Maximum: {F['F_max']*1e-6:.2f} MN/m³")
    print(f"   Interface max: {F_int_max*1e-6:.2f} MN/m³")
    print(f"   Expected: {config.force_expected*1e-6:.2f} MN/m³")
    print(f"   Ratio (actual/expected): {F_ratio:.3f}")
    Fy_Fx = F.get('Fy_Fx_ratio', np.nan)
    if not np.isnan(Fy_Fx):
        print(f"   Tangentiality (|Fy|/|Fx|): {Fy_Fx:.4f}")
    F_loc = F.get('F_localization_ratio', np.nan)
    if not np.isnan(F_loc) and not np.isinf(F_loc):
        print(f"   Localization ratio: {F_loc:.2f}x")
    print(f"   STATUS: {F_status}")
    if F_status == "FAIL":
        print(f"   DEVIATION: Force magnitude {F_ratio:.2f}x expected (should be 0.1-10x)")

    # Conservation validation
    print("\n5. CONSERVATION LAWS")
    print("-" * 80)
    mass_status = "PASS" if conservation['mass_conserved'] else "FAIL"
    print(f"   Mass change: {conservation['mass_change_percent']:.3f}%")
    print(f"   STATUS: {mass_status}")
    if mass_status == "FAIL":
        print(f"   DEVIATION: Mass changed by {conservation['mass_change_percent']:.3f}% (should be <1%)")

    # Steady state
    print("\n6. STEADY STATE CONVERGENCE")
    print("-" * 80)
    if time_evolution.get('steady_state_reached', False):
        print(f"   Status: REACHED")
        print(f"   Final velocity: {time_evolution['v_final_mean']:.4f} ± {time_evolution['v_final_std']:.4f} m/s")
        print(f"   Variation: {time_evolution['v_final_cv']*100:.2f}%")
    else:
        print(f"   Status: NOT REACHED (still evolving)")

    # Overall assessment
    print("\n" + "="*80)
    all_passed = all([
        T_status == "PASS",
        f_status == "PASS",
        v_status == "PASS",
        F_status == "PASS",
        mass_status == "PASS"
    ])

    if all_passed:
        print("OVERALL ASSESSMENT: ALL CHECKS PASSED")
        print("Physics implementation is CORRECT")
    else:
        print("OVERALL ASSESSMENT: DEVIATIONS DETECTED")
        print("Physics implementation requires attention to failed checks above")
    print("="*80)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Paths
    VTK_DIR = Path("/home/yzk/LBMProject/build/marangoni_benchmark_output")
    OUTPUT_DIR = Path("/home/yzk/LBMProject/analysis/marangoni_benchmark")

    print("="*80)
    print("MARANGONI BENCHMARK VTK ANALYSIS")
    print("="*80)
    print(f"VTK directory: {VTK_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nConfiguration:")
    print(f"  Domain: {config.nx} × {config.ny} × {config.nz} cells")
    print(f"  Resolution: {config.dx*1e6:.1f} μm")
    print(f"  Temperature: {config.T_hot} K → {config.T_cold} K")
    print(f"  Expected gradient: {config.dT_dx_expected*1e-6:.3f} K/μm")
    print(f"  Expected force: {config.force_expected*1e-6:.2f} MN/m³")
    print(f"  Expected velocity scale: {config.velocity_scale:.4f} m/s")
    print("="*80)

    # Run analysis
    analyze_all_vtk_files(VTK_DIR, OUTPUT_DIR)
