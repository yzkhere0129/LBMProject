#!/usr/bin/env python3
"""
Marangoni Benchmark VTK Analysis Script (Simplified)

Analyzes available fields from Marangoni benchmark VTK output.
Works with Temperature and Velocity fields only.

Project: /home/yzk/LBMProject
VTK files: /home/yzk/LBMProject/build/marangoni_benchmark_output/

Author: VTK Analysis Specialist
Date: 2025-12-03
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration parameters for Marangoni benchmark"""
    # Domain
    nx: int = 200
    ny: int = 100
    nz: int = 2
    dx: float = 2.0e-6  # meters
    dt: float = 1.0e-9  # seconds (output every 1000 steps = 1 μs)

    # Temperature (based on test configuration)
    T_hot: float = 2000.0   # K
    T_cold: float = 1800.0  # K

    # Material (Ti-6Al-4V)
    rho: float = 4420.0         # kg/m³
    mu: float = 4.0e-3          # Pa·s
    dsigma_dT: float = -0.26e-3 # N/(m·K)

    @property
    def dT_dx_expected(self) -> float:
        """Expected temperature gradient [K/m]"""
        Lx = (self.nx - 1) * self.dx
        return (self.T_hot - self.T_cold) / Lx

    @property
    def velocity_scale(self) -> float:
        """Characteristic Marangoni velocity [m/s]"""
        return abs(self.dsigma_dT) * (self.T_hot - self.T_cold) / self.mu

config = BenchmarkConfig()

# ============================================================================
# VTK LOADING
# ============================================================================

def load_vtk_file(filepath: Path):
    """Load VTK file"""
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    return reader.GetOutput()

def extract_fields(mesh):
    """Extract fields from VTK mesh"""
    fields = {}
    dims = mesh.GetDimensions()
    nx, ny, nz = dims

    point_data = mesh.GetPointData()
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        vtk_array = point_data.GetArray(i)
        np_array = vtk_to_numpy(vtk_array)

        if vtk_array.GetNumberOfComponents() == 3:
            fields[array_name] = np_array.reshape((nx, ny, nz, 3), order='F')
        else:
            fields[array_name] = np_array.reshape((nx, ny, nz), order='F')

    return fields, (nx, ny, nz)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_temperature(T: np.ndarray, dims: Tuple[int, int, int]) -> Dict:
    """Analyze temperature field"""
    nx, ny, nz = dims
    results = {}

    # Statistics
    results['T_min'] = np.min(T)
    results['T_max'] = np.max(T)
    results['T_mean'] = np.mean(T)
    results['T_std'] = np.std(T)

    # Boundary values
    results['T_left'] = np.mean(T[0, :, :])
    results['T_right'] = np.mean(T[-1, :, :])

    # Temperature profile along X (averaged over Y, Z)
    T_avg_x = np.mean(T, axis=(1, 2))
    x_cells = np.arange(nx)

    # Linear fit
    coeffs = np.polyfit(x_cells, T_avg_x, 1)
    dT_dx_measured = coeffs[0] * nx / ((nx - 1) * config.dx)

    results['dT_dx_measured'] = dT_dx_measured
    results['dT_dx_expected'] = config.dT_dx_expected
    results['dT_dx_error'] = abs(dT_dx_measured - config.dT_dx_expected) / abs(config.dT_dx_expected)

    # Linearity R²
    T_fit = np.polyval(coeffs, x_cells)
    ss_res = np.sum((T_avg_x - T_fit)**2)
    ss_tot = np.sum((T_avg_x - np.mean(T_avg_x))**2)
    results['T_linearity_R2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    # For plotting
    results['x_cells'] = x_cells
    results['T_profile'] = T_avg_x
    results['T_fit'] = T_fit

    return results

def analyze_velocity(v: np.ndarray, dims: Tuple[int, int, int]) -> Dict:
    """Analyze velocity field"""
    nx, ny, nz = dims
    results = {}

    # Extract components
    vx = v[:, :, :, 0]
    vy = v[:, :, :, 1]
    vz = v[:, :, :, 2]

    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Statistics
    results['v_max'] = np.max(v_mag)
    results['v_mean'] = np.mean(v_mag)
    results['vx_max'] = np.max(np.abs(vx))
    results['vy_max'] = np.max(np.abs(vy))
    results['vz_max'] = np.max(np.abs(vz))

    # Mean vx (should be positive for hot-to-cold flow)
    results['vx_mean'] = np.mean(vx)

    # Velocity profile along Y at center X
    x_center = nx // 2
    v_profile_y = np.mean(v_mag[x_center, :, :], axis=1)
    vx_profile_y = np.mean(vx[x_center, :, :], axis=1)

    results['y_cells'] = np.arange(ny)
    results['v_profile_y'] = v_profile_y
    results['vx_profile_y'] = vx_profile_y

    # Velocity at different heights
    y_bottom = ny // 4
    y_middle = ny // 2
    y_top = 3 * ny // 4

    results['v_bottom'] = np.mean(v_mag[:, y_bottom, :])
    results['v_middle'] = np.mean(v_mag[:, y_middle, :])
    results['v_top'] = np.mean(v_mag[:, y_top, :])

    results['v_scale_expected'] = config.velocity_scale

    return results

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_all_files(vtk_dir: Path, output_dir: Path):
    """Main analysis pipeline"""

    vtk_files = sorted(vtk_dir.glob("marangoni_*.vtk"))
    print(f"Found {len(vtk_files)} VTK files in {vtk_dir}\n")

    if len(vtk_files) == 0:
        print("ERROR: No VTK files found!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, filepath in enumerate(vtk_files):
        # Extract timestep from filename
        timestep = int(filepath.stem.split('_')[1])
        time = timestep * config.dt

        print(f"[{i+1}/{len(vtk_files)}] {filepath.name} (t = {time*1e6:.1f} μs)")

        mesh = load_vtk_file(filepath)
        fields, dims = extract_fields(mesh)

        # Check available fields
        if 'Temperature' not in fields or 'Velocity' not in fields:
            print(f"  WARNING: Missing required fields")
            continue

        T = fields['Temperature']
        v = fields['Velocity']

        T_results = analyze_temperature(T, dims)
        v_results = analyze_velocity(v, dims)

        result = {
            'timestep': timestep,
            'time': time,
            'temperature': T_results,
            'velocity': v_results
        }
        all_results.append(result)

        # Print summary
        print(f"  T: {T_results['T_min']:.1f} - {T_results['T_max']:.1f} K")
        print(f"  dT/dx: {T_results['dT_dx_measured']*1e-6:.3f} K/μm (expected: {config.dT_dx_expected*1e-6:.3f})")
        print(f"  v_max: {v_results['v_max']:.4f} m/s")
        print(f"  vx_mean: {v_results['vx_mean']:.4f} m/s")
        print()

    if len(all_results) == 0:
        print("ERROR: No valid results!")
        return

    # Generate plots
    print("="*80)
    print("GENERATING PLOTS")
    print("="*80)
    create_plots(all_results, output_dir)

    # Save CSV
    save_csv(all_results, output_dir)

    # Print summary
    print_summary(all_results)

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_plots(all_results: List[Dict], output_dir: Path):
    """Create analysis plots"""

    final = all_results[-1]
    T_r = final['temperature']
    v_r = final['velocity']

    # Extract time series
    times_us = np.array([r['time'] for r in all_results]) * 1e6
    v_max_series = [r['velocity']['v_max'] for r in all_results]
    T_grad_series = [r['temperature']['dT_dx_measured'] for r in all_results]

    # Create figure
    fig = plt.figure(figsize=(18, 10))

    # 1. Temperature profile
    ax1 = plt.subplot(2, 3, 1)
    x_um = T_r['x_cells'] * config.dx * 1e6
    ax1.plot(x_um, T_r['T_profile'], 'b-', linewidth=2, label='Measured')
    ax1.plot(x_um, T_r['T_fit'], 'r--', linewidth=2, label='Linear fit')
    ax1.axhline(config.T_hot, color='orange', linestyle=':', alpha=0.7)
    ax1.axhline(config.T_cold, color='cyan', linestyle=':', alpha=0.7)
    ax1.set_xlabel('X position [μm]', fontsize=12)
    ax1.set_ylabel('Temperature [K]', fontsize=12)
    ax1.set_title(f'Temperature Profile (R² = {T_r["T_linearity_R2"]:.6f})', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Velocity profile along Y
    ax2 = plt.subplot(2, 3, 2)
    y_um = v_r['y_cells'] * config.dx * 1e6
    ax2.plot(v_r['v_profile_y'], y_um, 'b-', linewidth=2, label='|v|')
    ax2.plot(v_r['vx_profile_y'], y_um, 'r--', linewidth=2, label='vx')
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Velocity [m/s]', fontsize=12)
    ax2.set_ylabel('Y position [μm]', fontsize=12)
    ax2.set_title(f'Velocity Profile at X=center', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Velocity vs time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times_us, v_max_series, 'b-', linewidth=2, marker='o', markersize=4)
    ax3.axhline(config.velocity_scale, color='green', linestyle='--',
                label=f'Expected scale = {config.velocity_scale:.2f} m/s')
    ax3.set_xlabel('Time [μs]', fontsize=12)
    ax3.set_ylabel('Maximum Velocity [m/s]', fontsize=12)
    ax3.set_title('Velocity Evolution', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Temperature gradient vs time
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(times_us, np.array(T_grad_series) * 1e-6, 'b-', linewidth=2, marker='o', markersize=4)
    ax4.axhline(config.dT_dx_expected * 1e-6, color='red', linestyle='--',
                label=f'Expected = {config.dT_dx_expected*1e-6:.3f} K/μm')
    ax4.set_xlabel('Time [μs]', fontsize=12)
    ax4.set_ylabel('Temperature Gradient [K/μm]', fontsize=12)
    ax4.set_title('Temperature Gradient Evolution', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Boundary temperatures
    ax5 = plt.subplot(2, 3, 5)
    T_left_series = [r['temperature']['T_left'] for r in all_results]
    T_right_series = [r['temperature']['T_right'] for r in all_results]
    ax5.plot(times_us, T_left_series, 'r-', linewidth=2, label='Left wall', marker='o', markersize=4)
    ax5.plot(times_us, T_right_series, 'b-', linewidth=2, label='Right wall', marker='o', markersize=4)
    ax5.axhline(config.T_hot, color='red', linestyle='--', alpha=0.5)
    ax5.axhline(config.T_cold, color='blue', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time [μs]', fontsize=12)
    ax5.set_ylabel('Temperature [K]', fontsize=12)
    ax5.set_title('Boundary Conditions', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary = "VALIDATION SUMMARY\n" + "="*50 + "\n\n"

    T_error = T_r['dT_dx_error'] * 100
    T_status = "PASS" if T_error < 5.0 else "FAIL"
    summary += f"1. Temperature Gradient:\n"
    summary += f"   Status: {T_status}\n"
    summary += f"   Measured: {T_r['dT_dx_measured']*1e-6:.3f} K/μm\n"
    summary += f"   Expected: {config.dT_dx_expected*1e-6:.3f} K/μm\n"
    summary += f"   Error: {T_error:.2f}%\n"
    summary += f"   Linearity (R²): {T_r['T_linearity_R2']:.6f}\n\n"

    v_ratio = v_r['v_max'] / config.velocity_scale
    v_status = "PASS" if 0.01 <= v_ratio <= 10.0 else "FAIL"
    summary += f"2. Velocity Magnitude:\n"
    summary += f"   Status: {v_status}\n"
    summary += f"   Max: {v_r['v_max']:.4f} m/s\n"
    summary += f"   Expected scale: {config.velocity_scale:.4f} m/s\n"
    summary += f"   Ratio: {v_ratio:.3f}\n\n"

    vx_status = "PASS" if v_r['vx_mean'] > 0 else "FAIL"
    summary += f"3. Flow Direction:\n"
    summary += f"   Status: {vx_status}\n"
    summary += f"   Mean vx: {v_r['vx_mean']:.4f} m/s\n"
    summary += f"   (Should be positive: hot→cold)\n\n"

    summary += f"4. Configuration:\n"
    summary += f"   Domain: {config.nx}×{config.ny}×{config.nz}\n"
    summary += f"   Resolution: {config.dx*1e6:.1f} μm\n"
    summary += f"   T: {config.T_hot}K → {config.T_cold}K\n"

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    png_path = output_dir / 'marangoni_analysis.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {png_path}")
    plt.close()

def save_csv(all_results: List[Dict], output_dir: Path):
    """Save results to CSV"""
    data = []
    for r in all_results:
        row = {
            'timestep': r['timestep'],
            'time_us': r['time'] * 1e6,
            'T_min': r['temperature']['T_min'],
            'T_max': r['temperature']['T_max'],
            'T_mean': r['temperature']['T_mean'],
            'T_left': r['temperature']['T_left'],
            'T_right': r['temperature']['T_right'],
            'dT_dx_measured': r['temperature']['dT_dx_measured'],
            'T_linearity_R2': r['temperature']['T_linearity_R2'],
            'v_max': r['velocity']['v_max'],
            'v_mean': r['velocity']['v_mean'],
            'vx_mean': r['velocity']['vx_mean'],
            'vx_max': r['velocity']['vx_max'],
            'vy_max': r['velocity']['vy_max'],
            'vz_max': r['velocity']['vz_max'],
        }
        data.append(row)

    df = pd.DataFrame(data)
    csv_path = output_dir / 'marangoni_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

def print_summary(all_results: List[Dict]):
    """Print validation summary"""
    print("\n" + "="*80)
    print("FINAL VALIDATION SUMMARY")
    print("="*80)

    final = all_results[-1]
    T = final['temperature']
    v = final['velocity']

    print("\n1. TEMPERATURE FIELD")
    print("-" * 80)
    T_error = T['dT_dx_error'] * 100
    T_status = "PASS" if T_error < 5.0 else "FAIL"
    print(f"   Gradient: {T['dT_dx_measured']*1e-6:.3f} K/μm (expected: {config.dT_dx_expected*1e-6:.3f} K/μm)")
    print(f"   Error: {T_error:.2f}%")
    print(f"   Linearity (R²): {T['T_linearity_R2']:.6f}")
    print(f"   Left boundary: {T['T_left']:.2f} K (expected: {config.T_hot:.2f} K)")
    print(f"   Right boundary: {T['T_right']:.2f} K (expected: {config.T_cold:.2f} K)")
    print(f"   STATUS: {T_status}")

    print("\n2. VELOCITY FIELD")
    print("-" * 80)
    v_ratio = v['v_max'] / config.velocity_scale
    v_status = "PASS" if 0.01 <= v_ratio <= 10.0 else "FAIL"
    print(f"   Maximum: {v['v_max']:.4f} m/s")
    print(f"   Expected scale: {config.velocity_scale:.4f} m/s")
    print(f"   Ratio (actual/expected): {v_ratio:.3f}")
    print(f"   Mean vx: {v['vx_mean']:.4f} m/s")
    print(f"   STATUS: {v_status}")

    print("\n3. FLOW DIRECTION")
    print("-" * 80)
    vx_status = "PASS" if v['vx_mean'] > 0 else "FAIL"
    print(f"   Mean vx: {v['vx_mean']:.4f} m/s")
    print(f"   Expected: Positive (hot→cold flow)")
    print(f"   STATUS: {vx_status}")

    print("\n" + "="*80)
    all_passed = (T_status == "PASS" and v_status == "PASS" and vx_status == "PASS")
    if all_passed:
        print("OVERALL: ALL CHECKS PASSED")
    else:
        print("OVERALL: DEVIATIONS DETECTED")
    print("="*80 + "\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
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
    print(f"  Expected dT/dx: {config.dT_dx_expected*1e-6:.3f} K/μm")
    print(f"  Expected v_scale: {config.velocity_scale:.4f} m/s")
    print("="*80 + "\n")

    analyze_all_files(VTK_DIR, OUTPUT_DIR)
