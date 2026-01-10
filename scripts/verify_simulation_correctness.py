#!/usr/bin/env python3
"""
VTK Simulation Correctness Validator

Validates LBM thermal solver output after critical bug fixes:
- CUDA error checking
- Evaporation division by zero
- Material validation
- Memory initialization

Checks for NaN/Inf values, energy conservation, mass conservation,
and numerical stability indicators.

Usage:
    python verify_simulation_correctness.py /path/to/vtk_files/
    python verify_simulation_correctness.py /path/to/vtk_files/ --timesteps 0,10,20,30
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from datetime import datetime


# === PARAMETERS ===
TEMPERATURE_MIN = 0.0  # K, absolute minimum physical temperature
TEMPERATURE_MAX = 50000.0  # K, maximum reasonable temperature
FILL_LEVEL_MIN = 0.0  # VOF minimum
FILL_LEVEL_MAX = 1.0  # VOF maximum
MAX_GRADIENT = 1e8  # K/m, threshold for extreme gradients
ENERGY_CHANGE_THRESHOLD = 0.05  # 5% max energy change tolerance


@dataclass
class ValidationResult:
    """Results from a single validation check."""
    check_name: str
    passed: bool
    message: str
    details: Dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class FieldStats:
    """Statistics for a scalar field."""
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    nan_count: int
    inf_count: int
    total_cells: int

    def has_invalid(self) -> bool:
        return self.nan_count > 0 or self.inf_count > 0


class VTKData:
    """Simple VTK data container."""
    def __init__(self):
        self.dimensions = None  # (nx, ny, nz)
        self.origin = None
        self.spacing = None
        self.points = None
        self.fields = {}  # field_name -> np.ndarray
        self.n_cells = 0


def read_legacy_vtk(filename: str) -> VTKData:
    """Read a legacy VTK ASCII file (STRUCTURED_POINTS)."""
    data = VTKData()

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # Parse dimensions
        if line.startswith('DIMENSIONS'):
            parts = line.split()
            data.dimensions = tuple(map(int, parts[1:4]))
            data.n_cells = np.prod(data.dimensions)

        # Parse origin
        elif line.startswith('ORIGIN'):
            parts = line.split()
            data.origin = tuple(map(float, parts[1:4]))

        # Parse spacing
        elif line.startswith('SPACING'):
            parts = line.split()
            data.spacing = tuple(map(float, parts[1:4]))

        # Parse scalar fields
        elif line.startswith('SCALARS'):
            parts = line.split()
            field_name = parts[1]
            # Skip LOOKUP_TABLE line
            i += 1
            # Read data
            i += 1
            field_data = []
            while i < len(lines) and len(field_data) < data.n_cells:
                line_data = lines[i].split()
                if not line_data or line_data[0].startswith(('SCALARS', 'VECTORS', 'FIELD')):
                    i -= 1
                    break
                field_data.extend([float(x) for x in line_data])
                i += 1
            data.fields[field_name] = np.array(field_data[:data.n_cells])

        i += 1

    return data


class SimulationValidator:
    """Validates VTK simulation output for correctness."""

    def __init__(self, vtk_dir: Path, output_dir: Path):
        self.vtk_dir = Path(vtk_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ValidationResult] = []
        self.timestep_data: Dict[int, VTKData] = {}

    def find_vtk_files(self, pattern: str = "*.vtk") -> List[Path]:
        """Find all VTK files matching pattern."""
        files = sorted(self.vtk_dir.glob(pattern))
        if not files:
            files = sorted(self.vtk_dir.glob("*.vtu"))
        return files

    def extract_timestep(self, filename: Path) -> Optional[int]:
        """Extract timestep number from filename."""
        match = re.search(r'(\d+)', filename.stem)
        if match:
            return int(match.group(1))
        return None

    def load_timesteps(self, timesteps: Optional[List[int]] = None) -> None:
        """Load VTK files for specified timesteps."""
        vtk_files = self.find_vtk_files()

        if not vtk_files:
            raise FileNotFoundError(f"No VTK files found in {self.vtk_dir}")

        print(f"Found {len(vtk_files)} VTK files")

        for vtk_file in vtk_files:
            ts = self.extract_timestep(vtk_file)
            if ts is not None:
                if timesteps is None or ts in timesteps:
                    try:
                        self.timestep_data[ts] = read_legacy_vtk(str(vtk_file))
                        print(f"  Loaded timestep {ts}: {vtk_file.name}")
                    except Exception as e:
                        print(f"  Warning: Failed to load {vtk_file.name}: {e}")

        if not self.timestep_data:
            raise ValueError("No valid timesteps could be loaded")

    def compute_field_stats(self, field_data: np.ndarray, field_name: str) -> FieldStats:
        """Compute statistics for a field."""
        total_cells = field_data.size

        # Check for NaN and Inf
        nan_mask = np.isnan(field_data)
        inf_mask = np.isinf(field_data)
        valid_mask = ~(nan_mask | inf_mask)

        nan_count = int(np.sum(nan_mask))
        inf_count = int(np.sum(inf_mask))

        # Compute stats on valid data
        if np.any(valid_mask):
            valid_data = field_data[valid_mask]
            stats = FieldStats(
                min_val=float(np.min(valid_data)),
                max_val=float(np.max(valid_data)),
                mean_val=float(np.mean(valid_data)),
                std_val=float(np.std(valid_data)),
                nan_count=int(nan_count),
                inf_count=int(inf_count),
                total_cells=int(total_cells)
            )
        else:
            stats = FieldStats(
                min_val=np.nan,
                max_val=np.nan,
                mean_val=np.nan,
                std_val=np.nan,
                nan_count=int(nan_count),
                inf_count=int(inf_count),
                total_cells=int(total_cells)
            )

        return stats

    def validate_temperature_field(self, timestep: int) -> ValidationResult:
        """Check temperature field for validity."""
        data = self.timestep_data[timestep]

        # Try to find temperature field
        temp_field = None
        temp_name = None
        for name in ['temperature', 'Temperature', 'T', 'temp']:
            if name in data.fields:
                temp_field = data.fields[name]
                temp_name = name
                break

        if temp_field is None:
            return ValidationResult(
                check_name=f"Temperature Field (t={timestep})",
                passed=False,
                message="Temperature field not found in VTK file",
                details={'available_fields': list(data.fields.keys())}
            )

        stats = self.compute_field_stats(temp_field, temp_name)

        # Check for invalid values
        issues = []
        if stats.has_invalid():
            issues.append(f"{stats.nan_count} NaN, {stats.inf_count} Inf")

        # Check bounds
        if not np.isnan(stats.min_val) and stats.min_val < TEMPERATURE_MIN:
            issues.append(f"min={stats.min_val:.2f}K < {TEMPERATURE_MIN}K")
        if not np.isnan(stats.max_val) and stats.max_val > TEMPERATURE_MAX:
            issues.append(f"max={stats.max_val:.2f}K > {TEMPERATURE_MAX}K")

        passed = len(issues) == 0
        message = "PASS: Valid T field" if passed else f"FAIL: {'; '.join(issues)}"

        return ValidationResult(
            check_name=f"Temperature Field (t={timestep})",
            passed=passed,
            message=message,
            details=asdict(stats)
        )

    def validate_fill_level(self, timestep: int) -> ValidationResult:
        """Check VOF fill level for validity."""
        data = self.timestep_data[timestep]

        # Try to find fill level field
        fill_field = None
        fill_name = None
        for name in ['fill_level', 'FillLevel', 'vof', 'VOF', 'f', 'alpha']:
            if name in data.fields:
                fill_field = data.fields[name]
                fill_name = name
                break

        if fill_field is None:
            return ValidationResult(
                check_name=f"Fill Level (t={timestep})",
                passed=False,
                message="Fill level field not found",
                details={'available_fields': list(data.fields.keys())}
            )

        stats = self.compute_field_stats(fill_field, fill_name)

        # Check for invalid values
        issues = []
        if stats.has_invalid():
            issues.append(f"{stats.nan_count} NaN, {stats.inf_count} Inf")

        # Check bounds (with small tolerance)
        if not np.isnan(stats.min_val) and stats.min_val < FILL_LEVEL_MIN - 1e-10:
            issues.append(f"min={stats.min_val:.6f} < {FILL_LEVEL_MIN}")
        if not np.isnan(stats.max_val) and stats.max_val > FILL_LEVEL_MAX + 1e-10:
            issues.append(f"max={stats.max_val:.6f} > {FILL_LEVEL_MAX}")

        passed = len(issues) == 0
        message = "PASS: Valid fill level" if passed else f"FAIL: {'; '.join(issues)}"

        return ValidationResult(
            check_name=f"Fill Level (t={timestep})",
            passed=passed,
            message=message,
            details=asdict(stats)
        )

    def compute_cell_volume(self, data: VTKData) -> float:
        """Compute cell volume for structured grid."""
        if data.spacing is not None:
            return data.spacing[0] * data.spacing[1] * data.spacing[2]
        return 1.0  # fallback

    def check_energy_conservation(self) -> ValidationResult:
        """Check energy conservation across timesteps."""
        if len(self.timestep_data) < 2:
            return ValidationResult(
                check_name="Energy Conservation",
                passed=True,
                message="SKIP: Need >= 2 timesteps",
                details={}
            )

        timesteps = sorted(self.timestep_data.keys())
        energies = []
        times = []

        for ts in timesteps:
            data = self.timestep_data[ts]

            # Find temperature field
            temp_field = None
            for name in ['temperature', 'Temperature', 'T', 'temp']:
                if name in data.fields:
                    temp_field = data.fields[name]
                    break

            if temp_field is None:
                continue

            # Material properties (typical metal)
            rho = 7000.0  # kg/m³
            cp = 500.0    # J/(kg·K)
            dV = self.compute_cell_volume(data)

            # Compute total energy: E = Σ(ρ·cp·T·dV)
            valid_mask = np.isfinite(temp_field)
            if np.any(valid_mask):
                energy = np.sum(rho * cp * temp_field[valid_mask] * dV)
                energies.append(energy)
                times.append(ts)

        if len(energies) < 2:
            return ValidationResult(
                check_name="Energy Conservation",
                passed=False,
                message="FAIL: Could not compute energy",
                details={}
            )

        energies = np.array(energies)
        times = np.array(times)

        # Compute relative change from initial
        energy_change = np.abs(energies - energies[0]) / (np.abs(energies[0]) + 1e-10)
        max_change = np.max(energy_change)

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(times, energies / 1e6, 'b.-', linewidth=2, markersize=8)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Total Energy (MJ)')
        ax1.set_title('Total Thermal Energy vs Time')
        ax1.grid(True, alpha=0.3)

        ax2.plot(times, energy_change * 100, 'r.-', linewidth=2, markersize=8)
        ax2.axhline(y=ENERGY_CHANGE_THRESHOLD * 100, color='orange',
                    linestyle='--', label=f'{ENERGY_CHANGE_THRESHOLD*100}% threshold')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Relative Energy Change (%)')
        ax2.set_title('Energy Conservation Check')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.output_dir / 'energy_conservation.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        passed = max_change <= ENERGY_CHANGE_THRESHOLD
        message = (f"PASS: Max change {max_change*100:.2f}% < {ENERGY_CHANGE_THRESHOLD*100}%"
                   if passed else
                   f"FAIL: Max change {max_change*100:.2f}% > {ENERGY_CHANGE_THRESHOLD*100}%")

        return ValidationResult(
            check_name="Energy Conservation",
            passed=passed,
            message=message,
            details={
                'timesteps': times.tolist(),
                'energies_J': energies.tolist(),
                'max_relative_change': float(max_change),
                'plot_saved': str(plot_file)
            }
        )

    def check_mass_conservation(self) -> ValidationResult:
        """Check mass conservation via VOF fill level."""
        if len(self.timestep_data) < 2:
            return ValidationResult(
                check_name="Mass Conservation",
                passed=True,
                message="SKIP: Need >= 2 timesteps",
                details={}
            )

        timesteps = sorted(self.timestep_data.keys())
        masses = []
        times = []

        for ts in timesteps:
            data = self.timestep_data[ts]

            # Find fill level field
            fill_field = None
            for name in ['fill_level', 'FillLevel', 'vof', 'VOF', 'f', 'alpha']:
                if name in data.fields:
                    fill_field = data.fields[name]
                    break

            if fill_field is None:
                continue

            # Material properties
            rho = 7000.0  # kg/m³
            dV = self.compute_cell_volume(data)

            # Compute total mass: M = Σ(f·ρ·dV)
            valid_mask = np.isfinite(fill_field)
            if np.any(valid_mask):
                mass = np.sum(fill_field[valid_mask] * rho * dV)
                masses.append(mass)
                times.append(ts)

        if len(masses) < 2:
            return ValidationResult(
                check_name="Mass Conservation",
                passed=False,
                message="FAIL: Could not compute mass",
                details={}
            )

        masses = np.array(masses)
        times = np.array(times)

        # Compute relative change
        mass_change = np.abs(masses - masses[0]) / (np.abs(masses[0]) + 1e-10)
        max_change = np.max(mass_change)

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(times, masses, 'b.-', linewidth=2, markersize=8)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Total Mass (kg)')
        ax1.set_title('Total Mass vs Time')
        ax1.grid(True, alpha=0.3)

        ax2.plot(times, mass_change * 100, 'r.-', linewidth=2, markersize=8)
        ax2.axhline(y=1.0, color='orange', linestyle='--', label='1% threshold')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Relative Mass Change (%)')
        ax2.set_title('Mass Conservation Check')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.output_dir / 'mass_conservation.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        # Stricter threshold for mass (1%)
        mass_threshold = 0.01
        passed = max_change <= mass_threshold
        message = (f"PASS: Max change {max_change*100:.2f}% < {mass_threshold*100}%"
                   if passed else
                   f"FAIL: Max change {max_change*100:.2f}% > {mass_threshold*100}%")

        return ValidationResult(
            check_name="Mass Conservation",
            passed=passed,
            message=message,
            details={
                'timesteps': times.tolist(),
                'masses_kg': masses.tolist(),
                'max_relative_change': float(max_change),
                'plot_saved': str(plot_file)
            }
        )

    def check_temperature_gradients(self, timestep: int) -> ValidationResult:
        """Check for extreme temperature gradients indicating instability."""
        data = self.timestep_data[timestep]

        # Find temperature field
        temp_field = None
        for name in ['temperature', 'Temperature', 'T', 'temp']:
            if name in data.fields:
                temp_field = data.fields[name]
                break

        if temp_field is None:
            return ValidationResult(
                check_name=f"Temperature Gradients (t={timestep})",
                passed=False,
                message="Temperature field not found",
                details={}
            )

        if data.dimensions is None or data.spacing is None:
            return ValidationResult(
                check_name=f"Temperature Gradients (t={timestep})",
                passed=True,
                message="SKIP: Cannot compute gradients (no grid info)",
                details={}
            )

        try:
            # Reshape to 3D grid
            nx, ny, nz = data.dimensions
            T = temp_field.reshape((nx, ny, nz))

            dx, dy, dz = data.spacing

            # Compute central differences for interior points
            grad_x = np.zeros_like(T)
            grad_y = np.zeros_like(T)
            grad_z = np.zeros_like(T)

            grad_x[1:-1, :, :] = (T[2:, :, :] - T[:-2, :, :]) / (2 * dx)
            grad_y[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / (2 * dy)
            grad_z[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / (2 * dz)

            # Gradient magnitude
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2).flatten()

            # Check for extreme gradients
            valid_mask = np.isfinite(grad_magnitude) & (grad_magnitude > 0)
            if np.any(valid_mask):
                max_grad = np.max(grad_magnitude[valid_mask])
                mean_grad = np.mean(grad_magnitude[valid_mask])
                extreme_count = np.sum(grad_magnitude[valid_mask] > MAX_GRADIENT)

                passed = extreme_count == 0
                message = (f"PASS: No extreme grads (max={max_grad:.2e} K/m)"
                           if passed else
                           f"FAIL: {extreme_count} cells with |∇T| > {MAX_GRADIENT:.2e} K/m")

                # Histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                valid_grads = grad_magnitude[valid_mask]

                ax.hist(np.log10(valid_grads), bins=50, edgecolor='black', alpha=0.7)
                ax.axvline(x=np.log10(MAX_GRADIENT), color='red',
                          linestyle='--', linewidth=2, label=f'Threshold: {MAX_GRADIENT:.2e} K/m')
                ax.set_xlabel('log₁₀(|∇T|) [K/m]')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Temperature Gradient Distribution (t={timestep})')
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plot_file = self.output_dir / f'gradients_t{timestep}.png'
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()

                return ValidationResult(
                    check_name=f"Temperature Gradients (t={timestep})",
                    passed=passed,
                    message=message,
                    details={
                        'max_gradient_K_per_m': float(max_grad),
                        'mean_gradient_K_per_m': float(mean_grad),
                        'extreme_gradient_count': int(extreme_count),
                        'threshold_K_per_m': MAX_GRADIENT,
                        'plot_saved': str(plot_file)
                    }
                )
            else:
                return ValidationResult(
                    check_name=f"Temperature Gradients (t={timestep})",
                    passed=False,
                    message="FAIL: All gradients invalid (NaN/Inf)",
                    details={}
                )

        except Exception as e:
            return ValidationResult(
                check_name=f"Temperature Gradients (t={timestep})",
                passed=False,
                message=f"FAIL: Could not compute gradients: {str(e)}",
                details={'error': str(e)}
            )

    def run_all_checks(self) -> None:
        """Run all validation checks."""
        print("\n" + "="*70)
        print("VTK SIMULATION CORRECTNESS VALIDATION")
        print("="*70)
        print(f"VTK Directory: {self.vtk_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Timesteps loaded: {sorted(self.timestep_data.keys())}")
        print("="*70 + "\n")

        # Check each timestep individually
        for ts in sorted(self.timestep_data.keys()):
            print(f"\n--- Timestep {ts} ---")

            # Temperature field validation
            result = self.validate_temperature_field(ts)
            self.results.append(result)
            self._print_result(result)

            # Fill level validation
            result = self.validate_fill_level(ts)
            self.results.append(result)
            self._print_result(result)

            # Gradient check
            result = self.check_temperature_gradients(ts)
            self.results.append(result)
            self._print_result(result)

        # Conservation checks across timesteps
        print("\n--- Conservation Checks ---")

        result = self.check_energy_conservation()
        self.results.append(result)
        self._print_result(result)

        result = self.check_mass_conservation()
        self.results.append(result)
        self._print_result(result)

        # Summary
        self._print_summary()

    def _print_result(self, result: ValidationResult) -> None:
        """Print a validation result."""
        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"  {status} {result.check_name}: {result.message}")

    def _print_summary(self) -> None:
        """Print overall summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total checks: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed checks:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.check_name}: {result.message}")

        print("="*70)

        if failed == 0:
            print("\n[PASS] ALL CHECKS PASSED - Simulation appears correct")
        else:
            print(f"\n[FAIL] {failed} CHECK(S) FAILED - Review output for issues")

    def save_results(self) -> None:
        """Save results to JSON file."""
        output_file = self.output_dir / 'validation_results.json'

        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'vtk_directory': str(self.vtk_dir),
            'timesteps_analyzed': sorted(self.timestep_data.keys()),
            'parameters': {
                'temperature_min_K': TEMPERATURE_MIN,
                'temperature_max_K': TEMPERATURE_MAX,
                'fill_level_min': FILL_LEVEL_MIN,
                'fill_level_max': FILL_LEVEL_MAX,
                'max_gradient_K_per_m': MAX_GRADIENT,
                'energy_change_threshold': ENERGY_CHANGE_THRESHOLD
            },
            'results': [
                {
                    'check_name': r.check_name,
                    'passed': r.passed,
                    'message': r.message,
                    'details': r.details
                }
                for r in self.results
            ],
            'summary': {
                'total_checks': len(self.results),
                'passed': sum(1 for r in self.results if r.passed),
                'failed': sum(1 for r in self.results if not r.passed)
            }
        }

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate VTK simulation output for correctness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all timesteps in directory
  python verify_simulation_correctness.py /path/to/vtk_files/

  # Analyze specific timesteps
  python verify_simulation_correctness.py /path/to/vtk_files/ --timesteps 0,10,20,30

  # Specify custom output directory
  python verify_simulation_correctness.py /path/to/vtk_files/ --output ./my_results/
        """
    )

    parser.add_argument('vtk_dir', type=str,
                        help='Directory containing VTK files')
    parser.add_argument('--timesteps', type=str, default=None,
                        help='Comma-separated list of timesteps to analyze (default: all)')
    parser.add_argument('--output', type=str,
                        default='/home/yzk/LBMProject/scripts/verification_results',
                        help='Output directory for results and plots')

    args = parser.parse_args()

    # Parse timesteps
    timesteps = None
    if args.timesteps:
        try:
            timesteps = [int(ts.strip()) for ts in args.timesteps.split(',')]
        except ValueError:
            print(f"Error: Invalid timesteps format: {args.timesteps}")
            sys.exit(1)

    # Create validator
    try:
        validator = SimulationValidator(
            vtk_dir=Path(args.vtk_dir),
            output_dir=Path(args.output)
        )

        # Load data
        print("Loading VTK files...")
        validator.load_timesteps(timesteps)

        # Run validation
        validator.run_all_checks()

        # Save results
        validator.save_results()

        # Exit with appropriate code
        failed_count = sum(1 for r in validator.results if not r.passed)
        sys.exit(1 if failed_count > 0 else 0)

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
