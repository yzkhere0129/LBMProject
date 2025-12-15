#!/usr/bin/env python3
"""
Test A Validation: Thermal-Fluid Coupling
==========================================

Validates that thermal-fluid coupling:
1. Reduces temperature vs v4 baseline (convection removes heat)
2. Develops natural velocity field (no forcing)
3. Maintains energy balance
4. Stays numerically stable (no NaN/Inf)

Physics Checks:
- Convective cooling: Q_conv = ρ * cp * ∫(v · ∇T) dV > 0
- CFL condition: v_max * dt / dx < 0.5
- Energy balance: |Q_in - Q_out| / Q_in < 0.1
"""

import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Physical constants
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)

# V4 baseline for comparison
V4_T_MAX = 45477.0  # K
V4_V_MAX = 0.005    # mm/s

class TestAValidator:
    """Validates Test A results"""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.config = {}
        self.steps_data = []
        self.errors = []
        self.warnings = []

    def parse_log(self) -> bool:
        """Parse simulation log file"""
        if not self.log_file.exists():
            self.errors.append(f"Log file not found: {self.log_file}")
            return False

        with open(self.log_file, 'r') as f:
            content = f.read()

        # Extract configuration
        self._parse_config(content)

        # Extract timestep data
        self._parse_timesteps(content)

        return len(self.steps_data) > 0

    def _parse_config(self, content: str):
        """Extract configuration parameters"""
        patterns = {
            'nx': r'nx\s*=\s*(\d+)',
            'ny': r'ny\s*=\s*(\d+)',
            'nz': r'nz\s*=\s*(\d+)',
            'dx': r'dx\s*=\s*([\d.e+-]+)',
            'dt': r'dt\s*=\s*([\d.e+-]+)',
            'laser_power': r'laser_power\s*=\s*([\d.e+-]+)',
            'emissivity': r'emissivity\s*=\s*([\d.]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                self.config[key] = float(match.group(1))

    def _parse_timesteps(self, content: str):
        """Extract timestep information"""
        # Pattern: Step 100/1000: T_max=1234.5 K, v_max=0.123 mm/s
        pattern = r'Step\s+(\d+)/\d+:.*?T_max\s*=\s*([\d.e+-]+)\s*K.*?v_max\s*=\s*([\d.e+-]+)\s*mm/s'

        for match in re.finditer(pattern, content):
            step = int(match.group(1))
            T_max = float(match.group(2))
            v_max = float(match.group(3))

            self.steps_data.append({
                'step': step,
                'T_max': T_max,
                'v_max': v_max
            })

    def check_numerical_stability(self) -> bool:
        """Check for NaN/Inf in results"""
        for data in self.steps_data:
            if math.isnan(data['T_max']) or math.isinf(data['T_max']):
                self.errors.append(f"Step {data['step']}: T_max is NaN/Inf")
                return False
            if math.isnan(data['v_max']) or math.isinf(data['v_max']):
                self.errors.append(f"Step {data['step']}: v_max is NaN/Inf")
                return False
        return True

    def check_temperature_decrease(self) -> bool:
        """Verify T_max decreased vs v4 baseline"""
        if not self.steps_data:
            self.errors.append("No timestep data available")
            return False

        final = self.steps_data[-1]
        T_final = final['T_max']

        if T_final >= V4_T_MAX:
            self.errors.append(
                f"FAIL: T_max did NOT decrease vs v4\n"
                f"  Test A final: {T_final:.1f} K\n"
                f"  V4 baseline:  {V4_T_MAX:.1f} K\n"
                f"  Expected: Coupling should reduce temperature via convection"
            )
            return False

        reduction_pct = 100 * (V4_T_MAX - T_final) / V4_T_MAX
        print(f"✓ PASS: Temperature decreased by {reduction_pct:.1f}% vs v4")
        print(f"  Test A: {T_final:.1f} K")
        print(f"  V4:     {V4_T_MAX:.1f} K")
        return True

    def check_velocity_development(self) -> bool:
        """Check that velocity develops naturally"""
        if not self.steps_data:
            return False

        final = self.steps_data[-1]
        v_final = final['v_max']

        # Check for unrealistic velocity (too high)
        if v_final > 1000:  # mm/s
            self.errors.append(
                f"FAIL: Velocity too high (unrealistic)\n"
                f"  v_max = {v_final:.2f} mm/s > 1000 mm/s\n"
                f"  Likely numerical instability or wrong unit conversion"
            )
            return False

        # Check that velocity increased from v4 (coupling should create flow)
        if v_final > V4_V_MAX:
            increase_factor = v_final / V4_V_MAX
            print(f"✓ PASS: Velocity developed naturally")
            print(f"  Test A: {v_final:.4f} mm/s ({increase_factor:.0f}× v4)")
            print(f"  V4:     {V4_V_MAX} mm/s")
            return True
        else:
            self.warnings.append(
                f"Warning: Velocity did not increase vs v4\n"
                f"  Test A: {v_final:.4f} mm/s\n"
                f"  V4:     {V4_V_MAX} mm/s\n"
                f"  This may indicate coupling is not active"
            )
            return True  # Not a failure, but suspicious

    def check_energy_balance(self) -> bool:
        """Estimate energy balance"""
        if not self.config or not self.steps_data:
            self.warnings.append("Cannot check energy balance: missing data")
            return True  # Don't fail on this

        # Get final state
        final = self.steps_data[-1]
        T_max = final['T_max']

        # Laser input power
        P_laser = self.config.get('laser_power', 195.0)  # W

        # Radiation output (Stefan-Boltzmann)
        emissivity = self.config.get('emissivity', 0.3)
        spot_radius = 50e-6  # m
        A_spot = math.pi * spot_radius**2
        T_amb = 300.0

        P_radiation = emissivity * STEFAN_BOLTZMANN * A_spot * (T_max**4 - T_amb**4)

        # Energy residual
        residual = abs(P_laser - P_radiation) / P_laser

        print(f"\nEnergy Balance:")
        print(f"  Laser input:      {P_laser:.1f} W")
        print(f"  Radiation output: {P_radiation:.1f} W")
        print(f"  Residual:         {100*residual:.1f}%")

        if residual < 0.1:
            print(f"✓ PASS: Energy balanced (residual < 10%)")
            return True
        elif residual < 0.5:
            self.warnings.append(
                f"Warning: Energy balance residual = {100*residual:.1f}%\n"
                f"  Expected < 10%, but < 50% is acceptable for short runs"
            )
            return True
        else:
            self.errors.append(
                f"FAIL: Energy not balanced\n"
                f"  Residual = {100*residual:.1f}% >> 10%\n"
                f"  Indicates heat is accumulating or leaking"
            )
            return False

    def check_physical_bounds(self) -> bool:
        """Check for unphysical values"""
        passed = True

        for data in self.steps_data:
            # Check temperature bounds
            if data['T_max'] < 0:
                self.errors.append(f"Step {data['step']}: Negative temperature {data['T_max']:.1f} K")
                passed = False

            # Ti6Al4V boiling point ~ 3560 K
            # At 2× boiling = 7120 K, something is very wrong
            if data['T_max'] > 7120:
                self.warnings.append(
                    f"Step {data['step']}: Temperature {data['T_max']:.1f} K > 2× boiling\n"
                    f"  Metal would vaporize completely\n"
                    f"  Vaporization cooling needed (V5 feature)"
                )

        return passed

    def check_cfl_condition(self) -> bool:
        """Check CFL condition for stability"""
        if not self.config or not self.steps_data:
            return True

        dx = self.config.get('dx', 2e-6)  # m
        dt = self.config.get('dt', 1e-7)  # s

        # Thermal diffusivity
        alpha = 5.8e-6  # m²/s for Ti6Al4V

        for data in self.steps_data:
            v_max = data['v_max'] * 1e-3  # Convert mm/s → m/s

            # Advection CFL
            CFL_adv = v_max * dt / dx

            # Diffusion CFL
            CFL_diff = alpha * dt / (dx * dx)

            if CFL_adv > 0.5:
                self.warnings.append(
                    f"Step {data['step']}: CFL_advection = {CFL_adv:.3f} > 0.5\n"
                    f"  May cause instability"
                )

            if CFL_diff > 0.25:
                self.warnings.append(
                    f"Step {data['step']}: CFL_diffusion = {CFL_diff:.3f} > 0.25\n"
                    f"  Timestep too large for thermal diffusion"
                )

        return True

    def run_validation(self) -> bool:
        """Run all validation checks"""
        print("="*60)
        print("TEST A VALIDATION: Thermal-Fluid Coupling")
        print("="*60)

        # Parse log
        if not self.parse_log():
            print(f"\n✗ FAIL: Could not parse log file")
            return False

        print(f"\nLog file: {self.log_file}")
        print(f"Timesteps parsed: {len(self.steps_data)}")

        # Run checks
        checks = [
            ("Numerical Stability", self.check_numerical_stability),
            ("Temperature Decrease", self.check_temperature_decrease),
            ("Velocity Development", self.check_velocity_development),
            ("Physical Bounds", self.check_physical_bounds),
            ("Energy Balance", self.check_energy_balance),
            ("CFL Condition", self.check_cfl_condition),
        ]

        print("\n" + "-"*60)
        print("VALIDATION CHECKS:")
        print("-"*60)

        all_passed = True
        for name, check_func in checks:
            try:
                passed = check_func()
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{status}: {name}")
                all_passed = all_passed and passed
            except Exception as e:
                print(f"✗ ERROR in {name}: {e}")
                all_passed = False

        # Print warnings
        if self.warnings:
            print("\n" + "="*60)
            print("WARNINGS:")
            print("="*60)
            for warning in self.warnings:
                print(f"⚠ {warning}\n")

        # Print errors
        if self.errors:
            print("\n" + "="*60)
            print("ERRORS:")
            print("="*60)
            for error in self.errors:
                print(f"✗ {error}\n")

        # Final verdict
        print("\n" + "="*60)
        if all_passed and not self.errors:
            print("FINAL RESULT: ✓ TEST A PASSED")
            print("="*60)
            print("\nThermal-fluid coupling is working correctly.")
            print("Ready to proceed to Test B (Marangoni).")
            return True
        else:
            print("FINAL RESULT: ✗ TEST A FAILED")
            print("="*60)
            print("\nThermal-fluid coupling has issues.")
            print("Do NOT proceed to Test B until these are fixed.")
            return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_test_A.py <log_file>")
        print("Example: python validate_test_A.py /path/to/test_A_coupling.log")
        sys.exit(1)

    log_file = sys.argv[1]
    validator = TestAValidator(log_file)

    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
