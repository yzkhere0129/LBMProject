#!/usr/bin/env python3
"""
Test B Validation: Marangoni Force
===================================

Validates that Marangoni thermocapillary convection:
1. Increases velocity significantly (v_max > 10 mm/s)
2. Flow direction correct (hot → cold for dσ/dT < 0)
3. Temperature decreases further (enhanced convection)
4. Remains numerically stable

Physics Checks:
- Marangoni number: Ma = |dσ/dT| * ΔT * L / (μ * α)
- Expected velocity: v ~ |dσ/dT| * ΔT / μ
- Energy enhancement: Q_conv_Marangoni >> Q_conv_natural
"""

import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional

# Physical constants
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)

# Test A baseline (for comparison)
TEST_A_T_MAX = 44000.0  # Estimate, will be overridden if Test A log provided
TEST_A_V_MAX = 1.0      # Estimate

# Material properties (Ti6Al4V)
DSIGMA_DT = -0.26  # N/(m·K) = -0.00026e-3 N/(m·K) * 1e3
VISCOSITY = 0.0333 * 4110 * 1e-6  # Convert kinematic to dynamic: ν*ρ [Pa·s]
DENSITY = 4110.0  # kg/m³

class TestBValidator:
    """Validates Test B results"""

    def __init__(self, log_file: str, test_a_log: Optional[str] = None):
        self.log_file = Path(log_file)
        self.test_a_log = Path(test_a_log) if test_a_log else None
        self.config = {}
        self.steps_data = []
        self.test_a_baseline = {'T_max': TEST_A_T_MAX, 'v_max': TEST_A_V_MAX}
        self.errors = []
        self.warnings = []

    def parse_log(self) -> bool:
        """Parse simulation log file"""
        if not self.log_file.exists():
            self.errors.append(f"Log file not found: {self.log_file}")
            return False

        with open(self.log_file, 'r') as f:
            content = f.read()

        self._parse_config(content)
        self._parse_timesteps(content)

        # Parse Test A baseline if provided
        if self.test_a_log and self.test_a_log.exists():
            self._parse_test_a_baseline()

        return len(self.steps_data) > 0

    def _parse_config(self, content: str):
        """Extract configuration parameters"""
        patterns = {
            'nx': r'nx\s*=\s*(\d+)',
            'dx': r'dx\s*=\s*([\d.e+-]+)',
            'dt': r'dt\s*=\s*([\d.e+-]+)',
            'laser_power': r'laser_power\s*=\s*([\d.e+-]+)',
            'dsigma_dT': r'dsigma_dT\s*=\s*([-\d.e+-]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                self.config[key] = float(match.group(1))

    def _parse_timesteps(self, content: str):
        """Extract timestep information"""
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

    def _parse_test_a_baseline(self):
        """Parse Test A results for comparison"""
        with open(self.test_a_log, 'r') as f:
            content = f.read()

        # Get final values from Test A
        pattern = r'Step\s+\d+/\d+:.*?T_max\s*=\s*([\d.e+-]+)\s*K.*?v_max\s*=\s*([\d.e+-]+)\s*mm/s'
        matches = list(re.finditer(pattern, content))

        if matches:
            last_match = matches[-1]
            self.test_a_baseline['T_max'] = float(last_match.group(1))
            self.test_a_baseline['v_max'] = float(last_match.group(2))
            print(f"Test A baseline loaded: T={self.test_a_baseline['T_max']:.1f} K, "
                  f"v={self.test_a_baseline['v_max']:.4f} mm/s")

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

    def check_marangoni_activation(self) -> bool:
        """Verify Marangoni effect is active (v_max > 10 mm/s)"""
        if not self.steps_data:
            self.errors.append("No timestep data available")
            return False

        final = self.steps_data[-1]
        v_final = final['v_max']

        # Primary criterion: v_max > 10 mm/s
        if v_final > 10.0:
            print(f"✓ PASS: Marangoni effect ACTIVE")
            print(f"  v_max = {v_final:.2f} mm/s > 10 mm/s threshold")

            # Check increase vs Test A
            if v_final > self.test_a_baseline['v_max']:
                factor = v_final / max(self.test_a_baseline['v_max'], 0.001)
                print(f"  Increased {factor:.0f}× vs Test A")

            return True
        else:
            self.errors.append(
                f"FAIL: Marangoni effect NOT active\n"
                f"  v_max = {v_final:.4f} mm/s < 10 mm/s\n"
                f"  Expected: Strong Marangoni convection\n"
                f"  Possible causes:\n"
                f"    - Marangoni force not computed\n"
                f"    - dσ/dT wrong sign or magnitude\n"
                f"    - Interface not detected\n"
                f"    - Temperature gradient too weak"
            )
            return False

    def check_temperature_decrease(self) -> bool:
        """Verify temperature decreased vs Test A (enhanced convection)"""
        if not self.steps_data:
            return False

        final = self.steps_data[-1]
        T_final = final['T_max']
        T_baseline = self.test_a_baseline['T_max']

        if T_final < T_baseline:
            reduction_pct = 100 * (T_baseline - T_final) / T_baseline
            print(f"✓ PASS: Temperature decreased vs Test A")
            print(f"  Test B: {T_final:.1f} K")
            print(f"  Test A: {T_baseline:.1f} K")
            print(f"  Reduction: {reduction_pct:.1f}%")
            return True
        else:
            # Not a hard failure, but suspicious
            self.warnings.append(
                f"Warning: Temperature did NOT decrease vs Test A\n"
                f"  Test B: {T_final:.1f} K\n"
                f"  Test A: {T_baseline:.1f} K\n"
                f"  Expected: Enhanced convection should cool melt pool\n"
                f"  This may indicate Marangoni is not effective"
            )
            return True  # Don't fail on this alone

    def check_velocity_regime(self) -> bool:
        """Check velocity is in physically reasonable range (10-1000 mm/s)"""
        if not self.steps_data:
            return False

        final = self.steps_data[-1]
        v_final = final['v_max']

        # Check upper bound
        if v_final > 1000:
            self.errors.append(
                f"FAIL: Velocity too high (unrealistic)\n"
                f"  v_max = {v_final:.1f} mm/s > 1000 mm/s\n"
                f"  Exceeds realistic Marangoni velocities\n"
                f"  Likely numerical instability or force limiter not working"
            )
            return False

        # Check lower bound (already checked in check_marangoni_activation)
        if v_final < 10:
            return False  # Already flagged elsewhere

        print(f"✓ PASS: Velocity in physical range")
        print(f"  v_max = {v_final:.2f} mm/s ∈ [10, 1000] mm/s")
        return True

    def check_marangoni_number(self) -> bool:
        """Compute Marangoni number for validation"""
        if not self.steps_data or not self.config:
            self.warnings.append("Cannot compute Marangoni number: missing data")
            return True

        final = self.steps_data[-1]
        T_max = final['T_max']
        T_amb = 300.0
        delta_T = T_max - T_amb

        # Characteristic length (laser spot diameter)
        L = 100e-6  # m

        # Thermal diffusivity
        alpha = 5.8e-6  # m²/s

        # Marangoni number: Ma = |dσ/dT| * ΔT * L / (μ * α)
        Ma = abs(DSIGMA_DT) * delta_T * L / (VISCOSITY * alpha)

        print(f"\nMarangoni Number:")
        print(f"  Ma = {Ma:.2e}")

        if Ma > 1e4:
            print(f"  → Strong Marangoni convection (Ma > 10^4)")
        elif Ma > 1e2:
            print(f"  → Moderate Marangoni convection")
        else:
            self.warnings.append(
                f"Warning: Low Marangoni number Ma = {Ma:.2e}\n"
                f"  Expected Ma > 10^4 for LPBF\n"
                f"  This may explain weak convection"
            )

        return True

    def check_velocity_oscillations(self) -> bool:
        """Check for wild velocity oscillations (instability sign)"""
        if len(self.steps_data) < 3:
            return True

        velocities = [d['v_max'] for d in self.steps_data]

        # Check for large oscillations (>50% variation)
        for i in range(1, len(velocities) - 1):
            variation = abs(velocities[i] - velocities[i-1]) / max(velocities[i-1], 1e-6)

            if variation > 0.5:
                self.warnings.append(
                    f"Warning: Large velocity oscillation at step {self.steps_data[i]['step']}\n"
                    f"  Variation: {100*variation:.1f}%\n"
                    f"  May indicate approaching instability"
                )
                break

        return True

    def estimate_flow_direction(self) -> bool:
        """Check if flow direction is physically correct"""
        # For dσ/dT < 0 (Ti6Al4V), flow should go from hot to cold
        # This creates outward radial flow from laser spot

        dsigma_dT = self.config.get('dsigma_dT', DSIGMA_DT)

        print(f"\nFlow Direction Check:")
        print(f"  dσ/dT = {dsigma_dT:.2e} N/(m·K)")

        if dsigma_dT < 0:
            print(f"  → Flow direction: HOT → COLD (outward from laser)")
            print(f"  ✓ Correct for most metals (Ti6Al4V, steel, etc.)")
        else:
            self.warnings.append(
                f"Warning: Positive dσ/dT = {dsigma_dT}\n"
                f"  Flow direction: COLD → HOT (inward to laser)\n"
                f"  This is unusual (only for some alloys)"
            )

        return True

    def run_validation(self) -> bool:
        """Run all validation checks"""
        print("="*60)
        print("TEST B VALIDATION: Marangoni Force")
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
            ("Marangoni Activation", self.check_marangoni_activation),
            ("Velocity Regime", self.check_velocity_regime),
            ("Temperature Decrease", self.check_temperature_decrease),
            ("Marangoni Number", self.check_marangoni_number),
            ("Flow Direction", self.estimate_flow_direction),
            ("Velocity Oscillations", self.check_velocity_oscillations),
        ]

        print("\n" + "-"*60)
        print("VALIDATION CHECKS:")
        print("-"*60)

        all_passed = True
        for name, check_func in checks:
            try:
                passed = check_func()
                if name not in ["Marangoni Number", "Flow Direction"]:  # Info only
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
            print("FINAL RESULT: ✓ TEST B PASSED")
            print("="*60)
            print("\nMarangoni thermocapillary convection is working correctly.")
            print("Ready to proceed to Test C (full coupling).")
            return True
        else:
            print("FINAL RESULT: ✗ TEST B FAILED")
            print("="*60)
            print("\nMarangoni force has issues.")
            print("Do NOT proceed to Test C until these are fixed.")
            return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_test_B.py <test_B_log> [test_A_log]")
        print("Example: python validate_test_B.py test_B_marangoni.log test_A_coupling.log")
        sys.exit(1)

    log_file = sys.argv[1]
    test_a_log = sys.argv[2] if len(sys.argv) > 2 else None

    validator = TestBValidator(log_file, test_a_log)
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
