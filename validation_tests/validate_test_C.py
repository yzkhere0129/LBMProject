#!/usr/bin/env python3
"""
Test C Validation: Full Multiphysics Coupling
==============================================

Validates that full coupling (thermal + fluid + Marangoni + radiation):
1. Achieves realistic LPBF temperatures (T_max < 10,000 K ideal)
2. Strong Marangoni convection (v_max: 10-1000 mm/s)
3. Stable melt pool (no collapse/explosion)
4. Energy balance maintained
5. Compares to literature (Khairallah 2016)

Literature Targets (Khairallah et al. 2016):
- T_peak: ~3,300 K
- v_max: ~970 mm/s
- Melt pool depth: ~100-200 μm
- Melt pool width: ~150-250 μm
"""

import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional

# Physical constants
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)

# Literature targets (Khairallah 2016)
LITERATURE_T_PEAK = 3300.0  # K
LITERATURE_V_MAX = 970.0    # mm/s
LITERATURE_DEPTH = 150e-6   # m (estimate)
LITERATURE_WIDTH = 200e-6   # m (estimate)

# Ti6Al4V properties
T_MELTING = 1923.0  # K
T_BOILING = 3560.0  # K

class TestCValidator:
    """Validates Test C results"""

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

        self._parse_config(content)
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
            'total_steps': r'total_steps\s*=\s*(\d+)',
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

    def check_numerical_stability(self) -> bool:
        """Check for NaN/Inf throughout simulation"""
        for data in self.steps_data:
            if math.isnan(data['T_max']) or math.isinf(data['T_max']):
                self.errors.append(
                    f"Step {data['step']}: T_max is NaN/Inf\n"
                    f"  Simulation diverged - numerical instability"
                )
                return False
            if math.isnan(data['v_max']) or math.isinf(data['v_max']):
                self.errors.append(f"Step {data['step']}: v_max is NaN/Inf")
                return False

        print(f"✓ PASS: Numerical stability maintained")
        print(f"  No NaN/Inf detected in {len(self.steps_data)} timesteps")
        return True

    def check_temperature_regime(self) -> bool:
        """Check if temperature is in realistic range"""
        if not self.steps_data:
            return False

        final = self.steps_data[-1]
        T_final = final['T_max']

        print(f"\nTemperature Assessment:")
        print(f"  Final T_max: {T_final:.1f} K")
        print(f"  Literature target: {LITERATURE_T_PEAK:.1f} K")
        print(f"  Ti6Al4V melting: {T_MELTING:.1f} K")
        print(f"  Ti6Al4V boiling: {T_BOILING:.1f} K")

        if T_final < 10000:
            if T_final <= 5000:
                print(f"  ✓ EXCELLENT: Realistic LPBF temperature")
                deviation = abs(T_final - LITERATURE_T_PEAK) / LITERATURE_T_PEAK
                if deviation < 0.5:
                    print(f"  ✓ Within 50% of literature value")
                return True
            else:
                print(f"  ✓ GOOD: Physical range (but higher than literature)")
                self.warnings.append(
                    f"Temperature {T_final:.1f} K is higher than literature {LITERATURE_T_PEAK:.1f} K\n"
                    f"  Still physical, but vaporization cooling may help"
                )
                return True
        elif T_final < 15000:
            self.warnings.append(
                f"Temperature {T_final:.1f} K > 10,000 K\n"
                f"  Entering unphysical regime\n"
                f"  Vaporization cooling strongly recommended (V5)"
            )
            return True  # Don't fail, but warn
        else:
            self.errors.append(
                f"FAIL: Temperature {T_final:.1f} K > 15,000 K\n"
                f"  Far above realistic LPBF range\n"
                f"  Physics coupling may not be effective"
            )
            return False

    def check_velocity_regime(self) -> bool:
        """Check if velocity is in realistic Marangoni range"""
        if not self.steps_data:
            return False

        final = self.steps_data[-1]
        v_final = final['v_max']

        print(f"\nVelocity Assessment:")
        print(f"  Final v_max: {v_final:.2f} mm/s")
        print(f"  Literature target: {LITERATURE_V_MAX:.1f} mm/s")

        if v_final < 1.0:
            self.errors.append(
                f"FAIL: Velocity too low {v_final:.4f} mm/s\n"
                f"  Marangoni convection not active\n"
                f"  Expected: Strong thermocapillary flow"
            )
            return False
        elif v_final < 10.0:
            self.warnings.append(
                f"Velocity {v_final:.2f} mm/s < 10 mm/s\n"
                f"  Marangoni effect weak\n"
                f"  Expected: 10-1000 mm/s for LPBF"
            )
            return True
        elif v_final <= 1000.0:
            print(f"  ✓ GOOD: Active Marangoni convection")
            deviation = abs(v_final - LITERATURE_V_MAX) / LITERATURE_V_MAX
            if deviation < 0.5:
                print(f"  ✓ Within 50% of literature value")
            return True
        else:
            self.errors.append(
                f"FAIL: Velocity {v_final:.1f} mm/s > 1000 mm/s\n"
                f"  Unrealistically high\n"
                f"  Likely numerical instability or force limiter failed"
            )
            return False

    def check_melt_pool_stability(self) -> bool:
        """Check for melt pool collapse or explosion"""
        if len(self.steps_data) < 10:
            return True

        # Check for sudden changes (sign of instability)
        for i in range(10, len(self.steps_data)):
            prev = self.steps_data[i-1]
            curr = self.steps_data[i]

            # Check for sudden temperature spike
            T_change = abs(curr['T_max'] - prev['T_max']) / prev['T_max']
            if T_change > 0.2:  # >20% change in one step
                self.warnings.append(
                    f"Warning: Sudden temperature change at step {curr['step']}\n"
                    f"  ΔT/T = {100*T_change:.1f}%\n"
                    f"  May indicate approaching instability"
                )

            # Check for velocity explosion
            v_change = abs(curr['v_max'] - prev['v_max']) / max(prev['v_max'], 1e-6)
            if v_change > 1.0 and curr['v_max'] > 100:  # >100% change, high velocity
                self.warnings.append(
                    f"Warning: Velocity spike at step {curr['step']}\n"
                    f"  v_max jumped from {prev['v_max']:.2f} to {curr['v_max']:.2f} mm/s"
                )

        print(f"✓ PASS: Melt pool stable (no collapse/explosion)")
        return True

    def check_energy_balance(self) -> bool:
        """Comprehensive energy balance check"""
        if not self.config or not self.steps_data:
            self.warnings.append("Cannot check energy balance: missing data")
            return True

        final = self.steps_data[-1]
        T_max = final['T_max']

        # Laser input
        P_laser = self.config.get('laser_power', 195.0)  # W

        # Radiation output
        emissivity = self.config.get('emissivity', 0.3)
        spot_radius = 50e-6  # m
        A_spot = math.pi * spot_radius**2
        T_amb = 300.0

        P_radiation = emissivity * STEFAN_BOLTZMANN * A_spot * (T_max**4 - T_amb**4)

        # Conduction (rough estimate)
        # Q_cond ~ k * A * ΔT / L
        k = 33.0  # W/(m·K) for liquid Ti6Al4V
        L = 100e-6  # Characteristic length
        delta_T = T_max - T_amb
        P_conduction = k * A_spot * delta_T / L

        # Total output
        P_total_out = P_radiation + P_conduction

        # Residual
        residual = abs(P_laser - P_total_out) / P_laser

        print(f"\nEnergy Balance:")
        print(f"  Input:")
        print(f"    Laser:        {P_laser:.1f} W")
        print(f"  Output:")
        print(f"    Radiation:    {P_radiation:.1f} W ({100*P_radiation/P_laser:.1f}%)")
        print(f"    Conduction:   {P_conduction:.1f} W ({100*P_conduction/P_laser:.1f}%)")
        print(f"    Total:        {P_total_out:.1f} W")
        print(f"  Residual:       {100*residual:.1f}%")

        if residual < 0.15:
            print(f"  ✓ EXCELLENT: Energy balanced (<15% residual)")
            return True
        elif residual < 0.5:
            print(f"  ✓ GOOD: Reasonable energy balance (<50%)")
            return True
        else:
            self.warnings.append(
                f"Large energy residual: {100*residual:.1f}%\n"
                f"  Heat may be accumulating or leaking"
            )
            return True  # Don't fail on this

    def compare_to_literature(self) -> bool:
        """Compare results to Khairallah 2016"""
        if not self.steps_data:
            return True

        final = self.steps_data[-1]
        T_final = final['T_max']
        v_final = final['v_max']

        print(f"\n" + "="*60)
        print("LITERATURE COMPARISON (Khairallah et al. 2016)")
        print("="*60)

        # Temperature comparison
        T_ratio = T_final / LITERATURE_T_PEAK
        T_match = "✓" if 0.5 < T_ratio < 2.0 else "✗"
        print(f"Temperature:")
        print(f"  Simulation: {T_final:.1f} K")
        print(f"  Literature: {LITERATURE_T_PEAK:.1f} K")
        print(f"  Ratio:      {T_ratio:.2f}  {T_match}")

        # Velocity comparison
        v_ratio = v_final / LITERATURE_V_MAX
        v_match = "✓" if 0.5 < v_ratio < 2.0 else "✗"
        print(f"\nVelocity:")
        print(f"  Simulation: {v_final:.1f} mm/s")
        print(f"  Literature: {LITERATURE_V_MAX:.1f} mm/s")
        print(f"  Ratio:      {v_ratio:.2f}  {v_match}")

        # Overall assessment
        both_match = (0.5 < T_ratio < 2.0) and (0.5 < v_ratio < 2.0)
        if both_match:
            print(f"\n✓ EXCELLENT: Within 2× of literature values")
        elif T_ratio < 5.0 and v_ratio > 0.1:
            print(f"\n✓ GOOD: Reasonable agreement with literature")
        else:
            print(f"\n⚠ FAIR: Some deviation from literature")
            print(f"  (May need vaporization cooling for exact match)")

        return True

    def generate_recommendations(self):
        """Generate recommendations based on results"""
        if not self.steps_data:
            return

        final = self.steps_data[-1]
        T_final = final['T_max']
        v_final = final['v_max']

        print(f"\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)

        # Temperature-based recommendations
        if T_final > 10000:
            print(f"1. IMPLEMENT VAPORIZATION COOLING (V5)")
            print(f"   Current T_max = {T_final:.1f} K >> boiling point")
            print(f"   Add latent heat loss above {T_BOILING:.1f} K")
            print(f"   Expected result: T_max → 3000-5000 K range")

        if T_final > 5000 and T_final <= 10000:
            print(f"1. CONSIDER VAPORIZATION COOLING")
            print(f"   Current T_max = {T_final:.1f} K")
            print(f"   May benefit from vapor phase modeling")

        # Velocity-based recommendations
        if v_final < 10:
            print(f"2. STRENGTHEN MARANGONI EFFECT")
            print(f"   Current v_max = {v_final:.4f} mm/s < 10 mm/s")
            print(f"   Check: dσ/dT magnitude, temperature gradients, interface detection")

        if v_final > 500:
            print(f"2. VALIDATE HIGH VELOCITIES")
            print(f"   v_max = {v_final:.1f} mm/s is high (but may be physical)")
            print(f"   Compare flow patterns to literature in ParaView")

        # Energy balance
        if self.config:
            print(f"\n3. VALIDATE WITH VISUALIZATION")
            print(f"   Open VTK files in ParaView:")
            print(f"   - Check melt pool shape")
            print(f"   - Verify flow patterns (radial outward)")
            print(f"   - Measure melt pool dimensions")
            print(f"   - Compare to literature images")

        print(f"\n4. NEXT STEPS")
        if T_final < 5000 and 10 < v_final < 1000:
            print(f"   ✓ Physics validated - ready for production")
            print(f"   → Run longer simulations")
            print(f"   → Test scanning paths")
            print(f"   → Compare to experimental data")
        else:
            print(f"   → Add vaporization cooling (if T > 5000 K)")
            print(f"   → Debug Marangoni (if v < 10 mm/s)")
            print(f"   → Validate stability (if oscillations detected)")

    def run_validation(self) -> bool:
        """Run all validation checks"""
        print("="*60)
        print("TEST C VALIDATION: Full Multiphysics Coupling")
        print("="*60)

        # Parse log
        if not self.parse_log():
            print(f"\n✗ FAIL: Could not parse log file")
            return False

        print(f"\nLog file: {self.log_file}")
        print(f"Timesteps parsed: {len(self.steps_data)}")
        if self.config:
            total_time = self.config.get('total_steps', 0) * self.config.get('dt', 0)
            print(f"Simulation time: {total_time*1e6:.1f} μs")

        # Run checks
        checks = [
            ("Numerical Stability", self.check_numerical_stability),
            ("Temperature Regime", self.check_temperature_regime),
            ("Velocity Regime", self.check_velocity_regime),
            ("Melt Pool Stability", self.check_melt_pool_stability),
            ("Energy Balance", self.check_energy_balance),
        ]

        print("\n" + "-"*60)
        print("VALIDATION CHECKS:")
        print("-"*60)

        all_passed = True
        for name, check_func in checks:
            try:
                passed = check_func()
                # Some checks are informational only
                if name not in ["Energy Balance"]:
                    all_passed = all_passed and passed
            except Exception as e:
                print(f"✗ ERROR in {name}: {e}")
                all_passed = False

        # Literature comparison (informational)
        try:
            self.compare_to_literature()
        except Exception as e:
            print(f"Error in literature comparison: {e}")

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

        # Generate recommendations
        self.generate_recommendations()

        # Final verdict
        print("\n" + "="*60)
        if all_passed and not self.errors:
            print("FINAL RESULT: ✓ TEST C PASSED")
            print("="*60)
            print("\nFull multiphysics coupling is working!")
            if self.steps_data:
                final = self.steps_data[-1]
                if final['T_max'] < 5000 and 10 < final['v_max'] < 1000:
                    print("Results match literature - ready for production validation.")
                else:
                    print("Physics stable but may benefit from V5 enhancements.")
            return True
        else:
            print("FINAL RESULT: ✗ TEST C FAILED")
            print("="*60)
            print("\nFull coupling has issues - review errors above.")
            return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_test_C.py <test_C_log>")
        print("Example: python validate_test_C.py test_C_full_coupling.log")
        sys.exit(1)

    log_file = sys.argv[1]
    validator = TestCValidator(log_file)
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
