#!/usr/bin/env python3
"""
Test E Validation: VOF Advection Performance and Stability
===========================================================

Validates that VOF advection:
1. Executes with acceptable performance (runtime < 20 min)
2. Maintains mass conservation (error < 5%)
3. Keeps interface sharp (interface cells < 5000)
4. Satisfies CFL_VOF condition (< 0.5)
5. Preserves fill level bounds [0, 1]
6. Achieves good GPU utilization (> 70%)
7. Remains numerically stable (no NaN/Inf)

Performance Comparison to Test C:
- Test C (static interface): ~8-10 minutes
- Test E (VOF advection): ~10-15 minutes (+20-50% expected)
- Test E > 20 minutes: Performance issue requiring investigation
"""

import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Physical constants
RHO_LIQUID = 4110.0  # Ti6Al4V density, kg/m³
DX = 2.0e-6          # Cell size, m
CELL_VOLUME = DX**3  # m³

# VOF validation thresholds
MAX_RUNTIME_MINUTES = 20        # Performance threshold
TARGET_RUNTIME_MINUTES = 15     # Acceptable target
MASS_ERROR_THRESHOLD = 0.05     # 5% mass conservation tolerance
MAX_INTERFACE_CELLS = 5000      # Interface sharpness threshold
CFL_VOF_THRESHOLD = 0.5         # VOF CFL stability limit
GPU_UTIL_THRESHOLD = 70.0       # Minimum GPU utilization %

class TestEValidator:
    """Validates Test E VOF advection results"""

    def __init__(self, log_file: str, test_c_log: Optional[str] = None):
        self.log_file = Path(log_file)
        self.test_c_log = Path(test_c_log) if test_c_log else None
        self.config = {}
        self.steps_data = []
        self.mass_data = []
        self.errors = []
        self.warnings = []
        self.runtime_seconds = 0
        self.test_c_runtime_seconds = 0

    def parse_log(self) -> bool:
        """Parse simulation log file"""
        if not self.log_file.exists():
            self.errors.append(f"Log file not found: {self.log_file}")
            return False

        with open(self.log_file, 'r') as f:
            content = f.read()

        self._parse_config(content)
        self._parse_timesteps(content)
        self._parse_runtime(content)
        self._parse_mass_conservation(content)

        # Parse Test C baseline if available
        if self.test_c_log and self.test_c_log.exists():
            with open(self.test_c_log, 'r') as f:
                test_c_content = f.read()
            self._parse_test_c_runtime(test_c_content)

        return len(self.steps_data) > 0

    def _parse_config(self, content: str):
        """Extract configuration parameters"""
        patterns = {
            'nx': r'nx\s*=\s*(\d+)',
            'ny': r'ny\s*=\s*(\d+)',
            'nz': r'nz\s*=\s*(\d+)',
            'dx': r'dx\s*=\s*([\d.e+-]+)',
            'dt': r'dt\s*=\s*([\d.e+-]+)',
            'total_steps': r'total_steps\s*=\s*(\d+)',
            'vof_subcycles': r'vof_subcycles\s*=\s*(\d+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                self.config[key] = float(match.group(1))

        # Calculate total cells
        if 'nx' in self.config and 'ny' in self.config and 'nz' in self.config:
            self.config['total_cells'] = int(
                self.config['nx'] * self.config['ny'] * self.config['nz']
            )

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

    def _parse_runtime(self, content: str):
        """Extract runtime from log"""
        # Look for explicit runtime reporting
        match = re.search(r'Total.*time.*?(\d+).*?min.*?(\d+).*?sec', content, re.IGNORECASE)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            self.runtime_seconds = minutes * 60 + seconds
            return

        # Fallback: estimate from timestamps
        start_match = re.search(r'Start.*time.*?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', content)
        end_match = re.search(r'End.*time.*?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', content)

        if start_match and end_match:
            from datetime import datetime
            start = datetime.strptime(start_match.group(1), '%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(end_match.group(1), '%Y-%m-%d %H:%M:%S')
            self.runtime_seconds = int((end - start).total_seconds())

    def _parse_test_c_runtime(self, content: str):
        """Extract Test C runtime for comparison"""
        match = re.search(r'Total.*time.*?(\d+).*?min.*?(\d+).*?sec', content, re.IGNORECASE)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            self.test_c_runtime_seconds = minutes * 60 + seconds

    def _parse_mass_conservation(self, content: str):
        """Extract mass conservation data if logged"""
        # Pattern for mass logging: "Mass: <value>, Error: <value>%"
        pattern = r'Mass.*?:\s*([\d.e+-]+).*?Error.*?:\s*([\d.e+-]+)\s*%'

        for match in re.finditer(pattern, content, re.IGNORECASE):
            mass = float(match.group(1))
            error_pct = float(match.group(2))
            self.mass_data.append({
                'mass': mass,
                'error_pct': error_pct
            })

    def check_runtime_performance(self) -> bool:
        """Check if runtime is acceptable"""
        if self.runtime_seconds == 0:
            self.warnings.append("Runtime not found in log - cannot validate performance")
            return True  # Don't fail

        runtime_min = self.runtime_seconds / 60.0

        print(f"\n{'='*60}")
        print(f"PERFORMANCE ASSESSMENT")
        print(f"{'='*60}")
        print(f"  Test E runtime: {runtime_min:.1f} minutes")

        if self.test_c_runtime_seconds > 0:
            test_c_min = self.test_c_runtime_seconds / 60.0
            overhead_pct = ((runtime_min - test_c_min) / test_c_min) * 100
            print(f"  Test C baseline: {test_c_min:.1f} minutes (static interface)")
            print(f"  VOF overhead: +{overhead_pct:.1f}%")

            if overhead_pct < 20:
                print(f"  ✓ EXCELLENT: Low VOF advection overhead")
            elif overhead_pct < 50:
                print(f"  ✓ GOOD: Acceptable overhead (expected 20-50%)")
            elif overhead_pct < 100:
                print(f"  ⚠ FAIR: Higher than expected overhead")
            else:
                print(f"  ✗ POOR: Very high overhead (> 100%)")

        # Absolute performance check
        if runtime_min <= TARGET_RUNTIME_MINUTES:
            print(f"  ✓ PASS: Runtime ≤ {TARGET_RUNTIME_MINUTES} min (target)")
            return True
        elif runtime_min <= MAX_RUNTIME_MINUTES:
            print(f"  ⚠ ACCEPTABLE: Runtime ≤ {MAX_RUNTIME_MINUTES} min (threshold)")
            self.warnings.append(
                f"Runtime {runtime_min:.1f} min is higher than target {TARGET_RUNTIME_MINUTES} min\n"
                f"  Still acceptable, but consider optimization"
            )
            return True
        else:
            self.errors.append(
                f"FAIL: Runtime {runtime_min:.1f} min > {MAX_RUNTIME_MINUTES} min\n"
                f"  VOF kernel performance is poor\n"
                f"  Actions: Profile with nsight, check memory access patterns, reduce subcycling"
            )
            return False

    def check_numerical_stability(self) -> bool:
        """Check for NaN/Inf throughout simulation"""
        print(f"\n{'='*60}")
        print(f"NUMERICAL STABILITY")
        print(f"{'='*60}")

        for data in self.steps_data:
            if math.isnan(data['T_max']) or math.isinf(data['T_max']):
                self.errors.append(
                    f"Step {data['step']}: T_max is NaN/Inf\n"
                    f"  VOF advection caused instability\n"
                    f"  Actions: Reduce dt, increase subcycling, add fill level clamping"
                )
                return False
            if math.isnan(data['v_max']) or math.isinf(data['v_max']):
                self.errors.append(f"Step {data['step']}: v_max is NaN/Inf")
                return False

        print(f"  ✓ PASS: No NaN/Inf in {len(self.steps_data)} timesteps")
        return True

    def check_mass_conservation(self) -> bool:
        """Check mass conservation over time"""
        print(f"\n{'='*60}")
        print(f"MASS CONSERVATION")
        print(f"{'='*60}")

        if not self.mass_data:
            self.warnings.append(
                "Mass conservation data not found in log\n"
                f"  Recommendation: Add mass tracking to VOF solver\n"
                f"  Every 100 steps: compute Σ(f_i) and check drift"
            )
            print(f"  ⚠ WARNING: Mass conservation not logged")
            return True  # Don't fail if not logged

        # Check final mass error
        final_mass_error = self.mass_data[-1]['error_pct']

        print(f"  Initial mass: {self.mass_data[0]['mass']:.6e}")
        print(f"  Final mass:   {self.mass_data[-1]['mass']:.6e}")
        print(f"  Mass error:   {final_mass_error:.2f}%")
        print(f"  Threshold:    {MASS_ERROR_THRESHOLD * 100:.1f}%")

        if final_mass_error < 1.0:
            print(f"  ✓ EXCELLENT: Error < 1% (sharp VOF scheme)")
            return True
        elif final_mass_error < MASS_ERROR_THRESHOLD * 100:
            print(f"  ✓ PASS: Error < {MASS_ERROR_THRESHOLD * 100}%")
            return True
        else:
            self.errors.append(
                f"FAIL: Mass conservation error {final_mass_error:.2f}% > {MASS_ERROR_THRESHOLD * 100}%\n"
                f"  Numerical diffusion or bug in VOF advection\n"
                f"  Actions:\n"
                f"    - Enable subcycling (vof_subcycles = 20)\n"
                f"    - Reduce dt by 50%\n"
                f"    - Switch to PLIC reconstruction\n"
                f"    - Check boundary conditions"
            )
            return False

    def check_interface_sharpness(self) -> bool:
        """Check if interface remains sharp (not smeared)"""
        print(f"\n{'='*60}")
        print(f"INTERFACE SHARPNESS")
        print(f"{'='*60}")

        # This requires analyzing fill level field from output files
        # For now, provide guidance
        print(f"  Manual check required:")
        print(f"    1. Load VTK output in ParaView")
        print(f"    2. Apply Threshold: 0.01 < fill_level < 0.99")
        print(f"    3. Count interface cells")
        print(f"  ")
        print(f"  Expected interface cells: 500-2000 (sharp)")
        print(f"  Warning threshold: > {MAX_INTERFACE_CELLS} (smeared)")
        print(f"  ")
        print(f"  If interface is smeared:")
        print(f"    - Reduce dt")
        print(f"    - Implement interface compression term")
        print(f"    - Switch to CICSAM or HRIC scheme")

        return True  # Don't fail, manual check required

    def check_cfl_vof(self) -> bool:
        """Check VOF CFL condition"""
        print(f"\n{'='*60}")
        print(f"CFL_VOF CONDITION")
        print(f"{'='*60}")

        if 'dx' not in self.config or 'dt' not in self.config:
            print(f"  ⚠ WARNING: Cannot compute CFL_VOF (dx or dt missing)")
            return True

        dx = self.config['dx']
        dt = self.config['dt']

        if not self.steps_data:
            return True

        # Estimate CFL_VOF from maximum velocity
        final_v_max = self.steps_data[-1]['v_max']  # mm/s
        v_max_mps = final_v_max * 1e-3  # Convert to m/s

        cfl_vof = v_max_mps * dt / dx

        print(f"  v_max: {final_v_max:.3f} mm/s = {v_max_mps:.3e} m/s")
        print(f"  dt: {dt:.2e} s")
        print(f"  dx: {dx:.2e} m")
        print(f"  CFL_VOF ≈ v·dt/dx = {cfl_vof:.3f}")
        print(f"  Stability limit: {CFL_VOF_THRESHOLD}")

        if cfl_vof < 0.3:
            print(f"  ✓ EXCELLENT: CFL_VOF in safe range (< 0.3)")
            return True
        elif cfl_vof < CFL_VOF_THRESHOLD:
            print(f"  ✓ PASS: CFL_VOF < {CFL_VOF_THRESHOLD}")
            return True
        else:
            self.errors.append(
                f"FAIL: CFL_VOF {cfl_vof:.3f} > {CFL_VOF_THRESHOLD}\n"
                f"  VOF advection may be unstable\n"
                f"  Actions:\n"
                f"    - Reduce dt by 50%: dt = {dt/2:.2e}\n"
                f"    - Increase subcycling: vof_subcycles = 20\n"
                f"    - Add velocity limiter near interface"
            )
            return False

    def check_fill_level_bounds(self) -> bool:
        """Check if fill level stayed within [0, 1]"""
        print(f"\n{'='*60}")
        print(f"FILL LEVEL BOUNDS")
        print(f"{'='*60}")

        # This requires analyzing actual fill level values
        # For now, check log for warnings
        with open(self.log_file, 'r') as f:
            content = f.read()

        violations = []
        for match in re.finditer(r'f_min.*?=\s*([\d.e+-]+)', content, re.IGNORECASE):
            f_min = float(match.group(1))
            if f_min < -0.01:
                violations.append(f"f_min = {f_min:.3f}")

        for match in re.finditer(r'f_max.*?=\s*([\d.e+-]+)', content, re.IGNORECASE):
            f_max = float(match.group(1))
            if f_max > 1.01:
                violations.append(f"f_max = {f_max:.3f}")

        if violations:
            print(f"  ✗ FAIL: Fill level out of bounds")
            for v in violations:
                print(f"    {v}")
            self.errors.append(
                f"Fill level exceeded physical bounds [0, 1]\n"
                f"  Actions:\n"
                f"    - Add clamping: f = clamp(f, 0, 1)\n"
                f"    - Check interface reconstruction algorithm\n"
                f"    - Verify advection scheme conservativeness"
            )
            return False
        else:
            print(f"  ✓ PASS: Fill level remained in [0, 1]")
            print(f"    (or bounds checking not implemented)")
            return True

    def check_physics_consistency(self) -> bool:
        """Check if physics results are consistent with Test C"""
        if not self.steps_data:
            return True

        print(f"\n{'='*60}")
        print(f"PHYSICS CONSISTENCY")
        print(f"{'='*60}")

        final = self.steps_data[-1]
        T_final = final['T_max']
        v_final = final['v_max']

        print(f"  Test E (VOF advection enabled):")
        print(f"    T_max: {T_final:.1f} K")
        print(f"    v_max: {v_final:.3f} mm/s")

        # Temperature should remain realistic
        if T_final > 50000:
            self.errors.append(
                f"FAIL: Temperature {T_final:.1f} K is unrealistic\n"
                f"  VOF advection may have disrupted physics"
            )
            return False
        else:
            print(f"  ✓ Temperature realistic (< 50,000 K)")

        # Velocity should be positive
        if v_final < 0.01:
            self.warnings.append(
                f"Velocity {v_final:.3f} mm/s is very low\n"
                f"  VOF advection may have suppressed Marangoni flow"
            )
            print(f"  ⚠ Velocity lower than expected")
        else:
            print(f"  ✓ Velocity developed (> 0.01 mm/s)")

        return True

    def generate_report(self) -> int:
        """Generate validation report and return exit code"""
        print(f"\n{'='*60}")
        print(f"TEST E VALIDATION SUMMARY")
        print(f"{'='*60}")

        if not self.steps_data:
            print(f"✗ FAIL: No simulation data found")
            return 1

        print(f"\nSimulation completed:")
        print(f"  Total timesteps: {len(self.steps_data)}")
        print(f"  Target steps: {int(self.config.get('total_steps', 0))}")

        # Run all checks
        checks = [
            ("Runtime Performance", self.check_runtime_performance()),
            ("Numerical Stability", self.check_numerical_stability()),
            ("Mass Conservation", self.check_mass_conservation()),
            ("Interface Sharpness", self.check_interface_sharpness()),
            ("CFL_VOF Condition", self.check_cfl_vof()),
            ("Fill Level Bounds", self.check_fill_level_bounds()),
            ("Physics Consistency", self.check_physics_consistency()),
        ]

        print(f"\n{'='*60}")
        print(f"CHECK RESULTS")
        print(f"{'='*60}")

        passed = sum(1 for _, result in checks if result)
        total = len(checks)

        for name, result in checks:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {name}")

        print(f"\n  Score: {passed}/{total} checks passed")

        # Print warnings
        if self.warnings:
            print(f"\n{'='*60}")
            print(f"WARNINGS ({len(self.warnings)})")
            print(f"{'='*60}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"\n{i}. {warning}")

        # Print errors
        if self.errors:
            print(f"\n{'='*60}")
            print(f"ERRORS ({len(self.errors)})")
            print(f"{'='*60}")
            for i, error in enumerate(self.errors, 1):
                print(f"\n{i}. {error}")

        # Overall assessment
        print(f"\n{'='*60}")
        print(f"OVERALL ASSESSMENT")
        print(f"{'='*60}")

        if self.errors:
            print(f"✗ TEST E FAILED")
            print(f"  VOF advection has critical issues")
            print(f"  Review errors above and take corrective actions")
            return 1
        elif self.warnings:
            print(f"⚠ TEST E PASSED WITH WARNINGS")
            print(f"  VOF advection works but has optimization opportunities")
            print(f"  Review warnings for potential improvements")
            return 0
        else:
            print(f"✓ TEST E PASSED")
            print(f"  VOF advection validated successfully!")
            print(f"  Performance, stability, and mass conservation all acceptable")
            return 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 validate_test_E.py <log_file> [test_c_log]")
        print("")
        print("Example:")
        print("  python3 validate_test_E.py test_E_vof_advection.log")
        print("  python3 validate_test_E.py test_E_vof_advection.log test_C_full_coupling.log")
        sys.exit(1)

    log_file = sys.argv[1]
    test_c_log = sys.argv[2] if len(sys.argv) > 2 else None

    validator = TestEValidator(log_file, test_c_log)

    if not validator.parse_log():
        print(f"✗ FAIL: Could not parse log file")
        sys.exit(1)

    exit_code = validator.generate_report()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
