#!/usr/bin/env python3
"""
Real-time Simulation Monitor
=============================

Monitors simulation progress and provides real-time diagnostics:
- Temperature and velocity trends
- Energy balance tracking
- CFL condition monitoring
- Early warning for instabilities

Usage:
    python monitor_simulation.py <log_file> [--interval 5]
"""

import sys
import time
import re
import math
from pathlib import Path
from typing import List, Dict, Optional
from collections import deque

class SimulationMonitor:
    """Real-time simulation monitoring"""

    def __init__(self, log_file: str, interval: int = 5):
        self.log_file = Path(log_file)
        self.interval = interval
        self.last_position = 0
        self.data_history = deque(maxlen=20)  # Keep last 20 data points
        self.config = {}

    def parse_config_line(self, line: str):
        """Extract configuration from log line"""
        patterns = {
            'dx': r'dx\s*=\s*([\d.e+-]+)',
            'dt': r'dt\s*=\s*([\d.e+-]+)',
            'laser_power': r'laser_power\s*=\s*([\d.e+-]+)',
            'emissivity': r'emissivity\s*=\s*([\d.]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                self.config[key] = float(match.group(1))

    def parse_step_line(self, line: str) -> Optional[Dict]:
        """Parse timestep data from log line"""
        # Pattern: Step 100/1000: T_max=1234.5 K, v_max=0.123 mm/s
        pattern = r'Step\s+(\d+)/(\d+):.*?T_max\s*=\s*([\d.e+-]+)\s*K.*?v_max\s*=\s*([\d.e+-]+)\s*mm/s'
        match = re.search(pattern, line)

        if match:
            return {
                'step': int(match.group(1)),
                'total_steps': int(match.group(2)),
                'T_max': float(match.group(3)),
                'v_max': float(match.group(4))
            }
        return None

    def read_new_lines(self) -> List[str]:
        """Read new lines from log file"""
        if not self.log_file.exists():
            return []

        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
                return new_lines
        except Exception as e:
            print(f"Error reading log: {e}")
            return []

    def check_cfl_condition(self, v_max: float) -> str:
        """Check CFL condition"""
        if not self.config:
            return "?"

        dx = self.config.get('dx', 2e-6)
        dt = self.config.get('dt', 1e-7)
        alpha = 5.8e-6  # Thermal diffusivity

        v_max_ms = v_max * 1e-3  # mm/s → m/s

        CFL_adv = v_max_ms * dt / dx
        CFL_diff = alpha * dt / (dx * dx)

        status = "OK"
        if CFL_adv > 0.5:
            status = "WARN"
        if CFL_adv > 1.0:
            status = "CRITICAL"

        return f"{CFL_adv:.3f} ({status})"

    def estimate_energy_balance(self, T_max: float) -> str:
        """Estimate energy balance"""
        if not self.config:
            return "?"

        P_laser = self.config.get('laser_power', 195.0)
        emissivity = self.config.get('emissivity', 0.3)

        # Stefan-Boltzmann radiation
        spot_radius = 50e-6
        A_spot = math.pi * spot_radius**2
        T_amb = 300.0
        sigma = 5.67e-8

        P_rad = emissivity * sigma * A_spot * (T_max**4 - T_amb**4)

        residual = abs(P_laser - P_rad) / P_laser

        return f"{100*residual:.1f}%"

    def detect_trends(self) -> Dict[str, str]:
        """Detect temperature and velocity trends"""
        if len(self.data_history) < 3:
            return {'T_trend': '...', 'v_trend': '...'}

        recent = list(self.data_history)[-3:]

        # Temperature trend
        T_values = [d['T_max'] for d in recent]
        if T_values[-1] > T_values[0] * 1.1:
            T_trend = "↑ RISING"
        elif T_values[-1] < T_values[0] * 0.9:
            T_trend = "↓ FALLING"
        else:
            T_trend = "→ STABLE"

        # Velocity trend
        v_values = [d['v_max'] for d in recent]
        if v_values[-1] > v_values[0] * 1.5:
            v_trend = "↑ RISING"
        elif v_values[-1] < v_values[0] * 0.7:
            v_trend = "↓ FALLING"
        else:
            v_trend = "→ STABLE"

        return {'T_trend': T_trend, 'v_trend': v_trend}

    def check_for_warnings(self, data: Dict) -> List[str]:
        """Check for warning conditions"""
        warnings = []

        # Check for NaN/Inf
        if math.isnan(data['T_max']) or math.isinf(data['T_max']):
            warnings.append("CRITICAL: T_max is NaN/Inf - DIVERGENCE!")

        if math.isnan(data['v_max']) or math.isinf(data['v_max']):
            warnings.append("CRITICAL: v_max is NaN/Inf - DIVERGENCE!")

        # Check temperature bounds
        if data['T_max'] > 100000:
            warnings.append("CRITICAL: Temperature > 100,000 K - RUNAWAY!")
        elif data['T_max'] > 50000:
            warnings.append("WARNING: Temperature > 50,000 K - Very high!")

        # Check velocity bounds
        if data['v_max'] > 10000:  # mm/s = 10 m/s
            warnings.append("CRITICAL: Velocity > 10 m/s - UNSTABLE!")
        elif data['v_max'] > 1000:
            warnings.append("WARNING: Velocity > 1 m/s - High!")

        # Check for sudden changes
        if len(self.data_history) > 1:
            prev = self.data_history[-1]
            T_change = abs(data['T_max'] - prev['T_max']) / prev['T_max']
            v_change = abs(data['v_max'] - prev['v_max']) / max(prev['v_max'], 1e-6)

            if T_change > 0.3:
                warnings.append(f"WARNING: Sudden T change ({100*T_change:.1f}%)")

            if v_change > 2.0 and data['v_max'] > 10:
                warnings.append(f"WARNING: Sudden v change ({100*v_change:.1f}%)")

        return warnings

    def print_status(self, data: Dict):
        """Print current status"""
        trends = self.detect_trends()
        cfl = self.check_cfl_condition(data['v_max'])
        energy = self.estimate_energy_balance(data['T_max'])
        warnings = self.check_for_warnings(data)

        # Progress
        progress = 100 * data['step'] / data['total_steps']
        bar_length = 30
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)

        # Clear line and print status
        print(f"\r[{bar}] {progress:.1f}%  ", end='')
        print(f"Step {data['step']}/{data['total_steps']}  ", end='')
        print(f"T={data['T_max']:.1f}K {trends['T_trend']}  ", end='')
        print(f"v={data['v_max']:.4f}mm/s {trends['v_trend']}  ", end='')
        print(f"CFL={cfl}  Energy={energy}  ", end='')

        # Print warnings on new line if any
        if warnings:
            print()
            for warning in warnings:
                print(f"  ⚠ {warning}")
            print("  ", end='')  # Restore position for next status line
        else:
            print(end='', flush=True)

    def monitor(self):
        """Main monitoring loop"""
        print("="*80)
        print("SIMULATION MONITOR - Press Ctrl+C to stop")
        print("="*80)
        print(f"Watching: {self.log_file}")
        print(f"Update interval: {self.interval}s")
        print()

        try:
            while True:
                new_lines = self.read_new_lines()

                for line in new_lines:
                    # Parse config
                    self.parse_config_line(line)

                    # Parse step data
                    data = self.parse_step_line(line)
                    if data:
                        self.data_history.append(data)
                        self.print_status(data)

                # Check if simulation finished
                if self.data_history and \
                   self.data_history[-1]['step'] == self.data_history[-1]['total_steps']:
                    print("\n\n✓ Simulation completed!")
                    break

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
        except Exception as e:
            print(f"\n\nError: {e}")

        # Final summary
        if self.data_history:
            print("\n" + "="*80)
            print("FINAL STATUS")
            print("="*80)
            final = self.data_history[-1]
            print(f"Final step:     {final['step']}/{final['total_steps']}")
            print(f"Final T_max:    {final['T_max']:.1f} K")
            print(f"Final v_max:    {final['v_max']:.4f} mm/s")

            trends = self.detect_trends()
            print(f"T trend:        {trends['T_trend']}")
            print(f"v trend:        {trends['v_trend']}")
            print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor_simulation.py <log_file> [--interval N]")
        print("Example: python monitor_simulation.py test_A_coupling.log --interval 5")
        sys.exit(1)

    log_file = sys.argv[1]
    interval = 5

    # Parse optional interval
    if '--interval' in sys.argv:
        try:
            idx = sys.argv.index('--interval')
            interval = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Invalid interval value")
            sys.exit(1)

    monitor = SimulationMonitor(log_file, interval)
    monitor.monitor()


if __name__ == '__main__':
    main()
