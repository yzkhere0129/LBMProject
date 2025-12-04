#!/usr/bin/env python3
"""
Energy Balance Analyzer for LPBF Steady-State Testing

This script analyzes energy balance CSV output from Tier 2/3 tests and
provides automated GO/NO-GO assessment for Week 3 readiness.

Usage:
    python3 analyze_energy_balance.py <energy_balance.csv>

Example:
    cd /home/yzk/LBMProject/build/tier2_energy_balance
    python3 ../../scripts/analyze_energy_balance.py energy_balance.csv

Output:
    - Quantitative metrics printed to stdout
    - PNG plots saved to current directory
    - GO/NO-GO recommendation
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_energy_data(filename):
    """Load energy balance CSV (skip comment lines starting with #)"""
    try:
        data = np.loadtxt(filename, comments='#')
        return data
    except Exception as e:
        print(f"ERROR: Failed to load {filename}")
        print(f"  {e}")
        sys.exit(1)

def analyze_energy_balance(data, tier='Tier2'):
    """
    Analyze energy balance data and assess GO/NO-GO criteria

    CSV columns (from energy_balance.h):
     0: time [s]
     1: step [-]
     2: E_thermal [J]
     3: E_kinetic [J]
     4: E_latent [J]
     5: E_total [J]
     6: P_laser [W]
     7: P_evap [W]
     8: P_rad [W]
     9: P_substrate [W]
    10: dE/dt_computed [W]
    11: dE/dt_balance [W]
    12: error_percent [%]
    """

    # Extract columns
    time_s = data[:, 0]
    step = data[:, 1].astype(int)
    E_thermal = data[:, 2]
    E_kinetic = data[:, 3]
    E_latent = data[:, 4]
    E_total = data[:, 5]
    P_laser = data[:, 6]
    P_evap = data[:, 7]
    P_rad = data[:, 8]
    P_substrate = data[:, 9]
    dE_dt_computed = data[:, 10]
    dE_dt_balance = data[:, 11]
    error_pct = data[:, 12]

    # Convert time to microseconds for readability
    time_us = time_s * 1e6

    # Convert energy to microjoules for plotting
    E_total_uJ = E_total * 1e6

    print("\n" + "="*70)
    print(f"  ENERGY BALANCE ANALYSIS - {tier}")
    print("="*70)

    # Basic statistics
    print(f"\nDataset Summary:")
    print(f"  Total timesteps:     {len(time_s)}")
    print(f"  Time range:          {time_us[0]:.1f} - {time_us[-1]:.1f} μs")
    print(f"  Physical duration:   {time_us[-1] - time_us[0]:.1f} μs")
    print(f"  Sampling interval:   {np.mean(np.diff(time_us)):.2f} μs")

    # Energy components (final state)
    print(f"\nFinal Energy State (t = {time_us[-1]:.1f} μs):")
    print(f"  E_thermal:  {E_thermal[-1]*1e6:10.3f} μJ  ({E_thermal[-1]/E_total[-1]*100:5.1f}%)")
    print(f"  E_kinetic:  {E_kinetic[-1]*1e6:10.3f} μJ  ({E_kinetic[-1]/E_total[-1]*100:5.1f}%)")
    print(f"  E_latent:   {E_latent[-1]*1e6:10.3f} μJ  ({E_latent[-1]/E_total[-1]*100:5.1f}%)")
    print(f"  E_total:    {E_total[-1]*1e6:10.3f} μJ")

    # Power balance (average over last 20% of simulation)
    late_mask = time_us > (time_us[-1] * 0.8)
    P_laser_avg = np.mean(P_laser[late_mask])
    P_evap_avg = np.mean(P_evap[late_mask])
    P_rad_avg = np.mean(P_rad[late_mask])
    P_sub_avg = np.mean(P_substrate[late_mask])
    P_total_out = P_evap_avg + P_rad_avg + P_sub_avg

    print(f"\nPower Balance (average over last 20% of run):")
    print(f"  P_laser:        {P_laser_avg:8.3f} W  (input)")
    print(f"  P_evaporation:  {P_evap_avg:8.3f} W  (output)")
    print(f"  P_radiation:    {P_rad_avg:8.3f} W  (output)")
    print(f"  P_substrate:    {P_sub_avg:8.3f} W  (output)")
    print(f"  P_total_out:    {P_total_out:8.3f} W")
    print(f"  Imbalance:      {P_laser_avg - P_total_out:8.3f} W  ({abs(P_laser_avg-P_total_out)/P_laser_avg*100:.1f}%)")

    # Convergence analysis
    print(f"\nConvergence Analysis:")

    # Split into early and late phases
    mid_idx = len(time_us) // 2
    early_mask = np.arange(len(time_us)) < mid_idx
    late_mask = ~early_mask

    # dE/dt statistics
    dE_dt_mean_early = np.mean(np.abs(dE_dt_computed[early_mask]))
    dE_dt_mean_late = np.mean(np.abs(dE_dt_computed[late_mask]))
    dE_dt_std_early = np.std(dE_dt_computed[early_mask])
    dE_dt_std_late = np.std(dE_dt_computed[late_mask])

    print(f"  Mean |dE/dt| (first half):  {dE_dt_mean_early:7.3f} W")
    print(f"  Mean |dE/dt| (second half): {dE_dt_mean_late:7.3f} W")
    print(f"  Reduction factor:           {dE_dt_mean_early/dE_dt_mean_late:7.2f}x")

    print(f"\n  Std(dE/dt) (first half):    {dE_dt_std_early:7.3f} W")
    print(f"  Std(dE/dt) (second half):   {dE_dt_std_late:7.3f} W")

    # Error statistics
    error_mean_early = np.mean(error_pct[early_mask])
    error_mean_late = np.mean(error_pct[late_mask])
    error_max = np.max(error_pct)
    error_final = error_pct[-1]

    print(f"\nEnergy Balance Error:")
    print(f"  Mean error (first half):   {error_mean_early:6.2f}%")
    print(f"  Mean error (second half):  {error_mean_late:6.2f}%")
    print(f"  Maximum error:             {error_max:6.2f}%")
    print(f"  Final error:               {error_final:6.2f}%")

    # Steady-state assessment (last 20% of run)
    ss_mask = time_us > (time_us[-1] * 0.8)
    dE_dt_ss_mean = np.mean(dE_dt_computed[ss_mask])
    dE_dt_ss_std = np.std(dE_dt_computed[ss_mask])
    error_ss_mean = np.mean(error_pct[ss_mask])

    print(f"\nSteady-State Assessment (last 20% of run):")
    print(f"  Mean dE/dt:     {dE_dt_ss_mean:7.3f} ± {dE_dt_ss_std:.3f} W")
    print(f"  Mean error:     {error_ss_mean:7.2f}%")

    # GO/NO-GO decision logic
    print("\n" + "="*70)
    print("  GO/NO-GO ASSESSMENT")
    print("="*70 + "\n")

    score = 0
    max_score = 0

    # Criterion 1: Convergence (dE/dt decreasing)
    max_score += 30
    if dE_dt_mean_late < dE_dt_mean_early * 0.5:
        score += 30
        print("[✓] Criterion 1: PASS - dE/dt decreased by > 50%")
        print(f"    {dE_dt_mean_early:.3f} W → {dE_dt_mean_late:.3f} W")
    elif dE_dt_mean_late < dE_dt_mean_early * 0.75:
        score += 20
        print("[~] Criterion 1: PARTIAL - dE/dt decreased by 25-50%")
        print(f"    {dE_dt_mean_early:.3f} W → {dE_dt_mean_late:.3f} W")
    else:
        print("[✗] Criterion 1: FAIL - dE/dt not converging")
        print(f"    {dE_dt_mean_early:.3f} W → {dE_dt_mean_late:.3f} W (< 25% reduction)")

    # Criterion 2: Energy balance error
    max_score += 40
    if error_ss_mean < 5.0:
        score += 40
        print(f"[✓] Criterion 2: PASS - Steady-state error < 5% ({error_ss_mean:.2f}%)")
    elif error_ss_mean < 10.0:
        score += 30
        print(f"[~] Criterion 2: PARTIAL - Steady-state error 5-10% ({error_ss_mean:.2f}%)")
    elif error_ss_mean < 15.0:
        score += 15
        print(f"[~] Criterion 2: MARGINAL - Steady-state error 10-15% ({error_ss_mean:.2f}%)")
    else:
        print(f"[✗] Criterion 2: FAIL - Steady-state error > 15% ({error_ss_mean:.2f}%)")

    # Criterion 3: Steady-state trend
    max_score += 30
    if abs(dE_dt_ss_mean) < 1.0:
        score += 30
        print(f"[✓] Criterion 3: PASS - Near steady-state (|dE/dt| < 1 W)")
    elif abs(dE_dt_ss_mean) < 2.0:
        score += 20
        print(f"[~] Criterion 3: PARTIAL - Approaching steady-state (|dE/dt| < 2 W)")
    elif abs(dE_dt_ss_mean) < 5.0:
        score += 10
        print(f"[~] Criterion 3: MARGINAL - Far from steady-state (|dE/dt| = {abs(dE_dt_ss_mean):.2f} W)")
    else:
        print(f"[✗] Criterion 3: FAIL - Not converging (|dE/dt| = {abs(dE_dt_ss_mean):.2f} W)")

    # Final decision
    print(f"\n{'='*70}")
    print(f"  FINAL SCORE: {score}/{max_score} ({score/max_score*100:.1f}%)")
    print(f"{'='*70}\n")

    if score >= 90:
        decision = "FULL GO"
        color = "\033[92m"  # Green
        recommendation = "Proceed to Week 3 with high confidence."
    elif score >= 70:
        decision = "CONDITIONAL GO"
        color = "\033[93m"  # Yellow
        recommendation = "Proceed to Week 3 with caution. Monitor energy balance closely."
    elif score >= 50:
        decision = "MARGINAL"
        color = "\033[93m"  # Yellow
        recommendation = "Consider running Tier 3 for full validation before Week 3."
    else:
        decision = "NO GO"
        color = "\033[91m"  # Red
        recommendation = "Fix energy conservation issues before proceeding to Week 3."

    reset = "\033[0m"
    print(f"{color}DECISION: {decision}{reset}")
    print(f"  {recommendation}\n")

    return {
        'time_us': time_us,
        'E_total_uJ': E_total_uJ,
        'dE_dt_computed': dE_dt_computed,
        'dE_dt_balance': dE_dt_balance,
        'error_pct': error_pct,
        'P_laser': P_laser,
        'P_evap': P_evap,
        'P_rad': P_rad,
        'P_substrate': P_substrate,
        'decision': decision,
        'score': score,
        'max_score': max_score
    }

def plot_energy_balance(results, output_file='energy_balance_analysis.png'):
    """Generate comprehensive energy balance plots"""

    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Total energy vs time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(results['time_us'], results['E_total_uJ'], 'b-', linewidth=1.5)
    ax1.set_xlabel('Time [μs]')
    ax1.set_ylabel('Total Energy [μJ]')
    ax1.set_title('Total Energy vs Time')
    ax1.grid(True, alpha=0.3)

    # Plot 2: dE/dt comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(results['time_us'], results['dE_dt_computed'], 'b-',
             label='dE/dt (computed)', linewidth=1.5)
    ax2.plot(results['time_us'], results['dE_dt_balance'], 'r--',
             label='dE/dt (balance)', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time [μs]')
    ax2.set_ylabel('Power [W]')
    ax2.set_title('Energy Balance Comparison')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error percentage
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(results['time_us'], results['error_pct'], 'r-', linewidth=1.5)
    ax3.axhline(y=5.0, color='g', linestyle='--', label='5% (target)', linewidth=1.5)
    ax3.axhline(y=10.0, color='orange', linestyle='--', label='10% (acceptable)', linewidth=1.5)
    ax3.set_xlabel('Time [μs]')
    ax3.set_ylabel('Error [%]')
    ax3.set_title('Energy Balance Error')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, min(30, results['error_pct'].max() * 1.1)])

    # Plot 4: Power terms stacked
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(results['time_us'], results['P_laser'], 'r-',
             label='Laser (input)', linewidth=2)
    ax4.plot(results['time_us'], results['P_evap'], 'b-',
             label='Evaporation', linewidth=1.5)
    ax4.plot(results['time_us'], results['P_rad'], 'g-',
             label='Radiation', linewidth=1.5)
    ax4.plot(results['time_us'], results['P_substrate'], 'm-',
             label='Substrate', linewidth=1.5)
    ax4.set_xlabel('Time [μs]')
    ax4.set_ylabel('Power [W]')
    ax4.set_title('Power Terms')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # Plot 5: dE/dt trend (focus on convergence)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(results['time_us'], results['dE_dt_computed'], 'b-', linewidth=1.5)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax5.axhline(y=1.0, color='g', linestyle='--', label='±1 W (target)', linewidth=1.5)
    ax5.axhline(y=-1.0, color='g', linestyle='--', linewidth=1.5)
    ax5.set_xlabel('Time [μs]')
    ax5.set_ylabel('dE/dt [W]')
    ax5.set_title('Energy Rate (should approach 0)')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Decision summary (text)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    decision_text = f"""
    DECISION: {results['decision']}

    Score: {results['score']}/{results['max_score']} ({results['score']/results['max_score']*100:.1f}%)

    Criteria:
    • Convergence (dE/dt ↓)
    • Energy error < 10%
    • Steady-state trend

    Final State:
    • dE/dt: {results['dE_dt_computed'][-1]:.3f} W
    • Error: {results['error_pct'][-1]:.2f}%
    • Time: {results['time_us'][-1]:.1f} μs
    """

    # Color-code decision
    if results['decision'] == 'FULL GO':
        bbox_color = 'lightgreen'
    elif 'CONDITIONAL' in results['decision']:
        bbox_color = 'yellow'
    else:
        bbox_color = 'lightcoral'

    ax6.text(0.1, 0.5, decision_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Energy balance plots saved to: {output_file}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_energy_balance.py <energy_balance.csv>")
        print("\nExample:")
        print("  cd /home/yzk/LBMProject/build/tier2_energy_balance")
        print("  python3 ../../scripts/analyze_energy_balance.py energy_balance.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not Path(csv_file).exists():
        print(f"ERROR: File not found: {csv_file}")
        sys.exit(1)

    # Determine tier from filename/path
    tier = 'Tier2'
    if 'tier1' in csv_file.lower():
        tier = 'Tier1'
    elif 'tier3' in csv_file.lower() or 'steady_state' in csv_file.lower():
        tier = 'Tier3'

    # Load and analyze
    data = load_energy_data(csv_file)
    results = analyze_energy_balance(data, tier=tier)

    # Generate plots
    output_png = str(Path(csv_file).parent / 'energy_balance_analysis.png')
    plot_energy_balance(results, output_file=output_png)

    print("="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == '__main__':
    main()
