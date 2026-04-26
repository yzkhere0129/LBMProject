#!/usr/bin/env python3
"""Side-by-side LBM-old vs LBM-new comparison for one frame each.

Reuses the metric extractors from phase1_summary.py via subprocess so we don't
duplicate logic. Prints a table with the four headline metrics and the delta
(new - old) for each.

Headline metrics (same convention as phase1_summary.py):
  * v_z @ x_offset = -150 um (centerline, m/s, physical)
  * side-ridge peak Δh in band [laser_x-500, laser_x-100] (μm vs z0)
  * centerline Δh 95%ile in [laser_start+200, laser_x-100] (μm vs z0)
  * groove depth at x = laser_x - 200 um (z0 - z_center, μm)

Usage:
    python phase1_compare_baseline.py <new_vtk> <old_vtk> [z0_um] [laser_start_um] [v_scan]

Defaults: z0_um=160, laser_start_um=500, v_scan=0.8
"""
import sys
import re
import subprocess
from pathlib import Path

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

new_vtk         = sys.argv[1]
old_vtk         = sys.argv[2]
z0_um           = sys.argv[3] if len(sys.argv) > 3 else "160"
laser_start_um  = sys.argv[4] if len(sys.argv) > 4 else "500"
v_scan          = sys.argv[5] if len(sys.argv) > 5 else "0.8"

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY = SCRIPT_DIR / "phase1_summary.py"
if not SUMMARY.is_file():
    print(f"ERROR: cannot find {SUMMARY}")
    sys.exit(2)

def run_summary(vtk_path):
    cmd = [sys.executable, str(SUMMARY), vtk_path, z0_um, laser_start_um, v_scan]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"ERROR running phase1_summary.py on {vtk_path}:\n{p.stderr}")
        sys.exit(p.returncode)
    return p.stdout

def parse_metrics(text):
    """Pull headline numbers out of phase1_summary.py decision-table output.

    Patterns we accept (each shows a number followed by ' um' or ' m/s' or NaN):
        v_z @ -150 um (m/s)            +0.123
        side-ridge peak Δh             +5.20 um
        centerline Δh (95%ile)         +3.40 um
        groove depth @ -200 um         +1.80 um
    """
    out = {'vz_150': None, 'side_ridge_dh': None, 'center_dh_p95': None, 'groove': None}
    # Use a permissive number regex that matches signed floats or NaN.
    num_re = r'([+-]?\d+\.\d+|NaN)'
    pats = {
        'vz_150':        rf'v_z @ -150 um \(m/s\)\s+{num_re}',
        'side_ridge_dh': rf'side-ridge peak Δh\s+{num_re}',
        'center_dh_p95': rf'centerline Δh \(95%ile\)\s+{num_re}',
        'groove':        rf'groove depth @ -200 um\s+{num_re}',
    }
    for key, pat in pats.items():
        m = re.search(pat, text)
        if m:
            tok = m.group(1)
            try:
                out[key] = float(tok) if tok != 'NaN' else float('nan')
            except ValueError:
                out[key] = float('nan')
    return out

text_new = run_summary(new_vtk)
text_old = run_summary(old_vtk)
m_new = parse_metrics(text_new)
m_old = parse_metrics(text_old)

def fmt(v, units):
    if v is None:
        return f"{'(parse fail)':>14}"
    if v != v:    # NaN check
        return f"{'NaN':>14}"
    return f"{v:>+10.3f} {units}"

def fmt_delta(a, b, units):
    if a is None or b is None or a != a or b != b:
        return f"{'--':>14}"
    return f"{a - b:>+10.3f} {units}"

print("=" * 92)
print(f"PHASE-1 LBM COMPARE  (new - old)")
print("=" * 92)
print(f"  NEW : {new_vtk}")
print(f"  OLD : {old_vtk}")
print()
print(f"  {'metric':<32} {'NEW':>14} {'OLD':>14} {'Δ (NEW-OLD)':>16}")
print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*16}")
rows = [
    ('v_z @ -150 um',        'vz_150',        'm/s'),
    ('side-ridge peak Δh',   'side_ridge_dh', 'um '),
    ('centerline Δh (95%ile)','center_dh_p95', 'um '),
    ('groove depth @ -200um','groove',        'um '),
]
for label, key, units in rows:
    print(f"  {label:<32} {fmt(m_new[key], units):>14} {fmt(m_old[key], units):>14} {fmt_delta(m_new[key], m_old[key], units):>16}")
print("=" * 92)

# Quick verbal interpretation
def interp(label, dv, expect_increase=True, tol=0.5):
    if dv is None or dv != dv:
        return f"  {label:<32}: cannot evaluate (NaN)"
    if abs(dv) < tol:
        return f"  {label:<32}: ~unchanged (|Δ|={abs(dv):.2f})"
    if (dv > 0) == expect_increase:
        return f"  {label:<32}: improved (Δ={dv:+.2f})"
    return f"  {label:<32}: regressed (Δ={dv:+.2f})"

dvs = {k: (m_new[k] - m_old[k]) if (m_new[k] is not None and m_old[k] is not None
                                    and m_new[k] == m_new[k] and m_old[k] == m_old[k])
       else None for k in m_new}
print()
print("Interpretation (vs F3D-favorable direction):")
print(interp("v_z @ -150 um (want > 0)",       dvs['vz_150'],        True,  0.05))
print(interp("side-ridge dh (want toward +4)", dvs['side_ridge_dh'], True,  0.50))
print(interp("center Δh (want toward +4)",     dvs['center_dh_p95'], True,  0.50))
print(interp("groove (want toward 0, smaller)",dvs['groove'],        False, 0.50))
