#!/usr/bin/env python3
"""Phase-1 decision automation. Reads phase1_summary.py output for the
final phase-1 frame, applies the protocol's decision tree, prints next
phase to execute.

Decision tree (from night protocol):
  IF v_z@-150 > 0.10 m/s AND side-ridge < +8μm AND groove shallower
    -> SUCCESS: jump to Phase 4
  ELIF v_z@-150 > 0.05 m/s AND (ridge or center bad)
    -> PARTIAL: jump to Phase 2 (mass conservation)
  ELSE (v_z still ~0 OR NaN)
    -> FAIL: revert + Phase 3 (τ→0.55)

Usage:
  python phase1_decide.py <vtk_frame>
"""
import re
import sys
import subprocess
import os

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]

scripts_dir = os.path.dirname(os.path.abspath(__file__))
out = subprocess.run(['python3', f'{scripts_dir}/phase1_summary.py', path],
                     capture_output=True, text=True).stdout

def parse(field, line_pat):
    m = re.search(line_pat, out)
    if not m: return None
    try: return float(m.group(1))
    except: return None

vz_150 = parse('vz', r'v_z @ -150 um \(m/s\)\s+([+-]?[\d.]+)')
ridge_pk = parse('rp', r'side-ridge peak Δh\s+([+-]?[\d.]+) um')
ridge_100 = parse('r100', r'x_off=-100 um\s+->\s+z_side_max=[\d.]+ um, Δh=\s*([+-]?[\d.]+) um')
ridge_200 = parse('r200', r'x_off=-200 um\s+->\s+z_side_max=[\d.]+ um, Δh=\s*([+-]?[\d.]+) um')
delta_h_p95 = parse('dh', r'Δh \(95%ile\)\s+([+-]?[\d.]+) um')

# Detect NaN / empty
if vz_150 is None or delta_h_p95 is None:
    print("FAIL: cannot parse summary (NaN, crash, or empty pool)")
    print(out[-1000:])
    print("DECISION: PHASE 3 (BGK τ stretch) — assumes Phase 1 produced no usable trailing data")
    sys.exit(2)

# Use trailing-zone (not splash-region) ridge as signal: avg of -100/-200
trailing_ridge = (ridge_100 + ridge_200) / 2 if (ridge_100 is not None and ridge_200 is not None) else ridge_pk

print("=== Phase-1 metrics ===")
print(f"  v_z @ -150 μm     : {vz_150:+.3f} m/s")
print(f"  side ridge -100   : {ridge_100} μm")
print(f"  side ridge -200   : {ridge_200} μm")
print(f"  trailing ridge avg: {trailing_ridge:+.1f} μm")
print(f"  centerline Δh p95 : {delta_h_p95:+.2f} μm")
print()

# Decision — calibrated to actual baseline numbers seen earlier:
# baseline: v_z=-0.003, ridge=+2, Δh=-20.
# success = clear improvement on all three.
SUCCESS = (vz_150 > 0.10) and (delta_h_p95 > -8) and (trailing_ridge < 8)
PARTIAL = (vz_150 > 0.05) and (delta_h_p95 > -15)
FAIL    = (vz_150 < 0.05) or (delta_h_p95 < -18)

if SUCCESS:
    print("DECISION: SUCCESS — proceed to Phase 4 (final 2 ms validation)")
elif PARTIAL:
    print("DECISION: PARTIAL — proceed to Phase 2 (mass conservation check)")
else:
    print("DECISION: FAIL — proceed to Phase 3 (τ stretch test)")
