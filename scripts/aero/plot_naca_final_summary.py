"""NACA0012 final summary plot — pre-fix vs post-fix, D/dx=40 vs D/dx=80.

Relabels pre-fix runs into standard aero convention (our pre-fix --alpha = std -α).
Reference: Kurtulus 2015, NACA0012 Re=1000:
    α=+8°: Cl_avg ≈ +0.49, Cd_avg ≈ 0.21
"""
from __future__ import annotations
import csv, os, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/yzk/CompressibleCFD'

# (label, csv_path, sign_flip_due_to_pre_fix_convention, color, linestyle)
runs = [
    ('STAIR std α=-8 (D/dx=40, pre-fix)',
     f'{BASE}/output_naca_a8_kurtulus/forces.csv',     False, 'C0', '-'),
    ('QBB   std α=-8 (D/dx=40, pre-fix)',
     f'{BASE}/output_qbb_a8_steady/forces.csv',        False, 'C1', '-'),
    ('QBB   std α=+8 (D/dx=40, pre-fix --alpha -8)',
     f'{BASE}/output_qbb_aminus8_steady/forces.csv',   False, 'C2', '-'),
    ('QBB   std α=+8 (D/dx=80, POST-fix --alpha +8)',
     f'{BASE}/output_qbb_dx80_a8_prod/forces.csv',     False, 'C3', '-'),
]

KURTULUS_CL = 0.49
KURTULUS_CD = 0.21

fig, (ax_cl, ax_cd) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
ax_cl.set_xlim(2000, 32000)
ax_cl.set_ylim(-0.9, 0.9)
ax_cd.set_xlim(2000, 32000)
ax_cd.set_ylim(0.0, 0.30)
summary = []
for name, path, _flip, c, ls in runs:
    if not os.path.exists(path):
        print(f"  missing: {path}")
        continue
    rows = list(csv.DictReader(open(path)))
    s  = [int(r['step'])  for r in rows]
    cl = [float(r['Cl']) for r in rows]
    cd = [float(r['Cd']) for r in rows]
    ax_cl.plot(s, cl, c+ls, label=name, lw=0.8, alpha=0.85)
    ax_cd.plot(s, cd, c+ls, label=name, lw=0.8, alpha=0.85)
    # half-window mean (last half of run)
    n_total = len(s)
    cutoff  = s[n_total // 2]
    cl_h = [v for st_, v in zip(s, cl) if st_ >= cutoff]
    cd_h = [v for st_, v in zip(s, cd) if st_ >= cutoff]
    if cl_h:
        cl_mean, cl_std = st.mean(cl_h), st.stdev(cl_h)
        cd_mean = st.mean(cd_h)
        ax_cl.axhline(cl_mean, color=c, linestyle=':', alpha=0.6, lw=1)
        ax_cd.axhline(cd_mean, color=c, linestyle=':', alpha=0.6, lw=1)
        summary.append((name, cl_mean, cl_std, cd_mean, len(cl_h), cutoff, s[-1]))

ax_cl.axhline(+KURTULUS_CL, color='k', linestyle='--', alpha=0.7, lw=1.2,
              label=f'Kurtulus std α=+8°  →  Cl=+{KURTULUS_CL}')
ax_cl.axhline(-KURTULUS_CL, color='k', linestyle='--', alpha=0.4, lw=1,
              label=f'Kurtulus std α=-8°  →  Cl=-{KURTULUS_CL}')
ax_cd.axhline(KURTULUS_CD, color='k', linestyle='--', alpha=0.7, lw=1.2,
              label=f'Kurtulus std α=±8°  →  Cd={KURTULUS_CD}')

ax_cl.axhline(0, color='gray', lw=0.5)
ax_cl.set_ylabel('Cl')
ax_cl.set_title('NACA0012 D3Q27 Cumulant Re=1000  —  std aero convention')
ax_cl.legend(loc='best', fontsize=8)
ax_cl.grid(True, alpha=0.3)

ax_cd.set_xlabel('step')
ax_cd.set_ylabel('Cd')
ax_cd.legend(loc='best', fontsize=8)
ax_cd.grid(True, alpha=0.3)

plt.tight_layout()
out = f'{BASE}/naca_final_summary.png'
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}\n')
print('Half-window means (last half of each run):')
print(f'  {"case":52s}  {"Cl_mean":>9s}  {"Cl_std":>7s}  {"Cd_mean":>8s}  N    steps')
for name, cl_m, cl_s, cd_m, n, cut, end in summary:
    print(f'  {name:52s}  {cl_m:+9.4f}  {cl_s:7.4f}  {cd_m:8.4f}  {n:3d}  [{cut},{end}]')

# Kurtulus magnitude analysis
print(f'\nKurtulus reference: |Cl|={KURTULUS_CL}, |Cd|={KURTULUS_CD}')
for name, cl_m, _, cd_m, _, _, _ in summary:
    print(f'  {name:52s}  |Cl|/Kurt={abs(cl_m)/KURTULUS_CL:.0%}  |Cd|/Kurt={cd_m/KURTULUS_CD:.0%}')
