"""Three-way comparison: AMR-OFF vs AMR-ON (±0.10c patch BUG) vs AMR-ON (auto-expand FIX).

Run after the 30k patch-fix rerun completes. Outputs the final verdict
on whether the patch truncation was the root cause of the Cd anomaly.

  images/patch_fix_verdict.png   — Cl/Cd/L/D bar with lit reference
  images/patch_fix_traj.png       — Cl(t), Cd(t), mass(t) for all three
  Console: gap analysis vs §12/§13 expected outcome
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"
CASES = [
    ("AMR-OFF (baseline)",                "output_30k_amr_off",            "C0"),
    ("AMR-ON ±0.10c bilin (BUG patch)",   "output_30k_amr_on_bilin_time",  "C3"),
    ("AMR-ON expand bilin (§13 FIX)",     "output_30k_amr_patch_fixed",    "C1"),
    ("AMR-ON 1D-cubic R1b",               "output_30k_amr_r1b",            "C2"),
]
LIT_CL = 0.57
LIT_CD = 0.15
LIT_LD = LIT_CL / LIT_CD


def settled(p):
    if not os.path.exists(p): return None
    try:
        d = np.loadtxt(p, delimiter=",", skiprows=1)
    except Exception:
        return None
    if d.ndim < 2 or d.shape[0] < 10: return None
    n = d.shape[0]; s = int(n*2/3)
    return dict(
        step=d[:, 0], t=d[:, 1],
        Cd=d[:, 7], Cl=d[:, 8], mass=d[:, 9],
        Cd_s=float(d[s:, 7].mean()), Cd_e=float(d[s:, 7].std(ddof=1)),
        Cl_s=float(d[s:, 8].mean()), Cl_e=float(d[s:, 8].std(ddof=1)),
        n_settled=n - s, n_total=n,
    )


results = []
print(f"{'case':32s} {'N':>5s} {'Cl':>9s} {'Cd':>9s} {'L/D':>7s} {'Cl_gap':>7s} {'L/D_gap':>8s}")
print("-" * 90)
print(f"{'literature (Kurtulus extrap)':32s}     {'-':>5s} {LIT_CL:>9.4f} {LIT_CD:>9.4f} {LIT_LD:>7.3f}")
for label, sub, color in CASES:
    r = settled(os.path.join(ROOT, sub, "forces.csv"))
    if r is None:
        print(f"{label:32s} {'MISS':>5s}")
        results.append(None); continue
    r["L_D"] = r["Cl_s"] / r["Cd_s"]
    r["cl_gap"] = abs(r["Cl_s"] - LIT_CL) / LIT_CL * 100
    r["ld_gap"] = abs(r["L_D"] - LIT_LD) / LIT_LD * 100
    r["label"] = label; r["color"] = color
    results.append(r)
    print(f"{label:32s} {r['n_total']:>5d} "
          f"{r['Cl_s']:>6.4f}±{r['Cl_e']:.3f} "
          f"{r['Cd_s']:>6.4f}±{r['Cd_e']:.3f} {r['L_D']:>7.3f} "
          f"{r['cl_gap']:>6.1f}% {r['ld_gap']:>7.1f}%")

valid = [r for r in results if r]
if len(valid) < 2:
    raise SystemExit("Need ≥2 cases for comparison")

# ===== Verdict =====
if len(valid) >= 4:
    bilin_fix = valid[2]; r1b = valid[3]
    print("\n=== R1b verdict (vs §13 bilin-FIX) ===")
    cd_change = (r1b["Cd_s"] - bilin_fix["Cd_s"]) / bilin_fix["Cd_s"] * 100
    cl_change = (r1b["Cl_s"] - bilin_fix["Cl_s"]) / bilin_fix["Cl_s"] * 100
    rms_change = (r1b["Cl_e"] - bilin_fix["Cl_e"]) / bilin_fix["Cl_e"] * 100
    print(f"  Cl change bilin → R1b:    {cl_change:+.1f}%   (closer to lit 0.57 = better)")
    print(f"  Cd change bilin → R1b:    {cd_change:+.1f}%   (within plateau band → physical)")
    print(f"  Cl_rms change:            {rms_change:+.1f}%   (smaller = more stable)")
    print(f"  Cl gap to lit 0.57:       {r1b['cl_gap']:.1f}%   (vs bilin-FIX {bilin_fix['cl_gap']:.1f}%)")
    print()
    if r1b["cl_gap"] < 15 and r1b["Cl_e"] < bilin_fix["Cl_e"] * 1.5:
        print(f"  ✓ Cl gap < 15% with controlled rms — R1b SUCCESS")
    elif r1b["cl_gap"] < bilin_fix["cl_gap"]:
        print(f"  ~ Cl improved (gap {bilin_fix['cl_gap']:.0f}%→{r1b['cl_gap']:.0f}%) but still > 15%")
    else:
        print(f"  ✗ Cl gap did not improve — accept bilin-FIX")
    if r1b["Cl_e"] > bilin_fix["Cl_e"] * 2:
        print(f"  ✗ Cl_rms blown up — instability (compare to R1 27× bicubic disaster)")

# ===== Plot 1: bars =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
labels = [r["label"] for r in valid]
colors = [r["color"] for r in valid]
cls    = [r["Cl_s"]  for r in valid]; cls_e = [r["Cl_e"] for r in valid]
cds    = [r["Cd_s"]  for r in valid]; cds_e = [r["Cd_e"] for r in valid]
lds    = [r["L_D"]   for r in valid]
xs = np.arange(len(labels))

axes[0].bar(xs, cls, yerr=cls_e, color=colors, edgecolor='k', capsize=5)
axes[0].axhline(LIT_CL, color='k', ls='--', lw=2, label=f'lit~{LIT_CL}')
axes[0].axhspan(LIT_CL*0.75, LIT_CL*1.25, color='gray', alpha=0.08, label='±25%')
axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
axes[0].set_ylabel("Cl_settled"); axes[0].set_title("Cl"); axes[0].legend(fontsize=8)
for x, c in zip(xs, cls): axes[0].text(x, c, f"{c:.3f}", ha='center', va='bottom', fontweight='bold')
axes[0].grid(True, ls=':', alpha=0.4, axis='y')

axes[1].bar(xs, cds, yerr=cds_e, color=colors, edgecolor='k', capsize=5)
axes[1].axhline(LIT_CD, color='k', ls='--', lw=2, label=f'lit~{LIT_CD}')
axes[1].axhspan(LIT_CD*0.7, LIT_CD*1.3, color='gray', alpha=0.08, label='lit ±30%')
axes[1].set_xticks(xs); axes[1].set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
axes[1].set_ylabel("Cd_settled"); axes[1].set_title("Cd — key test of fix"); axes[1].legend(fontsize=8)
for x, c in zip(xs, cds): axes[1].text(x, c, f"{c:.3f}", ha='center', va='bottom', fontweight='bold')
axes[1].grid(True, ls=':', alpha=0.4, axis='y')

axes[2].bar(xs, lds, color=colors, edgecolor='k')
axes[2].axhline(LIT_LD, color='k', ls='--', lw=2, label=f'lit~{LIT_LD:.2f}')
axes[2].set_xticks(xs); axes[2].set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
axes[2].set_ylabel("L/D"); axes[2].set_title("L/D — physical credibility"); axes[2].legend(fontsize=8)
for x, c in zip(xs, lds): axes[2].text(x, c, f"{c:.2f}", ha='center', va='bottom', fontweight='bold')
axes[2].grid(True, ls=':', alpha=0.4, axis='y')

fig.suptitle("30k AMR — patch truncation fix verdict", fontsize=12, fontweight='bold')
plt.tight_layout()
out1 = os.path.join(ROOT, "images", "patch_fix_verdict.png")
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=130, bbox_inches='tight'); print(f"\nSaved: {out1}")

# ===== Plot 2: trajectories =====
fig2, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for r in valid:
    ax[0].plot(r["step"], r["Cl"], lw=1.0, color=r["color"], label=r["label"], alpha=0.85)
    ax[1].plot(r["step"], r["Cd"], lw=1.0, color=r["color"], label=r["label"], alpha=0.85)
    drift = (r["mass"] - r["mass"][0]) / r["mass"][0]
    ax[2].plot(r["step"], drift, lw=1.0, color=r["color"], label=r["label"], alpha=0.85)
ax[0].axhline(LIT_CL, color='k', ls='--', lw=1.5, alpha=0.6, label=f'lit~{LIT_CL}')
ax[0].set_ylabel("Cl"); ax[0].legend(loc='lower right', fontsize=8); ax[0].grid(True, ls=':', alpha=0.4)
ax[1].axhline(LIT_CD, color='k', ls='--', lw=1.5, alpha=0.6, label=f'lit~{LIT_CD}')
ax[1].set_ylabel("Cd"); ax[1].legend(loc='upper right', fontsize=8); ax[1].grid(True, ls=':', alpha=0.4)
ax[2].set_xlabel("step"); ax[2].set_ylabel("Mass drift (rel)")
ax[2].axhspan(-1e-5, 1e-5, color='gray', alpha=0.15, label='FP32 noise')
ax[2].legend(loc='upper right', fontsize=8); ax[2].grid(True, ls=':', alpha=0.4)
fig2.suptitle("30k trajectories — AMR patch-fix vs prior", fontsize=12)
plt.tight_layout()
out2 = os.path.join(ROOT, "images", "patch_fix_traj.png")
plt.savefig(out2, dpi=130, bbox_inches='tight'); print(f"Saved: {out2}")
