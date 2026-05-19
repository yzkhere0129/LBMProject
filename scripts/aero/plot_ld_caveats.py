"""L/D comparison on existing 30k forces — surfaces the Cd anomaly.

The 30k AMR comparison previously reported Cl gap 41%→16.5% as PASS.
But Cd went 0.13→0.24 (+87%) while lit Cd~0.15 — L/D dropped 2.60→1.97
(worse than AMR-OFF). Root cause: AMR patch ±0.10c cuts through lower
TE at α=8°, see images/amr_patch_truncation_bug.png.

Output: images/ld_audit_30k.png + console summary.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"
CASES = [
    ("AMR-OFF (no patch)",       "output_30k_amr_off",           "C0"),
    ("AMR-ON ±0.10c patch (BUG)", "output_30k_amr_on_bilin_time", "C3"),
]
LIT_CL = 0.57
LIT_CD = 0.15
LIT_LD = LIT_CL / LIT_CD


def settled(p):
    d = np.loadtxt(p, delimiter=",", skiprows=1); n = d.shape[0]; s = int(n*2/3)
    return dict(Cd=d[s:,7].mean(), Cd_rms=d[s:,7].std(ddof=1),
                Cl=d[s:,8].mean(), Cl_rms=d[s:,8].std(ddof=1), n=n-s)


results = []
print(f"{'case':30s} {'Cl':>8s} {'Cd':>8s} {'L/D':>8s} {'Cl_gap':>8s} {'L/D_gap':>9s}")
print("-" * 80)
print(f"{'literature (Kurtulus extrap)':30s} {LIT_CL:>8.4f} {LIT_CD:>8.4f} {LIT_LD:>8.3f}")
for label, sub, color in CASES:
    r = settled(os.path.join(ROOT, sub, "forces.csv"))
    r["L_D"] = r["Cl"] / r["Cd"]
    r["cl_gap"] = abs(r["Cl"] - LIT_CL) / LIT_CL * 100
    r["ld_gap"] = abs(r["L_D"] - LIT_LD) / LIT_LD * 100
    r["label"] = label; r["color"] = color
    results.append(r)
    print(f"{label:30s} {r['Cl']:>8.4f} {r['Cd']:>8.4f} {r['L_D']:>8.3f} "
          f"{r['cl_gap']:>7.1f}% {r['ld_gap']:>8.1f}%")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
labels = [r["label"] for r in results]
colors = [r["color"] for r in results]

cls = [r["Cl"] for r in results]; cls_e = [r["Cl_rms"] for r in results]
cds = [r["Cd"] for r in results]; cds_e = [r["Cd_rms"] for r in results]
lds = [r["L_D"] for r in results]
xs = np.arange(len(labels))

axes[0].bar(xs, cls, yerr=cls_e, color=colors, edgecolor='k', capsize=5)
axes[0].axhline(LIT_CL, color='C2', ls='--', lw=2, label=f'lit~{LIT_CL}')
axes[0].axhspan(LIT_CL*0.75, LIT_CL*1.25, color='C2', alpha=0.1)
axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, rotation=12, ha='right')
axes[0].set_ylabel("Cl_settled"); axes[0].set_title("Cl (lit ±25%)")
for x, c in zip(xs, cls): axes[0].text(x, c, f"{c:.3f}", ha='center', va='bottom', fontweight='bold')
axes[0].legend(); axes[0].grid(True, ls=':', alpha=0.4, axis='y')

axes[1].bar(xs, cds, yerr=cds_e, color=colors, edgecolor='k', capsize=5)
axes[1].axhline(LIT_CD, color='C2', ls='--', lw=2, label=f'lit~{LIT_CD}')
axes[1].axhspan(LIT_CD*0.7, LIT_CD*1.3, color='C2', alpha=0.1, label='lit ±30%')
axes[1].set_xticks(xs); axes[1].set_xticklabels(labels, rotation=12, ha='right')
axes[1].set_ylabel("Cd_settled"); axes[1].set_title("Cd — AMR-ON +87%")
for x, c in zip(xs, cds): axes[1].text(x, c, f"{c:.3f}", ha='center', va='bottom', fontweight='bold')
axes[1].legend(); axes[1].grid(True, ls=':', alpha=0.4, axis='y')

axes[2].bar(xs, lds, color=colors, edgecolor='k')
axes[2].axhline(LIT_LD, color='C2', ls='--', lw=2, label=f'lit~{LIT_LD:.2f}')
axes[2].set_xticks(xs); axes[2].set_xticklabels(labels, rotation=12, ha='right')
axes[2].set_ylabel("L/D"); axes[2].set_title("Lift-to-drag — AMR-ON WORSE")
for x, c in zip(xs, lds): axes[2].text(x, c, f"{c:.2f}", ha='center', va='bottom', fontweight='bold')
axes[2].legend(); axes[2].grid(True, ls=':', alpha=0.4, axis='y')

fig.suptitle("30k settled — Cl PASS hid Cd FAIL. AMR patch ±0.10c truncates 42% of α=8° lower surface.\n"
             "Fix: auto-expand patch to airfoil bbox + 0.06c (committed 2026-05-20)",
             fontsize=11, fontweight='bold')
plt.tight_layout()
out = os.path.join(ROOT, "images", "ld_audit_30k.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\nSaved: {out}")
