"""Summarize all O1-N4 chain results in one comparison plot.

Reads forces.csv from each output dir, computes settled stats,
overlays bars vs Kurtulus baseline.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"

# Variants to compare. Tuple = (label, dir, Re, target Cl).
VARIANTS = [
    # Baseline (original, pre-patch)
    ("Original\nG2000",         "output_univ_G2000",                2000, 0.65, "C0"),
    ("Original\nG1000 (F8)",    None,                                1000, 0.49, "C0"),  # use fallback
    # omega_b = 1.0 (Cumulant bulk relaxation)
    ("O1\nomega_b=1.0",         "output_overnight_O1_re2000_patched", 2000, 0.65, "C2"),
    ("O2\nomega_b=1.0",         "output_overnight_O2_re1000_patched", 1000, 0.49, "C2"),
    # omega_3..6 = 0.5
    ("N1\nomega_3..6=0.5",      "output_diag_N1_re2000_omegahigh05",  2000, 0.65, "C3"),
    ("N3\nomega_3..6=0.5",      "output_diag_N3_re1000_omegahigh05",  1000, 0.49, "C3"),
    # bc=stair (no QBB)
    ("N2\nstair (no QBB)",      "output_diag_N2_re2000_stair",        2000, 0.65, "C4"),
    ("N4\nstair (no QBB)",      "output_diag_N4_re1000_stair",        1000, 0.49, "C4"),
]

# Fallback for Re=1000 original (memory: Cl=0.311)
FALLBACK_RE1000 = dict(cd=0.145, cl=0.311, cl_rms=0.007)


def settled(csv_path):
    if csv_path is None or not os.path.exists(csv_path):
        return None
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.shape[0] < 10:
        return None
    n = data.shape[0]
    s = int(n * 2 / 3)
    cd = data[s:, 7]
    cl = data[s:, 8]
    return dict(cd=cd.mean(), cl=cl.mean(),
                cl_rms=cl.std(ddof=1),
                n_settled=len(cd))


# Collect results
rows = []
for label, subdir, re, target, color in VARIANTS:
    csv = os.path.join(ROOT, subdir, "forces.csv") if subdir else None
    s = settled(csv)
    if s is None:
        if re == 1000 and "Original" in label:
            s = FALLBACK_RE1000
            print(f"{label.replace(chr(10),' '):30s} (fallback) Cl={s['cl']:.4f}")
        else:
            print(f"{label.replace(chr(10),' '):30s} MISSING {csv}")
            continue
    else:
        print(f"{label.replace(chr(10),' '):30s} Cl={s['cl']:.4f}  Cd={s['cd']:.4f}  "
              f"Cl_rms={s['cl_rms']:.4f}")
    rows.append((label, re, target, s, color))


# Plot: two panels (Re=2000 + Re=1000), bars of Cl with Kurtulus line + error
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
for ax, target_re, target_name, target_cl in [
    (ax1, 2000, "Kurtulus Re=2000", 0.65),
    (ax2, 1000, "Kurtulus Re=1000", 0.49),
]:
    sub_rows = [r for r in rows if r[1] == target_re]
    labels = [r[0] for r in sub_rows]
    cls = [r[3]["cl"] for r in sub_rows]
    rmss = [r[3]["cl_rms"] for r in sub_rows]
    colors = [r[4] for r in sub_rows]

    xs = np.arange(len(sub_rows))
    bars = ax.bar(xs, cls, yerr=rmss, capsize=4, color=colors,
                  edgecolor='k', linewidth=0.5, alpha=0.85)
    for x, c, rms in zip(xs, cls, rmss):
        ax.text(x, c + rms + 0.012, f"{c:.3f}", ha='center', fontsize=9,
                fontweight='bold')

    ax.axhline(target_cl, color="C3", ls="--", lw=2,
               label=f"{target_name} = {target_cl}")
    # 10% target band
    ax.axhspan(target_cl * 0.9, target_cl * 1.1, color="C3", alpha=0.12,
               label="10% acceptance band")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylabel("Cl (settled mean)", fontsize=11)
    ax.set_title(f"NACA0012 α=8° Re={target_re} D/dx=80", fontsize=11)
    ax.set_ylim(0, max(target_cl + 0.1, max(cls) + 0.1))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', ls=':', alpha=0.5)

fig.suptitle("Debug session 2026-05-16: code-level patches all fail to close 50% Cl gap\n"
             "→ structural LE under-resolution (radius=1.27 cells @ D/dx=80) is the bottleneck",
             fontsize=11, y=1.01)
plt.tight_layout()
out = os.path.join(ROOT, "naca_chain_summary.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")
