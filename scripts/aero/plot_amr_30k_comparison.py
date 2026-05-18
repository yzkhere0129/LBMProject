"""30k step AMR-OFF vs AMR-ON settled-mean comparison plot.

Produces:
  images/amr_30k_cl_cd_trajectory.png   — Cl(t), Cd(t), mass drift
  images/amr_30k_settled_summary.png    — bar chart of settled vs Kurtulus
  Console: settled mean table + verdict vs ≤25% target
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"

CASES = [
    ("AMR-OFF baseline",   "output_30k_amr_off",           "C0"),
    ("AMR-ON bilin+time",  "output_30k_amr_on_bilin_time", "C2"),
]
KURTULUS_RE2000_CL = 0.57   # extrapolation from Kurtulus 2015 Re=1000 0.49
KURTULUS_RE2000_CD = 0.15   # approx
TARGET_ACCEPTANCE = 0.25    # 25% gap = Cl ≥ 0.43

def settled(csv_path):
    if not os.path.exists(csv_path): return None
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    n = data.shape[0]
    if n < 10: return None
    s = int(n * 2 / 3)
    return dict(
        step=data[:, 0], t=data[:, 1],
        Cd=data[:, 7], Cl=data[:, 8], mass=data[:, 9],
        ux_mean=data[:, 10] if data.shape[1] > 10 else None,
        ux_std=data[:, 11] if data.shape[1] > 11 else None,
        ux_max_dev=data[:, 12] if data.shape[1] > 12 else None,
        Cd_settled=data[s:, 7].mean(),
        Cl_settled=data[s:, 8].mean(),
        Cl_rms_settled=data[s:, 8].std(ddof=1),
        Cd_rms_settled=data[s:, 7].std(ddof=1),
        mass_drift_max=float(np.max(np.abs(data[1:, 9] - data[0, 9]) / data[0, 9])),
        n_settled=n - s,
    )

results = []
for label, subdir, color in CASES:
    csv = os.path.join(ROOT, subdir, "forces.csv")
    r = settled(csv)
    if r is None:
        print(f"{label:30s} — MISSING {csv}")
        continue
    r["label"] = label
    r["color"] = color
    results.append(r)
    print(f"{label:30s} N_settled={r['n_settled']:4d}  Cd={r['Cd_settled']:.4f}±{r['Cd_rms_settled']:.4f}  "
          f"Cl={r['Cl_settled']:.4f}±{r['Cl_rms_settled']:.4f}  "
          f"mass_drift_max={r['mass_drift_max']:.2e}")

if len(results) < 2:
    print("\n[ABORT] Need both AMR-OFF and AMR-ON runs to compare.")
    raise SystemExit(0)

off, on = results[0], results[1]
off_gap = abs(off["Cl_settled"] - KURTULUS_RE2000_CL) / KURTULUS_RE2000_CL
on_gap  = abs(on["Cl_settled"]  - KURTULUS_RE2000_CL) / KURTULUS_RE2000_CL

print(f"\n=== AMR effect on settled Cl ===")
print(f"  AMR-OFF: Cl_settled = {off['Cl_settled']:.4f}  (gap to lit 0.57: {off_gap*100:.1f}%)")
print(f"  AMR-ON:  Cl_settled = {on['Cl_settled']:.4f}  (gap to lit 0.57: {on_gap*100:.1f}%)")
print(f"  Δ AMR:   {(on['Cl_settled'] - off['Cl_settled'])/off['Cl_settled']*100:+.1f}%")
print(f"  Acceptance (gap ≤ {TARGET_ACCEPTANCE*100:.0f}%): "
      f"OFF={'PASS' if off_gap <= TARGET_ACCEPTANCE else 'FAIL'}  "
      f"ON={'PASS'  if on_gap  <= TARGET_ACCEPTANCE else 'FAIL'}")

# ===== Plot 1: trajectory =====
fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for r in results:
    ax[0].plot(r["step"], r["Cl"], lw=1.0, color=r["color"], label=r["label"], alpha=0.85)
    ax[1].plot(r["step"], r["Cd"], lw=1.0, color=r["color"], label=r["label"], alpha=0.85)
    drift = (r["mass"] - r["mass"][0]) / r["mass"][0]
    ax[2].plot(r["step"], drift, lw=1.0, color=r["color"], label=r["label"], alpha=0.85)
ax[0].axhline(KURTULUS_RE2000_CL, color="C3", ls="--", lw=2,
              label=f"Kurtulus extrap Cl={KURTULUS_RE2000_CL}")
ax[0].axhspan(KURTULUS_RE2000_CL*0.75, KURTULUS_RE2000_CL*1.25,
              color="C3", alpha=0.08, label="±25% gap target")
ax[0].set_ylabel("Cl"); ax[0].grid(True, ls=":", alpha=0.5); ax[0].legend(loc="upper right", fontsize=9)
ax[1].axhline(KURTULUS_RE2000_CD, color="C3", ls="--", lw=2, label=f"Kurtulus ~Cd={KURTULUS_RE2000_CD}")
ax[1].set_ylabel("Cd"); ax[1].grid(True, ls=":", alpha=0.5); ax[1].legend(loc="upper right", fontsize=9)
ax[2].set_xlabel("step"); ax[2].set_ylabel("Mass drift (rel)"); ax[2].grid(True, ls=":", alpha=0.5)
ax[2].axhspan(-1e-5, 1e-5, color="gray", alpha=0.15, label="FP32 noise band")
ax[2].legend(loc="upper right", fontsize=9)
fig.suptitle("30k step settled comparison: NACA0012 Re=2000 α=8 D/dx=80 — AMR effect", fontsize=12)
plt.tight_layout()
out1 = os.path.join(ROOT, "images", "amr_30k_cl_cd_trajectory.png")
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out1}")

# ===== Plot 2: settled bar chart =====
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
labels = [r["label"] for r in results]
colors = [r["color"] for r in results]
cls    = [r["Cl_settled"] for r in results]
cls_e  = [r["Cl_rms_settled"] for r in results]
cds    = [r["Cd_settled"] for r in results]
cds_e  = [r["Cd_rms_settled"] for r in results]
drifts = [r["mass_drift_max"] for r in results]

xs = np.arange(len(labels))
axes[0].bar(xs, cls, yerr=cls_e, color=colors, edgecolor='k', capsize=5)
axes[0].axhline(KURTULUS_RE2000_CL, color="C3", ls="--", lw=2)
axes[0].axhspan(KURTULUS_RE2000_CL*0.75, KURTULUS_RE2000_CL*1.25,
                color="C3", alpha=0.1, label="±25% target")
axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, rotation=10); axes[0].set_ylabel("Cl_settled")
for x, c in zip(xs, cls):
    axes[0].text(x, c, f"{c:.3f}", ha='center', va='bottom', fontweight='bold')
axes[0].legend()
axes[0].set_title("Cl vs Kurtulus extrap (0.57)")

axes[1].bar(xs, cds, yerr=cds_e, color=colors, edgecolor='k', capsize=5)
axes[1].axhline(KURTULUS_RE2000_CD, color="C3", ls="--", lw=2, label=f"Kurtulus ~{KURTULUS_RE2000_CD}")
axes[1].set_xticks(xs); axes[1].set_xticklabels(labels, rotation=10); axes[1].set_ylabel("Cd_settled")
for x, c in zip(xs, cds):
    axes[1].text(x, c, f"{c:.3f}", ha='center', va='bottom', fontweight='bold')
axes[1].legend(); axes[1].set_title("Cd")

axes[2].bar(xs, drifts, color=colors, edgecolor='k')
axes[2].axhline(1e-5, color="C3", ls="--", lw=2, label="FP32 noise upper")
axes[2].set_yscale("log")
axes[2].set_xticks(xs); axes[2].set_xticklabels(labels, rotation=10); axes[2].set_ylabel("Max mass drift (rel)")
for x, d in zip(xs, drifts):
    axes[2].text(x, d*1.3, f"{d:.1e}", ha='center', va='bottom', fontsize=9)
axes[2].legend(); axes[2].set_title("Mass conservation (FP32 noise ~1e-5)")

fig2.suptitle(f"30k settled comparison — verdict ≤25% gap target: "
              f"OFF={'PASS' if off_gap<=TARGET_ACCEPTANCE else 'FAIL'}  "
              f"ON={'PASS' if on_gap<=TARGET_ACCEPTANCE else 'FAIL'}",
              fontsize=12, fontweight='bold')
plt.tight_layout()
out2 = os.path.join(ROOT, "images", "amr_30k_settled_summary.png")
plt.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")
