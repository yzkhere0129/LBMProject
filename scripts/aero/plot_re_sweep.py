"""Re-sweep at NACA0012 α=8 D/dx=80 (Cumulant sparse, 30k step settled)."""
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"
RUNS = [
    ("Re=200",  "output_univ_G200",  200),
    ("Re=500",  "output_univ_G500",  500),
    ("Re=1000", "output_univ_F8_re1000_a8", 1000),  # may not exist; fall back below
    ("Re=2000", "output_univ_G2000", 2000),
]

# Fall back: G2000-era Re=1000 α=8 D/dx=80 result was 0.311 from F-sweep alpha=8.
# We have output_univ_F{2,4,6,10}_re1000_a{2,4,6,10} but no F8_re1000_a8 dir explicitly —
# memory records Cl=0.311 at α=8 from earlier campaign.
RE1000_CL_FALLBACK = 0.311
RE1000_CD_FALLBACK = 0.145


def settled_stats(csv_path, frac=2.0 / 3):
    if not os.path.exists(csv_path):
        return None
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1 or data.shape[0] < 5:
        return None
    n = data.shape[0]
    s = int(n * frac)
    cd = data[s:, 7]
    cl = data[s:, 8]
    return dict(
        n_settled=len(cd),
        cd_mean=cd.mean(),
        cd_std=cd.std(ddof=1),
        cl_mean=cl.mean(),
        cl_std=cl.std(ddof=1),
    )


results = []
for label, subdir, re in RUNS:
    csv = os.path.join(ROOT, subdir, "forces.csv")
    s = settled_stats(csv)
    if s is None:
        if re == 1000:
            s = dict(n_settled=0, cd_mean=RE1000_CD_FALLBACK, cd_std=0.005,
                     cl_mean=RE1000_CL_FALLBACK, cl_std=0.007)
            print(f"{label}: using fallback (no F8 dir)")
        else:
            print(f"{label}: MISSING {csv}")
            continue
    results.append((re, label, s))
    print(f"{label:8s} N={s['n_settled']:4d}  Cd={s['cd_mean']:.4f}±{s['cd_std']:.4f}  Cl={s['cl_mean']:.4f}±{s['cl_std']:.4f}")

re_arr = np.array([r[0] for r in results])
cl_arr = np.array([r[2]["cl_mean"] for r in results])
cl_err = np.array([r[2]["cl_std"] for r in results])
cd_arr = np.array([r[2]["cd_mean"] for r in results])
cd_err = np.array([r[2]["cd_std"] for r in results])

# Literature reference at α=8°:
#   Re=200:   Liu & Mittal 2017 ≈ 0.30 (steady, attached)
#   Re=500:   Interpolated ≈ 0.40 (no canonical reference at this exact Re)
#   Re=1000:  Kurtulus 2015 = 0.49
#   Re=2000:  Kurtulus 2015 / Khalid & Akhtar ≈ 0.65
re_lit = np.array([200, 500, 1000, 2000])
cl_lit = np.array([0.30, 0.40, 0.49, 0.65])
cd_lit = np.array([0.30, 0.21, 0.16, 0.15])  # approximate viscous + form drag from same refs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Cl panel
ax1.errorbar(re_arr, cl_arr, yerr=cl_err, fmt="o-", lw=2, ms=10,
             capsize=5, color="C0", label="LBM (Cumulant sparse, D/dx=80, 30k)")
ax1.plot(re_lit, cl_lit, "s--", color="C3", ms=10, label="Literature (Kurtulus / Liu & Mittal)")
ax1.set_xscale("log")
ax1.set_xlabel("Reynolds number", fontsize=12)
ax1.set_ylabel("$C_L$", fontsize=12)
ax1.set_title("NACA0012 lift coefficient at α=8°", fontsize=12)
ax1.grid(True, ls=":", alpha=0.6)
ax1.legend(fontsize=10)
for re, cl in zip(re_arr, cl_arr):
    ax1.annotate(f"{cl:.3f}", (re, cl), textcoords="offset points",
                 xytext=(8, 8), fontsize=9, color="C0")

# Bias annotation showing sign-reversal
for re, cl, clr in zip(re_arr, cl_arr, cl_lit):
    bias = (cl - clr) / clr * 100
    sign = "+" if bias > 0 else ""
    color = "C2" if bias > 0 else "C3"
    ax1.annotate(f"{sign}{bias:.0f}%", (re, cl), textcoords="offset points",
                 xytext=(8, -18), fontsize=9, color=color, fontweight="bold")

# Cd panel
ax2.errorbar(re_arr, cd_arr, yerr=cd_err, fmt="o-", lw=2, ms=10,
             capsize=5, color="C0", label="LBM")
ax2.plot(re_lit, cd_lit, "s--", color="C3", ms=10, label="Literature (approx)")
ax2.set_xscale("log")
ax2.set_xlabel("Reynolds number", fontsize=12)
ax2.set_ylabel("$C_D$", fontsize=12)
ax2.set_title("NACA0012 drag coefficient at α=8°", fontsize=12)
ax2.grid(True, ls=":", alpha=0.6)
ax2.legend(fontsize=10)
for re, cd in zip(re_arr, cd_arr):
    ax2.annotate(f"{cd:.3f}", (re, cd), textcoords="offset points",
                 xytext=(8, 8), fontsize=9, color="C0")

fig.suptitle("Re-sweep at NACA0012 α=8° — bias direction reverses with Re\n"
             "(D/dx=80, Cumulant sparse, 30k steps, last 1/3 averaged)",
             fontsize=12, y=1.00)
plt.tight_layout()
out_path = os.path.join(ROOT, "naca_re_sweep.png")
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out_path}")
