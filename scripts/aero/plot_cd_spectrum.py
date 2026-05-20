"""FFT / power-spectrum analysis of Cd(t) and Cl(t).

Hypothesis B: AMR introduces numerical-period spikes in the force trace
(at coarse-step / fine-step / sub-step frequencies).

If we see a clean dominant peak at Strouhal frequency only → AMR is silent.
If we see harmonics at integer multiples of AMR step → AMR injects noise.

Output: images/cd_spectrum.png with three subplots (one per case).
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"
CASES = [
    ("AMR-OFF",                    "output_30k_amr_off",           "C0"),
    ("AMR-ON ±0.10c (BUG)",        "output_30k_amr_on_bilin_time", "C3"),
    ("AMR-ON auto-expand (FIX)",   "output_30k_amr_patch_fixed",   "C2"),
]


def load(p):
    d = np.loadtxt(p, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1], d[:, 7], d[:, 8]  # step, t, Cd, Cl


def settled_window(arr, frac=2/3):
    n = len(arr); s = int(n * frac)
    return arr[s:]


fig, axes = plt.subplots(3, 2, figsize=(14, 10))
results = []
for row, (label, sub, color) in enumerate(CASES):
    p = os.path.join(ROOT, sub, "forces.csv")
    if not os.path.exists(p):
        continue
    step, t, Cd, Cl = load(p)

    # Detrend (settled portion only): remove mean
    Cd_s = settled_window(Cd) - settled_window(Cd).mean()
    Cl_s = settled_window(Cl) - settled_window(Cl).mean()
    t_s = settled_window(t)

    # Sample rate: probe-every=100 LBM steps, dt comes from CSV
    dt_sample = t_s[1] - t_s[0]  # physical sec between probes

    # FFT
    n = len(Cd_s)
    freqs = np.fft.rfftfreq(n, d=dt_sample)
    Cd_fft = np.abs(np.fft.rfft(Cd_s)) / n
    Cl_fft = np.abs(np.fft.rfft(Cl_s)) / n

    # Strouhal: f * c / U; c=1, U_phys=U_LU * dx/dt → 我们没有 U_phys 直接
    # use dimensionless f_norm = f * dt_sample (cycles per sample)
    # Or report as just frequency in Hz; the user can convert.

    # Time series
    axes[row, 0].plot(t_s, Cd_s + settled_window(Cd).mean(),
                      lw=0.8, color=color, alpha=0.8)
    axes[row, 0].axhline(settled_window(Cd).mean(), ls='--', color='k', lw=1)
    axes[row, 0].set_ylabel(f"{label}\nCd(t)")
    axes[row, 0].grid(True, ls=':', alpha=0.4)
    if row == 0: axes[row, 0].set_title("Settled Cd time series")
    if row == 2: axes[row, 0].set_xlabel("t [s]")

    # Power spectrum (log Y)
    # Find dominant peak (skip DC)
    nonzero = freqs > 0.1
    dom_freq = freqs[nonzero][np.argmax(Cd_fft[nonzero])]
    dom_amp = Cd_fft[nonzero].max()
    axes[row, 1].semilogy(freqs[1:], Cd_fft[1:], color=color, lw=1.2,
                          label=f'peak @ f={dom_freq:.3f} Hz')
    axes[row, 1].grid(True, ls=':', alpha=0.4, which='both')
    axes[row, 1].set_ylabel("|FFT(Cd)|")
    axes[row, 1].legend(loc='upper right', fontsize=9)
    if row == 0: axes[row, 1].set_title("Cd power spectrum")
    if row == 2: axes[row, 1].set_xlabel("frequency [Hz]")

    results.append((label, dom_freq, dom_amp,
                    settled_window(Cd).mean(), settled_window(Cd).std(ddof=1)))

print(f"{'case':30s} {'Cd_mean':>9s} {'Cd_rms':>9s} {'dom_f[Hz]':>10s} {'amp':>9s}")
print("-" * 75)
for r in results:
    print(f"{r[0]:30s} {r[3]:>9.4f} {r[4]:>9.4f} {r[1]:>10.4f} {r[2]:>9.4f}")

# Strouhal context line: NACA0012 Re=2000 from various refs ≈ 0.16-0.20.
# f_strouhal_phys = St * U / c. Our U_phys = U_LU * dx/dt.
# We can estimate: in 30000 steps × dt=0.000625 s = 18.75 s sim time.
# Average period: ~50 LBM steps = 50*dt = 0.03125 s → St_freq ≈ 32 Hz too high.
# Actually U_phys ≈ 1 m/s, c=1 m, so St=0.18 → f≈0.18 Hz.

fig.suptitle("Cd(t) + FFT — looking for AMR-injected harmonics vs natural shedding peak\n"
             "Compare peak frequency across cases — should be the SAME (Strouhal) if AMR silent",
             fontsize=11)
plt.tight_layout()
out = os.path.join(ROOT, "images", "cd_spectrum.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\nSaved: {out}")
