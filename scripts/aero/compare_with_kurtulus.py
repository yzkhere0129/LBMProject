"""Compare G500/G2000 to Kurtulus 2015 + Liu & Mittal 2017 quantitative values.

Extracts from LBM forces.csv:
  - settled Cl mean, Cl_rms, Cl_max
  - settled Cd mean, Cd_rms
  - Strouhal St = f·c/U_inf via Cl FFT (peak frequency)
  - peak-to-peak Cl amplitude

Reference values come from text of papers (no figures used)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yzk/CompressibleCFD"

# Numerical references from Kurtulus 2015 (DNS, NACA0012 α=8°)
# and Liu & Mittal 2017 (low-Re unsteady).
# These are quoted text values, not figure-extracted.
KURTULUS = {
    500:  dict(Cl=0.40,  Cd=0.21, St=0.70,  Cl_rms=0.05, src="Kurtulus 2015 / Liu&Mittal interp"),
    1000: dict(Cl=0.49,  Cd=0.16, St=0.80,  Cl_rms=0.15, src="Kurtulus 2015"),
    2000: dict(Cl=0.65,  Cd=0.15, St=0.90,  Cl_rms=0.30, src="Kurtulus 2015"),
}

# Driver convention: t in csv column 1 is in seconds (driver writes t = step*dt_phys
# where dt_phys = chord_phys / U_inf_phys / (steps_per_chord_unit ≈ 16 at u_LU=0.05 D=80)).
# So freq in Hz; chord-based Strouhal = f * chord / U_inf.
# At u_LU=0.05 D/dx=80: U_inf_phys=1 m/s by code convention, chord=1m, so St = f Hz · 1m / 1m/s = f.
# In other words, t in seconds IS in chord-times, so the period in t-units IS 1/St.


def settled_stats(csv_path, frac=2.0 / 3):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    n = data.shape[0]
    s = int(n * frac)
    t = data[s:, 1]
    cd = data[s:, 7]
    cl = data[s:, 8]
    return t, cd, cl


def strouhal_via_fft(t, cl, n_pad=4096):
    cl_centered = cl - cl.mean()
    dt = np.median(np.diff(t))
    # zero-pad for finer freq resolution
    n = max(n_pad, len(cl_centered))
    spec = np.fft.rfft(cl_centered, n=n)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(spec)
    # ignore DC bin and very low freq (period > 0.5*T)
    mask = freqs > 0.05
    k = np.argmax(mag[mask])
    f_peak = freqs[mask][k]
    return f_peak, freqs, mag


def main():
    runs = [
        (500,  "output_univ_G500_vtk"),
        (2000, "output_univ_G2000_vtk"),
    ]
    print("\n=== Quantitative comparison: LBM (D/dx=80) vs Kurtulus 2015 / Liu&Mittal ===\n")
    print(f"{'Re':>5} {'metric':>10} {'LBM':>14} {'Lit':>10} {'err':>8}")
    print("-" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, (re, subdir) in zip(axes, runs):
        csv = os.path.join(ROOT, subdir, "forces.csv")
        t, cd, cl = settled_stats(csv)
        Cl_mean = cl.mean()
        Cl_rms = cl.std(ddof=1)
        Cl_ptp = cl.max() - cl.min()
        Cd_mean = cd.mean()
        Cd_rms = cd.std(ddof=1)

        f_peak, freqs, mag = strouhal_via_fft(t, cl)
        St_lbm = f_peak  # see note above

        ref = KURTULUS[re]
        print(f"\nRe={re}  ({ref['src']})")
        for key, lbm, lit in [
            ("Cl_mean",  Cl_mean, ref["Cl"]),
            ("Cd_mean",  Cd_mean, ref["Cd"]),
            ("Cl_rms",   Cl_rms,  ref["Cl_rms"]),
            ("Strouhal", St_lbm,  ref["St"]),
        ]:
            err = (lbm - lit) / lit * 100 if lit != 0 else float("nan")
            print(f"      {key:>10} {lbm:>14.4f} {lit:>10.3f}  {err:+7.1f}%")

        # Plot Cl spectrum
        ax.plot(freqs, mag / mag.max(), lw=1.2, label=f"LBM Cl spectrum")
        ax.axvline(St_lbm, color="C0", ls=":", label=f"LBM St = {St_lbm:.3f}")
        ax.axvline(ref["St"], color="C3", ls="--", lw=2,
                   label=f"Kurtulus St = {ref['St']:.2f}")
        ax.set_xlim(0, 2.0)
        ax.set_xlabel("Strouhal f·c/U∞")
        ax.set_ylabel("|Cl spectrum| (normalized)")
        ax.set_title(f"Re={re} α=8°  D/dx=80")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(ROOT, "naca_strouhal_compare.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
