#!/usr/bin/env python3
"""
Strouhal-number FFT analysis for Schäfer-Turek 2D-2 cylinder benchmark.

Reads forces.csv produced by aero_schaefer_turek_3d, identifies the dominant
shedding frequency via FFT on Cl(t), and reports:

  - Cd_mean, Cd_max  (DFG 2D-2 reference: 3.22 - 3.24)
  - Cl_max           (DFG 2D-2 reference: 0.99 - 1.01)
  - St = f * D / U_avg  (DFG 2D-2 reference: 0.295 - 0.305)

Discards an initial transient before computing statistics. Writes a 2-row
matplotlib figure (Cd/Cl time series + Cl spectrum) next to the CSV.

Usage:
    python3 strouhal_fft.py <output_dir> [--D 0.1] [--U-avg 1.0]
                                         [--transient-frac 0.5]
"""

import argparse
import os
import sys
import csv
import math

import numpy as np


def load_forces(csv_path):
    """Return numpy arrays (step, t, Cd, Cl)."""
    step, t, Cd, Cl = [], [], [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                step.append(int(row["step"]))
                t.append(float(row["t"]))
                Cd.append(float(row["Cd"]))
                Cl.append(float(row["Cl"]))
            except (ValueError, KeyError):
                continue
    return (np.asarray(step), np.asarray(t),
            np.asarray(Cd), np.asarray(Cl))


def strouhal_fft(t, Cl):
    """Return (St_freq, spectrum_f, spectrum_amp) for non-uniform t.

    Resamples Cl to a uniform grid (median dt) before FFT. Returns the peak
    frequency excluding DC.
    """
    if len(t) < 16:
        return float("nan"), np.array([]), np.array([])
    dt = float(np.median(np.diff(t)))
    t_uniform = np.arange(t[0], t[-1], dt)
    Cl_uniform = np.interp(t_uniform, t, Cl)
    # Window to reduce spectral leakage
    Cl_uniform = (Cl_uniform - Cl_uniform.mean()) * np.hanning(len(Cl_uniform))
    spec = np.abs(np.fft.rfft(Cl_uniform))
    freqs = np.fft.rfftfreq(len(Cl_uniform), d=dt)
    # Exclude DC
    if len(freqs) > 1:
        i_peak = 1 + int(np.argmax(spec[1:]))
        f_peak = float(freqs[i_peak])
    else:
        f_peak = float("nan")
    return f_peak, freqs, spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir",
                    help="Directory containing forces.csv from "
                         "aero_schaefer_turek_3d")
    ap.add_argument("--D", type=float, default=0.1,
                    help="Cylinder diameter [m] (default 0.1)")
    ap.add_argument("--U-avg", type=float, default=1.0,
                    help="Mean inlet velocity [m/s] for St normalisation "
                         "(default 1.0 = 2D-2). Use 0.2 for 2D-1.")
    ap.add_argument("--transient-frac", type=float, default=0.5,
                    help="Fraction of time series to discard as transient "
                         "before computing statistics (default 0.5)")
    ap.add_argument("--plot", default=None,
                    help="Output figure path (default: <dir>/strouhal_fft.png)")
    ap.add_argument("--assert-pass", action="store_true",
                    help="Exit non-zero if any DFG 2D-2 metric is outside its "
                         "tolerance band. Use in CI / make check.")
    ap.add_argument("--cd-tol", type=float, default=0.20,
                    help="Relative tolerance for Cd_max (default 0.20 = 20%, "
                         "appropriate for stair-step BC + D/dx=20). Tighten "
                         "to 0.05 once QBB curved BC is enabled.")
    ap.add_argument("--cl-tol", type=float, default=0.30,
                    help="Relative tolerance for Cl_max (default 0.30)")
    ap.add_argument("--st-tol", type=float, default=0.10,
                    help="Relative tolerance for Strouhal (default 0.10)")
    args = ap.parse_args()

    csv_path = os.path.join(args.output_dir, "forces.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    step, t, Cd, Cl = load_forces(csv_path)
    if len(t) == 0:
        print(f"ERROR: no data in {csv_path}.", file=sys.stderr)
        sys.exit(1)

    n_keep = int((1.0 - args.transient_frac) * len(t))
    n_keep = max(n_keep, 16)
    t_w, Cd_w, Cl_w = t[-n_keep:], Cd[-n_keep:], Cl[-n_keep:]

    Cd_mean = float(np.mean(Cd_w))
    Cd_max = float(np.max(Cd_w))
    Cl_max = float(np.max(np.abs(Cl_w)))

    f_peak, freqs, spec = strouhal_fft(t_w, Cl_w)
    St = f_peak * args.D / args.U_avg if math.isfinite(f_peak) else float("nan")

    # ---- DFG 2D-2 reference bands (Schäfer & Turek 1996) -------------------
    DFG_2D2 = {
        "Cd_max": (3.22, 3.24),
        "Cl_max": (0.99, 1.01),
        "St":     (0.295, 0.305),
    }

    def in_band(x, band):
        return band[0] <= x <= band[1]

    def in_relband(x, band, tol):
        """Within +/- tol relative of the band centre."""
        c = 0.5 * (band[0] + band[1])
        return abs(x - c) <= tol * c

    print()
    print("================================================================")
    print(f"Schäfer-Turek statistics over t ∈ [{t_w[0]:.3f}, {t_w[-1]:.3f}] s")
    print(f"  ({n_keep} samples after discarding {args.transient_frac*100:.0f}% transient)")
    print("================================================================")
    print(f"  Cd_mean    = {Cd_mean:8.4f}")
    print(f"  Cd_max     = {Cd_max:8.4f}    DFG 2D-2 ref: "
          f"[{DFG_2D2['Cd_max'][0]:.2f}, {DFG_2D2['Cd_max'][1]:.2f}]"
          f"   {'PASS' if in_band(Cd_max, DFG_2D2['Cd_max']) else 'OUT'}")
    print(f"  Cl_max     = {Cl_max:8.4f}    DFG 2D-2 ref: "
          f"[{DFG_2D2['Cl_max'][0]:.2f}, {DFG_2D2['Cl_max'][1]:.2f}]"
          f"   {'PASS' if in_band(Cl_max, DFG_2D2['Cl_max']) else 'OUT'}")
    print(f"  f_peak     = {f_peak:8.4f} Hz")
    print(f"  St         = {St:8.4f}    DFG 2D-2 ref: "
          f"[{DFG_2D2['St'][0]:.3f}, {DFG_2D2['St'][1]:.3f}]"
          f"   {'PASS' if in_band(St, DFG_2D2['St']) else 'OUT'}")
    print("================================================================")
    print()

    # ---- Plot --------------------------------------------------------------
    plot_path = args.plot or os.path.join(args.output_dir, "strouhal_fft.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes[0].plot(t, Cd, label="Cd", color="C0", lw=0.8)
        axes[0].plot(t, Cl, label="Cl", color="C1", lw=0.8, alpha=0.7)
        axes[0].axvspan(t_w[0], t_w[-1], alpha=0.08, color="C2",
                        label="analysis window")
        axes[0].set_xlabel("t [s]")
        axes[0].set_ylabel("Cd, Cl")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="best")
        axes[0].set_title(
            f"Cd_mean={Cd_mean:.3f}, Cd_max={Cd_max:.3f}, "
            f"Cl_max={Cl_max:.3f}")

        if len(freqs) > 1:
            axes[1].semilogy(freqs[1:], spec[1:], color="C1", lw=0.8)
            if math.isfinite(f_peak):
                axes[1].axvline(f_peak, ls="--", color="C3",
                                label=f"f={f_peak:.3f} Hz, St={St:.3f}")
            axes[1].set_xlabel("frequency [Hz]")
            axes[1].set_ylabel("|FFT(Cl)|")
            axes[1].grid(alpha=0.3, which="both")
            axes[1].legend(loc="best")
            axes[1].set_title(f"St = f·D/U_avg, D={args.D}, U_avg={args.U_avg}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        print(f"Figure: {plot_path}")
    except ImportError:
        print("(matplotlib not available; skipping plot)", file=sys.stderr)

    # ---- Gating exit code --------------------------------------------------
    if args.assert_pass:
        fails = []
        if not in_relband(Cd_max, DFG_2D2["Cd_max"], args.cd_tol):
            fails.append(f"Cd_max={Cd_max:.4f} outside ±{args.cd_tol*100:.0f}% "
                         f"of {DFG_2D2['Cd_max']}")
        if not in_relband(Cl_max, DFG_2D2["Cl_max"], args.cl_tol):
            fails.append(f"Cl_max={Cl_max:.4f} outside ±{args.cl_tol*100:.0f}% "
                         f"of {DFG_2D2['Cl_max']}")
        if math.isfinite(St) and not in_relband(St, DFG_2D2["St"], args.st_tol):
            fails.append(f"St={St:.4f} outside ±{args.st_tol*100:.0f}% "
                         f"of {DFG_2D2['St']}")
        if fails:
            print("GATE FAIL:", file=sys.stderr)
            for m in fails:
                print(f"  - {m}", file=sys.stderr)
            sys.exit(1)
        print("GATE PASS")


if __name__ == "__main__":
    main()
