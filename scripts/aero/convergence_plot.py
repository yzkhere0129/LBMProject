#!/usr/bin/env python3
"""
Plot Cd_max / Cl_max / St convergence across resolutions for the
Schaefer-Turek 2D-2 benchmark.

Usage:
    python3 convergence_plot.py \
        D20:output_aero_2d2_qbb \
        D40:output_aero_2d2_qbb_d40 \
        D80:output_aero_2d2_qbb_d80 \
        --U-avg 1.0 --transient-frac 0.5 \
        --out convergence.png
"""

import argparse
import os
import sys
import csv
import math
import numpy as np


def load_forces(p):
    step, t, Cd, Cl = [], [], [], []
    with open(p) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                step.append(int(row["step"]))
                t.append(float(row["t"]))
                Cd.append(float(row["Cd"]))
                Cl.append(float(row["Cl"]))
            except (ValueError, KeyError):
                continue
    return np.asarray(step), np.asarray(t), np.asarray(Cd), np.asarray(Cl)


def fft_peak(t, Cl):
    if len(t) < 16:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], dt)
    cu = np.interp(tu, t, Cl)
    cu = (cu - cu.mean()) * np.hanning(len(cu))
    spec = np.abs(np.fft.rfft(cu))
    freqs = np.fft.rfftfreq(len(cu), d=dt)
    if len(freqs) > 1:
        ipk = 1 + int(np.argmax(spec[1:]))
        return float(freqs[ipk])
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entries", nargs="+",
                    help="LABEL:dir entries (e.g. D40:output_aero_2d2_qbb_d40)")
    ap.add_argument("--D", type=float, default=0.1)
    ap.add_argument("--U-avg", type=float, default=1.0)
    ap.add_argument("--transient-frac", type=float, default=0.5)
    ap.add_argument("--out", default="convergence.png")
    args = ap.parse_args()

    rows = []
    for entry in args.entries:
        if ":" not in entry:
            print(f"skip malformed: {entry}", file=sys.stderr); continue
        label, d = entry.split(":", 1)
        p = os.path.join(d, "forces.csv")
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr); continue
        step, t, Cd, Cl = load_forces(p)
        nk = max(int((1 - args.transient_frac) * len(t)), 16)
        tw, Cdw, Clw = t[-nk:], Cd[-nk:], Cl[-nk:]
        f = fft_peak(tw, Clw)
        St = f * args.D / args.U_avg if math.isfinite(f) else float("nan")
        rows.append({
            "label": label,
            "Cd_max": float(np.max(Cdw)),
            "Cl_max": float(np.max(np.abs(Clw))),
            "St": St,
        })

    DFG = {"Cd_max": (3.22, 3.24), "Cl_max": (0.99, 1.01),
           "St": (0.295, 0.305)}

    print(f"{'Label':10s} {'Cd_max':10s} {'Cl_max':10s} {'St':10s}")
    for r in rows:
        print(f"{r['label']:10s} {r['Cd_max']:10.4f} {r['Cl_max']:10.4f} "
              f"{r['St']:10.4f}")
    print(f"{'DFG':10s} {DFG['Cd_max'][0]:.2f}-{DFG['Cd_max'][1]:.2f}    "
          f"{DFG['Cl_max'][0]:.2f}-{DFG['Cl_max'][1]:.2f}    "
          f"{DFG['St'][0]:.3f}-{DFG['St'][1]:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        labels = [r["label"] for r in rows]
        for ax, key, ttl in zip(
                axes,
                ["Cd_max", "Cl_max", "St"],
                ["Cd_max", "Cl_max", "Strouhal St"]):
            vals = [r[key] for r in rows]
            ax.bar(labels, vals, color="C0")
            ax.axhspan(DFG[key][0], DFG[key][1], alpha=0.18,
                       color="C2", label="DFG band")
            for x, v in zip(labels, vals):
                ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax.set_title(ttl)
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3)
        plt.suptitle("Schaefer-Turek 2D-2 convergence (LBM + QBB + corrected MEM)")
        plt.tight_layout()
        plt.savefig(args.out, dpi=120)
        print(f"figure: {args.out}")
    except ImportError:
        print("(matplotlib unavailable)", file=sys.stderr)


if __name__ == "__main__":
    main()
