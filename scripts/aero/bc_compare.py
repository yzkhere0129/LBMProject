#!/usr/bin/env python3
"""
Print Cd_max / Cl_max / St across multiple BC variants at the same resolution
side-by-side, with relative error vs DFG centre.
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
                step.append(int(row["step"])); t.append(float(row["t"]))
                Cd.append(float(row["Cd"])); Cl.append(float(row["Cl"]))
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
    return float(freqs[1 + int(np.argmax(spec[1:]))]) if len(freqs) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entries", nargs="+", help="LABEL:dir entries")
    ap.add_argument("--D", type=float, default=0.1)
    ap.add_argument("--U-avg", type=float, default=1.0)
    ap.add_argument("--transient-frac", type=float, default=0.5)
    args = ap.parse_args()

    DFG = {"Cd_max": (3.22, 3.24), "Cl_max": (0.99, 1.01), "St": (0.295, 0.305)}
    centre = {k: 0.5 * (a + b) for k, (a, b) in DFG.items()}

    rows = []
    for entry in args.entries:
        if ":" not in entry: continue
        label, d = entry.split(":", 1)
        p = os.path.join(d, "forces.csv")
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr); continue
        _, t, Cd, Cl = load_forces(p)
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

    # Header
    print(f"{'BC variant':<20s} {'Cd_max':>10s} {'err%':>7s} "
          f"{'Cl_max':>10s} {'err%':>7s} {'St':>8s} {'err%':>7s}")
    print("-" * 78)
    for r in rows:
        cd_err = 100 * (r['Cd_max'] - centre['Cd_max']) / centre['Cd_max']
        cl_err = 100 * (r['Cl_max'] - centre['Cl_max']) / centre['Cl_max']
        st_err = 100 * (r['St'] - centre['St']) / centre['St'] \
                 if math.isfinite(r['St']) else float('nan')
        print(f"{r['label']:<20s} {r['Cd_max']:>10.4f} {cd_err:>+7.2f} "
              f"{r['Cl_max']:>10.4f} {cl_err:>+7.2f} {r['St']:>8.4f} {st_err:>+7.2f}")
    print("-" * 78)
    print(f"{'DFG strict (centre)':<20s} {centre['Cd_max']:>10.4f} {'':>7s} "
          f"{centre['Cl_max']:>10.4f} {'':>7s} {centre['St']:>8.4f} {'':>7s}")
    print(f"{'DFG strict band':<20s} "
          f"[{DFG['Cd_max'][0]:.2f}, {DFG['Cd_max'][1]:.2f}]   "
          f"[{DFG['Cl_max'][0]:.2f}, {DFG['Cl_max'][1]:.2f}]   "
          f"[{DFG['St'][0]:.3f}, {DFG['St'][1]:.3f}]")


if __name__ == "__main__":
    main()
