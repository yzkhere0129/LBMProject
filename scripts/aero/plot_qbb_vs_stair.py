#!/usr/bin/env python3
"""Plot Cl(t) for QBB-MEM vs stair-step at α=+8°, NACA0012.

Visualises the partial-fix result: step-0 artifact reduced 32%, but steady-state
Cl mean unchanged (-0.31), with QBB showing wider oscillations including
brief positive excursions (cf. stair tightly locked at [-0.39, -0.22]).

Reference: Kurtulus 2015 expects Cl ≈ +0.49 at α=+8°, Re=1000 — POSITIVE sign.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    t, cd, cl = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"]))
            cd.append(float(r["Cd"]))
            cl.append(float(r["Cl"]))
    return t, cd, cl


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.normpath(os.path.join(here, "..", ".."))

    qbb_path = os.path.join(root, "output_qbb_a8_steady", "forces.csv")
    stair_path = os.path.join(root, "output_naca_a8_kurtulus", "forces.csv")

    if not os.path.exists(qbb_path):
        print(f"Missing: {qbb_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(stair_path):
        print(f"Missing: {stair_path}", file=sys.stderr)
        sys.exit(1)

    t_q, cd_q, cl_q = load(qbb_path)
    t_s, cd_s, cl_s = load(stair_path)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    # Drop the step-0 outlier (huge artifact transient) from the plot range
    ax0.plot(t_q[1:], cl_q[1:], label="QBB-MEM single-node", color="#1f77b4", lw=1.0)
    ax0.plot(t_s[1:], cl_s[1:], label="stair-step BB", color="#d62728", lw=1.0, alpha=0.7)
    ax0.axhline(+0.49, ls="--", color="green", lw=1.2,
                label="Kurtulus 2015 expected Cl ≈ +0.49")
    ax0.axhline(0.0, ls=":", color="black", lw=0.7)
    ax0.set_ylabel("Cl")
    ax0.set_title("NACA0012 α=+8°, Re=1000, D3Q27 Cumulant — QBB MEM vs stair-step")
    ax0.legend(loc="lower right")
    ax0.grid(alpha=0.3)

    ax1.plot(t_q[1:], cd_q[1:], label="QBB-MEM", color="#1f77b4", lw=1.0)
    ax1.plot(t_s[1:], cd_s[1:], label="stair-step", color="#d62728", lw=1.0, alpha=0.7)
    ax1.axhline(0.17, ls="--", color="green", lw=1.0,
                label="Kurtulus 2015 expected Cd ≈ 0.17")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("Cd")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(root, "qbb_vs_stair_naca_a8.png")
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")

    # Print summary
    nlast = 80
    import statistics
    print("\nSummary (last 80 samples each):")
    print(f"  QBB-MEM      : Cl mean={statistics.mean(cl_q[-nlast:]):+.3f}  "
          f"min={min(cl_q[-nlast:]):+.3f}  max={max(cl_q[-nlast:]):+.3f}  "
          f"Cd mean={statistics.mean(cd_q[-nlast:]):+.3f}")
    print(f"  stair-step   : Cl mean={statistics.mean(cl_s[-nlast:]):+.3f}  "
          f"min={min(cl_s[-nlast:]):+.3f}  max={max(cl_s[-nlast:]):+.3f}  "
          f"Cd mean={statistics.mean(cd_s[-nlast:]):+.3f}")
    print(f"  Kurtulus 2015: Cl ≈ +0.49 (POSITIVE), Cd ≈ 0.17")


if __name__ == "__main__":
    main()
