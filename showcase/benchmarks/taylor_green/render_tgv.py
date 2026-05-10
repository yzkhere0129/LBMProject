"""
render_tgv.py — Taylor-Green 2D vortex decay vs analytical.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


def main():
    base = "/home/yzk/LBMProject/round37_showcase_specs/results"
    decay = pd.read_csv(f"{base}/tg_energy_decay.csv")
    field0 = pd.read_csv(f"{base}/tg_field_t0.csv")
    field1 = pd.read_csv(f"{base}/tg_field_t_tau.csv")

    nx = field0["i"].max() + 1
    ny = field0["j"].max() + 1
    omega0 = field0["vorticity"].to_numpy().reshape((ny, nx))
    omega1 = field1["vorticity"].to_numpy().reshape((ny, nx))
    x = field0["x"].to_numpy().reshape((ny, nx))[0]
    y = field0["y"].to_numpy().reshape((ny, nx))[:, 0]

    t = decay["t_over_tau"].to_numpy()
    E_sim = decay["E_sim_over_E0"].to_numpy()
    E_exact = decay["E_exact_over_E0"].to_numpy()

    log_sim = np.log(np.maximum(E_sim, 1e-12))
    log_exact = np.log(np.maximum(E_exact, 1e-12))
    valid = (t > 0.05) & (E_sim > 1e-6)
    if valid.any():
        slope_sim = np.polyfit(t[valid], log_sim[valid], 1)[0]
        slope_exact = np.polyfit(t[valid], log_exact[valid], 1)[0]
        slope_err = abs(slope_sim - slope_exact) / abs(slope_exact) * 100
    else:
        slope_sim = slope_exact = slope_err = float("nan")

    L2_E_pct = float(np.sqrt(np.mean(((E_sim - E_exact) / np.maximum(E_exact, 1e-12)) ** 2)) * 100)

    fig = plt.figure(figsize=(11, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4])

    omega_lim = float(max(abs(omega0).max(), abs(omega1).max()))
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(x, y, omega0, cmap="RdBu_r",
                         vmin=-omega_lim, vmax=omega_lim, shading="auto")
    ax1.set_aspect("equal"); ax1.set_xlabel(r"$x$"); ax1.set_ylabel(r"$y$")
    ax1.set_title(r"$\omega_z(t=0)$ — initial vortex array", fontsize=10)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(x, y, omega1, cmap="RdBu_r",
                         vmin=-omega_lim, vmax=omega_lim, shading="auto")
    ax2.set_aspect("equal"); ax2.set_xlabel(r"$x$"); ax2.set_ylabel(r"$y$")
    ax2.set_title(r"$\omega_z(t=\tau_{\rm decay})$ — viscously decayed", fontsize=10)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(t, E_sim, "C0o", markersize=5, label="LBM (D3Q19 BGK)",
                 markerfacecolor="white", markeredgewidth=1.2)
    ax3.semilogy(t, E_exact, "k--", linewidth=1.2,
                 label=r"analytical $\exp(-4\nu k^2 t)$")
    ax3.set_xlabel(r"$t / \tau_{\rm decay}$")
    ax3.set_ylabel(r"$E(t) / E(0)$")
    ax3.set_title(rf"Kinetic-energy decay  ($L_2^E={L2_E_pct:.2f}\%$, "
                  rf"slope err = {slope_err:.2f}\%)", fontsize=10)
    ax3.legend(loc="lower left", fontsize=8.5)
    ax3.grid(which="both", alpha=0.3)

    out = "/home/yzk/LBMProject_showcase/showcase/benchmarks/taylor_green/tgv_decay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  L2 energy = {L2_E_pct:.2f}%, slope err = {slope_err:.2f}%")

    with open("/home/yzk/LBMProject_showcase/showcase/benchmarks/taylor_green/result.json", "w") as f:
        json.dump({"case": "B3 Taylor-Green 2D",
                   "grid": "128x128x3", "Re": 100, "tau": 0.7, "collision": "BGK",
                   "L2_energy_pct": L2_E_pct, "slope_err_pct": float(slope_err),
                   "decay_rate_sim": float(slope_sim), "decay_rate_exact": float(slope_exact)},
                  f, indent=2)


if __name__ == "__main__":
    main()
