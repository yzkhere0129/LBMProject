"""
render_natconv.py — Natural convection in square cavity, Ra=1e4 vs De Vahl Davis 1983.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    base = "/home/yzk/LBMProject/round37_showcase_specs/results"
    field = pd.read_csv(f"{base}/natconv_ra1e4_field.csv")
    metrics = json.load(open(f"{base}/natconv_ra1e4_metrics.json"))

    nx = field["i"].max() + 1
    ny = field["j"].max() + 1
    T = field["T"].to_numpy().reshape((ny, nx))
    u = field["u"].to_numpy().reshape((ny, nx))
    v = field["v"].to_numpy().reshape((ny, nx))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    L = 1.0

    Tref = 300.0
    Thot = float(T[:, 0].mean())
    Tcold = float(T[:, -1].mean())
    dT = Thot - Tcold
    T_norm = (T - Tcold) / max(dT, 1e-9)
    speed = np.hypot(u, v)
    speed_max = float(speed.max())
    if speed_max > 0:
        u_norm = u / speed_max
        v_norm = v / speed_max
    else:
        u_norm = u; v_norm = v

    dx = float(x[1] - x[0])
    Nu_local_hot = -(T[:, 1] - T[:, 0]) / dx
    Nu_local_hot_norm = Nu_local_hot * L / max(dT, 1e-9)
    Nu_avg_from_field = float(np.mean(Nu_local_hot_norm))

    fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    pcm = ax1.pcolormesh(x, y, T_norm, cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
    ax1.streamplot(x, y, u_norm, v_norm, color="black", density=1.6, linewidth=0.5,
                   arrowsize=0.7)
    ax1.contour(x, y, T_norm, levels=np.linspace(0.1, 0.9, 9),
                colors="white", linewidths=0.4, alpha=0.6)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x/L$"); ax1.set_ylabel(r"$y/L$")
    ax1.set_title(r"Natural convection Ra=$10^4$ — $T$ + streamlines", fontsize=10)
    fig.colorbar(pcm, ax=ax1, fraction=0.046, pad=0.02).set_label(r"$(T-T_c)/\Delta T$")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(Nu_local_hot_norm, y, "C0-", linewidth=1.4,
             label=fr"$\mathrm{{Nu}}_{{\rm local}}(y)$ on hot wall (LBM)")
    ax2.axvline(metrics["Nu_ref"], color="C1", linestyle="--", linewidth=1.2,
                label=fr"De Vahl Davis 1983: $\overline{{\mathrm{{Nu}}}} = {metrics['Nu_ref']:.3f}$")
    ax2.axvline(metrics["Nu_hot"], color="C0", linestyle=":", linewidth=1.2,
                label=fr"LBM $\overline{{\mathrm{{Nu}}}}_{{\rm hot}} = {metrics['Nu_hot']:.3f}$")
    ax2.set_xlabel(r"$\mathrm{Nu}$")
    ax2.set_ylabel(r"$y/L$")
    ax2.set_xlim(0, max(8, max(Nu_local_hot_norm.max(), metrics["Nu_hot"]) * 1.1))
    ax2.set_title(rf"Hot-wall Nu profile  "
                  rf"($\overline{{Nu}}_{{\rm err}}={metrics['Nu_err_pct']:.2f}\%$)",
                  fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    out = "/home/yzk/LBMProject_showcase/showcase/benchmarks/natural_convection/nat_conv_Ra1e4.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  Nu_hot = {metrics['Nu_hot']:.3f} vs ref 2.243 ({metrics['Nu_err_pct']:.2f}% err)")

    with open("/home/yzk/LBMProject_showcase/showcase/benchmarks/natural_convection/result.json", "w") as f:
        json.dump({**metrics,
                   "case_label": "B4 Natural Convection Ra=1e4",
                   "Nu_avg_from_field_recompute": Nu_avg_from_field}, f, indent=2)


if __name__ == "__main__":
    main()
