"""
render_lid_driven.py — Lid-driven cavity Re=1000 vs Ghia 1982.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


GHIA_RE1000_U = np.array([
    [0.0000,  0.00000], [0.0547, -0.18109], [0.0625, -0.20196], [0.0703, -0.22220],
    [0.1016, -0.29730], [0.1719, -0.38289], [0.2813, -0.27805], [0.4531, -0.10648],
    [0.5000, -0.06080], [0.6172,  0.05702], [0.7344,  0.18719], [0.8516,  0.33304],
    [0.9531,  0.46604], [0.9609,  0.51117], [0.9688,  0.57492], [0.9766,  0.65928],
    [1.0000,  1.00000],
])

GHIA_RE1000_V = np.array([
    [0.0000,  0.00000], [0.0625,  0.27485], [0.0703,  0.29012], [0.0781,  0.30353],
    [0.0938,  0.32627], [0.1563,  0.37095], [0.2266,  0.33075], [0.2344,  0.32235],
    [0.5000,  0.02526], [0.8047, -0.31966], [0.8594, -0.42665], [0.9063, -0.51550],
    [0.9453, -0.39188], [0.9531, -0.33714], [0.9609, -0.27669], [0.9688, -0.21388],
    [1.0000,  0.00000],
])


def main(field_csv, u_csv, v_csv, png_out):
    field = pd.read_csv(field_csv)
    cu = pd.read_csv(u_csv)
    cv = pd.read_csv(v_csv)

    nx = field["i"].max() + 1
    ny = field["j"].max() + 1
    ux = field["ux"].to_numpy().reshape((ny, nx))
    uy = field["uy"].to_numpy().reshape((ny, nx))
    speed = np.hypot(ux, uy)
    U_lid = float(speed[-1, :].max() if speed[-1, :].max() > 0 else cu["u_over_Ulid"].max() or 0.1)
    speed_norm = speed / U_lid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    u_LBM_y = cu["y"].to_numpy()
    u_LBM = cu["u_over_Ulid"].to_numpy()
    v_LBM_x = cv["x"].to_numpy()
    v_LBM = cv["v_over_Ulid"].to_numpy()

    ghia_y = GHIA_RE1000_U[:, 0]
    ghia_u = GHIA_RE1000_U[:, 1]
    ghia_x = GHIA_RE1000_V[:, 0]
    ghia_v = GHIA_RE1000_V[:, 1]

    u_at_ghia = np.interp(ghia_y, u_LBM_y, u_LBM)
    v_at_ghia = np.interp(ghia_x, v_LBM_x, v_LBM)
    L2_u = float(np.sqrt(np.mean((u_at_ghia - ghia_u) ** 2)))
    L2_v = float(np.sqrt(np.mean((v_at_ghia - ghia_v) ** 2)))
    Linf_u = float(np.max(np.abs(u_at_ghia - ghia_u)))
    Linf_v = float(np.max(np.abs(v_at_ghia - ghia_v)))

    fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])
    pcm = ax1.pcolormesh(x, y, speed_norm, cmap="viridis",
                         vmin=0, vmax=float(np.percentile(speed_norm, 99)),
                         shading="auto")
    ax1.streamplot(x, y, ux, uy, color="white", density=1.6, linewidth=0.5,
                   arrowsize=0.7)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x/L$"); ax1.set_ylabel(r"$y/L$")
    ax1.set_title(r"Lid-driven cavity Re=1000, BGK, $257^2$ grid",
                  fontsize=10)
    fig.colorbar(pcm, ax=ax1, fraction=0.046, pad=0.02).set_label(r"$|\vec u|/U_{\rm lid}$")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(u_LBM, u_LBM_y, "C0-", linewidth=1.4, label=r"$u(y)$ at $x=0.5L$ (LBM)")
    ax2.plot(ghia_u, ghia_y, "C0o", markersize=5, label="Ghia 1982 (17 pts)",
             markerfacecolor="white", markeredgewidth=1.2)
    ax2.plot(v_LBM_x, v_LBM, "C3-", linewidth=1.4, label=r"$v(x)$ at $y=0.5L$ (LBM)")
    ax2.plot(ghia_x, ghia_v, "C3s", markersize=5, label="Ghia 1982",
             markerfacecolor="white", markeredgewidth=1.2)
    ax2.axhline(0, color="grey", linewidth=0.4)
    ax2.set_xlabel(r"$u$ or $x$ position"); ax2.set_ylabel(r"$y$ or $v$")
    ax2.set_title(rf"Centerline profiles vs Ghia 1982"
                  rf"  ($L_2^u={L2_u*100:.2f}\%$, $L_2^v={L2_v*100:.2f}\%$)",
                  fontsize=10)
    ax2.legend(loc="lower right", fontsize=7.5)
    ax2.grid(alpha=0.3)

    fig.savefig(png_out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_out}")
    print(f"  L2 u = {L2_u*100:.2f}%, L2 v = {L2_v*100:.2f}%")
    print(f"  Linf u = {Linf_u*100:.2f}%, Linf v = {Linf_v*100:.2f}%")
    return {"L2_u_pct": L2_u * 100, "L2_v_pct": L2_v * 100,
            "Linf_u_pct": Linf_u * 100, "Linf_v_pct": Linf_v * 100}


if __name__ == "__main__":
    base = "/home/yzk/LBMProject/round37_showcase_specs/results"
    out = "/home/yzk/LBMProject_showcase/showcase/benchmarks/lid_driven/lid_driven_cavity_Re1000.png"
    metrics = main(f"{base}/cavity_re1000_field.csv",
                   f"{base}/cavity_re1000_centerline_u.csv",
                   f"{base}/cavity_re1000_centerline_v.csv",
                   out)
    import json
    with open("/home/yzk/LBMProject_showcase/showcase/benchmarks/lid_driven/result.json", "w") as f:
        json.dump({"case": "B1 Lid-driven Re=1000", "grid": "257x257x3", "tau": 0.577,
                   "U_lid_LU": 0.1, "reference": "Ghia, Ghia & Shin 1982",
                   **metrics}, f, indent=2)
