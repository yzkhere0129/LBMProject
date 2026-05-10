"""
render_cylinder.py — Publication-quality re-render of the Schäfer-Turek 2D-2
cylinder cross-flow case (D3Q27 Cumulant + Ladd inlet, Re=100, D/dx=20).

Produces 3 figures:
  1. cylinder_vorticity_snapshot.png  — single high-quality vorticity snapshot
  2. cylinder_speed_snapshot.png      — single high-quality speed snapshot
  3. cylinder_forces_strouhal.png     — Cd/Cl time history + St FFT, with DFG comparison
"""
import os
import sys
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


CASE_DIR = "/home/yzk/CompressibleCFD/output_BEST_cumulant_d20_st2d2"
OUT_DIR = "/home/yzk/LBMProject_showcase/showcase/cylinder/figures"
DATA_DIR = "/home/yzk/LBMProject_showcase/showcase/cylinder/data"

D = 0.1
CYL_X = 0.2 + D / 2
CYL_Y = 0.2
U_avg = 1.0


def load_velocity_slice(vtk_path: str):
    g = pv.read(vtk_path)
    nx, ny, nz = g.dimensions
    vel = g.point_data["velocity"].reshape((nz, ny, nx, 3))
    bounds = g.bounds
    dx = (bounds.x_max - bounds.x_min) / (nx - 1)
    k_mid = nz // 2
    u = vel[k_mid, :, :, 0]
    v = vel[k_mid, :, :, 1]
    speed = np.hypot(u, v)
    dudy, dudx = np.gradient(u, dx, axis=(0, 1))
    dvdy, dvdx = np.gradient(v, dx, axis=(0, 1))
    omega = dvdx - dudy
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    return x, y, u, v, speed, omega, dx


def render_vorticity(x, y, omega, png):
    fig, ax = plt.subplots(figsize=(9.5, 2.5), constrained_layout=True)
    omega_lim = float(np.percentile(np.abs(omega), 99.0))
    im = ax.imshow(omega, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                   cmap="RdBu_r", vmin=-omega_lim, vmax=omega_lim,
                   aspect="equal", interpolation="bilinear")
    cyl = Circle((CYL_X, CYL_Y), D / 2, edgecolor="black", facecolor="white",
                 linewidth=1.0, zorder=5)
    ax.add_patch(cyl)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_title(r"Schäfer-Turek 2D-2 cylinder, Re=100, D3Q27 Cumulant + Ladd inlet — "
                 r"vorticity $\omega_z$ at $t\!\approx\!5$ s "
                 r"($C_d\!=\!3.16$ vs DFG 3.22, $S_t\!=\!0.280$ vs DFG 0.30)",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                        ticks=[-omega_lim, 0, omega_lim])
    cbar.set_label(r"$\omega_z$ [1/s]")
    cbar.ax.set_yticklabels([f"{-omega_lim:.0f}", "0", f"{omega_lim:.0f}"])
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, y[-1])
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")


def render_speed(x, y, u, v, speed, png):
    fig, ax = plt.subplots(figsize=(9.5, 2.5), constrained_layout=True)
    im = ax.imshow(speed, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                   cmap="viridis", vmin=0, vmax=float(np.percentile(speed, 99.5)),
                   aspect="equal", interpolation="bilinear")
    cyl = Circle((CYL_X, CYL_Y), D / 2, edgecolor="white", facecolor="black",
                 linewidth=1.0, zorder=5)
    ax.add_patch(cyl)
    skip = 12
    yy, xx = np.meshgrid(y[::skip], x[::skip], indexing="ij")
    ax.streamplot(x, y, u, v, color="white", linewidth=0.4, density=1.4,
                  arrowsize=0.6)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_title(r"Speed magnitude $|\vec u|$ + streamlines, von Kármán vortex street",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label(r"$|\vec u|$ [m/s]")
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, y[-1])
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")


def render_forces_strouhal(forces_csv: str, png: str):
    df = pd.read_csv(forces_csv)
    df = df[df["step"] > 50].reset_index(drop=True)
    t = df["t"].values
    Cd = df["Cd"].values
    Cl = df["Cl"].values

    win_lo = float(t[-1]) * 0.5
    mask = t > win_lo
    Cd_win = Cd[mask]
    Cl_win = Cl[mask]
    t_win = t[mask]

    Cd_mean = float(Cd_win.mean())
    Cd_std = float(Cd_win.std())
    Cl_amp = float((Cl_win.max() - Cl_win.min()) / 2.0)

    Cl_demean = Cl_win - Cl_win.mean()
    if len(Cl_demean) > 64:
        dt = float(np.diff(t_win).mean())
        n = len(Cl_demean)
        freqs = np.fft.rfftfreq(n, d=dt)
        spec = np.abs(np.fft.rfft(Cl_demean))
        valid = freqs > 0.5
        if valid.any():
            i_peak = np.argmax(spec[valid]) + valid.argmax()
            f_peak = float(freqs[i_peak])
        else:
            f_peak = float(freqs[np.argmax(spec)])
    else:
        f_peak = float("nan")
    St_LBM = f_peak * D / U_avg

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), constrained_layout=True,
                              gridspec_kw={"height_ratios": [1.0, 1.0]})

    ax = axes[0]
    ax.plot(t, Cd, color="C0", linewidth=0.7, label=r"$C_d$ (LBM, D3Q27 Cumulant)")
    ax.plot(t, Cl, color="C1", linewidth=0.7, label=r"$C_l$ (LBM)")
    ax.axhspan(3.22, 3.24, color="C0", alpha=0.20, label=r"DFG 2D-2 band: $C_d\in[3.22,3.24]$")
    ax.axvspan(t[mask].min(), t[mask].max(), color="grey", alpha=0.10,
               label=f"analysis window ({len(t_win)} samples)")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$C_d$, $C_l$")
    ax.set_title(rf"Force coefficients — $\overline{{C_d}}={Cd_mean:.3f}\pm{Cd_std:.3f}$, "
                 rf"$\hat C_l={Cl_amp:.3f}$, "
                 rf"DFG offset = ${(Cd_mean - 3.23) / 3.23 * 100:+.1f}\%$",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if len(Cl_demean) > 64:
        ax.semilogy(freqs[1:], spec[1:], color="C2", linewidth=0.8)
        ax.axvline(f_peak, color="C3", linestyle="--",
                   label=rf"$f_{{\rm peak}}={f_peak:.3f}$ Hz, $S_t = {St_LBM:.3f}$")
        ax.axvspan(0.295 / D * U_avg, 0.305 / D * U_avg, color="C0", alpha=0.20,
                   label=r"DFG 2D-2: $S_t\in[0.295,0.305]$")
        ax.set_xlim(0, 8)
        ax.set_xlabel(r"frequency [Hz]")
        ax.set_ylabel(r"$|{\rm FFT}(C_l)|$")
        ax.set_title(r"Strouhal spectrum from $C_l$ in the analysis window", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(which="both", alpha=0.3)

    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")

    return {
        "Cd_mean": Cd_mean,
        "Cd_std": Cd_std,
        "Cl_amplitude": Cl_amp,
        "f_peak_Hz": f_peak,
        "St_LBM": St_LBM,
        "DFG_Cd_band": [3.22, 3.24],
        "DFG_St_band": [0.295, 0.305],
        "Cd_pct_offset": (Cd_mean - 3.23) / 3.23 * 100.0,
        "St_pct_offset": (St_LBM - 0.30) / 0.30 * 100.0,
        "n_samples_in_window": int(len(t_win)),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    vtk_path = os.path.join(CASE_DIR, "snap_0030000.vtk")
    forces_csv = os.path.join(CASE_DIR, "forces.csv")

    x, y, u, v, speed, omega, dx = load_velocity_slice(vtk_path)
    print(f"loaded {vtk_path} — dims = ({len(y)}, {len(x)}), dx = {dx:.4g}")

    render_vorticity(x, y, omega, os.path.join(OUT_DIR, "cylinder_vorticity_snapshot.png"))
    render_speed(x, y, u, v, speed, os.path.join(OUT_DIR, "cylinder_speed_snapshot.png"))
    metrics = render_forces_strouhal(forces_csv, os.path.join(OUT_DIR, "cylinder_forces_strouhal.png"))

    import json
    metrics_path = os.path.join(DATA_DIR, "cylinder_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "case": "Schaefer-Turek 2D-2 cylinder cross-flow",
            "Re": 100,
            "D_over_dx": 20,
            "scheme": "D3Q27 Cumulant + Ladd moving-wall inlet",
            **metrics,
        }, f, indent=2)
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
