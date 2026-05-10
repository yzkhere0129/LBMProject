"""
render_sections.py — Render 3-panel cross-section figure from extracted .npz.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm


def main(npz_in: str, png_out: str) -> None:
    d = np.load(npz_in)
    dx = float(d["dx"])
    nx, ny, nz = int(d["nx"]), int(d["ny"]), int(d["nz"])
    long_T = d["long_T"]
    long_fl = d["long_fl"]
    long_speed = d["long_speed"]
    trans_T = d["trans_T"]
    trans_fl = d["trans_fl"]
    top_T = d["top_T"]
    top_speed = d["top_speed"]
    substrate_k = int(d["substrate_top_k"])

    z_ref = substrate_k * dx * 1e6
    x = (np.arange(nx) * dx) * 1e6
    y = (np.arange(ny) * dx) * 1e6
    z = (np.arange(nz) * dx) * 1e6 - z_ref

    x = x - x.mean()
    y = y - y.mean()

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    T_min, T_max = 300, 3500
    cmap_T = cm.get_cmap("inferno")

    ax1 = fig.add_subplot(gs[0, :])
    extent_xz = [x[0], x[-1], z[0], z[-1]]
    im1 = ax1.imshow(long_T, origin="lower", extent=extent_xz,
                     cmap=cmap_T, vmin=T_min, vmax=T_max, aspect="equal",
                     interpolation="bilinear")
    ax1.contour(x, z, long_fl, levels=[0.5], colors="cyan", linewidths=1.2,
                linestyles="-")
    ax1.contour(x, z, long_T, levels=[1700], colors="lime", linewidths=0.8,
                linestyles="--", alpha=0.7)
    ax1.contour(x, z, long_T, levels=[3000], colors="white", linewidths=0.6,
                linestyles=":", alpha=0.6)
    ax1.axhline(0, color="white", linestyle=":", linewidth=0.5, alpha=0.4)
    ax1.set_xlabel(r"$x$ (scan direction) [$\mu$m]")
    ax1.set_ylabel(r"$z$ — substrate-top reference [$\mu$m]")
    ax1.set_title(r"Longitudinal section ($y\!=\!0$): keyhole + melt pool, $T$ field "
                  r"+ VOF=0.5 (cyan), $T_{\rm melt}$=1700K (--), $T_{\rm boil}$≈3000K (:)",
                  fontsize=11)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.025, pad=0.01)
    cbar1.set_label(r"$T$ [K]")

    ax2 = fig.add_subplot(gs[1, 0])
    extent_yz = [y[0], y[-1], z[0], z[-1]]
    im2 = ax2.imshow(trans_T, origin="lower", extent=extent_yz,
                     cmap=cmap_T, vmin=T_min, vmax=T_max, aspect="equal",
                     interpolation="bilinear")
    ax2.contour(y, z, trans_fl, levels=[0.5], colors="cyan", linewidths=1.2)
    ax2.contour(y, z, trans_T, levels=[1700], colors="lime", linewidths=0.8,
                linestyles="--", alpha=0.7)
    ax2.axhline(0, color="white", linestyle=":", linewidth=0.5, alpha=0.4)
    ax2.set_xlabel(r"$y$ [$\mu$m]")
    ax2.set_ylabel(r"$z$ [$\mu$m]")
    ax2.set_title(r"Transverse section (mid-pool, behind laser)", fontsize=11)
    fig.colorbar(im2, ax=ax2, fraction=0.05, pad=0.02).set_label(r"$T$ [K]")

    ax3 = fig.add_subplot(gs[1, 1])
    extent_xy = [x[0], x[-1], y[0], y[-1]]
    speed_max = float(np.percentile(top_speed, 99))
    im3 = ax3.imshow(top_speed, origin="lower", extent=extent_xy,
                     cmap="viridis", vmin=0, vmax=speed_max, aspect="equal",
                     interpolation="bilinear")
    ax3.contour(x, y, top_T, levels=[1700], colors="orange", linewidths=1.0,
                linestyles="-")
    ax3.contour(x, y, top_T, levels=[3000], colors="red", linewidths=0.8,
                linestyles="--", alpha=0.8)
    ax3.set_xlabel(r"$x$ [$\mu$m]")
    ax3.set_ylabel(r"$y$ [$\mu$m]")
    ax3.set_title(r"Top-down view (just below substrate top): "
                  r"$|\vec u|$ + $T_{\rm melt}$ (orange) + $T_{\rm boil}$ (red)",
                  fontsize=11)
    fig.colorbar(im3, ax=ax3, fraction=0.05, pad=0.02).set_label(r"$|\vec u|$ [m/s]")

    fig.suptitle(
        r"316L LPBF — laser P=150 W, $v_{\rm scan}$=0.8 m/s, $\sigma_0$=50 $\mu$m, $t$=240 $\mu$s "
        r"(D3Q19 + EDM + TRT, ray-tracing keyhole, ESM thermal, VOF/PLIC)",
        fontsize=12.5, weight="bold", y=1.02
    )

    fig.savefig(png_out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {png_out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
