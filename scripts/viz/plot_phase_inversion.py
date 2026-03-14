#!/usr/bin/env python3
"""
Phase Inversion: At=0.5 RT instability with 10:1 viscosity ratio.
4-panel figure showing full inversion from heavy-over-light to settled state.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

DIR = Path(__file__).parent
NX, NY = 128, 512

STEPS = [3000, 8000, 14000, 25000]
LABELS = [
    "Step 3000\nLinear growth",
    "Step 8000\nMushroom cap",
    "Step 14000\nAdvanced roll-up",
    "Step 25000\nFull inversion",
]

def load(step):
    path = DIR / f"phase_inversion_step{step:05d}.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    f   = data["fill_level"].reshape(NY, NX)
    vx  = data["vx_ms"].reshape(NY, NX)
    vy  = data["vy_ms"].reshape(NY, NX)
    vm  = data["vmag_ms"].reshape(NY, NX)
    return f, vx, vy, vm

cmap = LinearSegmentedColormap.from_list(
    "heavy_light",
    [(0.0, "#2166ac"),   # light (f=0) — blue
     (0.45, "#92c5de"),
     (0.5, "#f7f7f7"),   # interface
     (0.55, "#f4a582"),
     (1.0, "#b2182b")],  # heavy (f=1) — red
)

fig, axes = plt.subplots(1, 4, figsize=(16, 14))

for idx, (step, label) in enumerate(zip(STEPS, LABELS)):
    ax = axes[idx]
    f, vx, vy, vm = load(step)

    im = ax.imshow(f, origin="lower", cmap=cmap, vmin=0, vmax=1,
                   aspect="equal", extent=[0, NX, 0, NY])

    # Interface contour
    ax.contour(np.arange(NX), np.arange(NY), f,
               levels=[0.5], colors="black", linewidths=0.8)

    # Velocity vectors
    skip = 8
    xs = np.arange(0, NX, skip) + skip//2
    ys = np.arange(0, NY, skip) + skip//2
    X, Y = np.meshgrid(xs, ys)
    VX = vx[ys][:, xs]
    VY = vy[ys][:, xs]
    VM = vm[ys][:, xs]

    v_max = vm.max()
    mask = VM > max(1e-6, 0.08 * v_max)
    if mask.any():
        ax.quiver(X[mask], Y[mask], VX[mask], VY[mask],
                  color="k", alpha=0.5,
                  scale=v_max*10, width=0.004, headwidth=3)

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("x [cells]")
    if idx == 0:
        ax.set_ylabel("y [cells]")

    ma_val = v_max / 0.577
    ax.text(0.97, 0.01,
            f"|u|$_{{max}}$={v_max:.4f}\nMa={ma_val:.3f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label(r"Fill level $f$  (0=light, 1=heavy)", fontsize=10)

fig.suptitle(
    "Phase Inversion: Rayleigh-Taylor Instability (At=0.5)\n"
    r"$\rho_H$=3, $\rho_L$=1  |  $\mu_H/\mu_L$=10:1  |  "
    r"$g_{LB}$=1e-5  |  PLIC + TRT  |  $\sigma$=0",
    fontsize=12, fontweight="bold", y=0.98,
)

plt.subplots_adjust(wspace=0.12, right=0.90, top=0.93, bottom=0.03)
out = DIR / "phase_inversion.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close()
