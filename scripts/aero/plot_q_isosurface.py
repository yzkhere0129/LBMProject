"""Q-criterion isosurface visualization — TUM-engineering-style.

Q = 1/2 (||Ω||² - ||S||²) where S = symmetric, Ω = antisymmetric parts of ∇u.
Q > 0 cells are rotation-dominated → vortex.

TUM-engineering style: nested TRANSLUCENT Q shells colored by |u| with a soft
green map, over a bright sheened aircraft (real STL), framed nose-upstream-left
with the wake trailing right.

Pipeline:
  1. Load snap_<step>.vtk (velocity vectors); compute Q via compute_derivative.
  2. Draw 3 nested Q-isosurfaces at {4%, 10%, 22%} of Q_max, opacity 0.25/0.42/
     0.68 (faint outer shell -> bright core), colored by |u| (cmap YlGn_r,
     tight clim around freestream so the small in-core deficit shows).
  3. Body = REAL transformed aircraft STL (scale/translate parsed from
     <out_dir>/run.log; path re-resolved to this repo if logged elsewhere).
     Colored by Cp if a sibling rho3d_<step>.vtk density field exists
     (Cp=(rho-1)/(1.5 U^2)), else a bright grey sheen. Mask box is the fallback.
  4. Clip vortices to a tight focus box (body + <wake_chords> downstream);
     camera nose-left / wake-right, 3/4 from above-side; white bg.

NOTE: a steady/laminar/coarse field has few vortices -> expect sparse shells,
not a turbulent plume. The plume needs a turbulent re-run (higher Re + finer
mesh), not a render tweak.

Usage:
  plot_q_isosurface.py <vtk> [<mask.txt>] [<out_png>] [<wake_chords>] [<body_mode>]
  body_mode = auto (Cp if density present else grey) | cp | proxy | grey

  python3 scripts/aero/plot_q_isosurface.py \
      output_f18_3d_5k/snap_0005000.vtk output_f18_3d_5k/mask_zmid.txt \
      images/f18_q_iso.png 1.5
"""
import os
import re
import sys
import numpy as np
import pyvista as pv


def parse_run_log(out_dir):
    """Read the STL path + transform (scale, tx, ty, tz) the solver actually
    used, from <out_dir>/run.log. Returns (stl_path, scale, (tx,ty,tz)) or None."""
    log = os.path.join(out_dir, "run.log")
    if not os.path.exists(log):
        return None
    pat = re.compile(
        r"Loading STL:\s+(\S+)\s+\(scale\s+([\d.eE+-]+),\s*"
        r"translate\s+([\d.eE+-]+),([\d.eE+-]+),([\d.eE+-]+)\)")
    for line in open(log, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            return (m.group(1), float(m.group(2)),
                    (float(m.group(3)), float(m.group(4)), float(m.group(5))))
    return None


def load_stl_body(stl_path, scale, t):
    """Load the aligned STL and apply the solver's scale-then-translate so it
    sits exactly in the flow domain (same order as transform_mesh in the app)."""
    if not os.path.exists(stl_path):
        return None
    surf = pv.read(stl_path)
    surf.points = surf.points * scale + np.asarray(t, dtype=surf.points.dtype)
    return surf


def auto_q_threshold(qcrit, fraction_high=0.97):
    """Pick Q_thresh as a high quantile of positive Q values."""
    pos = qcrit[qcrit > 0]
    if pos.size < 100:
        return 1e-6
    return float(np.quantile(pos, fraction_high))


def load_vtk_grid(path):
    """Load legacy STRUCTURED_POINTS VTK with velocity vectors."""
    mesh = pv.read(path)
    # Ensure velocity is in active vectors / point_data
    if "velocity" not in mesh.point_data and "vectors" in mesh.point_data:
        mesh.rename_array("vectors", "velocity")
    return mesh


def build_body_mesh_from_mask(mask_path, mesh):
    """Reconstruct a body surface from the mid-z mask text file.

    The mask is 2D (ny × nx). Extrude across z to match the 3D grid, then
    contour at 0.5 to produce a solid surface mesh.
    """
    if not os.path.exists(mask_path):
        return None
    with open(mask_path) as f:
        lines = [l for l in f if not l.startswith("#")]
    mask2d = np.array([[int(v) for v in l.split()] for l in lines if l.strip()],
                      dtype=np.float32)
    ny, nx = mask2d.shape
    nx_g, ny_g, nz_g = mesh.dimensions
    # Sanity: the mask should match the xy dims of the grid.
    if nx_g != nx or ny_g != ny:
        print(f"WARN: mask shape ({ny}×{nx}) doesn't match grid xy ({ny_g}×{nx_g})")
        return None
    # Build a structured grid with the same dims but cell-data = mask.
    mask3d = np.broadcast_to(mask2d[None, :, :], (nz_g, ny, nx)).copy()
    body_grid = pv.ImageData(
        dimensions=(nx, ny, nz_g),
        spacing=mesh.spacing,
        origin=mesh.origin,
    )
    # mask3d shape is (nz, ny, nx). PyVista ImageData expects point data in
    # (k slowest, j, i fastest) order = numpy C-order flatten of (nz, ny, nx).
    body_grid.point_data["solid"] = mask3d.flatten(order='C')
    surf = body_grid.contour(isosurfaces=[0.5], scalars="solid")
    return surf


def find_density_vtk(vtk_path):
    """The solver writes a sibling rho3d_<step>.vtk (SCALARS density) next to
    each snap_<step>.vtk. Return its path if present, else None."""
    cand = vtk_path.replace("snap_", "rho3d_")
    return cand if (cand != vtk_path and os.path.exists(cand)) else None


def resolve_body(vtk_path, mask_path, mesh):
    """Load the real transformed aircraft STL (transform from run.log; path
    re-resolved to this repo if logged from another machine). Mask box fallback."""
    out_dir = os.path.dirname(os.path.abspath(vtk_path))
    rl = parse_run_log(out_dir)
    if rl is not None:
        stl_path, scale, t = rl
        if not os.path.exists(stl_path):
            cand = os.path.join(os.path.dirname(out_dir), "test_data",
                                os.path.basename(stl_path))
            if os.path.exists(cand):
                stl_path = cand
        print(f"Body: real STL {stl_path} (scale {scale}, translate {t})")
        body = load_stl_body(stl_path, scale, t)
        if body is not None:
            print(f"  STL body points = {body.n_points}")
            return body
    if mask_path:
        print("Body: mask-broadcast box (fallback)...")
        return build_body_mesh_from_mask(mask_path, mesh)
    return None


def main(vtk_path, mask_path=None, out_path=None, down=1.5, body_mode="auto"):
    if out_path is None:
        out_path = "images/q_iso.png"
    print(f"Loading {vtk_path}...")
    mesh = load_vtk_grid(vtk_path)
    print(f"  dims = {mesh.dimensions}, n_points = {mesh.n_points}")

    print("Computing gradient + Q criterion...")
    mesh = mesh.compute_derivative(scalars="velocity", qcriterion="qcrit",
                                   gradient=False)
    q = mesh.point_data["qcrit"]
    qmax = float(q.max())
    mesh.point_data["umag"] = np.linalg.norm(mesh.point_data["velocity"], axis=1)
    U = float(np.median(mesh.point_data["umag"]))          # ~ freestream LU speed
    print(f"  Q in [{q.min():.3e}, {qmax:.3e}], U_inf~={U:.4f} LU")

    body = resolve_body(vtk_path, mask_path, mesh)
    if body is None or body.n_points == 0:
        print("ERROR: no body geometry resolved."); return

    bb = body.bounds
    L = bb[1] - bb[0]                                       # body length (~1 chord)
    focus = [bb[0] - 0.15 * L, bb[1] + down * L,            # x: nose margin + wake
             bb[2] - 0.40 * L, bb[3] + 0.40 * L,            # y: tip-vortex spread
             bb[4] - 0.50 * L, bb[5] + 0.50 * L]            # z: fin/keel wake

    # ---- body coloring: real Cp if a density field exists, else grey sheen ----
    rho_vtk = find_density_vtk(vtk_path)
    use_cp = (body_mode == "cp") or (body_mode == "auto" and rho_vtk is not None)
    body_scalar = None
    if use_cp and rho_vtk is not None:
        print(f"Body Cp from density field {rho_vtk}")
        rho = load_vtk_grid(rho_vtk)
        # p = cs^2 rho, p_inf = cs^2 (rho_inf=1) -> Cp = (rho-1)/(1.5 U^2).
        rho.point_data["Cp"] = (rho.point_data["density"] - 1.0) / (1.5 * U * U)
        body = body.sample(rho)
        body_scalar = "Cp"
    elif body_mode == "proxy":
        print("Body Cp = 1-(|u|/U)^2 Bernoulli proxy (inviscid; blocky near wall)")
        body = body.sample(mesh)
        body.point_data["Cp"] = 1.0 - (body.point_data["umag"] / U) ** 2
        body_scalar = "Cp"

    # ---- render ----
    print("Rendering...")
    pv.set_plot_theme("document")
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    try:
        pl.enable_depth_peeling(10)                        # correct nested transparency
    except Exception:
        pass
    pl.background_color = "white"

    # Nested translucent Q shells, soft green, colored by |u| (tight clim so the
    # small velocity deficit in the cores actually shows). Faint outer -> bright core.
    clim = [0.55 * U, 1.12 * U]
    n_drawn = 0
    for lev, op in [(0.04 * qmax, 0.25), (0.10 * qmax, 0.42), (0.22 * qmax, 0.68)]:
        iso = mesh.contour(isosurfaces=[lev], scalars="qcrit")
        if iso.n_points == 0:
            continue
        iso = iso.clip_box(focus, invert=False)
        if iso.n_points == 0:
            continue
        iso = iso.sample(mesh)
        last = (lev >= 0.22 * qmax)
        pl.add_mesh(iso, scalars="umag", cmap="YlGn_r", clim=clim, opacity=op,
                    smooth_shading=True, specular=0.1, show_scalar_bar=last,
                    scalar_bar_args={"title": "|u| (LU)", "n_labels": 4})
        n_drawn += 1
    if n_drawn == 0:
        print("WARN: no Q-isosurface in focus box (flow may be steady/attached).")

    # Body: bright sheened surface (grey hero look, or Cp if available).
    if body_scalar:
        pl.add_mesh(body, scalars=body_scalar, cmap="coolwarm", clim=[-1.0, 1.0],
                    smooth_shading=True, specular=0.5, specular_power=15,
                    scalar_bar_args={"title": "Cp", "n_labels": 5})
    else:
        pl.add_mesh(body, color="#d9d2c5", smooth_shading=True, specular=0.55,
                    specular_power=18, ambient=0.25, diffuse=0.7,
                    show_scalar_bar=False)

    # Camera: nose upstream-LEFT, wake trailing RIGHT, 3/4 from above-side.
    # (camera on -y side + above +z => downstream +x projects to screen-right.)
    # Focus on the FOCUS-BOX center so framing adapts to `down` (small down ->
    # centered on body for steady baselines; large down -> includes the wake).
    fc = np.array([0.5 * (focus[0] + focus[1]), 0.5 * (focus[2] + focus[3]),
                   0.5 * (focus[4] + focus[5])])
    span = max(focus[1] - focus[0], focus[3] - focus[2], focus[5] - focus[4])
    off = (span / np.sqrt(0.36 + 0.64 + 0.16)) * np.array([-0.6, -0.8, 0.4])
    pl.camera_position = [tuple(fc + off), tuple(fc), (0, 0, 1)]
    pl.camera.zoom(1.35)                                   # crop residual margin

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pl.screenshot(out_path, transparent_background=False)
    print(f"Saved: {out_path}")
    pl.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    vtk = sys.argv[1]
    mask = sys.argv[2] if len(sys.argv) > 2 else None
    out  = sys.argv[3] if len(sys.argv) > 3 else None
    down = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5  # wake length (chords)
    mode = sys.argv[5] if len(sys.argv) > 5 else "auto"      # auto|cp|proxy|grey
    main(vtk, mask, out, down=down, body_mode=mode)
