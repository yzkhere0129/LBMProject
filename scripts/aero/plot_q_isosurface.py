"""Q-criterion isosurface visualization — TUM-engineering-style.

Q = 1/2 (||Ω||² - ||S||²) where S = symmetric, Ω = antisymmetric parts of ∇u.
Q > 0 cells are rotation-dominated → vortex.

Pipeline:
  1. Load .vtk snapshot (structured grid with velocity vectors).
  2. Compute gradient + Q via PyVista's compute_derivative.
  3. Extract isosurface at Q = max(top-1% quantile, 4% of Q_max) so we get
     tight vortex cores rather than a fat low-Q envelope.
  4. Color the isosurface by velocity magnitude (|u|).
  5. Add the REAL aircraft surface: read the STL path + scale/translate the
     solver used from <out_dir>/run.log and apply them (falls back to the
     crude mask-broadcast box if run.log/STL are unavailable).
  6. Clip the vortices to a focus box around the aircraft + near wake and
     frame the camera tightly on it (no empty far field).
  7. Render with off_screen Plotter → PNG.

Usage:
  python3 scripts/aero/plot_q_isosurface.py <vtk_file> [<mask.txt>] [<out_png>] [<wake_chords>]

  python3 scripts/aero/plot_q_isosurface.py \
      output_f18_3d_5k/snap_0005000.vtk \
      output_f18_3d_5k/mask_zmid.txt \
      images/f18_q_iso.png 1.3
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


def main(vtk_path, mask_path=None, out_path=None, q_frac=0.99, down=1.3):
    if out_path is None:
        out_path = "images/q_iso.png"
    print(f"Loading {vtk_path}...")
    mesh = load_vtk_grid(vtk_path)
    print(f"  dims = {mesh.dimensions}, n_points = {mesh.n_points}")

    print("Computing gradient + Q criterion...")
    mesh = mesh.compute_derivative(scalars="velocity", qcriterion="qcrit",
                                    gradient=False)
    qmin, qmax = mesh.point_data["qcrit"].min(), mesh.point_data["qcrit"].max()
    print(f"  Q ∈ [{qmin:.3e}, {qmax:.3e}]")

    # Floor the auto threshold at a fraction of Q_max so we get tight vortex
    # cores, not a fat low-Q envelope that swallows the aircraft.
    q_thresh = max(auto_q_threshold(mesh.point_data["qcrit"], q_frac),
                   0.04 * float(qmax))
    print(f"  Q_thresh = {q_thresh:.3e}  (Q_max={qmax:.3e})")

    # Add velocity magnitude for coloring
    vel = mesh.point_data["velocity"]
    mesh.point_data["umag"] = np.linalg.norm(vel, axis=1)

    print("Extracting Q isosurface...")
    iso = mesh.contour(isosurfaces=[q_thresh], scalars="qcrit")
    print(f"  isosurface points = {iso.n_points}, cells = {iso.n_cells}")

    # Color iso by velocity magnitude (sample velocity onto iso surface)
    iso_pts = iso.points
    iso = iso.sample(mesh)  # interpolate all scalars from mesh

    # Body: prefer the real transformed STL (read transform from run.log);
    # fall back to the crude mask-broadcast box only if that fails.
    body = None
    out_dir = os.path.dirname(os.path.abspath(vtk_path))
    rl = parse_run_log(out_dir)
    if rl is not None:
        stl_path, scale, t = rl
        # run.log records an absolute path from the machine it ran on; if that
        # doesn't exist here, resolve to this repo's test_data/<basename>.
        if not os.path.exists(stl_path):
            repo = os.path.dirname(out_dir)
            cand = os.path.join(repo, "test_data", os.path.basename(stl_path))
            if os.path.exists(cand):
                stl_path = cand
        print(f"Loading real STL body: {stl_path} (scale {scale}, translate {t})")
        body = load_stl_body(stl_path, scale, t)
        if body is not None:
            print(f"  STL body points = {body.n_points}")
    if body is None and mask_path:
        print("Building body surface from mask (fallback)...")
        body = build_body_mesh_from_mask(mask_path, mesh)
        if body is not None:
            print(f"  body points = {body.n_points}")

    # Focus box: tight on the aircraft + near wake so the empty far field is
    # cropped out. Margins are in body-length (L) units; downstream gets more
    # so the forming tip/wake vortices stay in frame.
    if body is not None and body.n_points > 0:
        bb = body.bounds
        L = bb[1] - bb[0]                       # body length (~1 chord)
        focus = [bb[0] - 0.10 * L, bb[1] + down * L,
                 bb[2] - 0.12 * L, bb[3] + 0.12 * L,
                 bb[4] - 0.30 * L, bb[5] + 0.30 * L]
    else:
        focus = list(mesh.bounds)

    # Clip the vortex isosurface to the focus box (keep cells fully inside).
    if iso.n_points > 0:
        p = iso.points
        keep = ((p[:, 0] >= focus[0]) & (p[:, 0] <= focus[1]) &
                (p[:, 1] >= focus[2]) & (p[:, 1] <= focus[3]) &
                (p[:, 2] >= focus[4]) & (p[:, 2] <= focus[5]))
        if keep.any():
            iso = iso.extract_points(keep, adjacent_cells=False)

    print("Rendering...")
    pv.set_plot_theme("document")
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1000))

    if iso.n_points > 0:
        pl.add_mesh(iso, scalars="umag", cmap="turbo",
                    clim=[0.0, max(0.05, float(iso.point_data["umag"].max()))],
                    show_scalar_bar=True,
                    scalar_bar_args={"title": "|u| (LU)", "n_labels": 4})
    else:
        print("WARN: no Q-isosurface points in focus box.")

    if body is not None and body.n_points > 0:
        pl.add_mesh(body, color="#888c94", opacity=1.0,
                    specular=0.3, smooth_shading=True, show_scalar_bar=False)

    # Front-3/4 view (from upstream -x, one side -y, above +z), fitted tightly
    # to the focus box via reset_camera(bounds=...) then cropped with zoom.
    fc = np.array([0.5 * (focus[0] + focus[1]),
                   0.5 * (focus[2] + focus[3]),
                   0.5 * (focus[4] + focus[5])])
    off = np.array([-1.3, -1.6, 0.95])          # view direction (magnitude irrelevant)
    pl.camera_position = [tuple(fc + off), tuple(fc), (0, 0, 1)]
    pl.reset_camera()                            # fit all actors (body + clipped vortices)
    pl.camera.zoom(1.5)

    pl.add_axes()
    pl.background_color = "white"

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
    down = float(sys.argv[4]) if len(sys.argv) > 4 else 1.3  # wake length (chords)
    main(vtk, mask, out, down=down)
