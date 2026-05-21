"""Q-criterion isosurface visualization — TUM-engineering-style.

Q = 1/2 (||Ω||² - ||S||²) where S = symmetric, Ω = antisymmetric parts of ∇u.
Q > 0 cells are rotation-dominated → vortex.

Pipeline:
  1. Load .vtk snapshot (structured grid with velocity vectors).
  2. Compute gradient + Q via PyVista's compute_derivative.
  3. Extract isosurface at Q = Q_thresh.
  4. Color the isosurface by velocity magnitude (|u|).
  5. Add body surface (extracted as contour of solid mask at level 0.5).
  6. Render with off_screen Plotter → PNG.

Usage:
  python3 scripts/aero/plot_q_isosurface.py <vtk_file> [<mask_zmid.txt>] [<out_png>]

  python3 scripts/aero/plot_q_isosurface.py \
      output_30k_amr_off/snap_0030000.vtk \
      output_30k_amr_off/mask_zmid.txt \
      images/naca_q_iso.png
"""
import os
import sys
import numpy as np
import pyvista as pv


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


def main(vtk_path, mask_path=None, out_path=None, q_frac=0.99, body_zoom=True):
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

    q_thresh = auto_q_threshold(mesh.point_data["qcrit"], q_frac)
    print(f"  Q_thresh (top {(1-q_frac)*100:.0f}% of positive Q) = {q_thresh:.3e}")

    # Add velocity magnitude for coloring
    vel = mesh.point_data["velocity"]
    mesh.point_data["umag"] = np.linalg.norm(vel, axis=1)

    print("Extracting Q isosurface...")
    iso = mesh.contour(isosurfaces=[q_thresh], scalars="qcrit")
    print(f"  isosurface points = {iso.n_points}, cells = {iso.n_cells}")

    # Color iso by velocity magnitude (sample velocity onto iso surface)
    iso_pts = iso.points
    iso = iso.sample(mesh)  # interpolate all scalars from mesh

    body = None
    if mask_path:
        print("Building body surface from mask...")
        body = build_body_mesh_from_mask(mask_path, mesh)
        if body is not None:
            print(f"  body points = {body.n_points}")

    # Render off-screen
    print("Rendering...")
    pv.set_plot_theme("document")
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1000))

    if iso.n_points > 0:
        pl.add_mesh(iso, scalars="umag",
                    cmap="turbo", clim=[0.0, max(0.05, iso.point_data["umag"].max())],
                    show_scalar_bar=True,
                    scalar_bar_args={"title": "|u| (LU)", "n_labels": 4})
    else:
        print("WARN: no Q-isosurface points — Q field may be all zero/negative.")

    if body is not None and body.n_points > 0:
        pl.add_mesh(body, color="#444444", opacity=1.0, show_scalar_bar=False)

    # Camera setup: zoom to body bbox + downstream wake region.
    mesh_dims = mesh.dimensions
    is_quasi_2d = mesh_dims[2] <= 8

    if body is not None and body.n_points > 0:
        bb = body.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
        cx, cy, cz = body.center
        body_chord = max(bb[1] - bb[0], bb[3] - bb[2])
        # Frame: body + 3 chords downstream + 1.5 chord cross
        frame_x = (bb[0] - 0.5 * body_chord, bb[1] + 3.0 * body_chord)
        frame_y = (cy - 1.5 * body_chord, cy + 1.5 * body_chord)
        # Set camera focus to wake middle, view via xy plane
        cx_view = 0.5 * (frame_x[0] + frame_x[1])
        cy_view = cy
        cz_view = cz
        view_dx = (frame_x[1] - frame_x[0]) * 0.55  # half-extent + margin
    else:
        cx_view, cy_view, cz_view = mesh.center
        view_dx = (mesh.bounds[1] - mesh.bounds[0]) * 0.5

    if is_quasi_2d:
        # Top-down (looking -z), view_up = +y. parallel_scale = half-height of view.
        # Frame: airfoil + ~4 chords downstream wake. Center mid-wake.
        if body is not None and body.n_points > 0:
            cx_view = body.center[0] + body_chord * 1.5  # center mid-wake
            cy_view = body.center[1]
            # parallel_scale = half-height in world units. Window aspect 1600/1000 = 1.6.
            # We want horizontal extent ≥ body + 3 chords of wake = 4 chord wide.
            # horizontal_extent = 2 * pscale * 1.6 = 3.2 pscale → pscale ≥ 4/3.2 = 1.25 chord.
            pscale = body_chord * 1.5
        else:
            cx_view, cy_view, _ = mesh.center
            pscale = (mesh.bounds[3] - mesh.bounds[2]) * 0.2
        pl.camera_position = [
            (cx_view, cy_view, cz_view + 10.0),
            (cx_view, cy_view, cz_view),
            (0, 1, 0),
        ]
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = pscale
        pl.add_text("quasi-2D (nz≤8): Q-iso ≈ small blobs (no 3D vortex tubes)",
                    position="upper_left", font_size=11, color='red')
    else:
        # Oblique 3/4 view: TUM engineering style
        pl.camera_position = [
            (cx_view - view_dx * 0.6, cy_view - view_dx * 0.9, cz_view + view_dx * 0.6),
            (cx_view, cy_view, cz_view),
            (0, 0, 1),
        ]

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
    main(vtk, mask, out)
