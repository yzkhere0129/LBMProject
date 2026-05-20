"""Project a 3D STL onto the xy plane, then z-extrude into a thin silhouette
   STL suitable for quasi-2D LBM simulations (nz~4 slab).

Strategy:
  1. Load STL, find xy bbox.
  2. Rasterise all triangle xy-projections onto a high-resolution 2D mask.
  3. Run morphology close + small-dilate to fill thin gaps.
  4. Extract the boundary polygon via marching squares.
  5. Triangulate the polygon (top face, bottom face, side walls) into a
     new STL extruded between z_lo and z_hi.

Output STL is axis-aligned (silhouette = xy plane, z = thin slab).
The new STL has ~100-1000× fewer triangles than the original.
"""
import argparse
import struct
import numpy as np
import sys

ap = argparse.ArgumentParser()
ap.add_argument("input_stl")
ap.add_argument("output_stl")
ap.add_argument("--raster-cells", type=int, default=400,
                help="raster resolution along x; y matched by aspect")
ap.add_argument("--z-thickness", type=float, default=0.1,
                help="thickness of extruded silhouette (in model units)")
ap.add_argument("--projection", default="xy",
                choices=["xy", "xz", "yz"],
                help="which 2D plane to project onto (silhouette plane)")
ap.add_argument("--swap", default=None,
                choices=["none", "x↔y", "x↔z", "y↔z", "xyz"],
                help="permute axes BEFORE projection (handles non-standard STL orientation)")
args = ap.parse_args()


def load_stl(path):
    with open(path, "rb") as f:
        head = f.read(80)
        if head[:5] == b"solid":
            # might still be binary with "solid" header — sniff for facet
            f.seek(0); content = f.read(2048)
            if b"facet" in content:
                # ASCII
                f.seek(0)
                tris = []
                lines = f.read().decode().splitlines()
                i = 0
                while i < len(lines):
                    if "vertex" in lines[i]:
                        vs = []
                        for _ in range(3):
                            parts = lines[i].split()
                            vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
                            i += 1
                        tris.append(vs)
                    else:
                        i += 1
                return np.array(tris, dtype=np.float32)
            # else fall through to binary path
        f.seek(80)
        ntri = struct.unpack("<I", f.read(4))[0]
        arr = np.frombuffer(f.read(ntri * 50), dtype=np.uint8).reshape(ntri, 50)
        verts = np.frombuffer(arr[:, 12:48].tobytes(),
                              dtype=np.float32).reshape(ntri, 3, 3)
        return verts


def write_binary_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"silhouette STL by project_stl_to_silhouette.py".ljust(80, b"\x00")[:80])
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            v0, v1, v2 = tri
            e1 = v1 - v0; e2 = v2 - v0
            n = np.cross(e1, e2)
            ln = np.linalg.norm(n)
            n = n / ln if ln > 1e-20 else np.array([0., 0., 1.])
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))


def rasterise_triangles(verts2, nx, ny, lo, hi):
    """Render filled triangle list (in 2D) onto an nx×ny boolean mask using
       barycentric scanline. verts2: (ntri,3,2)."""
    mask = np.zeros((ny, nx), dtype=bool)
    dx = (hi[0] - lo[0]) / nx
    dy = (hi[1] - lo[1]) / ny
    for tri in verts2:
        x0, y0 = tri[0]; x1, y1 = tri[1]; x2, y2 = tri[2]
        # Triangle bbox in cell coords
        ix0 = max(0,    int((min(x0, x1, x2) - lo[0]) / dx) - 1)
        ix1 = min(nx-1, int((max(x0, x1, x2) - lo[0]) / dx) + 1)
        iy0 = max(0,    int((min(y0, y1, y2) - lo[1]) / dy) - 1)
        iy1 = min(ny-1, int((max(y0, y1, y2) - lo[1]) / dy) + 1)
        # Edge function
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-20: continue
        inv_d = 1.0 / denom
        for iy in range(iy0, iy1 + 1):
            y = lo[1] + (iy + 0.5) * dy
            for ix in range(ix0, ix1 + 1):
                x = lo[0] + (ix + 0.5) * dx
                a = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) * inv_d
                b = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) * inv_d
                c = 1 - a - b
                if a >= -1e-7 and b >= -1e-7 and c >= -1e-7:
                    mask[iy, ix] = True
    return mask


def main():
    print(f"Loading {args.input_stl}...", file=sys.stderr)
    verts = load_stl(args.input_stl)
    print(f"  {len(verts)} triangles", file=sys.stderr)

    # Axis permutation (handles STLs whose 'z' is not vertical).
    if args.swap == "x↔y":
        verts = verts[:, :, [1, 0, 2]]
    elif args.swap == "x↔z":
        verts = verts[:, :, [2, 1, 0]]
    elif args.swap == "y↔z":
        verts = verts[:, :, [0, 2, 1]]
    elif args.swap == "xyz":
        verts = verts[:, :, [2, 0, 1]]

    # Select 2D projection axes.
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[args.projection]
    z_axis = {"xy": 2, "xz": 1, "yz": 0}[args.projection]

    verts2 = verts[:, :, list(axes)]  # (ntri, 3, 2)
    lo = verts2.reshape(-1, 2).min(axis=0)
    hi = verts2.reshape(-1, 2).max(axis=0)
    print(f"  projection={args.projection}  bbox: {lo} → {hi}", file=sys.stderr)

    # Rasterise onto fine grid.
    nx = args.raster_cells
    aspect_y = (hi[1] - lo[1]) / (hi[0] - lo[0])
    ny = max(64, int(nx * aspect_y))
    print(f"  rasterising at {nx}×{ny} (subset of triangles)...", file=sys.stderr)
    # Subsample triangles for speed if huge; rasterised filling will catch missed
    # interiors via dense triangle layout.
    if len(verts2) > 200_000:
        idx = np.random.choice(len(verts2), 200_000, replace=False)
        verts2_sub = verts2[idx]
    else:
        verts2_sub = verts2
    mask = rasterise_triangles(verts2_sub, nx, ny, lo, hi)
    print(f"  rasterised {mask.sum()} solid cells / {mask.size} total", file=sys.stderr)

    # Morphological close to seal gaps (3x3 structuring element).
    from scipy.ndimage import binary_closing, binary_fill_holes
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    print(f"  after close+fill: {mask.sum()} solid cells", file=sys.stderr)

    # Extract boundary polygon via marching squares (skimage.measure.find_contours).
    from skimage import measure
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        print("ERROR: no contour found", file=sys.stderr); sys.exit(1)
    # Pick largest contour (the outer boundary).
    contour = max(contours, key=len)
    # Convert raster (row, col) → world (x, y) using bbox.
    dx_r = (hi[0] - lo[0]) / nx
    dy_r = (hi[1] - lo[1]) / ny
    px = lo[0] + (contour[:, 1] + 0.5) * dx_r
    py = lo[1] + (contour[:, 0] + 0.5) * dy_r
    poly = np.column_stack([px, py])
    print(f"  outer contour: {len(poly)} points", file=sys.stderr)

    # Simplify polygon (Douglas-Peucker via scikit-image approximation).
    from skimage.measure import approximate_polygon
    poly = approximate_polygon(poly, tolerance=0.5 * dx_r)
    print(f"  simplified polygon: {len(poly)} points", file=sys.stderr)

    # Triangulate filled polygon into top and bottom faces + side walls.
    # Top/bottom: fan triangulation from polygon centroid (works for convex
    # or mildly concave; for highly concave shapes use a real triangulator).
    z_lo = verts[:, :, z_axis].min()
    z_hi = z_lo + args.z_thickness
    c = poly.mean(axis=0)
    tris_out = []
    for i in range(len(poly) - 1):
        # Bottom face (CCW from outside i.e. z=z_lo looking up)
        v_c_lo = np.array([c[0], c[1], z_lo], dtype=np.float32)
        v0_lo  = np.array([poly[i][0],   poly[i][1],   z_lo], dtype=np.float32)
        v1_lo  = np.array([poly[i+1][0], poly[i+1][1], z_lo], dtype=np.float32)
        tris_out.append(np.array([v_c_lo, v1_lo, v0_lo], dtype=np.float32))  # CCW from below
        # Top face
        v_c_hi = np.array([c[0], c[1], z_hi], dtype=np.float32)
        v0_hi  = np.array([poly[i][0],   poly[i][1],   z_hi], dtype=np.float32)
        v1_hi  = np.array([poly[i+1][0], poly[i+1][1], z_hi], dtype=np.float32)
        tris_out.append(np.array([v_c_hi, v0_hi, v1_hi], dtype=np.float32))  # CCW from above
        # Side wall (2 triangles)
        tris_out.append(np.array([v0_lo, v1_lo, v0_hi], dtype=np.float32))
        tris_out.append(np.array([v0_hi, v1_lo, v1_hi], dtype=np.float32))

    # If projection wasn't xy, need to re-map xyz: the silhouette's "2D plane"
    # axes are `axes` and the extrusion axis is `z_axis`. Write coords back
    # in original convention so the model is in its proper orientation.
    if args.projection != "xy":
        # Currently tris_out has coords as (axes[0], axes[1], z_axis).
        # Need to permute back to (x, y, z).
        ax_to_world = {0: axes[0], 1: axes[1], 2: z_axis}
        # ax_to_world[k] = which world axis the k-th coord represents
        # We need world coord at axis w by selecting tris_out[..., k] where ax_to_world[k]=w
        inv = [None]*3
        for k, w in ax_to_world.items(): inv[w] = k
        tris_remapped = []
        for tri in tris_out:
            new = np.zeros_like(tri)
            for w in range(3):
                new[:, w] = tri[:, inv[w]]
            tris_remapped.append(new)
        tris_out = tris_remapped

    print(f"  output: {len(tris_out)} triangles", file=sys.stderr)
    write_binary_stl(args.output_stl, tris_out)
    print(f"  wrote {args.output_stl}", file=sys.stderr)


if __name__ == "__main__":
    main()
