"""Axis-align an aircraft STL for LBM external-aero simulation.

The FA-18E source model is exported in a steep nose-up pose (length axis
pitched ~53 deg in the y-z plane). The LBM CLI can only scale+translate,
not rotate, so we bake a clean rotation into a new STL:

    domain +x  = streamwise (flow direction); NOSE points to -x (upstream)
    domain  y  = wingspan (lateral)
    domain  z  = vertical (tail fins up)

Frame is found by PCA: largest principal axis = length, the clean model
symmetry axis (original x) = wingspan, third = vertical. Nose vs tail is
disambiguated by cross-section width (nose end is the narrow pointed one).
Output is recentered to the origin so CLI --stl-tx/ty/tz place it directly.
"""
import argparse
import struct
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("input_stl")
ap.add_argument("output_stl")
ap.add_argument("--pitch", type=float, default=0.0,
                help="nose-up pitch in deg about the span (y) axis = angle "
                     "of attack. Flow stays +x; positive lifts the nose.")
args = ap.parse_args()


def load_stl(path):
    with open(path, "rb") as f:
        head = f.read(80)
        if head[:5] == b"solid":
            f.seek(0); content = f.read(2048)
            if b"facet" in content:
                f.seek(0)
                tris, lines, i = [], f.read().decode().splitlines(), 0
                while i < len(lines):
                    if "vertex" in lines[i]:
                        vs = []
                        for _ in range(3):
                            p = lines[i].split()
                            vs.append([float(p[1]), float(p[2]), float(p[3])])
                            i += 1
                        tris.append(vs)
                    else:
                        i += 1
                return np.array(tris, dtype=np.float32)
        f.seek(80)
        ntri = struct.unpack("<I", f.read(4))[0]
        arr = np.frombuffer(f.read(ntri * 50), dtype=np.uint8).reshape(ntri, 50)
        return np.frombuffer(arr[:, 12:48].tobytes(),
                             dtype=np.float32).reshape(ntri, 3, 3)


def write_binary_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"axis-aligned by align_stl_axes.py".ljust(80, b"\x00")[:80])
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            v0, v1, v2 = tri
            n = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(n)
            n = n / ln if ln > 1e-20 else np.array([0., 0., 1.])
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))


v = load_stl(args.input_stl)
pts = v.reshape(-1, 3)
c = pts.mean(0)
P = pts - c

# PCA: eigenvectors of covariance, sorted by extent.
w, V = np.linalg.eigh(np.cov(P.T))
order = np.argsort(w)[::-1]          # large -> small
e_len, e_span, e_vert = V[:, order[0]], V[:, order[1]], V[:, order[2]]

# Orient nose to -x: nose is the NARROW end along the length axis.
s = P @ e_len
lo, hi = s.min(), s.max(); L = hi - lo
def end_width(mask):
    return P[mask][:, :].dot(e_span).ptp()  # lateral spread proxy
w_lo = end_width(s < lo + 0.12 * L)
w_hi = end_width(s > hi - 0.12 * L)
# nose = narrow end; we want nose at -x, i.e. nose should map to small x.
# After R, x = P·e_len. Flip e_len so the NOSE (narrow end) gets negative x.
nose_at_high_s = w_hi < w_lo
if nose_at_high_s:
    e_len = -e_len                  # so nose (high s) -> -x

# Build right-handed rotation: rows map model coords -> (x_len, y_span, z_vert)
e_vert = np.cross(e_len, e_span)
e_vert /= np.linalg.norm(e_vert)
R = np.stack([e_len, e_span, e_vert])   # 3x3, world = R @ model

Pr = P @ R.T
# Recenter to origin (centroid at 0).
Pr -= Pr.mean(0)

# Upright check: the twin vertical tails are the tallest structures at the
# tail (high x). Whichever z-sign they reach further in is "up". We want
# fins at +z -> if they point down, roll 180 deg about x (negate y and z,
# keeps det=+1 and nose still at -x).
tail = Pr[:, 0] > 0.5 * Pr[:, 0].max()
if abs(Pr[tail, 2].min()) > Pr[tail, 2].max():
    Pr[:, 1] *= -1.0
    Pr[:, 2] *= -1.0

# True aircraft length (invariant under pitch) = x-extent before pitching.
model_len = Pr[:, 0].ptp()

# Optional nose-up pitch about the span (y) axis: rotate in the x-z plane.
# Nose is at -x; positive theta sends it to +z (nose up = +AoA).
if args.pitch != 0.0:
    th = np.radians(args.pitch)
    c_, s_ = np.cos(th), np.sin(th)
    x, z = Pr[:, 0].copy(), Pr[:, 2].copy()
    Pr[:, 0] = x * c_ + z * s_
    Pr[:, 2] = -x * s_ + z * c_

tris = Pr.reshape(-1, 3, 3).astype(np.float32)
write_binary_stl(args.output_stl, tris)

span = Pr.max(0) - Pr.min(0)
print(f"model_len={model_len:.4f}")     # machine-readable: true chord length
print(f"aligned bbox span  x(len)={span[0]:.1f}  y(span)={span[1]:.1f}  z(vert)={span[2]:.1f}")
print(f"aspect  span/len={span[1]/span[0]:.2f}  vert/len={span[2]/span[0]:.2f}")
print(f"wrote {args.output_stl}  ({len(tris)} triangles)")
print("nose at -x (upstream), tail at +x (wake)")
