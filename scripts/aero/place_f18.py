"""Compute --stl-scale/-tx/-ty/-tz to place the axis-aligned F-18 in the
LBM domain, and print the resulting CLI flags + a memory estimate.

Convention (chord c = aircraft length = 1 m, U_inf = 1 m/s):
  x = streamwise   (nose upstream at x_nose, tail downstream)
  y = lateral/span (centred at Ly/2, free-slip far-field faces)
  z = vertical     (centred at Lz/2, periodic faces)

Usage:
  place_f18.py STL --res 40 --lx 12 --ly 8 --lz 4 --x-nose 3.0
Prints a single line of CLI flags on stdout (the launcher evals it) and a
human summary on stderr.
"""
import argparse
import struct
import sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("stl")
ap.add_argument("--res", type=int, default=40)
ap.add_argument("--lx", type=float, default=12.0)
ap.add_argument("--ly", type=float, default=8.0)
ap.add_argument("--lz", type=float, default=4.0)
ap.add_argument("--x-nose", type=float, default=3.0,
                help="nose position from inlet, in chord units")
a = ap.parse_args()


def bbox(path):
    with open(path, "rb") as f:
        f.seek(80); n = struct.unpack("<I", f.read(4))[0]
        arr = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
        v = np.frombuffer(arr[:, 12:48].tobytes(),
                          dtype=np.float32).reshape(-1, 3)
    return v.min(0), v.max(0)


lo, hi = bbox(a.stl)
length_model = hi[0] - lo[0]          # x = length after alignment
scale = 1.0 / length_model            # -> aircraft length = 1 chord = 1 m

# Scaled bbox.
slo, shi = lo * scale, hi * scale
# x: place nose (min x) at x_nose.
tx = a.x_nose - slo[0]
# y, z: centre the body bbox in the domain.
ty = 0.5 * a.ly - 0.5 * (slo[1] + shi[1])
tz = 0.5 * a.lz - 0.5 * (slo[2] + shi[2])

nx = round(a.lx * a.res)
ny = round(a.ly * a.res) + 1
nz = max(4, round(a.lz * a.res))
ncell = nx * ny * nz
gb = ncell * 233 / 1e9                 # 2*27 f + 4 macro + 1 solid bytes/cell

span_y = shi[1] - slo[1]
span_z = shi[2] - slo[2]
print(f"# F-18 placement: scale={scale:.6f} (len 1c), nose@x={a.x_nose}c "
      f"tail@x={a.x_nose+1:.2f}c, wake={a.lx-(a.x_nose+1):.1f}c", file=sys.stderr)
print(f"# mesh {nx}x{ny}x{nz} = {ncell/1e6:.1f}M cells, ~{gb:.1f} GB VRAM "
      f"(fields only)", file=sys.stderr)
print(f"# blockage: span {span_y:.3f}c / {a.ly}c = {100*span_y/a.ly:.1f}% (y), "
      f"vert {span_z:.3f}c / {a.lz}c = {100*span_z/a.lz:.1f}% (z)", file=sys.stderr)
if gb > 7.0:
    print(f"# WARNING: ~{gb:.1f} GB exceeds safe 8 GB budget — reduce --res "
          f"or domain, or use an 11 GB card.", file=sys.stderr)

# stdout: the flags the launcher consumes.
print(f"--resolution {a.res} --lx-over-c {a.lx} --ly-over-c {a.ly} "
      f"--lz-over-c {a.lz} --stl-scale {scale:.6f} "
      f"--stl-tx {tx:.4f} --stl-ty {ty:.4f} --stl-tz {tz:.4f}")
