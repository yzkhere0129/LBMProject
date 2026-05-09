# LBM-CUDA Show Case — 316L LPBF melt-pool simulation

A self-contained snapshot of the project's current best result, intended
for external presentation.

![Melt pool 3D isosurface](figures/melt_pool_3d_316L_240us.png)

## What this shows

A laser-powder-bed-fusion (LPBF) line-scan simulation of 316L stainless
steel under a 150 W, 0.8 m/s, 50 μm-spot moving Gaussian laser, rendered
at t = 240 μs (3000 LBM steps). The surface is the VOF=0.5 isosurface,
coloured by local temperature. The bright (4000 K) plume marks the
boiling vapourisation zone; the cooler (300 K) tail is the just-frozen
solidified track behind the laser.

## Physics stack

| Module | Method |
|---|---|
| Fluid | D3Q19 BGK + EDM forcing, TRT collision |
| Free surface | VOF / PLIC interface tracking |
| Thermal | D3Q7 enthalpy method (mushy zone) |
| Laser | 16-bounce DDA ray-tracing, Born-Wolf complex Fresnel (n=2.9613, k=4.0133 @ 1064 nm for 316L) |
| Recoil | Anisimov–Knight pressure |
| Evaporation | Hertz-Knudsen cooling, α_evap = 0.18 |
| Surface forces | Marangoni stress, CSF surface tension |

## Numerics

* Grid 500 × 200 × 100, dx = 2 μm, dt = 80 ns
* CFL-strict mode, force substep, 200-step laser ramp
* GPU: single CUDA device

## Headline numbers at 240 μs

| Metric | Value |
|---|---|
| T_max | 4079 K |
| v_max | 6.0 m/s |
| Pool length | 220 μm |
| Pool width | 92 μm |
| Envelope min z (keyhole floor) | −103 μm |
| Ma_LU max | 0.50 |
| Mass drift | −0.33 % |

## How to reproduce the figure

The original VTK keyframe is 600 MB and not committed. The repo ships the
extracted isosurface (~210 KB):

```bash
python3 showcase/scripts/render_3d_pool.py \
    showcase/data/melt_pool_316L_240us.npz \
    showcase/figures/melt_pool_3d_316L_240us.png
```

To regenerate the `.npz` from a fresh VTK keyframe:

```bash
python3 showcase/scripts/extract_isosurface.py <path/to/keyframe.vtk> \
    <step> showcase/data/melt_pool_316L_240us.npz
```

## File layout

```
showcase/
├── README.md                              this file
├── data/
│   ├── melt_pool_316L_240us.npz           extracted VOF=0.5 verts + T (~210 KB)
│   └── metrics.json                       headline numbers + stack info
├── figures/
│   └── melt_pool_3d_316L_240us.png        the showcase 3D figure
└── scripts/
    ├── extract_isosurface.py              VTK → .npz extractor
    └── render_3d_pool.py                  .npz → PNG renderer
```
