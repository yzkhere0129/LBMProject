#!/bin/bash
# Phase 3.2 — finite-span NACA wing 3D demo (transfer-ready).
#
# REQUIRES: this branch's aero_naca0012_cumulant binary (commit fa9bf50+),
# which has --lz-over-c, --bc-z wall, --perz-force, --span-lo/hi flags.
#
# Setup:
#   D/dx=30, lx=15c, ly=10c, lz=2c
#   nz=60 cells → 8.1M total cells → ~2 GB GPU memory
#   NACA0012 stamped at z ∈ [0.5c, 1.5c] = central 1c span of 2c domain
#   Margins: 0.5c above + 0.5c below wing (between wing tip and z-wall)
#   z-walls at top/bottom (--bc-z wall)
#   QBB sub-cell curved BC + sparse qfrac
#   --perz-force for sectional Cl(z) plots
#
# Args:
#   $1 = steps (default 10000 — short demo; use 30000 for settled mean)
#
# On a faster GPU (e.g. RTX 5060, A100):
#   - 8.1M cells × 27 floats × 2 buffers ≈ 1.75 GB f arrays
#   - + qfrac sparse ~100 MB + buffers ~150 MB
#   - Total ~2 GB — fits any modern GPU
#   - Expect 50-200 MLUPS depending on hardware → 5000 steps in ~10-40 min

set -u
ROOT=${ROOT:-/home/yzk/CompressibleCFD}
BIN=$ROOT/build/aero_naca0012_cumulant
STEPS=${1:-10000}
OUT=$ROOT/output_phase3_2_wing_${STEPS}

if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN missing. Build first: cmake --build build --target aero_naca0012_cumulant"
    exit 1
fi

rm -rf "$OUT" && mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] Launching Phase 3.2 3D NACA wing demo"
echo "  out: $OUT  steps: $STEPS"
echo "  nz=60 (2c span × D/dx=30), wing in z ∈ [0.5c, 1.5c]"
echo "  bc-z=wall (no-slip top/bottom), bc=qbb-snode (curved at airfoil)"

"$BIN" \
    --resolution 30 --re 2000 --u-max-lu 0.05 --alpha 8 \
    --steps "$STEPS" --probe-every 100 --vtk-every $((STEPS/2)) \
    --lx-over-c 15 --ly-over-c 10 --lz-over-c 2.0 \
    --bc qbb-snode --sparse-qfrac --bc-z wall --perz-force \
    --span-lo 0.25 --span-hi 0.75 \
    --shape naca \
    --output-dir "$OUT" \
    | tee "$OUT/run.log"

echo ""
echo "[$(date +%H:%M:%S)] Done."
echo "Forces: $OUT/forces.csv"
echo "Per-z forces (sectional Cl(z)): $OUT/forces_perz.csv"
echo "VTK snapshots: $OUT/snap_*.vtk"
echo ""
echo "Next steps (run on this machine after copying $OUT back):"
echo "  python3 $ROOT/scripts/aero/plot_q_isosurface.py $OUT/snap_$(printf %07d $STEPS).vtk $OUT/mask_zmid.txt images/phase3_2_q_iso.png"
echo "  → Q-isosurface 3D vortex tubes (real 3D wingtip vortex!)"
