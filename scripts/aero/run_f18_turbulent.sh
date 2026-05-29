#!/bin/bash
# F/A-18E TURBULENT-wake run for a Q-criterion hero figure (lab GTX 1080, 8 GB).
#
# WHY this exists: the first F-18 run (run_f18_3d.sh: Re=2000, res=40, 5k steps,
# big 12x8x4 box) came out STEADY and ATTACHED (Cd dead-flat, no wake) — verified
# by field forensics. A Q-criterion render of steady laminar flow is just a few
# blobs, nothing like a turbulent reference. To get a real unsteady/turbulent
# wake you must raise Re and resolution; the 8 GB card caps you at ~30M cells, so
# we SHRINK the domain to a tight wake box to afford a finer grid.
#
# This config (8x5x3 chords, res=60, Re=15000, ILES via the Cumulant operator):
#   mesh 480x301x180 = 26.0M cells, ~6.1 GB  (fits 8 GB with headroom)
#   ~30 convective times = 36000 steps @ ~130 MLUPS ~= 2 h
#   dumps velocity (Q-criterion) AND density (rho3d_*, for surface Cp) each frame
#
# HONEST EXPECTATION: this gives a qualitatively turbulent, "coarse-LES" wake
# (unsteady shedding, shear-layer roll-up, first stage of 3D breakdown) and an
# OSCILLATING Cd — far better than the blobs. It will NOT match HPC/TUM renders
# (those use 1e8-1e9 cells, ~1-2 orders more than our 30M ceiling). There is NO
# explicit Smagorinsky model; regularization is the Cumulant's higher-order
# relaxation (implicit LES). Do NOT quote Cd from Re>=1e4 runs as validated.
#
# Args:  $1 = steps (default 36000)   $2 = tag (default turb)
# Env:   RES=60 LX=8 LY=5 LZ=3 XNOSE=2 RE=15000 UMAX=0.05 VTK_EVERY=6000

set -u
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
BIN=$ROOT/build/aero_naca0012_cumulant
SRC=$ROOT/Model/obj_1_FA-18E_Final01.stl
ALIGNED=$ROOT/test_data/f18_3d_aligned.stl

STEPS=${1:-36000}
TAG=${2:-turb}
RES=${RES:-60}; LX=${LX:-8}; LY=${LY:-5}; LZ=${LZ:-3}; XNOSE=${XNOSE:-2}
RE=${RE:-15000}; UMAX=${UMAX:-0.05}; VTK_EVERY=${VTK_EVERY:-6000}
OUT=$ROOT/output_f18_${TAG}

[ -f "$BIN" ] || { echo "ERROR: $BIN not built. cmake --build build -j --target aero_naca0012_cumulant"; exit 1; }
[ -f "$SRC" ] || { echo "ERROR: missing source STL $SRC (git-ignored; copy it to the lab machine)"; exit 1; }

if [ ! -f "$ALIGNED" ] || [ "$SRC" -nt "$ALIGNED" ]; then
    echo "[$(date +%H:%M:%S)] Axis-aligning STL (PCA)..."
    python3 $ROOT/scripts/aero/align_stl_axes.py "$SRC" "$ALIGNED"
fi

PLACE=$(python3 $ROOT/scripts/aero/place_f18.py "$ALIGNED" \
            --res $RES --lx $LX --ly $LY --lz $LZ --x-nose $XNOSE)
echo "[$(date +%H:%M:%S)] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""
echo "[$(date +%H:%M:%S)] Launching F-18 TURBULENT: Re=$RE res=$RES box=${LX}x${LY}x${LZ}c steps=$STEPS -> $OUT"
echo "  (dumps snap_*.vtk + rho3d_*.vtk every $VTK_EVERY steps; ASCII VTK is large)"

rm -rf $OUT && mkdir -p $OUT

$BIN \
    --re $RE --u-max-lu $UMAX --alpha 0 \
    --steps $STEPS --probe-every 50 --vtk-every $VTK_EVERY \
    --bc stair --bc-z periodic --shape stl --stl-file "$ALIGNED" \
    $PLACE --output-dir $OUT \
    | tee $OUT/run.log

echo ""
echo "[$(date +%H:%M:%S)] Done. Outputs in $OUT"
if [ -f "$OUT/forces.csv" ]; then
    echo "  Cd time series (did it go UNSTEADY? a non-flat Cd means we escaped the steady regime):"
    awk -F, 'NR>1{print $1, $8}' $OUT/forces.csv | tail -8 | awk '{printf "    step=%-6s Cd=%+.4f\n",$1,$2}'
    echo "  Cd min/max over last half (oscillation amplitude):"
    awk -F, 'NR>1{n++; cd[n]=$8} END{s=int(n/2); mn=cd[s]; mx=cd[s]; for(i=s;i<=n;i++){if(cd[i]<mn)mn=cd[i]; if(cd[i]>mx)mx=cd[i]} printf "    Cd in [%+.4f, %+.4f]  amp=%.4f\n", mn,mx, mx-mn}' $OUT/forces.csv
fi
echo ""
echo "Render the hero figure (copy $OUT back to a box with PyVista, or run here):"
echo "  export PYVISTA_OFF_SCREEN=true"
LAST=$(printf "snap_%07d.vtk" $STEPS)
echo "  python3 scripts/aero/plot_q_isosurface.py $OUT/$LAST $OUT/mask_zmid.txt images/f18_q_turb.png 1.5"
echo "  python3 scripts/aero/plot_f18_3d.py $OUT          # 2D wake slices"
