#!/bin/bash
# F/A-18E full-3D external-aero demo launcher (lab GTX 1080 / SM 6.1).
#
# Pipeline: PCA-align source STL -> place in domain -> D3Q27 Cumulant LBM.
# The aircraft is axis-aligned (nose into the flow at -x, wings along y,
# fins up in z) by align_stl_axes.py; placement by place_f18.py.
#
# Args:  $1 = steps (default 5000)   $2 = output tag (default 3d_5k)
# Env knobs (override before calling):
#   RES=40  LX=12  LY=8  LZ=4  XNOSE=3  RE=2000  UMAX=0.05  ALPHA=0
#
# Primary config (8 GB 1080): RES=40 -> 24.7M cells, ~5.7 GB.
# 11 GB 1080 Ti: try RES=48 LX=12 LY=8 LZ=4 (~9.9 GB) for sharper wings.
# OOM / GPU contention: drop to RES=32 LY=10 (15.8M, ~3.7 GB).

set -u
ROOT=/home/yzk/CompressibleCFD
BIN=$ROOT/build/aero_naca0012_cumulant
SRC=$ROOT/Model/obj_1_FA-18E_Final01.stl
ALIGNED=$ROOT/test_data/f18_3d_aligned.stl

STEPS=${1:-5000}
TAG=${2:-3d_5k}
RES=${RES:-40}; LX=${LX:-12}; LY=${LY:-8}; LZ=${LZ:-4}; XNOSE=${XNOSE:-3}
RE=${RE:-2000}; UMAX=${UMAX:-0.05}; ALPHA=${ALPHA:-0}
OUT=$ROOT/output_f18_${TAG}

[ -f "$BIN" ] || { echo "ERROR: $BIN not built. Run: cmake --build build -j --target aero_naca0012_cumulant"; exit 1; }
[ -f "$SRC" ] || { echo "ERROR: missing source STL $SRC"; exit 1; }

# Axis-align the source STL if missing or source is newer.
if [ ! -f "$ALIGNED" ] || [ "$SRC" -nt "$ALIGNED" ]; then
    echo "[$(date +%H:%M:%S)] Axis-aligning STL (PCA)..."
    python3 $ROOT/scripts/aero/align_stl_axes.py "$SRC" "$ALIGNED"
fi

# Compute placement flags (scale + translate) and print summary to stderr.
PLACE=$(python3 $ROOT/scripts/aero/place_f18.py "$ALIGNED" \
            --res $RES --lx $LX --ly $LY --lz $LZ --x-nose $XNOSE)
echo "[$(date +%H:%M:%S)] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo ""
echo "[$(date +%H:%M:%S)] Launching F-18 3D: steps=$STEPS Re=$RE alpha=$ALPHA -> $OUT"

rm -rf $OUT && mkdir -p $OUT

$BIN \
    --re $RE --u-max-lu $UMAX --alpha $ALPHA \
    --steps $STEPS --probe-every 50 --vtk-every $((STEPS/2)) \
    --bc stair --bc-z periodic --shape stl --stl-file "$ALIGNED" \
    $PLACE \
    --output-dir $OUT \
    | tee $OUT/run.log

echo ""
echo "[$(date +%H:%M:%S)] Done. Outputs in $OUT"
if [ -f "$OUT/forces.csv" ]; then
    echo "  forces.csv: $(wc -l < $OUT/forces.csv) samples"
    echo "  Final Cd/Cl (last 3 samples):"
    tail -3 $OUT/forces.csv | awk -F, '{printf "    step=%6s Cd=%+.4f Cl=%+.4f\n", $1, $8, $9}'
fi
