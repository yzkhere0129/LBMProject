#!/bin/bash
# F/A-18E silhouette demo launcher.
# Regenerates silhouette STL if missing, then runs the LBM simulation.
# Args:
#   $1 = step count (default 5000)
#   $2 = tag (output dir suffix, default "5k")
#
# Run only when GPU is solo (other processes will compete and 3-5× slow this down).

set -u
ROOT=/home/yzk/CompressibleCFD
BIN=$ROOT/build/aero_naca0012_cumulant
SILH=$ROOT/test_data/f18_silhouette_xy.stl
SRC=$ROOT/Model/obj_1_FA-18E_Final01.stl

STEPS=${1:-5000}
TAG=${2:-5k}
OUT=$ROOT/output_f18_${TAG}

if [ ! -f "$SRC" ]; then
    echo "ERROR: missing $SRC — drop the F/A-18E STL there first."
    exit 1
fi

# Regenerate silhouette if missing or source newer
if [ ! -f "$SILH" ] || [ "$SRC" -nt "$SILH" ]; then
    echo "[$(date +%H:%M:%S)] Regenerating silhouette STL..."
    python3 $ROOT/scripts/aero/project_stl_to_silhouette.py \
        $SRC $SILH \
        --raster-cells 400 --projection xy --z-thickness 20.0
fi

echo "[$(date +%H:%M:%S)] Checking GPU contention..."
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
echo ""
echo "[$(date +%H:%M:%S)] Launching F/A-18E demo: steps=$STEPS, out=$OUT"

rm -rf $OUT && mkdir $OUT

$BIN \
    --resolution 80 --re 2000 --u-max-lu 0.05 --alpha 0 \
    --steps $STEPS --probe-every 50 --vtk-every $((STEPS/2)) \
    --bc stair --shape stl \
    --stl-file $SILH \
    --stl-scale 0.00625 --stl-tx 9.937 --stl-ty 9.521 --stl-tz -0.04 \
    --output-dir $OUT \
    | tee $OUT/run.log

echo ""
echo "[$(date +%H:%M:%S)] Done. Outputs in $OUT"
echo "  forces.csv: $(wc -l < $OUT/forces.csv) samples"
echo "  Final Cd/Cl:"
tail -3 $OUT/forces.csv | awk -F, '{printf "  step=%5s Cd=%+.4f Cl=%+.4f\n", $1, $8, $9}'
