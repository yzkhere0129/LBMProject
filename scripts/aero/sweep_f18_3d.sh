#!/bin/bash
# F/A-18E 3D angle-of-attack sweep.
#
# --alpha in the solver only rotates the (NACA) stamp geometry, NOT the
# inflow, so it is a no-op for an STL body. We instead PITCH the aircraft
# nose-up by alpha about the span axis (align_stl_axes.py --pitch) — flow
# stays +x — which is the physically correct way to set AoA here.
#
# Lift is Fz_LU (col 5) and drag is Fx_LU (col 3): our vertical axis is z,
# so the solver's "Cl" column (Fy-based) is the SIDE force here, not lift.
# We report Fx, Fz and L/D = Fz/Fx, averaged over the last third of samples.
#
# Args:  $1 = steps per alpha (default 5000)   $2 = tag (default sweep)
# Env:   ALPHAS="0 4 8 12"  RES=40 LX=12 LY=8 LZ=4 XNOSE=3 RE=2000 UMAX=0.05

set -u
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
BIN=$ROOT/build/aero_naca0012_cumulant
SRC=$ROOT/Model/obj_1_FA-18E_Final01.stl

STEPS=${1:-5000}
TAG=${2:-sweep}
ALPHAS=${ALPHAS:-"0 4 8 12"}
RES=${RES:-40}; LX=${LX:-12}; LY=${LY:-8}; LZ=${LZ:-4}; XNOSE=${XNOSE:-3}
RE=${RE:-2000}; UMAX=${UMAX:-0.05}
SWEEP_DIR=$ROOT/output_f18_${TAG}
SUMMARY=$SWEEP_DIR/sweep_summary.csv

[ -f "$BIN" ] || { echo "ERROR: $BIN not built. cmake --build build -j --target aero_naca0012_cumulant"; exit 1; }
[ -f "$SRC" ] || { echo "ERROR: missing source STL $SRC (copy it to the lab machine — it is git-ignored)"; exit 1; }

mkdir -p $SWEEP_DIR
echo "alpha_deg,Fx_drag_LU,Fz_lift_LU,L_over_D" > $SUMMARY

for A in $ALPHAS; do
    STL=$ROOT/test_data/f18_3d_a${A}.stl
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] alpha=${A}deg  (pitch nose-up, flow +x)"
    echo "============================================================"
    MLEN=$(python3 $ROOT/scripts/aero/align_stl_axes.py "$SRC" "$STL" --pitch $A \
             | sed -n 's/^model_len=//p')
    PLACE=$(python3 $ROOT/scripts/aero/place_f18.py "$STL" \
              --res $RES --lx $LX --ly $LY --lz $LZ --x-nose $XNOSE --model-len $MLEN)
    OUT=$SWEEP_DIR/a${A}
    rm -rf $OUT && mkdir -p $OUT
    $BIN \
        --re $RE --u-max-lu $UMAX --alpha 0 \
        --steps $STEPS --probe-every 50 --vtk-every $STEPS \
        --bc stair --bc-z periodic --shape stl --stl-file "$STL" \
        $PLACE --output-dir $OUT \
        > $OUT/run.log 2>&1

    # Mean drag (col3) / lift (col5) over the last third of force samples.
    read FX FZ LD < <(awk -F, 'NR>1{n++; fx[n]=$3; fz[n]=$5}
        END{ if(n<3){print 0,0,0; exit}
             s=int(2*n/3)+1; sx=0; sz=0; c=0;
             for(i=s;i<=n;i++){sx+=fx[i]; sz+=fz[i]; c++}
             mx=sx/c; mz=sz/c; printf "%.5f %.5f %.4f", mx, mz, (mx!=0?mz/mx:0) }' $OUT/forces.csv)
    echo "${A},${FX},${FZ},${LD}" >> $SUMMARY
    echo "[$(date +%H:%M:%S)] alpha=${A}: drag Fx=${FX}  lift Fz=${FZ}  L/D=${LD}"
done

echo ""; echo "=== sweep summary ($SUMMARY) ==="; column -t -s, $SUMMARY

# Plot drag, lift, L/D vs alpha.
python3 - "$SUMMARY" "$SWEEP_DIR/sweep_plot.png" <<'PY'
import sys, csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=list(csv.DictReader(open(sys.argv[1])))
a=[float(r['alpha_deg']) for r in rows]
fx=[float(r['Fx_drag_LU']) for r in rows]
fz=[float(r['Fz_lift_LU']) for r in rows]
ld=[float(r['L_over_D']) for r in rows]
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].plot(a,fx,'o-',label='drag Fx'); ax[0].plot(a,fz,'s-',label='lift Fz')
ax[0].set_xlabel('alpha (deg)'); ax[0].set_ylabel('force (LU)'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title('F-18 3D: drag & lift vs AoA')
ax[1].plot(a,ld,'^-',color='green'); ax[1].set_xlabel('alpha (deg)'); ax[1].set_ylabel('L/D = Fz/Fx')
ax[1].grid(alpha=.3); ax[1].set_title('lift-to-drag ratio')
plt.tight_layout(); plt.savefig(sys.argv[2],dpi=100); print('plot ->',sys.argv[2])
PY
echo "[$(date +%H:%M:%S)] sweep done. Per-alpha runs + run.log in $SWEEP_DIR/a*/"
