#!/bin/bash
# Overnight chain: 30k settled mean comparison AMR-OFF vs AMR-ON.
# Per task brief and self-audit 2026-05-19: real Cl improvement
# evidence requires settled mean run (not 1000-step transient).

set -u
ROOT=/home/yzk/CompressibleCFD
BIN=$ROOT/build/aero_naca0012_cumulant
LOG=$ROOT/overnight_30k_amr.log
echo "30k AMR chain start $(date)" > $LOG

COMMON="--resolution 80 --re 2000 --alpha 8 --u-max-lu 0.05 \
        --steps 30000 --probe-every 100 --vtk-every 10000 \
        --sparse-qfrac --bc qbb-snode"

run_variant() {
    local tag=$1 ; shift
    local outdir=$ROOT/output_30k_${tag}
    rm -rf $outdir && mkdir -p $outdir
    echo "[$(date +%H:%M:%S)] ▶ ${tag}: $*" | tee -a $LOG
    $BIN $COMMON "$@" --output-dir $outdir \
        > $outdir/run.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ ${tag} FAILED rc=$rc" | tee -a $LOG
        tail -10 $outdir/run.log | tee -a $LOG
        return $rc
    fi
    # Settled stats (last 1/3 of samples)
    awk -F, 'NR==2{m0=$10} NR>1{
        c[++n]=$8; l[n]=$9; m[n]=$10; mn[n]=$11; ux[n]=$13
    } END{
        s=int(n*2/3); cd=0; cl=0; cl2=0
        for(i=s;i<=n;i++){cd+=c[i]; cl+=l[i]; cl2+=l[i]*l[i]}
        N=n-s+1
        cd_m=cd/N; cl_m=cl/N
        printf "  Settled (last %d): Cd=%.4f Cl=%.4f Cl_rms=%.4f\n",
               N, cd_m, cl_m, sqrt(cl2/N - cl_m*cl_m)
        printf "  Mass: M0=%.4e M_end=%.4e drift=%+.3e\n",
               m[1], m[n], (m[n]-m[1])/m[1]
        printf "  u_x: mean=%.6f max_dev=%.3e (last step)\n",
               mn[n], ux[n]
    }' $outdir/forces.csv | tee -a $LOG
    grep -A 8 "Per-stage profiling" $outdir/run.log | head -10 | tee -a $LOG
    echo "[$(date +%H:%M:%S)] ✓ ${tag}" | tee -a $LOG
}

# Baseline AMR-OFF (~80-100 min)
run_variant amr_off

# AMR-ON bilinear + time interp (~150-180 min)
run_variant amr_on_bilin_time --amr-enable

echo "30k AMR chain complete $(date)" | tee -a $LOG
