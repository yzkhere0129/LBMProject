#!/bin/bash
# Diagnostic chain (replaces broken overnight_chain.sh after bash-buffer issue).
# 4 high-priority variants to test what's left after omega_b/QMIN/xLE all null.
#
#   N1: Re=2000 + omega_3..6 = 0.5    (Cumulant higher-cumulant hypothesis)
#   N2: Re=2000 + bc=stair             (test if QBB even helps lift)
#   N3: Re=1000 + omega_3..6 = 0.5    (cross-Re for N1)
#   N4: Re=1000 + bc=stair             (cross-Re for N2)
# Each ~45 min. Total ~3 hours.

set -u
ROOT=/home/yzk/CompressibleCFD
BIN=$ROOT/build/aero_naca0012_cumulant
COMMON="--resolution 80 --u-max-lu 0.05 --steps 30000 --probe-every 100"
LOG=$ROOT/diagnostic_chain.log
echo "Diagnostic chain start $(date)" > $LOG

run_variant() {
    local tag=$1 ; shift
    local outdir=$ROOT/output_diag_${tag}
    local extra=("$@")
    rm -rf $outdir && mkdir -p $outdir
    echo "[$(date +%H:%M:%S)] ▶ ${tag}: ${extra[*]}" | tee -a $LOG
    $BIN $COMMON "${extra[@]}" --output-dir $outdir \
        > $outdir/run.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ ${tag} FAILED rc=$rc" | tee -a $LOG
        tail -5 $outdir/run.log | tee -a $LOG
        return $rc
    fi
    awk -F, 'NR>1{c[++n]=$8; l[n]=$9} END{
        s=int(n*2/3); cd=0; cl=0; cl2=0
        for(i=s;i<=n;i++){cd+=c[i]; cl+=l[i]; cl2+=l[i]*l[i]}
        m=n-s+1
        printf "  Cd=%.4f  Cl=%.4f  Cl_rms=%.4f  (last %d samples)\n",
               cd/m, cl/m, sqrt(cl2/m-(cl/m)*(cl/m)), m
    }' $outdir/forces.csv | tee -a $LOG
    grep -A 8 "Per-stage profiling" $outdir/run.log | tee -a $LOG
    echo "[$(date +%H:%M:%S)] ✓ ${tag}" | tee -a $LOG
}

# N1: omega_3..6 = 0.5 at Re=2000  (preserve more higher-order info → less damping)
run_variant N1_re2000_omegahigh05 \
    --re 2000 --alpha 8 --sparse-qfrac --bc qbb-snode \
    --omega-3 0.5 --omega-4 0.5 --omega-5 0.5 --omega-6 0.5

# N2: bc=stair at Re=2000 (no QBB) — does QBB even help lift?
run_variant N2_re2000_stair \
    --re 2000 --alpha 8 --bc stair

# N3: omega_3..6 = 0.5 at Re=1000 (cross-Re check)
run_variant N3_re1000_omegahigh05 \
    --re 1000 --alpha 8 --sparse-qfrac --bc qbb-snode \
    --omega-3 0.5 --omega-4 0.5 --omega-5 0.5 --omega-6 0.5

# N4: bc=stair at Re=1000 (cross-Re check)
run_variant N4_re1000_stair \
    --re 1000 --alpha 8 --bc stair

echo "Diagnostic chain complete $(date)" | tee -a $LOG
