#!/bin/bash
# Overnight chain: 5 NACA runs with omega_b=1.0 + profiling.
# Variants:
#   O1: Re=2000 baseline patched              (anchor — vs original 0.334)
#   O2: Re=1000 baseline patched              (vs original 0.311)
#   O3: Re=2000 + xLE=30 (test inlet distance)
#   O4: Re=500  baseline patched              (vs original 0.366)
#   O5: Re=200  baseline patched              (vs original 0.400)
# Each ~50 min. Total ~4 hours.
# Stop early with: pkill -f aero_naca0012_cumulant; echo "$$" > .killed

set -u
ROOT=/home/yzk/CompressibleCFD
BIN=$ROOT/build/aero_naca0012_cumulant
COMMON="--resolution 80 --u-max-lu 0.05 --steps 30000 --probe-every 100 --sparse-qfrac --bc qbb-snode"
LOG=$ROOT/overnight_chain.log
echo "Chain start $(date)" > $LOG

run_variant() {
    local tag=$1 ; shift
    local outdir=$ROOT/output_overnight_${tag}
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
    # Settled stats
    awk -F, 'NR>1{c[++n]=$8; l[n]=$9} END{
        s=int(n*2/3); cd=0; cl=0; cl2=0
        for(i=s;i<=n;i++){cd+=c[i]; cl+=l[i]; cl2+=l[i]*l[i]}
        m=n-s+1
        printf "  Cd=%.4f  Cl=%.4f  Cl_rms=%.4f  (last %d samples)\n",
               cd/m, cl/m, sqrt(cl2/m-(cl/m)*(cl/m)), m
    }' $outdir/forces.csv | tee -a $LOG
    # Profiling info
    grep -A 8 "Per-stage profiling" $outdir/run.log | tee -a $LOG
    echo "[$(date +%H:%M:%S)] ✓ ${tag}" | tee -a $LOG
}

# O1: Re=2000 patched baseline (PRIMARY anchor)
run_variant O1_re2000_patched --re 2000 --alpha 8

# O2: Re=1000 patched baseline
run_variant O2_re1000_patched --re 1000 --alpha 8

# O3: Re=2000 + xLE=30 (inlet distance test)
run_variant O3_re2000_xle30 --re 2000 --alpha 8 --xle-over-c 30

# O4: Re=2000 + omega_3..6 = 0.5 (preserve higher cumulants → less damping)
# Hypothesis: ω_3..6 = 1.0 instant-relax kills vorticity transport info;
# 0.5 keeps memory of higher moments → less effective numerical viscosity.
run_variant O4_re2000_omegahigh05 --re 2000 --alpha 8 \
    --omega-3 0.5 --omega-4 0.5 --omega-5 0.5 --omega-6 0.5

# O5: Re=2000 + bc=stair (NO QBB) — diagnostic of QBB's net effect.
# If stair gives similar/higher Cl, QBB isn't helping the lift mechanism.
run_variant O5_re2000_stair --re 2000 --alpha 8 --bc stair

# O6: Re=2000 + Ly=40c (test if y-domain is over-constraining lift)
# At Cl=0.65 the bound circulation creates wake that extends >5c. Ly=20
# may force this back through periodic-mirror constraints.
run_variant O6_re2000_ly40 --re 2000 --alpha 8 --ly-over-c 40

echo "Chain complete $(date)" | tee -a $LOG
