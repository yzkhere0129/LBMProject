#!/bin/bash
# Auto-decide next step after D/dx=80 + Ma=0.043 finishes.
#
# Reads err% from the strouhal_fft.py output. If err > 1%, queue
# D/dx=80 + Ma=0.025 in background. If err <= 1%, declare success.
#
# Usage: ./auto_next.sh

set -eu
cd /home/yzk/CompressibleCFD

OUTDIR_PRIMARY=output_aero_2d2_d80_lowma         # Ma=0.043
OUTDIR_NEXT=output_aero_2d2_d80_lowerma          # Ma=0.025 (if needed)

if [ ! -e "$OUTDIR_PRIMARY/forces.csv" ]; then
    echo "ERROR: $OUTDIR_PRIMARY/forces.csv missing — primary run not done"
    exit 2
fi

# Compute err% from strouhal output (extract Cd_max line, parse number)
CD_MAX=$(python3 scripts/aero/strouhal_fft.py "$OUTDIR_PRIMARY" \
    --U-avg 1.0 --transient-frac 0.5 2>&1 | \
    awk '/Cd_max/ { print $3; exit }')

if [ -z "$CD_MAX" ]; then
    echo "ERROR: failed to extract Cd_max"
    exit 3
fi

# DFG centre = 3.23
ERR=$(python3 -c "print(100.0 * abs($CD_MAX - 3.23) / 3.23)")
echo "Primary run Cd_max = $CD_MAX, err = $ERR%"

# Decision: <= 1% → done. Else queue Ma=0.025 next.
if (( $(python3 -c "print(1 if $ERR <= 1.0 else 0)") )); then
    echo "GATE PASS — within 1% of DFG. Phase 1c done."
    exit 0
fi

if [ -e "$OUTDIR_NEXT/forces.csv" ]; then
    echo "Next-step output dir already exists; skipping launch."
    exit 0
fi

echo "Queueing D/dx=80 + Ma=0.025 (~16 hr)..."
build/aero_schaefer_turek_3d --case 2d-2 --re 100 --resolution 80 \
    --bc qbb-quad --wall-bc halfway --u-max-lu 0.0125 \
    --steps 480000 --probe-every 400 \
    --output-dir "$OUTDIR_NEXT" \
    > "$OUTDIR_NEXT.log" 2>&1 &
echo "Launched as PID $!"
