#!/bin/bash
# After Phase-1 completes, automatically run the appropriate next phase
# based on phase1_decide.py verdict.
#
# Usage: ./run_phase_dispatcher.sh
#   Reads output_phase1/line_scan_010000.vtk, decides, launches Phase 2/3/4.
set -e
cd /home/yzk/LBMProject

P1_FINAL=output_phase1/line_scan_010000.vtk
LOG=night_run_log.md

if [ ! -f "$P1_FINAL" ]; then
    echo "ERROR: Phase-1 final frame not found at $P1_FINAL"
    exit 1
fi

echo "=== Phase-1 final frame analysis ==="
python3 scripts/flow3d/phase1_summary.py "$P1_FINAL" | tee /tmp/phase1_final.txt
echo ""
echo "=== Decision ==="
DECISION_OUT=$(python3 scripts/flow3d/phase1_decide.py "$P1_FINAL" 2>&1 | tee /tmp/phase1_decision.txt)
echo "$DECISION_OUT"

# Mass conservation snapshot
echo ""
echo "=== Mass conservation t=0 vs t=800 μs ==="
python3 scripts/flow3d/check_mass_conservation.py \
    output_phase1/line_scan_000000.vtk "$P1_FINAL" \
    | tee /tmp/phase1_mass.txt

# Append to night log
{
    echo ""
    echo "## Phase-1 final analysis ($(date '+%H:%M:%S'))"
    echo ""
    echo '```'
    cat /tmp/phase1_final.txt
    echo '```'
    echo ""
    echo '**Decision:**'
    echo '```'
    cat /tmp/phase1_decision.txt
    echo '```'
    echo ""
    echo '**Mass conservation:**'
    echo '```'
    cat /tmp/phase1_mass.txt
    echo '```'
    echo ""
} >> "$LOG"

# Branch
if grep -q "DECISION: SUCCESS" /tmp/phase1_decision.txt; then
    NEXT="Phase 4 (final 2 ms)"
    BIN=./build/sim_linescan_phase4
    OUT=output_phase4
elif grep -q "DECISION: PARTIAL" /tmp/phase1_decision.txt; then
    NEXT="Phase 2 (mass correction)"
    BIN=./build/sim_linescan_phase2
    OUT=output_phase2
else
    NEXT="Phase 3 (τ stretch)"
    BIN=./build/sim_linescan_phase3
    OUT=output_phase3
fi

echo ""
echo "=== Launching $NEXT ==="
{
    echo ""
    echo "## Launching $NEXT  ($(date '+%H:%M:%S'))"
    echo ""
} >> "$LOG"

rm -rf "$OUT"
mkdir -p "$OUT"
echo "Output: $OUT/"
echo "Binary: $BIN"
nohup stdbuf -oL "$BIN" > "$OUT/run.log" 2>&1 &
PID=$!
echo "PID=$PID"
echo "PID=$PID" >> "$LOG"
disown $PID
sleep 2
ls -la "$OUT/" | head -5
