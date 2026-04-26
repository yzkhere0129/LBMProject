#!/bin/bash
# Wait for Phase 3 to finish, analyze, launch Phase 4, wait, generate report.
set -e
cd /home/yzk/LBMProject

LOG=night_run_log.md
P3_FINAL=output_phase3/line_scan_005000.vtk
P4_FINAL=output_phase4/line_scan_025000.vtk

# Wait for Phase 3 final VTK or process death
while [ ! -f "$P3_FINAL" ]; do
    sleep 30
    if ! pgrep -f sim_linescan_phase3 > /dev/null; then
        if [ ! -f "$P3_FINAL" ]; then
            echo "Phase 3 died early — likely NaN at low τ. Falling back to Phase 1 config." | tee -a "$LOG"
            break
        fi
    fi
done

sleep 15

# Analyze Phase 3 if it produced a final
P3_DEC="FAIL"
if [ -f "$P3_FINAL" ]; then
    echo "" >> "$LOG"
    echo "## Phase-3 final (400 μs short test, $(date '+%H:%M:%S'))" >> "$LOG"
    echo '' >> "$LOG"
    echo '```' >> "$LOG"
    python3 scripts/flow3d/phase1_summary.py "$P3_FINAL" >> "$LOG" 2>&1
    echo '```' >> "$LOG"

    P3_DEC_OUT=$(python3 scripts/flow3d/phase1_decide.py "$P3_FINAL" 2>&1)
    echo '' >> "$LOG"
    echo '**Phase-3 decision:**' >> "$LOG"
    echo '```' >> "$LOG"
    echo "$P3_DEC_OUT" >> "$LOG"
    echo '```' >> "$LOG"
    P3_DEC=$(echo "$P3_DEC_OUT" | grep -oP "DECISION: \w+" | head -1 | awk '{print $2}')
fi

# Decide Phase 4 config
if [ "$P3_DEC" != "FAIL" ] && [ -f "$P3_FINAL" ]; then
    echo "" >> "$LOG"
    echo "Phase 3 didn't NaN crash → use τ=0.55 in Phase 4" >> "$LOG"
    sed -i 's/kinematic_viscosity      = 0.065f/kinematic_viscosity      = 0.0167f/' apps/sim_linescan_phase4.cu
    cmake --build build --target sim_linescan_phase4 -j 4 > /tmp/build_p4.log 2>&1
    echo "Phase 4 rebuilt with τ=0.55" >> "$LOG"
else
    echo "" >> "$LOG"
    echo "Phase 3 didn't help — Phase 4 uses Phase 1 config (τ=0.7, no mass corr)" >> "$LOG"
fi

# Launch Phase 4
echo "" >> "$LOG"
echo "## Phase-4 launching ($(date '+%H:%M:%S'))" >> "$LOG"
rm -rf output_phase4
mkdir -p output_phase4
nohup stdbuf -oL ./build/sim_linescan_phase4 > output_phase4/run.log 2>&1 &
P4_PID=$!
disown $P4_PID
echo "Phase 4 PID=$P4_PID" >> "$LOG"

# Wait for Phase 4 final
while [ ! -f "$P4_FINAL" ]; do
    sleep 60
    if ! pgrep -f sim_linescan_phase4 > /dev/null; then
        if [ ! -f "$P4_FINAL" ]; then
            echo "Phase 4 died before completion." >> "$LOG"
            break
        fi
    fi
done

sleep 15

# Analyze Phase 4
if [ -f "$P4_FINAL" ]; then
    echo "" >> "$LOG"
    echo "## Phase-4 final ($(date '+%H:%M:%S'))" >> "$LOG"
    echo '```' >> "$LOG"
    python3 scripts/flow3d/phase1_summary.py "$P4_FINAL" >> "$LOG" 2>&1
    echo '```' >> "$LOG"
    echo '**Mass conservation:**' >> "$LOG"
    echo '```' >> "$LOG"
    python3 scripts/flow3d/check_mass_conservation.py output_phase4/line_scan_000000.vtk "$P4_FINAL" >> "$LOG" 2>&1
    echo '```' >> "$LOG"
fi

# Generate morning report
bash scripts/flow3d/finalize_morning_report.sh
echo "" >> "$LOG"
echo "## DONE — morning report generated  ($(date '+%H:%M:%S'))" >> "$LOG"
