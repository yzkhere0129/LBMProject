#!/bin/bash
# Wait for Phase 2 to finish, analyze, launch Phase 4, wait, generate report.
set -e
cd /home/yzk/LBMProject

LOG=night_run_log.md

P2_FINAL=output_phase2/line_scan_010000.vtk
P4_FINAL=output_phase4/line_scan_025000.vtk

# Wait for Phase 2 final VTK
while [ ! -f "$P2_FINAL" ]; do
    sleep 30
    if ! pgrep -f sim_linescan_phase2 > /dev/null; then
        # sim died
        if [ ! -f "$P2_FINAL" ]; then
            echo "Phase 2 process died before completing — falling back to Phase 1 → Phase 4" | tee -a "$LOG"
            break
        fi
    fi
done

# Wait a bit more for sim to fully exit
sleep 15

if [ -f "$P2_FINAL" ]; then
    echo "" >> "$LOG"
    echo "## Phase-2 final  ($(date '+%H:%M:%S'))" >> "$LOG"
    echo '' >> "$LOG"
    echo '```' >> "$LOG"
    python3 scripts/flow3d/phase1_summary.py "$P2_FINAL" >> "$LOG" 2>&1
    echo '```' >> "$LOG"
    echo '' >> "$LOG"
    echo '**Mass conservation:**' >> "$LOG"
    echo '```' >> "$LOG"
    python3 scripts/flow3d/check_mass_conservation.py output_phase2/line_scan_000000.vtk "$P2_FINAL" >> "$LOG" 2>&1
    echo '```' >> "$LOG"
fi

# Launch Phase 4
echo "" >> "$LOG"
echo "## Phase-4 launching ($(date '+%H:%M:%S'))" >> "$LOG"
echo '' >> "$LOG"

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
    echo "## Phase-4 final  ($(date '+%H:%M:%S'))" >> "$LOG"
    echo '' >> "$LOG"
    echo '```' >> "$LOG"
    python3 scripts/flow3d/phase1_summary.py "$P4_FINAL" >> "$LOG" 2>&1
    echo '```' >> "$LOG"
    echo '' >> "$LOG"
    echo '**Mass conservation:**' >> "$LOG"
    echo '```' >> "$LOG"
    python3 scripts/flow3d/check_mass_conservation.py output_phase4/line_scan_000000.vtk "$P4_FINAL" >> "$LOG" 2>&1
    echo '```' >> "$LOG"
fi

# Generate morning report
bash scripts/flow3d/finalize_morning_report.sh
echo "" >> "$LOG"
echo "## DONE — morning report generated  ($(date '+%H:%M:%S'))" >> "$LOG"
