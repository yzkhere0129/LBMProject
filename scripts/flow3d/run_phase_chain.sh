#!/bin/bash
# Master chain: after Phase 1 done, run Phase 2/3 then Phase 4.
# Logs to night_run_log.md and finalizes Morning_Briefing_Report.md.
set -e
cd /home/yzk/LBMProject

LOG=night_run_log.md
log_section() {
    echo ""                          | tee -a "$LOG"
    echo "## $1  ($(date '+%H:%M:%S'))" | tee -a "$LOG"
    echo ""                          | tee -a "$LOG"
}

log_block() {
    echo '```'                       >> "$LOG"
    cat "$1"                         >> "$LOG"
    echo '```'                       >> "$LOG"
    echo ''                          >> "$LOG"
}

wait_for_vtk() {
    local DIR=$1
    local STEP=$2
    local PID=$3
    until [ -f "$DIR/line_scan_$(printf %06d $STEP).vtk" ] || ! kill -0 "$PID" 2>/dev/null; do
        sleep 30
    done
}

run_sim() {
    local PHASE=$1
    local BIN="./build/$PHASE"
    local OUT="output_${PHASE#sim_linescan_}"
    local FINAL_STEP=$2

    log_section "Launching $PHASE"
    rm -rf "$OUT"
    mkdir -p "$OUT"
    nohup stdbuf -oL "$BIN" > "$OUT/run.log" 2>&1 &
    local PID=$!
    disown $PID
    echo "PID=$PID  output=$OUT  final_step=$FINAL_STEP" | tee -a "$LOG"

    wait_for_vtk "$OUT" "$FINAL_STEP" "$PID"

    if ! kill -0 $PID 2>/dev/null; then
        echo "PID $PID exited (NaN/crash?). Last 20 lines of run.log:" | tee -a "$LOG"
        tail -20 "$OUT/run.log" | tee -a "$LOG"
        return 1
    fi

    # Wait for sim to actually finish (last VTK plus shutdown)
    while kill -0 $PID 2>/dev/null; do sleep 5; done
    echo "$PHASE completed normally" | tee -a "$LOG"
    return 0
}

# === Step 1: Phase 1 final analysis & decision ===
log_section "Phase-1 final summary"
python3 scripts/flow3d/phase1_summary.py output_phase1/line_scan_010000.vtk \
    > /tmp/p1_final.txt 2>&1
log_block /tmp/p1_final.txt

python3 scripts/flow3d/phase1_decide.py output_phase1/line_scan_010000.vtk \
    > /tmp/p1_decision.txt 2>&1
log_section "Phase-1 decision"
log_block /tmp/p1_decision.txt
DECISION=$(grep -oP "DECISION: \w+" /tmp/p1_decision.txt | head -1 | awk '{print $2}')
echo "  → $DECISION" | tee -a "$LOG"

# Mass conservation
python3 scripts/flow3d/check_mass_conservation.py \
    output_phase1/line_scan_000000.vtk output_phase1/line_scan_010000.vtk \
    > /tmp/p1_mass.txt 2>&1
log_section "Phase-1 mass conservation"
log_block /tmp/p1_mass.txt

# === Step 2: branch based on decision ===
case "$DECISION" in
    SUCCESS)
        echo "✓ Phase 1 SUCCESS → Phase 4 (using Phase 1 config)" | tee -a "$LOG"
        run_sim sim_linescan_phase4 25000 || echo "Phase 4 FAILED" | tee -a "$LOG"
        ;;
    PARTIAL)
        echo "⚠ Phase 1 PARTIAL → Phase 2 (mass correction)" | tee -a "$LOG"
        if run_sim sim_linescan_phase2 10000; then
            python3 scripts/flow3d/phase1_summary.py output_phase2/line_scan_010000.vtk \
                > /tmp/p2_final.txt 2>&1
            log_section "Phase-2 final summary"
            log_block /tmp/p2_final.txt
            python3 scripts/flow3d/phase1_decide.py output_phase2/line_scan_010000.vtk \
                > /tmp/p2_decision.txt 2>&1
            log_section "Phase-2 decision"
            log_block /tmp/p2_decision.txt

            # Phase 4: rebuild with mass correction enabled
            sed -i 's/enable_vof_mass_correction  = false/enable_vof_mass_correction  = true/' \
                apps/sim_linescan_phase4.cu
            cmake --build build --target sim_linescan_phase4 -j 4 > /tmp/build_p4.log 2>&1
            run_sim sim_linescan_phase4 25000 || echo "Phase 4 FAILED" | tee -a "$LOG"
        fi
        ;;
    FAIL)
        echo "✗ Phase 1 FAIL → Phase 3 (τ stretch test)" | tee -a "$LOG"
        if run_sim sim_linescan_phase3 5000; then
            python3 scripts/flow3d/phase1_summary.py output_phase3/line_scan_005000.vtk \
                > /tmp/p3_final.txt 2>&1
            log_section "Phase-3 final summary (400 μs short test)"
            log_block /tmp/p3_final.txt
            python3 scripts/flow3d/phase1_decide.py output_phase3/line_scan_005000.vtk \
                > /tmp/p3_decision.txt 2>&1
            log_section "Phase-3 decision"
            log_block /tmp/p3_decision.txt
            P3_DEC=$(grep -oP "DECISION: \w+" /tmp/p3_decision.txt | head -1 | awk '{print $2}')
            if [ "$P3_DEC" != "FAIL" ]; then
                echo "Phase 3 improved → Phase 4 with τ=0.55" | tee -a "$LOG"
                sed -i 's/kinematic_viscosity      = 0.065f/kinematic_viscosity      = 0.0167f/' \
                    apps/sim_linescan_phase4.cu
            else
                echo "Phase 3 no help → Phase 4 with Phase 1 config" | tee -a "$LOG"
            fi
            cmake --build build --target sim_linescan_phase4 -j 4 > /tmp/build_p4.log 2>&1
            run_sim sim_linescan_phase4 25000 || echo "Phase 4 FAILED" | tee -a "$LOG"
        else
            echo "Phase 3 NaN → Phase 4 with Phase 1 config" | tee -a "$LOG"
            run_sim sim_linescan_phase4 25000 || echo "Phase 4 FAILED" | tee -a "$LOG"
        fi
        ;;
    *)
        echo "UNKNOWN DECISION → falling back to Phase 4 with Phase 1 config" | tee -a "$LOG"
        run_sim sim_linescan_phase4 25000 || echo "Phase 4 FAILED" | tee -a "$LOG"
        ;;
esac

# === Step 3: finalize morning report ===
log_section "Generating Morning_Briefing_Report.md"
bash scripts/flow3d/finalize_morning_report.sh
echo "DONE" | tee -a "$LOG"
