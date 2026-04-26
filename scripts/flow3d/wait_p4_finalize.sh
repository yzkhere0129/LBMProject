#!/bin/bash
# Wait for Phase 4 (now alone on GPU) to finish, then finalize report.
set -e
cd /home/yzk/LBMProject

LOG=night_run_log.md
P4_FINAL=output_phase4/line_scan_025000.vtk

while [ ! -f "$P4_FINAL" ]; do
    sleep 60
    if ! pgrep -f sim_linescan_phase4 > /dev/null; then
        if [ ! -f "$P4_FINAL" ]; then
            echo "Phase 4 died before completion" >> "$LOG"
            break
        fi
    fi
done

sleep 15

if [ -f "$P4_FINAL" ]; then
    {
        echo ""
        echo "## Phase-4 final  ($(date '+%H:%M:%S'))"
        echo ''
        echo '```'
        python3 scripts/flow3d/phase1_summary.py "$P4_FINAL"
        echo '```'
        echo ''
        echo '**Mass conservation:**'
        echo '```'
        python3 scripts/flow3d/check_mass_conservation.py output_phase4/line_scan_000000.vtk "$P4_FINAL"
        echo '```'
    } 2>&1 >> "$LOG"
fi

bash scripts/flow3d/finalize_morning_report.sh
echo "" >> "$LOG"
echo "## DONE — morning report generated  ($(date '+%H:%M:%S'))" >> "$LOG"

# === Dawn protocol (only if main finished) ===
if [ -f Morning_Briefing_Report.md ] && [ -f "$P4_FINAL" ]; then
    echo "" >> "$LOG"
    echo "## Dawn-3: dσ/dT sensitivity test ($(date '+%H:%M:%S'))" >> "$LOG"
    
    # Build Dawn-3 variant if not exists
    if [ ! -f apps/sim_linescan_dawn3.cu ]; then
        cp apps/sim_linescan_phase1.cu apps/sim_linescan_dawn3.cu
        # Override dσ/dT to old "Iron" value -4.9e-4 (vs F3D 316L -4.3e-4)
        sed -i 's|config.material = MaterialDatabase::get316L();|config.material = MaterialDatabase::get316L();\n    config.material.dsigma_dT = -4.9e-4f;  // Dawn-3: stress test old "Iron" steeper Marangoni|' apps/sim_linescan_dawn3.cu
        sed -i 's/output_phase1/output_dawn3/g' apps/sim_linescan_dawn3.cu
        # Shorten to 400 μs for sensitivity
        sed -i 's/const float t_total  = 800.0e-6f/const float t_total  = 400.0e-6f/' apps/sim_linescan_dawn3.cu
    fi
    
    # Add CMake target if missing
    if ! grep -q sim_linescan_dawn3 CMakeLists.txt; then
        sed -i '/Phase-4 final 2 ms validation/i\
    # Dawn-3: dσ/dT sensitivity\
    add_executable(sim_linescan_dawn3 apps/sim_linescan_dawn3.cu)\
    set_target_properties(sim_linescan_dawn3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)\
    target_link_libraries(sim_linescan_dawn3 lbm_physics lbm_io CUDA::cudart)\
' CMakeLists.txt
        cmake -B build -S . > /tmp/cmake_dawn3.log 2>&1
    fi
    
    cmake --build build --target sim_linescan_dawn3 -j 4 > /tmp/build_dawn3.log 2>&1
    
    # Run
    rm -rf output_dawn3
    mkdir -p output_dawn3
    nohup stdbuf -oL ./build/sim_linescan_dawn3 > output_dawn3/run.log 2>&1 &
    DAWN_PID=$!
    disown $DAWN_PID
    echo "Dawn-3 PID=$DAWN_PID" >> "$LOG"
    
    # Wait
    DAWN_FINAL=output_dawn3/line_scan_005000.vtk
    while [ ! -f "$DAWN_FINAL" ]; do
        sleep 60
        if ! pgrep -f sim_linescan_dawn3 > /dev/null; then
            if [ ! -f "$DAWN_FINAL" ]; then
                echo "Dawn-3 died" >> "$LOG"
                break
            fi
        fi
    done
    
    sleep 10
    
    if [ -f "$DAWN_FINAL" ]; then
        {
            echo ''
            echo '**Dawn-3 final summary:**'
            echo '```'
            python3 scripts/flow3d/phase1_summary.py "$DAWN_FINAL"
            echo '```'
            echo ''
            echo '**Side-ridge comparison (Phase 1 -4.3e-4 vs Dawn-3 -4.9e-4):**'
            python3 -c "
import subprocess, re
out_p1 = subprocess.check_output(['python3', 'scripts/flow3d/phase1_summary.py', 'output_phase1/line_scan_005000.vtk']).decode()
out_d3 = subprocess.check_output(['python3', 'scripts/flow3d/phase1_summary.py', '$DAWN_FINAL']).decode()
def get(t, p): m = re.search(p, t); return m.group(1) if m else 'NA'
p_r = get(out_p1, r'side-ridge peak Δh\s+([+\-\d.]+) um')
d_r = get(out_d3, r'side-ridge peak Δh\s+([+\-\d.]+) um')
p_v = get(out_p1, r'v_z @ -150 um \(m/s\)\s+([+\-\d.]+)')
d_v = get(out_d3, r'v_z @ -150 um \(m/s\)\s+([+\-\d.]+)')
print(f'  side-ridge peak Δh:   Phase 1 (-4.3e-4) = {p_r} um   |   Dawn-3 (-4.9e-4) = {d_r} um')
print(f'  v_z @ -150:           Phase 1 (-4.3e-4) = {p_v} m/s |   Dawn-3 (-4.9e-4) = {d_v} m/s')
"
        } 2>&1 >> "$LOG"
    fi
fi
echo "" >> "$LOG"
echo "## ALL DONE  ($(date '+%H:%M:%S'))" >> "$LOG"
