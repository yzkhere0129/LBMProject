#!/bin/bash
# Manual Dawn-3 launch — runs AFTER Phase 4 completes and morning report exists.
set -e
cd /home/yzk/LBMProject

LOG=night_run_log.md

if [ ! -f Morning_Briefing_Report.md ]; then
    echo "ERROR: morning report not yet generated" >&2
    exit 1
fi

# Check if morning report has Phase 4 data (not just baseline)
if ! grep -q output_phase4 Morning_Briefing_Report.md 2>/dev/null; then
    echo "WARN: morning report doesn't reference output_phase4 — may be stale"
fi

# Build Dawn-3 variant
if [ ! -f apps/sim_linescan_dawn3.cu ]; then
    cp apps/sim_linescan_phase1.cu apps/sim_linescan_dawn3.cu
    sed -i 's|config.material = MaterialDatabase::get316L();|config.material = MaterialDatabase::get316L();\n    config.material.dsigma_dT = -4.9e-4f; // Dawn-3: stress test old "Iron" Marangoni|' apps/sim_linescan_dawn3.cu
    sed -i 's/output_phase1/output_dawn3/g' apps/sim_linescan_dawn3.cu
    sed -i 's/const float t_total  = 800.0e-6f/const float t_total  = 400.0e-6f/' apps/sim_linescan_dawn3.cu
fi

# Add CMake target if missing
if ! grep -q sim_linescan_dawn3 CMakeLists.txt; then
    cat >> CMakeLists.txt <<EOF

    # Dawn-3: dσ/dT sensitivity (-4.9e-4 vs -4.3e-4)
    add_executable(sim_linescan_dawn3 apps/sim_linescan_dawn3.cu)
    set_target_properties(sim_linescan_dawn3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(sim_linescan_dawn3 lbm_physics lbm_io CUDA::cudart)
EOF
fi

cmake -B build -S . > /tmp/cmake_dawn3.log 2>&1
cmake --build build --target sim_linescan_dawn3 -j 4 > /tmp/build_dawn3.log 2>&1

if [ ! -x build/sim_linescan_dawn3 ]; then
    echo "ERROR: dawn3 binary not built. Check /tmp/build_dawn3.log" >&2
    exit 1
fi

# Run
rm -rf output_dawn3
mkdir -p output_dawn3
nohup stdbuf -oL ./build/sim_linescan_dawn3 > output_dawn3/run.log 2>&1 &
DAWN_PID=$!
disown $DAWN_PID
echo "Dawn-3 PID=$DAWN_PID" | tee -a "$LOG"

# Wait for sim final
DAWN_FINAL=output_dawn3/line_scan_005000.vtk
while [ ! -f "$DAWN_FINAL" ]; do
    sleep 60
    if ! pgrep -f sim_linescan_dawn3 > /dev/null; then
        if [ ! -f "$DAWN_FINAL" ]; then
            echo "Dawn-3 died" | tee -a "$LOG"
            break
        fi
    fi
done

sleep 10

# Compare
if [ -f "$DAWN_FINAL" ]; then
    {
        echo ''
        echo '## Dawn-3 result'
        echo '```'
        python3 scripts/flow3d/phase1_summary.py "$DAWN_FINAL"
        echo '```'
        echo ''
        echo '**Comparison Phase 1 (-4.3e-4) vs Dawn-3 (-4.9e-4) at t=400μs:**'
        python3 -c "
import subprocess, re
out_p1 = subprocess.check_output(['python3', 'scripts/flow3d/phase1_summary.py', 'output_phase1/line_scan_005000.vtk']).decode()
out_d3 = subprocess.check_output(['python3', 'scripts/flow3d/phase1_summary.py', '$DAWN_FINAL']).decode()
def get(t, p): m = re.search(p, t); return m.group(1) if m else 'NA'
p_r = get(out_p1, r'side-ridge peak Δh\s+([+\-\d.]+) um')
d_r = get(out_d3, r'side-ridge peak Δh\s+([+\-\d.]+) um')
p_v = get(out_p1, r'v_z @ -150 um \(m/s\)\s+([+\-\d.]+)')
d_v = get(out_d3, r'v_z @ -150 um \(m/s\)\s+([+\-\d.]+)')
p_h = get(out_p1, r'centerline Δh \(95%ile\)\s+([+\-\d.]+) um')
d_h = get(out_d3, r'centerline Δh \(95%ile\)\s+([+\-\d.]+) um')
print(f'  side-ridge peak Δh:   Phase 1 (-4.3e-4) = {p_r} μm   Dawn-3 (-4.9e-4) = {d_r} μm')
print(f'  v_z @ -150 μm:        Phase 1 (-4.3e-4) = {p_v}      Dawn-3 (-4.9e-4) = {d_v}')
print(f'  centerline Δh 95%ile: Phase 1 (-4.3e-4) = {p_h} μm   Dawn-3 (-4.9e-4) = {d_h} μm')
"
    } >> "$LOG"
fi

echo "" >> "$LOG"
echo "## DAWN-3 DONE  ($(date '+%H:%M:%S'))" >> "$LOG"
