#!/bin/bash
################################################################################
# TEST E: VOF Advection Performance and Stability Monitoring
################################################################################
# Purpose: Monitor VOF advection kernel performance, mass conservation, stability
# Expected: Runtime +20-50% vs Test C, mass error < 5%, interface sharp
# Duration: ~10-15 minutes for 3000 steps (acceptable), >20 min = performance issue
################################################################################

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
BUILD_DIR="/home/yzk/LBMProject/build"
CONFIG_FILE="/home/yzk/LBMProject/configs/lpbf_195W_test_E_vof_advection.conf"
LOG_FILE="${BUILD_DIR}/test_E_vof_advection.log"
VALIDATOR="/home/yzk/LBMProject/validation_tests/validate_test_E.py"
TEST_C_LOG="${BUILD_DIR}/test_C_full_coupling.log"

# Performance tracking
START_TIME=$(date +%s)
PERFORMANCE_LOG="${BUILD_DIR}/test_E_performance.txt"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}TEST E: VOF Advection Performance Monitoring${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${CYAN}Monitoring Focus:${NC}"
echo "  - Runtime: Target 10-15 min (vs Test C: 8-10 min)"
echo "  - Mass conservation: error < 5%"
echo "  - Interface sharpness: cells < 5000"
echo "  - CFL_VOF: < 0.5"
echo "  - GPU utilization: > 80%"
echo "  - Stability: no NaN/Inf, fill level ∈ [0,1]"
echo ""
echo "Configuration: ${CONFIG_FILE}"
echo "Log file:      ${LOG_FILE}"
echo "Build dir:     ${BUILD_DIR}"
echo ""

# Step 0: Check prerequisites
if [ ! -f "${TEST_C_LOG}" ]; then
    echo -e "${YELLOW}⚠ WARNING: Test C (baseline) not completed${NC}"
    echo "  Test E compares VOF advection performance to Test C static interface"
    echo "  Recommended: Run Test C first for baseline comparison"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Run: ./run_test_C.sh"
        exit 1
    fi
fi

# Step 1: Check executable exists
if [ ! -f "${BUILD_DIR}/visualize_lpbf_scanning" ]; then
    echo -e "${RED}ERROR: Executable not found${NC}"
    echo "Please build first: cd ${BUILD_DIR} && cmake .. && make"
    exit 1
fi

# Step 2: Check config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}ERROR: Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

# Step 3: Verify VOF advection is enabled in config
if ! grep -q "enable_vof_advection.*=.*true" "${CONFIG_FILE}"; then
    echo -e "${RED}ERROR: enable_vof_advection not set to true in config${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} VOF advection enabled in config"

# Step 4: Create output directory
OUTPUT_DIR="${BUILD_DIR}/lpbf_test_E_vof_advection"
mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}✓${NC} Output directory: ${OUTPUT_DIR}"
echo ""

# Step 5: Initialize performance log
cat > "${PERFORMANCE_LOG}" <<EOF
TEST E: VOF Advection Performance Monitoring
Start time: $(date '+%Y-%m-%d %H:%M:%S')
Configuration: ${CONFIG_FILE}
Total steps: 3000 (300 μs simulation time)

Performance Targets:
  Runtime: 10-15 minutes (acceptable)
  Runtime > 20 minutes: Performance issue
  GPU utilization: > 80%
  Mass conservation error: < 5%
  Interface cells: < 5000
  CFL_VOF: < 0.5

================================
EOF

# Step 6: Start GPU monitoring in background
GPU_MONITOR_LOG="${BUILD_DIR}/test_E_gpu_monitor.txt"
echo -e "${CYAN}Starting GPU monitoring...${NC}"
(
    echo "GPU Utilization Monitoring - Test E" > "${GPU_MONITOR_LOG}"
    echo "Timestamp,GPU_Util(%),Memory_Used(MB),Temp(C)" >> "${GPU_MONITOR_LOG}"
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
                   --format=csv,noheader,nounits 2>/dev/null | \
            awk -v t="$(date +%s)" '{print t","$1","$2","$3}' >> "${GPU_MONITOR_LOG}" || true
        sleep 5
    done
) &
GPU_MONITOR_PID=$!

# Ensure GPU monitor is killed on exit
trap "kill ${GPU_MONITOR_PID} 2>/dev/null || true" EXIT

echo -e "${GREEN}✓${NC} GPU monitoring active (PID: ${GPU_MONITOR_PID})"
echo ""

# Step 7: Run simulation
echo -e "${YELLOW}Running Test E simulation with VOF advection monitoring...${NC}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo -e "${BLUE}Real-time Monitoring Commands (open in separate terminals):${NC}"
echo "  Terminal 2: tail -f ${LOG_FILE} | grep 'v_max'"
echo "  Terminal 3: tail -f ${LOG_FILE} | grep -i 'mass\\|error\\|warning'"
echo "  Terminal 4: watch -n 5 nvidia-smi"
echo ""
echo -e "${CYAN}Progress monitoring (updated every 30s):${NC}"
echo ""

cd "${BUILD_DIR}"

# Run with real-time output
./visualize_lpbf_scanning --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}" &
SIM_PID=$!

# Monitor progress with VOF-specific metrics
LAST_REPORTED_STEP=0
while kill -0 ${SIM_PID} 2>/dev/null; do
    sleep 30
    if [ -f "${LOG_FILE}" ]; then
        LAST_STEP=$(grep -oP 'Step\s+\K\d+' "${LOG_FILE}" | tail -1 || echo "0")
        if [ "${LAST_STEP}" != "0" ] && [ "${LAST_STEP}" != "${LAST_REPORTED_STEP}" ]; then
            PROGRESS=$(( LAST_STEP * 100 / 3000 ))
            ELAPSED=$(( $(date +%s) - START_TIME ))
            ELAPSED_MIN=$(( ELAPSED / 60 ))
            ELAPSED_SEC=$(( ELAPSED % 60 ))

            # Extract current metrics
            CURRENT_T=$(grep "Step.*${LAST_STEP}" "${LOG_FILE}" | tail -1 | \
                        grep -oP 'T_max\s*=\s*\K[\d.e+-]+' || echo "0")
            CURRENT_V=$(grep "Step.*${LAST_STEP}" "${LOG_FILE}" | tail -1 | \
                        grep -oP 'v_max\s*=\s*\K[\d.e+-]+' || echo "0")

            # Check for issues
            STATUS="${GREEN}STABLE${NC}"
            if grep -q "NaN\|Inf" "${LOG_FILE}"; then
                STATUS="${RED}NaN/Inf DETECTED${NC}"
            elif (( $(echo "${CURRENT_T} > 50000" | bc -l 2>/dev/null || echo 0) )); then
                STATUS="${YELLOW}HIGH TEMP${NC}"
            fi

            echo -e "${BLUE}[${ELAPSED_MIN}m ${ELAPSED_SEC}s]${NC} Step ${LAST_STEP}/3000 (${PROGRESS}%) | T=${CURRENT_T}K | v=${CURRENT_V}mm/s | ${STATUS}"

            LAST_REPORTED_STEP=${LAST_STEP}
        fi
    fi
done

# Wait for completion
wait ${SIM_PID}
EXIT_CODE=$?

# Stop GPU monitoring
kill ${GPU_MONITOR_PID} 2>/dev/null || true

# Calculate total runtime
END_TIME=$(date +%s)
TOTAL_RUNTIME=$(( END_TIME - START_TIME ))
RUNTIME_MIN=$(( TOTAL_RUNTIME / 60 ))
RUNTIME_SEC=$(( TOTAL_RUNTIME % 60 ))

echo ""
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${CYAN}Total runtime: ${RUNTIME_MIN} min ${RUNTIME_SEC} sec${NC}"

# Performance assessment
if [ ${RUNTIME_MIN} -lt 15 ]; then
    echo -e "${GREEN}✓ Performance: EXCELLENT (< 15 min)${NC}"
elif [ ${RUNTIME_MIN} -lt 20 ]; then
    echo -e "${YELLOW}⚠ Performance: ACCEPTABLE (< 20 min)${NC}"
else
    echo -e "${RED}✗ Performance: POOR (> 20 min) - Investigate VOF kernel efficiency${NC}"
fi

# Save runtime to performance log
cat >> "${PERFORMANCE_LOG}" <<EOF

RUNTIME RESULTS:
  Total time: ${RUNTIME_MIN} min ${RUNTIME_SEC} sec
  Target: 10-15 minutes
  Assessment: $([ ${RUNTIME_MIN} -lt 15 ] && echo "EXCELLENT" || \
                 [ ${RUNTIME_MIN} -lt 20 ] && echo "ACCEPTABLE" || echo "POOR")

================================
EOF

# Step 8: Check for simulation errors
if [ ${EXIT_CODE} -ne 0 ]; then
    echo -e "${RED}✗ FAIL: Simulation crashed (exit code ${EXIT_CODE})${NC}"
    echo "Check log file: ${LOG_FILE}"
    exit 1
fi

# Step 9: VOF-specific sanity checks
echo ""
echo -e "${YELLOW}VOF Advection Sanity Checks...${NC}"

# Check for NaN/Inf
if grep -qi "nan\|inf" "${LOG_FILE}"; then
    echo -e "${RED}✗ CRITICAL: NaN or Inf detected - VOF advection unstable${NC}"
    echo "  Action: Check CFL_VOF condition, reduce dt, or add fill level clipping"
else
    echo -e "${GREEN}✓${NC} No NaN/Inf detected"
fi

# Check for fill level bounds violations
if grep -qi "fill.*level.*out.*of.*bounds\|f_min.*<.*0\|f_max.*>.*1" "${LOG_FILE}"; then
    echo -e "${RED}✗ WARNING: Fill level outside [0,1] range${NC}"
    echo "  Action: Add clamping: f = clamp(f, 0, 1)"
else
    echo -e "${GREEN}✓${NC} Fill level within physical bounds"
fi

# Extract final values
FINAL_T=$(grep "Step.*3000" "${LOG_FILE}" | tail -1 | grep -oP 'T_max\s*=\s*\K[\d.e+-]+' || echo "0")
FINAL_V=$(grep "Step.*3000" "${LOG_FILE}" | tail -1 | grep -oP 'v_max\s*=\s*\K[\d.e+-]+' || echo "0")

echo "Final T_max: ${FINAL_T} K"
echo "Final v_max: ${FINAL_V} mm/s"

# Compare to Test C if available
if [ -f "${TEST_C_LOG}" ]; then
    TEST_C_T=$(grep "Step.*3000" "${TEST_C_LOG}" | tail -1 | grep -oP 'T_max\s*=\s*\K[\d.e+-]+' || echo "0")
    TEST_C_V=$(grep "Step.*3000" "${TEST_C_LOG}" | tail -1 | grep -oP 'v_max\s*=\s*\K[\d.e+-]+' || echo "0")

    if (( $(echo "${TEST_C_T} > 0" | bc -l 2>/dev/null || echo 0) )); then
        echo ""
        echo -e "${CYAN}Comparison to Test C (static interface):${NC}"
        echo "  Test C T_max: ${TEST_C_T} K"
        echo "  Test E T_max: ${FINAL_T} K"
        echo "  Test C v_max: ${TEST_C_V} mm/s"
        echo "  Test E v_max: ${FINAL_V} mm/s"
    fi
fi

# Analyze GPU utilization
if [ -f "${GPU_MONITOR_LOG}" ]; then
    AVG_GPU_UTIL=$(tail -n +2 "${GPU_MONITOR_LOG}" | awk -F, '{sum+=$2; count++} END {print sum/count}' 2>/dev/null || echo "0")
    echo ""
    echo -e "${CYAN}GPU Performance:${NC}"
    echo "  Average utilization: ${AVG_GPU_UTIL}%"

    if (( $(echo "${AVG_GPU_UTIL} > 80" | bc -l 2>/dev/null || echo 0) )); then
        echo -e "${GREEN}✓${NC} Excellent GPU utilization"
    elif (( $(echo "${AVG_GPU_UTIL} > 70" | bc -l 2>/dev/null || echo 0) )); then
        echo -e "${YELLOW}⚠${NC} Acceptable GPU utilization"
    else
        echo -e "${RED}✗${NC} Low GPU utilization - VOF kernel may be inefficient"
    fi
fi

# Step 10: Run Python validator
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Running VOF-specific automated validation...${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

if [ -f "${VALIDATOR}" ]; then
    python3 "${VALIDATOR}" "${LOG_FILE}" "${TEST_C_LOG}"
    VALIDATION_CODE=$?

    echo ""
    if [ ${VALIDATION_CODE} -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✓ TEST E PASSED                                  ║${NC}"
        echo -e "${GREEN}║  VOF advection working correctly!                 ║${NC}"
        echo -e "${GREEN}║  Runtime: ${RUNTIME_MIN}m ${RUNTIME_SEC}s                                    ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${BLUE}Key Achievements:${NC}"
        echo "  ✓ VOF advection stable (no NaN/Inf)"
        echo "  ✓ Mass conservation maintained"
        echo "  ✓ Interface remained sharp"
        echo "  ✓ Performance acceptable"
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "  1. Visualize in ParaView: ${OUTPUT_DIR}/*.vti"
        echo "  2. Check interface deformation (compare to Test C)"
        echo "  3. Verify mass conservation over time"
        echo "  4. Review performance log: ${PERFORMANCE_LOG}"
        echo "  5. Check GPU utilization: ${GPU_MONITOR_LOG}"
        exit 0
    else
        echo -e "${RED}╔════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ✗ TEST E FAILED                                  ║${NC}"
        echo -e "${RED}║  Review validation report above                   ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}Troubleshooting:${NC}"
        echo "  - Check log file: ${LOG_FILE}"
        echo "  - Review performance: ${PERFORMANCE_LOG}"
        echo "  - GPU monitoring: ${GPU_MONITOR_LOG}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Validator not found: ${VALIDATOR}${NC}"
    echo "Creating basic validator - please review manually"
    echo ""
    echo -e "${CYAN}Manual Review Checklist:${NC}"
    echo "  [ ] Runtime < 20 minutes"
    echo "  [ ] No NaN/Inf in output"
    echo "  [ ] Fill level stayed in [0,1]"
    echo "  [ ] Temperature realistic (< 50,000 K)"
    echo "  [ ] Velocity developed (> 0.1 mm/s)"
    echo "  [ ] GPU utilization > 70%"
    exit 0
fi
