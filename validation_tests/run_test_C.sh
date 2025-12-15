#!/bin/bash
################################################################################
# TEST C: Full Multiphysics Coupling
################################################################################
# Purpose: Validate complete LPBF physics with all couplings enabled
# Expected: Realistic temperatures and velocities matching literature
# Duration: ~5-10 minutes for 5000 steps
################################################################################

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BUILD_DIR="/home/yzk/LBMProject/build"
CONFIG_FILE="/home/yzk/LBMProject/configs/lpbf_195W_test_C_full_coupling.conf"
LOG_FILE="${BUILD_DIR}/test_C_full_coupling.log"
VALIDATOR="/home/yzk/LBMProject/validation_tests/validate_test_C.py"
TEST_A_LOG="${BUILD_DIR}/test_A_coupling.log"
TEST_B_LOG="${BUILD_DIR}/test_B_marangoni.log"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}TEST C: Full Multiphysics Coupling${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Configuration: ${CONFIG_FILE}"
echo "Log file:      ${LOG_FILE}"
echo "Build dir:     ${BUILD_DIR}"
echo ""

# Step 0: Check prerequisites
PREREQUISITES_OK=true

if [ ! -f "${TEST_A_LOG}" ]; then
    echo -e "${YELLOW}⚠ WARNING: Test A not completed${NC}"
    PREREQUISITES_OK=false
fi

if [ ! -f "${TEST_B_LOG}" ]; then
    echo -e "${YELLOW}⚠ WARNING: Test B not completed${NC}"
    PREREQUISITES_OK=false
fi

if [ "${PREREQUISITES_OK}" = false ]; then
    echo ""
    echo -e "${YELLOW}Recommended workflow:${NC}"
    echo "  1. Run Test A: ./run_test_A.sh"
    echo "  2. Run Test B: ./run_test_B.sh"
    echo "  3. Run Test C: ./run_test_C.sh (this script)"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
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

# Step 3: Create output directory
OUTPUT_DIR="${BUILD_DIR}/test_C_full_coupling"
mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}✓${NC} Output directory: ${OUTPUT_DIR}"
echo ""

# Step 4: Run simulation
echo -e "${YELLOW}Running Test C simulation (this will take several minutes)...${NC}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo -e "${BLUE}Progress monitoring:${NC}"
echo "  - Watch for 'Step' messages in output"
echo "  - Expected: 5000 steps total"
echo "  - Est. time: 5-10 minutes"
echo ""

cd "${BUILD_DIR}"

# Run with monitoring in background
./visualize_lpbf_scanning --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}" &
SIM_PID=$!

# Monitor progress every 30 seconds
while kill -0 ${SIM_PID} 2>/dev/null; do
    sleep 30
    if [ -f "${LOG_FILE}" ]; then
        LAST_STEP=$(grep -oP 'Step\s+\K\d+' "${LOG_FILE}" | tail -1 || echo "0")
        if [ "${LAST_STEP}" != "0" ]; then
            PROGRESS=$(( LAST_STEP * 100 / 5000 ))
            echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} Progress: ${LAST_STEP}/5000 (${PROGRESS}%)"
        fi
    fi
done

# Wait for completion
wait ${SIM_PID}
EXIT_CODE=$?

echo ""
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"

# Step 5: Check for simulation errors
if [ ${EXIT_CODE} -ne 0 ]; then
    echo -e "${RED}✗ FAIL: Simulation crashed (exit code ${EXIT_CODE})${NC}"
    echo "Check log file: ${LOG_FILE}"
    exit 1
fi

# Step 6: Quick sanity checks
echo ""
echo -e "${YELLOW}Quick sanity checks...${NC}"

# Check for NaN/Inf
if grep -qi "nan\|inf" "${LOG_FILE}"; then
    echo -e "${RED}✗ WARNING: NaN or Inf detected in output${NC}"
fi

# Extract final values
FINAL_T=$(grep "Step.*5000" "${LOG_FILE}" | tail -1 | grep -oP 'T_max\s*=\s*\K[\d.e+-]+' || echo "0")
FINAL_V=$(grep "Step.*5000" "${LOG_FILE}" | tail -1 | grep -oP 'v_max\s*=\s*\K[\d.e+-]+' || echo "0")

echo "Final T_max: ${FINAL_T} K"
echo "Final v_max: ${FINAL_V} mm/s"

# Literature comparison
LIT_T=3300
LIT_V=970

if (( $(echo "${FINAL_T} > 0 && ${FINAL_T} < 10000" | bc -l) )); then
    echo -e "${GREEN}✓${NC} Temperature in realistic range (< 10,000 K)"
else
    echo -e "${YELLOW}⚠${NC} Temperature may need vaporization cooling"
fi

if (( $(echo "${FINAL_V} > 10 && ${FINAL_V} < 1000" | bc -l) )); then
    echo -e "${GREEN}✓${NC} Velocity in Marangoni range (10-1000 mm/s)"
elif (( $(echo "${FINAL_V} > 1" | bc -l) )); then
    echo -e "${YELLOW}⚠${NC} Velocity lower than expected"
else
    echo -e "${RED}✗${NC} Velocity too low (Marangoni not active?)"
fi

# Step 7: Run Python validator
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Running automated validation...${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

if [ -f "${VALIDATOR}" ]; then
    python3 "${VALIDATOR}" "${LOG_FILE}"
    VALIDATION_CODE=$?

    echo ""
    if [ ${VALIDATION_CODE} -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✓ TEST C PASSED                                  ║${NC}"
        echo -e "${GREEN}║  Full multiphysics coupling validated!            ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "  1. Visualize in ParaView: ${OUTPUT_DIR}/*.vti"
        echo "  2. Compare melt pool shape to literature"
        echo "  3. Measure melt pool dimensions"
        echo "  4. If T > 5000 K, consider vaporization cooling (V5)"
        exit 0
    else
        echo -e "${RED}╔════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ✗ TEST C FAILED                                  ║${NC}"
        echo -e "${RED}║  Review validation report above                   ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════╝${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Validator not found: ${VALIDATOR}${NC}"
    echo "Manual review required"
    exit 0
fi
