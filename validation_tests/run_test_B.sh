#!/bin/bash
################################################################################
# TEST B: Marangoni Force
################################################################################
# Purpose: Verify Marangoni thermocapillary convection activates
# Expected: v_max > 10 mm/s, T_max lower than Test A
# Duration: ~30-60 seconds for 1000 steps
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
CONFIG_FILE="/home/yzk/LBMProject/configs/lpbf_195W_test_B_marangoni.conf"
LOG_FILE="${BUILD_DIR}/test_B_marangoni.log"
VALIDATOR="/home/yzk/LBMProject/validation_tests/validate_test_B.py"
TEST_A_LOG="${BUILD_DIR}/test_A_coupling.log"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}TEST B: Marangoni Force${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Configuration: ${CONFIG_FILE}"
echo "Log file:      ${LOG_FILE}"
echo "Build dir:     ${BUILD_DIR}"
echo ""

# Step 0: Check Test A completed
if [ ! -f "${TEST_A_LOG}" ]; then
    echo -e "${YELLOW}⚠ WARNING: Test A log not found${NC}"
    echo "Recommended: Run Test A first for baseline comparison"
    echo "Continuing anyway..."
    echo ""
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
OUTPUT_DIR="${BUILD_DIR}/test_B_marangoni"
mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}✓${NC} Output directory: ${OUTPUT_DIR}"
echo ""

# Step 4: Run simulation
echo -e "${YELLOW}Running Test B simulation...${NC}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd "${BUILD_DIR}"
./visualize_lpbf_scanning --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

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

# Extract final velocity
FINAL_V=$(grep "Step.*1000" "${LOG_FILE}" | tail -1 | grep -oP 'v_max\s*=\s*\K[\d.e+-]+' || echo "0")
echo "Final v_max: ${FINAL_V} mm/s"

# Check Marangoni activation
if (( $(echo "${FINAL_V} > 10.0" | bc -l) )); then
    echo -e "${GREEN}✓${NC} Marangoni effect ACTIVE (v_max > 10 mm/s)"
else
    echo -e "${RED}✗${NC} Marangoni effect NOT active (v_max < 10 mm/s)"
fi

# Step 7: Run Python validator
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Running automated validation...${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

if [ -f "${VALIDATOR}" ]; then
    if [ -f "${TEST_A_LOG}" ]; then
        python3 "${VALIDATOR}" "${LOG_FILE}" "${TEST_A_LOG}"
    else
        python3 "${VALIDATOR}" "${LOG_FILE}"
    fi
    VALIDATION_CODE=$?

    echo ""
    if [ ${VALIDATION_CODE} -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✓ TEST B PASSED                      ║${NC}"
        echo -e "${GREEN}║  Ready to proceed to Test C           ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
        exit 0
    else
        echo -e "${RED}╔════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ✗ TEST B FAILED                      ║${NC}"
        echo -e "${RED}║  Fix Marangoni issues before Test C   ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════╝${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Validator not found: ${VALIDATOR}${NC}"
    echo "Manual review required"
    exit 0
fi
