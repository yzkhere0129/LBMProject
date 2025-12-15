#!/bin/bash
################################################################################
# TEST A: Thermal-Fluid Coupling (Basic Advection)
################################################################################
# Purpose: Verify that thermal-fluid coupling reduces temperature via convection
# Expected: T_max < 45,477 K (v4 baseline)
# Duration: ~30-60 seconds for 1000 steps
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="/home/yzk/LBMProject/build"
CONFIG_FILE="/home/yzk/LBMProject/configs/lpbf_195W_test_A_coupling.conf"
LOG_FILE="${BUILD_DIR}/test_A_coupling.log"
VALIDATOR="/home/yzk/LBMProject/validation_tests/validate_test_A.py"
BASELINE_V4_LOG="${BUILD_DIR}/test_195W_v5_radiation.log"  # v4 reference

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}TEST A: Thermal-Fluid Coupling${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Configuration: ${CONFIG_FILE}"
echo "Log file:      ${LOG_FILE}"
echo "Build dir:     ${BUILD_DIR}"
echo ""

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
OUTPUT_DIR="${BUILD_DIR}/test_A_coupling"
mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}✓${NC} Output directory: ${OUTPUT_DIR}"
echo ""

# Step 4: Run simulation
echo -e "${YELLOW}Running Test A simulation...${NC}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run with real-time monitoring
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

# Extract final temperature
FINAL_T=$(grep "Step.*1000" "${LOG_FILE}" | tail -1 | grep -oP 'T_max\s*=\s*\K[\d.e+-]+' || echo "0")
echo "Final T_max: ${FINAL_T} K"

# Compare to v4 baseline
V4_T=45477
if (( $(echo "${FINAL_T} > 0" | bc -l) )); then
    if (( $(echo "${FINAL_T} < ${V4_T}" | bc -l) )); then
        echo -e "${GREEN}✓${NC} Temperature decreased vs v4 (${V4_T} K)"
    else
        echo -e "${RED}✗${NC} Temperature did NOT decrease vs v4"
    fi
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
        echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✓ TEST A PASSED                      ║${NC}"
        echo -e "${GREEN}║  Ready to proceed to Test B           ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
        exit 0
    else
        echo -e "${RED}╔════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ✗ TEST A FAILED                      ║${NC}"
        echo -e "${RED}║  Fix issues before Test B             ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════╝${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Validator not found: ${VALIDATOR}${NC}"
    echo "Manual review required"
    exit 0
fi
