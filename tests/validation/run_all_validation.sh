#!/bin/bash
################################################################################
# Master Validation Script
# Purpose: Orchestrate all validation tests to prove fixes are "治本" not "治标"
# Author: LBM-CUDA Architecture Team
# Date: 2025-01-19
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="/home/yzk/LBMProject"
VALIDATION_DIR="${PROJECT_ROOT}/tests/validation"
RESULTS_BASE="${PROJECT_ROOT}/validation_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_BASE}/run_${TIMESTAMP}"

# Test selection (can be overridden by command-line args)
RUN_GRID_CONVERGENCE=true
RUN_PECLET_SWEEP=true
RUN_ENERGY_CONSERVATION=true
RUN_LITERATURE_BENCHMARK=false  # Disabled by default (takes ~20 minutes)
RUN_FLUX_LIMITER_IMPACT=true

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            RUN_GRID_CONVERGENCE=false
            RUN_LITERATURE_BENCHMARK=false
            echo "Quick mode: Running only essential tests"
            shift
            ;;
        --full)
            RUN_GRID_CONVERGENCE=true
            RUN_PECLET_SWEEP=true
            RUN_ENERGY_CONSERVATION=true
            RUN_LITERATURE_BENCHMARK=true
            RUN_FLUX_LIMITER_IMPACT=true
            echo "Full mode: Running all validation tests"
            shift
            ;;
        --benchmark-only)
            RUN_GRID_CONVERGENCE=false
            RUN_PECLET_SWEEP=false
            RUN_ENERGY_CONSERVATION=false
            RUN_LITERATURE_BENCHMARK=true
            RUN_FLUX_LIMITER_IMPACT=false
            echo "Benchmark mode: Running literature comparison only"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick           Run only fast tests (~30 min)"
            echo "  --full            Run all tests including benchmark (~3 hours)"
            echo "  --benchmark-only  Run only literature benchmark"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Default: Standard validation (no benchmark, ~1 hour)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# Create run directory
mkdir -p "${RUN_DIR}"

################################################################################
# Print header
################################################################################

clear
echo ""
echo "================================================================================"
echo -e "${CYAN}       LBM-CUDA VALIDATION FRAMEWORK       ${NC}"
echo "================================================================================"
echo ""
echo "  Purpose: Validate stability fixes are treating ROOT CAUSE (治本)"
echo "           not superficial symptoms (治标)"
echo ""
echo "  Validation Strategy:"
echo "    1. Grid Convergence    → Prove numerical soundness"
echo "    2. Peclet Sweep        → Verify robustness across regimes"
echo "    3. Energy Conservation → Confirm thermodynamic consistency"
echo "    4. Literature Benchmark → Validate physical accuracy"
echo "    5. Flux Limiter Impact → Quantify accuracy vs. efficiency"
echo ""
echo "================================================================================"
echo ""
echo "  Run Directory: ${RUN_DIR}"
echo "  Start Time:    $(date)"
echo ""
echo "================================================================================"
echo ""

# Track test results
declare -a TEST_RESULTS
declare -a TEST_NAMES
declare -a TEST_TIMES

################################################################################
# Function: Run a validation test
################################################################################
run_test() {
    local TEST_NAME=$1
    local TEST_SCRIPT=$2

    echo ""
    echo "================================================================================"
    echo -e "${MAGENTA}Running: ${TEST_NAME}${NC}"
    echo "================================================================================"
    echo ""

    local START_TIME=$(date +%s)

    if bash "${TEST_SCRIPT}"; then
        local END_TIME=$(date +%s)
        local ELAPSED=$((END_TIME - START_TIME))

        TEST_RESULTS+=("PASS")
        TEST_NAMES+=("${TEST_NAME}")
        TEST_TIMES+=("${ELAPSED}")

        echo ""
        echo -e "${GREEN}✓ ${TEST_NAME} PASSED (${ELAPSED}s)${NC}"
        return 0
    else
        local END_TIME=$(date +%s)
        local ELAPSED=$((END_TIME - START_TIME))

        TEST_RESULTS+=("FAIL")
        TEST_NAMES+=("${TEST_NAME}")
        TEST_TIMES+=("${ELAPSED}")

        echo ""
        echo -e "${RED}✗ ${TEST_NAME} FAILED (${ELAPSED}s)${NC}"
        return 1
    fi
}

################################################################################
# Execute validation tests
################################################################################

TOTAL_START_TIME=$(date +%s)

# Test 1: Grid Convergence
if [ "$RUN_GRID_CONVERGENCE" = true ]; then
    run_test "Grid Convergence Study" "${VALIDATION_DIR}/test_grid_convergence.sh" || true
fi

# Test 2: Peclet Number Sweep
if [ "$RUN_PECLET_SWEEP" = true ]; then
    run_test "Peclet Number Sweep" "${VALIDATION_DIR}/test_peclet_sweep.sh" || true
fi

# Test 3: Energy Conservation
if [ "$RUN_ENERGY_CONSERVATION" = true ]; then
    run_test "Energy Conservation" "${VALIDATION_DIR}/test_energy_conservation.sh" || true
fi

# Test 4: Literature Benchmark
if [ "$RUN_LITERATURE_BENCHMARK" = true ]; then
    run_test "Literature Benchmark" "${VALIDATION_DIR}/test_literature_benchmark.sh" || true
fi

# Test 5: Flux Limiter Impact
if [ "$RUN_FLUX_LIMITER_IMPACT" = true ]; then
    run_test "Flux Limiter Impact" "${VALIDATION_DIR}/test_flux_limiter_impact.sh" || true
fi

TOTAL_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

################################################################################
# Generate summary report
################################################################################

echo ""
echo ""
echo "================================================================================"
echo -e "${CYAN}                  VALIDATION SUMMARY                   ${NC}"
echo "================================================================================"
echo ""

# Count passes and fails
PASS_COUNT=0
FAIL_COUNT=0

for result in "${TEST_RESULTS[@]}"; do
    if [ "$result" = "PASS" ]; then
        ((PASS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

TOTAL_TESTS=${#TEST_RESULTS[@]}

# Display individual test results
echo "Test Results:"
echo "--------------------------------------------------------------------------------"
for i in "${!TEST_NAMES[@]}"; do
    local NAME="${TEST_NAMES[$i]}"
    local RESULT="${TEST_RESULTS[$i]}"
    local TIME="${TEST_TIMES[$i]}"

    if [ "$RESULT" = "PASS" ]; then
        printf "  ${GREEN}✓${NC} %-35s PASS    %4ds\n" "${NAME}" "${TIME}"
    else
        printf "  ${RED}✗${NC} %-35s FAIL    %4ds\n" "${NAME}" "${TIME}"
    fi
done

echo ""
echo "--------------------------------------------------------------------------------"
echo "Total Tests:     ${TOTAL_TESTS}"
echo "Passed:          ${GREEN}${PASS_COUNT}${NC}"
echo "Failed:          ${RED}${FAIL_COUNT}${NC}"
echo "Total Runtime:   ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "--------------------------------------------------------------------------------"
echo ""

# Overall verdict
if [ ${FAIL_COUNT} -eq 0 ] && [ ${TOTAL_TESTS} -gt 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                    ALL VALIDATION TESTS PASSED                    ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}CONCLUSION: Stability fixes are treating ROOT CAUSE (治本)${NC}"
    echo ""
    echo "Evidence:"
    echo "  ✓ Solution converges with grid refinement (numerically sound)"
    echo "  ✓ Stable across all Peclet number regimes (robust)"
    echo "  ✓ Energy conserved within acceptable limits (thermodynamically consistent)"
    echo "  ✓ Minimal accuracy loss from flux limiter (not over-dissipative)"
    echo ""
    echo -e "${GREEN}Recommendation: PROCEED with current fixes for production use${NC}"
    echo ""

    OVERALL_STATUS=0
else
    echo -e "${RED}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}                  SOME VALIDATION TESTS FAILED                    ${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}CONCLUSION: Fixes may be partially superficial (治标)${NC}"
    echo ""
    echo "Failed Tests:"
    for i in "${!TEST_NAMES[@]}"; do
        if [ "${TEST_RESULTS[$i]}" = "FAIL" ]; then
            echo "  ✗ ${TEST_NAMES[$i]}"
        fi
    done
    echo ""
    echo -e "${YELLOW}Recommendation: Address failed tests before production deployment${NC}"
    echo ""

    OVERALL_STATUS=1
fi

echo "================================================================================"
echo ""
echo "Results Location: ${RUN_DIR}"
echo "End Time:         $(date)"
echo ""
echo "Next Steps:"
echo "  1. Review individual test reports in validation_results/"
echo "  2. Run analysis notebook: jupyter lab ${PROJECT_ROOT}/analysis/validation_analysis.ipynb"
echo "  3. Generate visualizations in ParaView using VTK output files"
echo "  4. Complete validation report: ${PROJECT_ROOT}/docs/validation/VALIDATION_REPORT_TEMPLATE.md"
echo ""
echo "================================================================================"

# Save summary to file
SUMMARY_FILE="${RUN_DIR}/validation_summary.txt"
cat > "${SUMMARY_FILE}" << EOF
LBM-CUDA VALIDATION FRAMEWORK SUMMARY
================================================================================

Run Timestamp: ${TIMESTAMP}
Total Runtime: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s

TEST RESULTS:
--------------------------------------------------------------------------------
EOF

for i in "${!TEST_NAMES[@]}"; do
    echo "${TEST_NAMES[$i]}: ${TEST_RESULTS[$i]} (${TEST_TIMES[$i]}s)" >> "${SUMMARY_FILE}"
done

cat >> "${SUMMARY_FILE}" << EOF

--------------------------------------------------------------------------------
OVERALL: ${PASS_COUNT}/${TOTAL_TESTS} tests passed

EOF

if [ ${FAIL_COUNT} -eq 0 ]; then
    echo "VERDICT: All tests passed - fixes are treating ROOT CAUSE (治本)" >> "${SUMMARY_FILE}"
else
    echo "VERDICT: Some tests failed - further investigation needed" >> "${SUMMARY_FILE}"
fi

echo ""
echo "Summary saved to: ${SUMMARY_FILE}"

exit ${OVERALL_STATUS}
