#!/bin/bash
################################################################################
# MASTER TEST SUITE: Comprehensive Thermal-Fluid Coupling Validation
################################################################################
# Executes Tests A, B, C in sequence with automated validation
#
# Test A: Thermal-Fluid Coupling (basic advection)
# Test B: Marangoni Force (thermocapillary convection)
# Test C: Full Multiphysics (complete LPBF simulation)
#
# Usage:
#   ./run_all_tests.sh              # Run all tests sequentially
#   ./run_all_tests.sh --skip-c     # Run A and B only (faster)
#   ./run_all_tests.sh --continue   # Continue from last failure
################################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/home/yzk/LBMProject/build"

# Parse arguments
SKIP_TEST_C=false
CONTINUE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-c)
            SKIP_TEST_C=true
            shift
            ;;
        --continue)
            CONTINUE_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-c] [--continue]"
            exit 1
            ;;
    esac
done

# Test tracking
declare -A TEST_STATUS
TEST_STATUS["A"]="pending"
TEST_STATUS["B"]="pending"
TEST_STATUS["C"]="pending"

LOG_FILES=(
    "${BUILD_DIR}/test_A_coupling.log"
    "${BUILD_DIR}/test_B_marangoni.log"
    "${BUILD_DIR}/test_C_full_coupling.log"
)

# Functions
print_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  $1"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_test_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_summary() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  TEST SUITE SUMMARY"
    echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"

    for test in A B C; do
        status="${TEST_STATUS[$test]}"
        case $status in
            "passed")
                icon="✓"
                color="${GREEN}"
                ;;
            "failed")
                icon="✗"
                color="${RED}"
                ;;
            "skipped")
                icon="⊘"
                color="${YELLOW}"
                ;;
            *)
                icon="⋯"
                color="${CYAN}"
                ;;
        esac
        printf "${CYAN}║${NC}  Test ${test}: ${color}${icon} %-50s${NC}\n" "${status^^}"
    done

    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

run_test() {
    local test_name=$1
    local test_script=$2

    print_test_header "TEST ${test_name}"

    # Check if already passed (continue mode)
    if [ "${CONTINUE_MODE}" = true ]; then
        if [ "${TEST_STATUS[$test_name]}" = "passed" ]; then
            echo -e "${GREEN}✓ Test ${test_name} already passed (skipping)${NC}"
            return 0
        fi
    fi

    # Run test
    if bash "${test_script}"; then
        TEST_STATUS[$test_name]="passed"
        echo -e "${GREEN}✓ Test ${test_name} PASSED${NC}"
        return 0
    else
        TEST_STATUS[$test_name]="failed"
        echo -e "${RED}✗ Test ${test_name} FAILED${NC}"
        return 1
    fi
}

# Main execution
print_header "MULTIPHYSICS COUPLING TEST SUITE"

echo "Test sequence:"
echo "  A. Thermal-Fluid Coupling (1000 steps, ~1 min)"
echo "  B. Marangoni Force (1000 steps, ~1 min)"
if [ "${SKIP_TEST_C}" = false ]; then
    echo "  C. Full Coupling (5000 steps, ~5-10 min)"
else
    echo "  C. Full Coupling (SKIPPED)"
fi
echo ""
echo "Total estimated time: $([ "${SKIP_TEST_C}" = true ] && echo "~2-3 min" || echo "~7-12 min")"
echo ""

read -p "Proceed with test suite? (Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Check continue mode
if [ "${CONTINUE_MODE}" = true ]; then
    echo -e "${YELLOW}Continue mode: Checking previous results...${NC}"

    if [ -f "${LOG_FILES[0]}" ]; then
        if python3 "${SCRIPT_DIR}/validate_test_A.py" "${LOG_FILES[0]}" > /dev/null 2>&1; then
            TEST_STATUS["A"]="passed"
            echo -e "${GREEN}✓ Test A previously passed${NC}"
        fi
    fi

    if [ -f "${LOG_FILES[1]}" ]; then
        if python3 "${SCRIPT_DIR}/validate_test_B.py" "${LOG_FILES[1]}" > /dev/null 2>&1; then
            TEST_STATUS["B"]="passed"
            echo -e "${GREEN}✓ Test B previously passed${NC}"
        fi
    fi

    if [ -f "${LOG_FILES[2]}" ]; then
        if python3 "${SCRIPT_DIR}/validate_test_C.py" "${LOG_FILES[2]}" > /dev/null 2>&1; then
            TEST_STATUS["C"]="passed"
            echo -e "${GREEN}✓ Test C previously passed${NC}"
        fi
    fi
    echo ""
fi

# Start timestamp
START_TIME=$(date +%s)

# Run Test A
if ! run_test "A" "${SCRIPT_DIR}/run_test_A.sh"; then
    print_summary
    echo -e "${RED}Test suite aborted at Test A${NC}"
    echo "Fix Test A issues before proceeding to Test B"
    exit 1
fi

# Run Test B
if ! run_test "B" "${SCRIPT_DIR}/run_test_B.sh"; then
    print_summary
    echo -e "${RED}Test suite aborted at Test B${NC}"
    echo "Fix Test B issues before proceeding to Test C"
    exit 1
fi

# Run Test C (if not skipped)
if [ "${SKIP_TEST_C}" = true ]; then
    TEST_STATUS["C"]="skipped"
    echo -e "${YELLOW}Test C skipped (--skip-c flag)${NC}"
else
    if ! run_test "C" "${SCRIPT_DIR}/run_test_C.sh"; then
        print_summary
        echo -e "${RED}Test suite completed with failures${NC}"
        exit 1
    fi
fi

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Final summary
print_summary

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ALL TESTS PASSED!                                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Total execution time: ${MINUTES}m ${SECONDS}s"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "  1. Review VTK output in ParaView"
echo "  2. Compare melt pool shapes to literature"
echo "  3. Measure melt pool dimensions"
echo "  4. If needed, implement vaporization cooling (V5)"
echo ""
echo -e "${GREEN}Physics validation complete!${NC}"

exit 0
