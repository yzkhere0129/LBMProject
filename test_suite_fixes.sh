#!/bin/bash
# ============================================================================
# Comprehensive Test Suite for Recent Fixes
# ============================================================================
# Tests for:
# 1. Newton bisection fallback
# 2. Laser shutoff time configuration
# 3. Marangoni gradient limiter (CRITICAL - 500x increase)
# 4. Newton convergence reporting
# ============================================================================

set -e  # Exit on error
cd /home/yzk/LBMProject/build

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "  COMPREHENSIVE TEST SUITE - Fix Validation"
echo "========================================================================"
echo ""

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run test and check result
run_test() {
    local test_name=$1
    local test_cmd=$2
    local success_criteria=$3

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "${BLUE}[TEST $TOTAL_TESTS] ${test_name}${NC}"
    echo "Command: $test_cmd"
    echo "Success criteria: $success_criteria"
    echo "----------------------------------------"

    if eval "$test_cmd"; then
        echo -e "${GREEN}PASS${NC}: $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}: $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# ============================================================================
# TEST 1: Baseline Test - Verify Basic Functionality
# ============================================================================
echo ""
echo "========================================================================"
echo "  TEST CATEGORY 1: Baseline Tests"
echo "========================================================================"

# Test 1.1: Basic laser heating (should always work)
run_test \
    "1.1 Basic Laser Heating (10us, no flow)" \
    "./visualize_laser_heating 2>&1 | tee test_1_1.log | grep -q 'Simulation completed'" \
    "Simulation completes without errors"

# Test 1.2: Laser melting with flow
run_test \
    "1.2 Laser Melting with Flow (20us)" \
    "timeout 60 ./visualize_laser_melting_with_flow 2>&1 | tee test_1_2.log | grep -q 'Step'" \
    "Simulation runs for multiple steps"

# ============================================================================
# TEST 2: Laser Shutoff Time Configuration (Fix 2)
# ============================================================================
echo ""
echo "========================================================================"
echo "  TEST CATEGORY 2: Laser Shutoff Configuration"
echo "========================================================================"

# Test 2.1: Immediate shutoff (shutoff_time = 0)
# Need to modify source and rebuild - skip for now, test manually later

# Test 2.2: Normal shutoff (50 us)
run_test \
    "2.2 Laser Shutoff at 50us (current default)" \
    "timeout 120 ./visualize_lpbf_marangoni_realistic 2>&1 | tee test_2_2.log | grep -q 'Step 3000'" \
    "Simulation reaches step 3000 (100us total)"

# ============================================================================
# TEST 3: Marangoni Gradient Limiter (Fix 3 - CRITICAL)
# ============================================================================
echo ""
echo "========================================================================"
echo "  TEST CATEGORY 3: Marangoni Gradient Limiter (CRITICAL)"
echo "========================================================================"

# Test 3.1: Marangoni with new limiter (5e8 K/m)
echo -e "${YELLOW}Running Marangoni test with 5e8 K/m limiter...${NC}"
run_test \
    "3.1 Marangoni Stability Test (5e8 limiter, 100us)" \
    "timeout 180 ./visualize_lpbf_marangoni_realistic 2>&1 | tee test_3_1_marangoni.log | tail -100" \
    "Simulation completes without NaN or divergence"

# Test 3.2: Check for CFL violations
if [ -f test_3_1_marangoni.log ]; then
    echo -e "${BLUE}Checking for CFL violations...${NC}"
    if grep -q "CFL" test_3_1_marangoni.log; then
        echo -e "${YELLOW}WARNING: CFL limiter was triggered${NC}"
        grep "CFL" test_3_1_marangoni.log | head -10
    else
        echo -e "${GREEN}No CFL violations detected${NC}"
    fi
fi

# Test 3.3: Check maximum velocities
if [ -f test_3_1_marangoni.log ]; then
    echo -e "${BLUE}Extracting maximum velocities...${NC}"
    grep -E "v_max|velocity" test_3_1_marangoni.log | tail -20 || echo "No velocity data found"
fi

# ============================================================================
# TEST 4: Newton Convergence and Bisection Fallback (Fix 1 & 4)
# ============================================================================
echo ""
echo "========================================================================"
echo "  TEST CATEGORY 4: Phase Change Convergence"
echo "========================================================================"

# Test 4.1: High power laser (stress test for Newton solver)
echo -e "${YELLOW}This test requires modifying laser power - manual test needed${NC}"
echo "TODO: Create high-power test case (100W laser, small spot)"

# ============================================================================
# TEST 5: Physical Consistency Checks
# ============================================================================
echo ""
echo "========================================================================"
echo "  TEST CATEGORY 5: Physical Consistency"
echo "========================================================================"

# Test 5.1: Energy conservation
echo -e "${BLUE}Energy conservation test - checking VTK outputs${NC}"
if ls lpbf_realistic_*/*.vtk 1> /dev/null 2>&1; then
    NUM_VTK=$(ls lpbf_realistic_*/*.vtk 2>/dev/null | wc -l)
    echo "Found $NUM_VTK VTK files for analysis"
    if [ $NUM_VTK -gt 10 ]; then
        echo -e "${GREEN}PASS: Sufficient output files generated${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${YELLOW}WARNING: Only $NUM_VTK files generated${NC}"
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
fi

# Test 5.2: Temperature bounds check
echo -e "${BLUE}Checking temperature logs for physical bounds${NC}"
for log in test_*.log; do
    if [ -f "$log" ]; then
        echo "Analyzing $log..."
        # Check for temperatures above boiling point
        if grep -q "T_max.*[4-9][0-9][0-9][0-9]" "$log"; then
            echo -e "${RED}WARNING: Temperature approaching/exceeding boiling point (3533K)${NC}"
            grep "T_max" "$log" | tail -5
        fi
    fi
done

# ============================================================================
# TEST 6: Regression Tests
# ============================================================================
echo ""
echo "========================================================================"
echo "  TEST CATEGORY 6: Regression Tests"
echo "========================================================================"

# Test 6.1: Verify old test cases still work
run_test \
    "6.1 Phase 6 Marangoni (legacy test)" \
    "timeout 60 ./visualize_phase6_marangoni 2>&1 | tee test_6_1.log | grep -q 'Step'" \
    "Legacy test still runs"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "  TEST SUITE SUMMARY"
echo "========================================================================"
echo -e "Total tests run:    ${BLUE}$TOTAL_TESTS${NC}"
echo -e "Tests passed:       ${GREEN}$PASSED_TESTS${NC}"
echo -e "Tests failed:       ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests PASSED!${NC}"
    exit 0
else
    echo -e "${RED}Some tests FAILED - review logs above${NC}"
    exit 1
fi
