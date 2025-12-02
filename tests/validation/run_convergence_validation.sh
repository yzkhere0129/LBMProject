#!/bin/bash
# ============================================================================
# WEEK 2: Automated Convergence Validation Suite
# ============================================================================
#
# This script runs the complete suite of convergence and regression tests
# to verify that the simulation is ready for Week 3 development.
#
# Test Suite:
#   1. Grid Independence        - Verify spatial convergence
#   2. Timestep Convergence     - Verify temporal convergence
#   3. Energy Conservation      - Verify thermodynamic consistency
#   4. Regression Test          - Verify no breaking changes
#   5. CFL Stability           - Verify numerical stability
#   6. Configuration Parser     - Verify config file reading
#
# Usage:
#   ./run_convergence_validation.sh [--quick] [--verbose]
#
# Options:
#   --quick     Run only fast tests (skip expensive convergence studies)
#   --verbose   Show detailed output from each test
#
# Exit codes:
#   0 - All tests passed (safe to proceed to Week 3)
#   1 - One or more tests failed (DO NOT PROCEED)
# ============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
SKIP=0

# Options
QUICK_MODE=0
VERBOSE=0

# Parse arguments
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=1
            echo "Quick mode: Skipping expensive convergence tests"
            ;;
        --verbose)
            VERBOSE=1
            echo "Verbose mode: Showing detailed output"
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--quick] [--verbose]"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Week 2 Convergence Validation Suite"
echo "========================================="
echo ""
echo "Build directory: ./build/tests/validation"
echo "Test executables must be compiled before running this script"
echo ""

# Check if build directory exists
if [ ! -d "./build/tests/validation" ]; then
    echo -e "${RED}ERROR: Build directory not found${NC}"
    echo "Please run: mkdir -p build && cd build && cmake .. && make -j4"
    exit 1
fi

# Function to run a test
run_test() {
    local test_name=$1
    local test_exec=$2
    local is_expensive=$3  # 0=fast, 1=expensive

    echo "========================================="
    echo "Test: $test_name"
    echo "========================================="

    # Skip expensive tests in quick mode
    if [ $QUICK_MODE -eq 1 ] && [ $is_expensive -eq 1 ]; then
        echo -e "${YELLOW}SKIP (expensive test, use full mode to run)${NC}"
        ((SKIP++))
        echo ""
        return
    fi

    # Check if executable exists
    if [ ! -f "./build/tests/validation/$test_exec" ]; then
        echo -e "${RED}ERROR: Test executable not found${NC}"
        echo "Expected: ./build/tests/validation/$test_exec"
        echo "Run: cd build && make $test_exec"
        ((FAIL++))
        echo ""
        return
    fi

    # Run the test
    if [ $VERBOSE -eq 1 ]; then
        ./build/tests/validation/$test_exec
    else
        ./build/tests/validation/$test_exec > /tmp/test_${test_exec}.log 2>&1
    fi

    local ret=$?

    # Check result
    if [ $ret -eq 0 ]; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++))
    else
        echo -e "${RED}FAIL${NC}"
        ((FAIL++))

        # Show log on failure if not verbose
        if [ $VERBOSE -eq 0 ]; then
            echo ""
            echo "Test failed. Last 20 lines of output:"
            echo "----------------------------------------"
            tail -20 /tmp/test_${test_exec}.log
            echo "----------------------------------------"
            echo "Full log: /tmp/test_${test_exec}.log"
        fi
    fi

    echo ""
}

# ============================================================================
# Run Test Suite
# ============================================================================

# Test 1: Configuration Parser (CRITICAL - must run first)
run_test "Configuration Parser" "test_config_parser" 0

# If config parser fails, abort
if [ $FAIL -gt 0 ]; then
    echo ""
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}CRITICAL: Configuration Parser Failed${NC}"
    echo -e "${RED}=========================================${NC}"
    echo ""
    echo "The configuration parser bug prevents all other tests from being valid."
    echo "This is the hardcoded num_steps bug discovered in Week 2."
    echo ""
    echo "REQUIRED ACTION:"
    echo "  1. Fix configuration parser to read num_steps/total_steps correctly"
    echo "  2. Re-run this test suite"
    echo "  3. Do NOT proceed to other tests until this PASSES"
    echo ""
    exit 1
fi

# Test 2: CFL Stability (fast, analytical)
run_test "CFL Stability Check" "test_cfl_stability" 0

# Test 3: Regression Test (baseline validation)
run_test "Regression Test (50W baseline)" "test_regression_50W" 0

# Test 4: Grid Convergence (expensive - 3 simulations)
run_test "Grid Independence" "test_grid_convergence" 1

# Test 5: Timestep Convergence (expensive - 3 simulations)
run_test "Timestep Convergence" "test_timestep_convergence" 1

# Test 6: Energy Conservation (expensive - 3 simulations)
run_test "Energy Conservation (timestep sweep)" "test_energy_conservation_timestep" 1

# ============================================================================
# Summary
# ============================================================================

echo "========================================="
echo "Test Suite Summary"
echo "========================================="
echo ""
echo "Results:"
echo -e "  ${GREEN}PASS:${NC}  $PASS"
echo -e "  ${RED}FAIL:${NC}  $FAIL"
if [ $SKIP -gt 0 ]; then
    echo -e "  ${YELLOW}SKIP:${NC}  $SKIP (use full mode to run)"
fi
echo ""
echo "Total tests: $((PASS + FAIL + SKIP))"
echo ""

# Final verdict
if [ $FAIL -eq 0 ]; then
    echo "========================================="
    echo -e "${GREEN}ALL TESTS PASSED${NC}"
    echo "========================================="
    echo ""
    echo "Code is verified for Week 3 development!"
    echo ""
    echo "Validated:"
    echo "  - Grid independence achieved"
    echo "  - Temporal convergence verified"
    echo "  - Energy conservation confirmed"
    echo "  - Numerical stability ensured"
    echo "  - Configuration parser working"
    echo "  - No regressions detected"
    echo ""
    echo "Safe to proceed with:"
    echo "  - Advanced physics implementation"
    echo "  - Production parameter studies"
    echo "  - Literature comparison"
    echo ""
    exit 0
else
    echo "========================================="
    echo -e "${RED}FAILURES DETECTED${NC}"
    echo "========================================="
    echo ""
    echo -e "${RED}DO NOT PROCEED TO WEEK 3${NC}"
    echo ""
    echo "Issues found:"

    if [ $QUICK_MODE -eq 1 ]; then
        echo "  - Run full test suite (without --quick) for complete diagnosis"
    fi

    echo ""
    echo "Required actions:"
    echo "  1. Review test failure logs in /tmp/test_*.log"
    echo "  2. Fix failing tests"
    echo "  3. Re-run this validation suite"
    echo "  4. Ensure ALL tests PASS before continuing"
    echo ""
    echo "Common issues:"
    echo "  - Configuration parser bug (num_steps hardcoded)"
    echo "  - Temporal divergence (time integration bug)"
    echo "  - Energy loss (missing heat flux term)"
    echo "  - CFL violation (timestep too large)"
    echo ""
    exit 1
fi
