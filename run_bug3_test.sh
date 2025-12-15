#!/bin/bash
#
# run_bug3_test.sh
#
# Convenience script to run Bug 3 regression test
#
# Purpose: Verify that energy conservation diagnostics work correctly
# across different timestep sizes and follow normal convergence behavior
# (fine timestep → best accuracy).
#
# Runtime: ~3-5 minutes (3 short simulations of 20μs each)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                   BUG 3 REGRESSION TEST - RUNNER SCRIPT${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Test Purpose:"
echo "  Verify energy diagnostic calculation handles different timestep sizes"
echo "  correctly and follows normal convergence (fine timestep → best accuracy)."
echo ""
echo "Bug 3 Summary:"
echo "  - BEFORE FIX: Fine timestep had WORST error (22.8%) - paradoxical!"
echo "  - AFTER FIX:  Fine timestep should have BEST error - normal convergence"
echo ""
echo "Test Configuration:"
echo "  - 3 simulations with dt = 0.2, 0.1, 0.05 μs"
echo "  - Each simulation runs for 20 μs (100-400 steps)"
echo "  - Small domain (50×50×25 cells) for speed"
echo "  - Pure storage scenario (no heat sinks enabled)"
echo ""
echo "Expected Runtime: ~3-5 minutes"
echo ""
echo -e "${BLUE}-------------------------------------------------------------------------------${NC}"

# Navigate to build directory
cd "$(dirname "$0")/build"

# Check if test executable exists, build if not
if [ ! -f "tests/validation/test_bug3_energy_diagnostic" ]; then
    echo -e "${YELLOW}Test executable not found. Building...${NC}"
    echo ""

    # Run cmake if needed
    if [ ! -f "Makefile" ]; then
        echo "Running cmake..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
        echo ""
    fi

    # Build the test
    echo "Compiling test_bug3_energy_diagnostic..."
    make test_bug3_energy_diagnostic -j8
    echo ""

    if [ ! -f "tests/validation/test_bug3_energy_diagnostic" ]; then
        echo -e "${RED}ERROR: Failed to build test_bug3_energy_diagnostic${NC}"
        exit 1
    fi

    echo -e "${GREEN}Build successful!${NC}"
    echo ""
fi

# Run the test
echo -e "${BLUE}-------------------------------------------------------------------------------${NC}"
echo -e "${BLUE}                           RUNNING BUG 3 TEST${NC}"
echo -e "${BLUE}-------------------------------------------------------------------------------${NC}"
echo ""

./tests/validation/test_bug3_energy_diagnostic

TEST_EXIT_CODE=$?

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                              TEST COMPLETE${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓✓✓ BUG 3 TEST PASSED ✓✓✓${NC}"
    echo ""
    echo "Energy diagnostic is working correctly:"
    echo "  - All simulations completed successfully"
    echo "  - Energy errors are reasonable (<20%)"
    echo "  - Fine timestep has BEST accuracy (Bug 3 FIXED)"
    echo "  - Normal convergence behavior observed"
    echo ""
    echo "Bug 3 will not regress."
    echo ""
    exit 0
else
    echo -e "${RED}✗✗✗ BUG 3 TEST FAILED ✗✗✗${NC}"
    echo ""
    echo "Energy diagnostic may have issues!"
    echo ""
    echo "Possible problems:"
    echo "  - Simulations failed to complete"
    echo "  - Energy errors exceed 20% threshold"
    echo "  - Fine timestep does NOT have best accuracy (Bug 3 present!)"
    echo ""
    echo "Action required:"
    echo "  1. Review energy diagnostic calculation code"
    echo "  2. Check dt scaling in computeTotalThermalEnergy()"
    echo "  3. Verify dE/dt calculation uses correct dt value"
    echo "  4. Review test output above for specific failures"
    echo ""
    exit 1
fi
