#!/bin/bash
#
# Master script to run all velocity diagnostic tests
#
# This script runs tests in sequence and stops at the first failure.
# The first failed test indicates the root cause of the zero-velocity bug.
#

echo "================================================================"
echo "  VELOCITY DIAGNOSTIC TEST SUITE"
echo "  Isolating zero-velocity bug in v5 validation"
echo "================================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test executables (relative to build directory)
BUILD_DIR="/home/yzk/LBMProject/build"

TEST1="${BUILD_DIR}/tests/test1_fluid_velocity_only"
TEST2="${BUILD_DIR}/tests/test2_buoyancy_force"
TEST3="${BUILD_DIR}/tests/test3_darcy_damping"
TEST4="${BUILD_DIR}/tests/test4_thermal_advection_coupling"
TEST5="${BUILD_DIR}/tests/test5_config_flags"

# Check if tests are built
if [ ! -f "$TEST1" ]; then
    echo -e "${RED}ERROR: Tests not built!${NC}"
    echo "Please run 'cd build && make' first."
    exit 1
fi

# Function to run a test and check result
run_test() {
    local test_name=$1
    local test_exe=$2

    echo ""
    echo "================================================================"
    echo "  Running: $test_name"
    echo "================================================================"

    # Run test
    $test_exe
    local result=$?

    if [ $result -eq 0 ]; then
        echo -e "${GREEN}[PASS]${NC} $test_name"
        return 0
    else
        echo -e "${RED}[FAIL]${NC} $test_name"
        echo ""
        echo -e "${YELLOW}STOP: First failed test indicates root cause!${NC}"
        echo ""
        echo "Root cause location: $test_name"
        return 1
    fi
}

# Run tests in sequence
echo "Starting diagnostic sequence..."
echo "Tests will run in order of increasing complexity."
echo "First failure indicates the root cause."
echo ""

# Test 1: Basic FluidLBM force->velocity
run_test "Test 1: Bare FluidLBM Velocity Computation" "$TEST1"
if [ $? -ne 0 ]; then
    echo ""
    echo "DIAGNOSIS: FluidLBM is not computing velocity from forces."
    echo "This is the most basic level - force application is broken."
    echo ""
    echo "ACTION: Fix FluidLBM::collisionBGK() or force kernel."
    exit 1
fi

# Test 2: Buoyancy force computation and application
run_test "Test 2: Buoyancy Force Magnitude" "$TEST2"
if [ $? -ne 0 ]; then
    echo ""
    echo "DIAGNOSIS: Buoyancy force computation or application is broken."
    echo "FluidLBM works, but buoyancy-specific code has issues."
    echo ""
    echo "ACTION: Fix computeBuoyancyForce() or its integration."
    exit 1
fi

# Test 3: Darcy damping isolation
run_test "Test 3: Darcy Damping Isolation" "$TEST3"
if [ $? -ne 0 ]; then
    echo ""
    echo "DIAGNOSIS: Darcy damping is killing flow in liquid regions."
    echo "Formula or liquid_fraction handling is incorrect."
    echo ""
    echo "ACTION: Fix applyDarcyDamping() formula or liquid_fraction field."
    exit 1
fi

# Test 4: Thermal-fluid velocity coupling
run_test "Test 4: Multiphysics Velocity Coupling" "$TEST4"
if [ $? -ne 0 ]; then
    echo ""
    echo "DIAGNOSIS: Velocity field not passed correctly to thermal solver."
    echo "Pointer passing or thermal advection implementation has issues."
    echo ""
    echo "ACTION: Fix velocity pointer passing in MultiphysicsSolver."
    exit 1
fi

# Test 5: Config flags
run_test "Test 5: Config Flag Propagation" "$TEST5"
if [ $? -ne 0 ]; then
    echo ""
    echo "DIAGNOSIS: Configuration flags are disabling physics modules."
    echo "Config file has wrong settings."
    echo ""
    echo "ACTION: Edit config file to enable all required physics."
    exit 1
fi

# All tests passed
echo ""
echo "================================================================"
echo -e "${GREEN}  ALL DIAGNOSTIC TESTS PASSED!${NC}"
echo "================================================================"
echo ""
echo "This is unexpected if v5 test shows zero velocity."
echo ""
echo "Possible explanations:"
echo "1. The bug only appears in the full multiphysics integration"
echo "2. Problem is in a physics module not tested here (e.g., VOF)"
echo "3. Issue is with specific parameter values used in v5 test"
echo "4. Bug is in visualization/output code, not the solver"
echo ""
echo "NEXT STEPS:"
echo "1. Run v5 test again and capture intermediate outputs"
echo "2. Check liquid_fraction field values"
echo "3. Add instrumentation to MultiphysicsSolver::step()"
echo "4. Compare parameter values between these tests and v5 test"
echo ""

exit 0
