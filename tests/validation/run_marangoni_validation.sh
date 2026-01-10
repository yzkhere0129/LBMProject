#!/bin/bash
#
# Marangoni Velocity Validation Test Suite
#
# This script runs the Marangoni velocity test and performs analytical comparison.
#
# Usage:
#   ./run_marangoni_validation.sh
#
# Output:
#   - VTK files in phase6_test2c_visualization/
#   - Analysis results in marangoni_validation_results/
#   - Validation plots and metrics

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build"
TEST_NAME="MarangoniVelocityValidation.RealisticVelocityMagnitude"

echo "========================================"
echo "Marangoni Velocity Validation Suite"
echo "========================================"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory not found: $BUILD_DIR"
    echo "Please build the project first with CMake"
    exit 1
fi

# Change to build directory
cd "$BUILD_DIR"

# Check if test executable exists
TEST_EXEC="./tests/validation/test_marangoni_velocity"
if [ ! -f "$TEST_EXEC" ]; then
    echo "ERROR: Test executable not found: $TEST_EXEC"
    echo "Please build the tests first:"
    echo "  cd $BUILD_DIR"
    echo "  make test_marangoni_velocity"
    exit 1
fi

# Step 1: Run the CUDA test
echo "Step 1: Running CUDA simulation..."
echo "=========================================="
echo ""

# Clean previous output
rm -rf phase6_test2c_visualization
mkdir -p phase6_test2c_visualization

# Run test with filter
$TEST_EXEC --gtest_filter="$TEST_NAME"

TEST_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "Test PASSED (exit code: $TEST_EXIT_CODE)"
else
    echo "Test FAILED (exit code: $TEST_EXIT_CODE)"
    echo "Check output above for errors"
fi
echo "=========================================="
echo ""

# Step 2: Check if VTK files were generated
VTK_COUNT=$(ls -1 phase6_test2c_visualization/marangoni_flow_*.vtk 2>/dev/null | wc -l)
if [ $VTK_COUNT -eq 0 ]; then
    echo "WARNING: No VTK files generated"
    echo "Analysis cannot proceed without simulation data"
    exit 1
fi

echo "Generated $VTK_COUNT VTK files"
echo ""

# Step 3: Run Python analysis
echo "Step 2: Running analytical comparison..."
echo "=========================================="
echo ""

ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyze_marangoni_validation.py"

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "ERROR: Analysis script not found: $ANALYSIS_SCRIPT"
    exit 1
fi

# Check if Python and required packages are available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3"
    exit 1
fi

# Run analysis
python3 "$ANALYSIS_SCRIPT"

ANALYSIS_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "Analysis PASSED (exit code: $ANALYSIS_EXIT_CODE)"
else
    echo "Analysis FAILED (exit code: $ANALYSIS_EXIT_CODE)"
fi
echo "=========================================="
echo ""

# Step 4: Summary
echo "========================================"
echo "VALIDATION SUMMARY"
echo "========================================"
echo ""

if [ -f "marangoni_validation_results/validation_metrics.txt" ]; then
    echo "Validation Metrics:"
    cat marangoni_validation_results/validation_metrics.txt
    echo ""
fi

echo "Output Files:"
echo "  VTK files: ${BUILD_DIR}/phase6_test2c_visualization/"
echo "  Analysis:  ${BUILD_DIR}/marangoni_validation_results/"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ] && [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "OVERALL RESULT: PASS"
    echo ""
    echo "Next steps:"
    echo "  1. Review plots in marangoni_validation_results/"
    echo "  2. Open VTK files in ParaView for visualization"
    echo "  3. Check validation_metrics.txt for detailed error analysis"
    exit 0
else
    echo "OVERALL RESULT: FAIL"
    echo ""
    echo "Debugging suggestions:"
    echo "  1. Check test output for CUDA errors"
    echo "  2. Verify material properties match physical values"
    echo "  3. Review temperature gradient magnitude"
    echo "  4. Check for numerical instabilities"
    exit 1
fi
