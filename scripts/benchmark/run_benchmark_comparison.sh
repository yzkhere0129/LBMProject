#!/bin/bash
#
# Benchmark Comparison Script: LBM-CUDA vs waLBerla
#
# This script runs validation tests and compares results between
# LBM-CUDA and waLBerla frameworks.
#
# Usage:
#   ./run_benchmark_comparison.sh [test_name] [options]
#
# Examples:
#   ./run_benchmark_comparison.sh pure_conduction
#   ./run_benchmark_comparison.sh stefan --walberla
#   ./run_benchmark_comparison.sh all
#
# Author: LBM-CUDA Validation Team
# Date: 2025-11-22

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

LBMPROJECT_DIR="/home/yzk/LBMProject"
WALBERLA_DIR="/home/yzk/walberla"
BUILD_DIR="${LBMPROJECT_DIR}/build"
RESULTS_DIR="${LBMPROJECT_DIR}/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default parameters
TEST_NAME="${1:-all}"
INCLUDE_WALBERLA=false
GRID_SIZE=2.0  # micrometers

# Parse options
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --walberla)
            INCLUDE_WALBERLA=true
            shift
            ;;
        --grid)
            GRID_SIZE=$2
            shift 2
            ;;
        --help)
            echo "Usage: $0 [test_name] [options]"
            echo ""
            echo "Tests:"
            echo "  pure_conduction  - 1D heat diffusion (analytical)"
            echo "  grid_convergence - Grid independence study"
            echo "  stefan           - Phase change (moving boundary)"
            echo "  gaussian_source  - Laser heat source"
            echo "  all              - Run all tests"
            echo ""
            echo "Options:"
            echo "  --walberla       Include waLBerla comparison"
            echo "  --grid SIZE      Grid size in micrometers (default: 2.0)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

echo "============================================================"
echo "LBM-CUDA Benchmark Comparison Suite"
echo "============================================================"
echo "Date: $(date)"
echo "Test: ${TEST_NAME}"
echo "Grid size: ${GRID_SIZE} um"
echo "Include waLBerla: ${INCLUDE_WALBERLA}"
echo "Results directory: ${RESULTS_DIR}/${TIMESTAMP}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}/${TIMESTAMP}"
cd "${RESULTS_DIR}/${TIMESTAMP}"

# Log file
LOG_FILE="benchmark_${TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# ============================================================================
# Check Prerequisites
# ============================================================================

echo "Checking prerequisites..."

# Check LBM-CUDA build
if [ ! -d "${BUILD_DIR}" ]; then
    echo "ERROR: LBM-CUDA build directory not found: ${BUILD_DIR}"
    echo "Please run: cd ${LBMPROJECT_DIR} && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

# Check for validation test executables
VALIDATION_TESTS=(
    "test_pure_conduction"
    "test_stefan_problem"
    "test_grid_convergence"
    "test_thermal_lbm"
    "test_laser_source"
)

for test in "${VALIDATION_TESTS[@]}"; do
    if [ -f "${BUILD_DIR}/${test}" ]; then
        echo "  Found: ${test}"
    else
        echo "  Missing: ${test} (will skip if needed)"
    fi
done

# Check waLBerla if requested
if [ "${INCLUDE_WALBERLA}" = true ]; then
    if [ ! -d "${WALBERLA_DIR}/build" ]; then
        echo "WARNING: waLBerla build not found. Skipping waLBerla comparison."
        INCLUDE_WALBERLA=false
    else
        echo "  waLBerla build found"
    fi
fi

echo ""

# ============================================================================
# Test Functions
# ============================================================================

run_pure_conduction() {
    echo "============================================================"
    echo "TEST 1: Pure 1D Heat Conduction"
    echo "============================================================"

    local test_exe="${BUILD_DIR}/test_pure_conduction"

    if [ -f "${test_exe}" ]; then
        echo "Running LBM-CUDA test..."
        "${test_exe}" 2>&1 | tee pure_conduction_lbmcuda.log

        # Extract results
        if grep -q "PASS" pure_conduction_lbmcuda.log; then
            echo "LBM-CUDA: PASS"
        else
            echo "LBM-CUDA: FAIL or INCOMPLETE"
        fi
    else
        echo "Executable not found: ${test_exe}"
        echo "Running Python validation instead..."
        python3 "${LBMPROJECT_DIR}/scripts/benchmark/compare_thermal_solutions.py" \
            --test pure_conduction \
            --grid "${GRID_SIZE}" \
            --output . \
            --json-output pure_conduction_results.json
    fi

    if [ "${INCLUDE_WALBERLA}" = true ]; then
        echo ""
        echo "Running waLBerla DiffusionTest..."
        local walberla_exe="${WALBERLA_DIR}/build/tests/lbm/DiffusionTest"
        if [ -f "${walberla_exe}" ]; then
            # Compute parameters
            local alpha="9.05e-6"  # Ti6Al4V thermal diffusivity
            local dx="${GRID_SIZE}e-6"
            local dt=$(python3 -c "print(0.1 * (${GRID_SIZE}e-6)**2 / ${alpha})")

            "${walberla_exe}" \
                -d "${alpha}" \
                -dx "${dx}" \
                -dt "${dt}" \
                -t 0.001 \
                --quiet \
                2>&1 | tee pure_conduction_walberla.log

            echo "waLBerla: Complete"
        else
            echo "waLBerla DiffusionTest not found"
        fi
    fi

    echo ""
}

run_grid_convergence() {
    echo "============================================================"
    echo "TEST 2: Grid Convergence Study"
    echo "============================================================"

    local test_exe="${BUILD_DIR}/test_grid_convergence"

    if [ -f "${test_exe}" ]; then
        echo "Running LBM-CUDA grid convergence..."
        "${test_exe}" 2>&1 | tee grid_convergence.log
    else
        echo "Executable not found, using Python implementation..."
        python3 "${LBMPROJECT_DIR}/scripts/benchmark/compare_thermal_solutions.py" \
            --test grid_convergence \
            --output . \
            --json-output grid_convergence_results.json
    fi

    echo ""
}

run_stefan_problem() {
    echo "============================================================"
    echo "TEST 3: Stefan Problem (Phase Change)"
    echo "============================================================"

    local test_exe="${BUILD_DIR}/test_stefan_problem"

    if [ -f "${test_exe}" ]; then
        echo "Running LBM-CUDA Stefan problem..."
        "${test_exe}" 2>&1 | tee stefan_problem.log

        if grep -q "PASS" stefan_problem.log; then
            echo "LBM-CUDA: PASS"
        else
            echo "LBM-CUDA: FAIL or INCOMPLETE"
        fi
    else
        echo "Executable not found, using Python implementation..."
        python3 "${LBMPROJECT_DIR}/scripts/benchmark/compare_thermal_solutions.py" \
            --test stefan \
            --output . \
            --json-output stefan_results.json
    fi

    echo ""
}

run_gaussian_source() {
    echo "============================================================"
    echo "TEST 4: Gaussian Heat Source"
    echo "============================================================"

    local test_exe="${BUILD_DIR}/test_laser_source"

    if [ -f "${test_exe}" ]; then
        echo "Running LBM-CUDA laser source tests..."
        "${test_exe}" 2>&1 | tee laser_source.log

        # Count passed tests
        local passed=$(grep -c "PASSED" laser_source.log || echo 0)
        local failed=$(grep -c "FAILED" laser_source.log || echo 0)
        echo "LBM-CUDA: ${passed} passed, ${failed} failed"
    else
        echo "Executable not found: ${test_exe}"
    fi

    echo ""
}

# ============================================================================
# Run Tests
# ============================================================================

case "${TEST_NAME}" in
    pure_conduction)
        run_pure_conduction
        ;;
    grid_convergence)
        run_grid_convergence
        ;;
    stefan)
        run_stefan_problem
        ;;
    gaussian|gaussian_source)
        run_gaussian_source
        ;;
    all)
        run_pure_conduction
        run_grid_convergence
        run_stefan_problem
        run_gaussian_source
        ;;
    *)
        echo "Unknown test: ${TEST_NAME}"
        echo "Valid tests: pure_conduction, grid_convergence, stefan, gaussian, all"
        exit 1
        ;;
esac

# ============================================================================
# Generate Summary Report
# ============================================================================

echo "============================================================"
echo "BENCHMARK SUMMARY REPORT"
echo "============================================================"
echo ""

# Collect results
echo "Test Results:"
echo "-------------"

if [ -f "pure_conduction_lbmcuda.log" ] || [ -f "pure_conduction_results.json" ]; then
    echo -n "  Pure Conduction: "
    if grep -q "PASS" pure_conduction_lbmcuda.log 2>/dev/null; then
        echo "PASS"
    elif grep -q "FAIL" pure_conduction_lbmcuda.log 2>/dev/null; then
        echo "FAIL"
    else
        echo "RUN (see logs)"
    fi
fi

if [ -f "grid_convergence.log" ] || [ -f "grid_convergence_results.json" ]; then
    echo -n "  Grid Convergence: "
    if grep -q "PASS" grid_convergence.log 2>/dev/null; then
        echo "PASS"
    elif grep -q "average_order" grid_convergence_results.json 2>/dev/null; then
        order=$(python3 -c "import json; d=json.load(open('grid_convergence_results.json')); print(f\"{d['grid_convergence']['average_order']:.2f}\")")
        echo "Order = ${order}"
    else
        echo "RUN (see logs)"
    fi
fi

if [ -f "stefan_problem.log" ] || [ -f "stefan_results.json" ]; then
    echo -n "  Stefan Problem: "
    if grep -q "PASS" stefan_problem.log 2>/dev/null; then
        echo "PASS"
    elif grep -q "FAIL" stefan_problem.log 2>/dev/null; then
        echo "FAIL"
    else
        echo "RUN (see logs)"
    fi
fi

if [ -f "laser_source.log" ]; then
    passed=$(grep -c "PASSED" laser_source.log 2>/dev/null || echo 0)
    failed=$(grep -c "FAILED" laser_source.log 2>/dev/null || echo 0)
    echo "  Laser Source: ${passed} passed, ${failed} failed"
fi

echo ""
echo "Output files:"
echo "  ${RESULTS_DIR}/${TIMESTAMP}/"
ls -la

echo ""
echo "============================================================"
echo "Benchmark completed at $(date)"
echo "============================================================"
