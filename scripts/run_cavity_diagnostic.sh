#!/bin/bash
# ============================================================================
# CAVITY DIAGNOSTIC TEST SUITE
# ============================================================================
# Purpose: Isolate the cause of abnormal cavity/depression behind melt pool
#
# Test Matrix:
#   1. Baseline (all effects ON) - reference case
#   2. No solidification shrinkage - test if shrinkage causes cavity
#   3. No evaporation mass loss - test if evaporation causes cavity
#
# Expected Results:
#   - If cavity disappears with no_shrinkage: shrinkage is the cause
#   - If cavity disappears with no_evaporation: evaporation is the cause
#   - If cavity persists in both: other physics (recoil pressure, advection)
#
# Usage:
#   cd /home/yzk/LBMProject/build
#   bash ../scripts/run_cavity_diagnostic.sh
# ============================================================================

set -e

# Configuration
BUILD_DIR="/home/yzk/LBMProject/build"
CONFIG_DIR="/home/yzk/LBMProject/config"
EXECUTABLE="${BUILD_DIR}/visualize_lpbf_scanning"

# Check if executable exists
if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: Executable not found at ${EXECUTABLE}"
    echo "Please build the project first:"
    echo "  cd ${BUILD_DIR} && cmake .. && make -j"
    exit 1
fi

# Function to run a test
run_test() {
    local config_file=$1
    local output_dir=$2
    local description=$3

    echo ""
    echo "============================================================================"
    echo "RUNNING: ${description}"
    echo "  Config: ${config_file}"
    echo "  Output: ${output_dir}"
    echo "============================================================================"
    echo ""

    # Create output directory
    mkdir -p "${BUILD_DIR}/${output_dir}"

    # Run simulation with timeout (10 minutes max)
    timeout 600 "${EXECUTABLE}" "${config_file}" --output "${output_dir}" 2>&1 | tee "${BUILD_DIR}/${output_dir}/simulation.log"

    echo ""
    echo "Test completed: ${description}"
    echo "Results saved to: ${BUILD_DIR}/${output_dir}/"
    echo ""
}

# Function to analyze results
analyze_results() {
    local output_dir=$1
    local log_file="${BUILD_DIR}/${output_dir}/simulation.log"

    if [ ! -f "${log_file}" ]; then
        echo "WARNING: Log file not found for ${output_dir}"
        return
    fi

    echo ""
    echo "=== Analysis: ${output_dir} ==="

    # Extract shrinkage diagnostics
    echo "Shrinkage Summary:"
    grep -E "SHRINKAGE DIAGNOSTIC|Max solidification rate|Expected max df" "${log_file}" | tail -20

    # Extract evaporation diagnostics
    echo ""
    echo "Evaporation Summary:"
    grep -E "EVAPORATION DIAGNOSTIC|Max J_evap|Expected max df" "${log_file}" | tail -20

    # Extract mass conservation
    echo ""
    echo "Mass Conservation:"
    grep -E "total mass|mass delta" "${log_file}" | tail -10
}

# Main execution
echo "============================================================================"
echo "CAVITY DIAGNOSTIC TEST SUITE"
echo "============================================================================"
echo ""
echo "This script runs three test cases to isolate the cause of cavity formation:"
echo "  1. Baseline (all effects enabled)"
echo "  2. No solidification shrinkage"
echo "  3. No evaporation mass loss"
echo ""

# Parse arguments
RUN_ALL=true
RUN_BASELINE=false
RUN_NO_SHRINK=false
RUN_NO_EVAP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-only)
            RUN_ALL=false
            RUN_BASELINE=true
            shift
            ;;
        --no-shrink-only)
            RUN_ALL=false
            RUN_NO_SHRINK=true
            shift
            ;;
        --no-evap-only)
            RUN_ALL=false
            RUN_NO_EVAP=true
            shift
            ;;
        --analyze-only)
            echo "=== Analyzing Previous Results ==="
            analyze_results "lpbf_baseline"
            analyze_results "lpbf_no_shrinkage"
            analyze_results "lpbf_no_evaporation"
            exit 0
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --baseline-only    Run only baseline test"
            echo "  --no-shrink-only   Run only no-shrinkage test"
            echo "  --no-evap-only     Run only no-evaporation test"
            echo "  --analyze-only     Analyze previous results without running"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run tests
if $RUN_ALL || $RUN_BASELINE; then
    run_test "${CONFIG_DIR}/diagnostic_baseline.cfg" "lpbf_baseline" "Baseline (all effects enabled)"
fi

if $RUN_ALL || $RUN_NO_SHRINK; then
    run_test "${CONFIG_DIR}/diagnostic_no_shrinkage.cfg" "lpbf_no_shrinkage" "No Solidification Shrinkage"
fi

if $RUN_ALL || $RUN_NO_EVAP; then
    run_test "${CONFIG_DIR}/diagnostic_no_evaporation.cfg" "lpbf_no_evaporation" "No Evaporation Mass Loss"
fi

# Summary
echo ""
echo "============================================================================"
echo "DIAGNOSTIC TEST SUITE COMPLETED"
echo "============================================================================"
echo ""
echo "Results directories:"
echo "  ${BUILD_DIR}/lpbf_baseline/"
echo "  ${BUILD_DIR}/lpbf_no_shrinkage/"
echo "  ${BUILD_DIR}/lpbf_no_evaporation/"
echo ""
echo "Next steps:"
echo "  1. Open ParaView and load VTK files from each directory"
echo "  2. Compare fill_level field at same timesteps"
echo "  3. Check if cavity exists in each case"
echo ""
echo "Analysis:"
analyze_results "lpbf_baseline"
analyze_results "lpbf_no_shrinkage"
analyze_results "lpbf_no_evaporation"

echo ""
echo "ParaView commands:"
echo "  paraview ${BUILD_DIR}/lpbf_baseline/lpbf_*.vtk &"
echo "  paraview ${BUILD_DIR}/lpbf_no_shrinkage/lpbf_*.vtk &"
echo "  paraview ${BUILD_DIR}/lpbf_no_evaporation/lpbf_*.vtk &"
echo ""
