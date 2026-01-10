#!/bin/bash
#
# Run Phase 1 Fluid Validation Suite
#
# This script executes all fluid validation analysis scripts and generates
# a comprehensive validation report.
#
# Usage:
#   ./run_fluid_validation.sh [options]
#
# Options:
#   --taylor-green <dir>   Taylor-Green VTK directory
#   --cavity-re100 <dir>   Cavity Re=100 VTK directory
#   --cavity-re400 <dir>   Cavity Re=400 VTK directory
#   --grid-study <dirs>    Grid convergence VTK directories (space-separated)
#   --output <dir>         Output directory for results
#   --help                 Show this help message
#

set -e  # Exit on error

# Default paths (adjust based on your setup)
TAYLOR_GREEN_DIR="tests/validation/output_taylor_green"
CAVITY_RE100_DIR="tests/validation/output_cavity_re100"
CAVITY_RE400_DIR="tests/validation/output_cavity_re400"
GRID_STUDY_DIRS=""
OUTPUT_DIR="scripts/validation/results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --taylor-green)
            TAYLOR_GREEN_DIR="$2"
            shift 2
            ;;
        --cavity-re100)
            CAVITY_RE100_DIR="$2"
            shift 2
            ;;
        --cavity-re400)
            CAVITY_RE400_DIR="$2"
            shift 2
            ;;
        --grid-study)
            GRID_STUDY_DIRS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            grep '^#' "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}         Phase 1 Fluid Validation Suite                                ${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo ""

# Track pass/fail status
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to run a test
run_test() {
    local test_name=$1
    local script=$2
    shift 2
    local args=("$@")

    echo -e "\n${YELLOW}>>> Running: ${test_name}${NC}"
    echo "Command: python $script ${args[@]}"

    if python "$script" "${args[@]}"; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Function to skip a test
skip_test() {
    local test_name=$1
    local reason=$2

    echo -e "\n${YELLOW}>>> Skipping: ${test_name}${NC}"
    echo -e "${YELLOW}Reason: ${reason}${NC}"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
}

# =========================================================================
# Test 1: Taylor-Green Vortex
# =========================================================================

if [ -d "$TAYLOR_GREEN_DIR" ]; then
    run_test "Taylor-Green Vortex" \
        "scripts/validation/fluid_taylor_green_analysis.py" \
        --vtk "$TAYLOR_GREEN_DIR" \
        --output "$OUTPUT_DIR"
else
    skip_test "Taylor-Green Vortex" \
        "Directory not found: $TAYLOR_GREEN_DIR"
fi

# =========================================================================
# Test 2: Lid-Driven Cavity Re=100
# =========================================================================

if [ -d "$CAVITY_RE100_DIR" ]; then
    run_test "Lid-Driven Cavity Re=100" \
        "scripts/validation/fluid_lid_driven_cavity_analysis.py" \
        --vtk "$CAVITY_RE100_DIR" \
        --re 100 \
        --output "$OUTPUT_DIR"
else
    skip_test "Lid-Driven Cavity Re=100" \
        "Directory not found: $CAVITY_RE100_DIR"
fi

# =========================================================================
# Test 3: Lid-Driven Cavity Re=400
# =========================================================================

if [ -d "$CAVITY_RE400_DIR" ]; then
    run_test "Lid-Driven Cavity Re=400" \
        "scripts/validation/fluid_lid_driven_cavity_analysis.py" \
        --vtk "$CAVITY_RE400_DIR" \
        --re 400 \
        --output "$OUTPUT_DIR"
else
    skip_test "Lid-Driven Cavity Re=400" \
        "Directory not found: $CAVITY_RE400_DIR"
fi

# =========================================================================
# Test 4: Grid Convergence Study (optional)
# =========================================================================

if [ -n "$GRID_STUDY_DIRS" ]; then
    # Check if directories exist
    ALL_EXIST=true
    for dir in $GRID_STUDY_DIRS; do
        if [ ! -d "$dir" ]; then
            ALL_EXIST=false
            break
        fi
    done

    if [ "$ALL_EXIST" = true ]; then
        run_test "Grid Convergence Study" \
            "scripts/validation/fluid_grid_convergence_analysis.py" \
            $GRID_STUDY_DIRS \
            --output "$OUTPUT_DIR"
    else
        skip_test "Grid Convergence Study" \
            "One or more directories not found: $GRID_STUDY_DIRS"
    fi
else
    skip_test "Grid Convergence Study" \
        "No grid study directories specified (use --grid-study)"
fi

# =========================================================================
# Summary
# =========================================================================

echo ""
echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}                   Validation Summary                                  ${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))

if [ $TESTS_PASSED -gt 0 ]; then
    echo -e "${GREEN}Passed:  ${TESTS_PASSED}${NC}"
fi

if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed:  ${TESTS_FAILED}${NC}"
fi

if [ $TESTS_SKIPPED -gt 0 ]; then
    echo -e "${YELLOW}Skipped: ${TESTS_SKIPPED}${NC}"
fi

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Generate combined report
REPORT_FILE="$OUTPUT_DIR/fluid_validation_report.txt"
echo "Fluid Validation Report" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "=======================================================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Test Results:" >> "$REPORT_FILE"
echo "  Passed:  $TESTS_PASSED" >> "$REPORT_FILE"
echo "  Failed:  $TESTS_FAILED" >> "$REPORT_FILE"
echo "  Skipped: $TESTS_SKIPPED" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Append individual test summaries if they exist
for summary in "$OUTPUT_DIR"/*.txt; do
    if [ -f "$summary" ] && [ "$summary" != "$REPORT_FILE" ]; then
        echo "=======================================================================" >> "$REPORT_FILE"
        echo "File: $(basename $summary)" >> "$REPORT_FILE"
        echo "=======================================================================" >> "$REPORT_FILE"
        cat "$summary" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
done

echo "Combined report: $REPORT_FILE"

# =========================================================================
# Exit with appropriate code
# =========================================================================

if [ $TESTS_FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}Some tests failed. See output above for details.${NC}"
    exit 1
elif [ $TOTAL_TESTS -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}No tests were run. Check VTK directory paths.${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
