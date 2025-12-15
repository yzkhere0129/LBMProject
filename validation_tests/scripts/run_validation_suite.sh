#!/bin/bash

################################################################################
# Marangoni Limiter Removal - Automated Validation Suite
#
# Purpose: Systematically test removal of gradient and CFL limiters
# Strategy: 6-stage progressive validation with automatic rollback
#
# Test Stages:
#   1. BASELINE      - Current code (reference)
#   2. GRAD-2X       - Gradient limit: 5e8 -> 1e9 K/m
#   3. GRAD-10X      - Gradient limit: 5e8 -> 5e9 K/m
#   4. GRAD-REMOVED  - Remove gradient limiter entirely
#   5. CFL-REMOVED   - Remove CFL limiter (keep gradient removed)
#   6. LONG-STABILITY - Extended run with all limiters removed
#
# Safety: Automatic backup/restore of source files
################################################################################

set -e  # Exit on error (disabled during tests to continue suite)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/yzk/LBMProject"
BUILD_DIR="$PROJECT_ROOT/build"
SRC_DIR="$PROJECT_ROOT/src"
VALIDATION_DIR="$PROJECT_ROOT/validation_tests"
DATA_DIR="$VALIDATION_DIR/data"
REPORT_DIR="$VALIDATION_DIR/reports"
BACKUP_DIR="$VALIDATION_DIR/backups"

# Source files to modify
MARANGONI_SRC="$SRC_DIR/physics/vof/marangoni.cu"
MULTIPHYSICS_SRC="$SRC_DIR/physics/multiphysics/multiphysics_solver.cu"

# Executable
EXECUTABLE="$BUILD_DIR/visualize_lpbf_scanning"

# Test configuration
TEST_TIMEOUT=1800  # 30 minutes per test
SHORT_STEPS=1000   # Steps for quick tests
LONG_STEPS=6000    # Steps for stability test

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================================
# Color output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Backup and Restore Functions
# ============================================================================

backup_sources() {
    log_info "Backing up source files..."

    if [ ! -f "$MARANGONI_SRC" ]; then
        log_error "Source file not found: $MARANGONI_SRC"
        exit 1
    fi

    if [ ! -f "$MULTIPHYSICS_SRC" ]; then
        log_error "Source file not found: $MULTIPHYSICS_SRC"
        exit 1
    fi

    cp "$MARANGONI_SRC" "$BACKUP_DIR/marangoni.cu.backup_$TIMESTAMP"
    cp "$MULTIPHYSICS_SRC" "$BACKUP_DIR/multiphysics_solver.cu.backup_$TIMESTAMP"

    log_success "Backup created: $TIMESTAMP"
}

restore_sources() {
    log_info "Restoring source files from backup..."

    if [ -f "$BACKUP_DIR/marangoni.cu.backup_$TIMESTAMP" ]; then
        cp "$BACKUP_DIR/marangoni.cu.backup_$TIMESTAMP" "$MARANGONI_SRC"
    fi

    if [ -f "$BACKUP_DIR/multiphysics_solver.cu.backup_$TIMESTAMP" ]; then
        cp "$BACKUP_DIR/multiphysics_solver.cu.backup_$TIMESTAMP" "$MULTIPHYSICS_SRC"
    fi

    log_success "Source files restored"
}

# Trap to ensure cleanup on exit or Ctrl+C
cleanup_on_exit() {
    log_warning "Caught interrupt or exit - restoring source files..."
    restore_sources
    log_info "Validation suite terminated"
    exit 1
}

trap cleanup_on_exit INT TERM EXIT

# ============================================================================
# Source Code Modification Functions
# ============================================================================

apply_grad_2x() {
    log_info "Modifying source: GRAD-2X (5e8 -> 1e9 K/m)"

    # Line 31: Change MAX_PHYSICAL_GRAD_T constant
    sed -i '31s/constexpr float MAX_PHYSICAL_GRAD_T = 5\.0e8f;/constexpr float MAX_PHYSICAL_GRAD_T = 1.0e9f;/' "$MARANGONI_SRC"

    # Verify modification
    if grep -q "MAX_PHYSICAL_GRAD_T = 1.0e9f" "$MARANGONI_SRC"; then
        log_success "GRAD-2X modification applied"
        return 0
    else
        log_error "GRAD-2X modification failed"
        return 1
    fi
}

apply_grad_10x() {
    log_info "Modifying source: GRAD-10X (5e8 -> 5e9 K/m)"

    sed -i '31s/constexpr float MAX_PHYSICAL_GRAD_T = 5\.0e8f;/constexpr float MAX_PHYSICAL_GRAD_T = 5.0e9f;/' "$MARANGONI_SRC"

    if grep -q "MAX_PHYSICAL_GRAD_T = 5.0e9f" "$MARANGONI_SRC"; then
        log_success "GRAD-10X modification applied"
        return 0
    else
        log_error "GRAD-10X modification failed"
        return 1
    fi
}

apply_grad_removed() {
    log_info "Modifying source: GRAD-REMOVED (commenting out limiter)"

    # Comment out lines 109-114 (first limiter block)
    sed -i '109,114s/^/\/\/ /' "$MARANGONI_SRC"

    # Comment out lines 199-204 (second limiter block)
    sed -i '199,204s/^/\/\/ /' "$MARANGONI_SRC"

    if grep -q "\/\/ *if (grad_T_mag > MAX_PHYSICAL_GRAD_T)" "$MARANGONI_SRC"; then
        log_success "GRAD-REMOVED modification applied"
        return 0
    else
        log_error "GRAD-REMOVED modification failed"
        return 1
    fi
}

apply_cfl_removed() {
    log_info "Modifying source: CFL-REMOVED (commenting out CFL limiter)"

    # First apply gradient removal
    apply_grad_removed || return 1

    # Comment out lines 624-636 (CFL limiter kernel call)
    sed -i '624,636s/^/\/\/ /' "$MULTIPHYSICS_SRC"

    if grep -q "\/\/ *limitForcesByCFL_kernel" "$MULTIPHYSICS_SRC"; then
        log_success "CFL-REMOVED modification applied"
        return 0
    else
        log_error "CFL-REMOVED modification failed"
        return 1
    fi
}

# ============================================================================
# Build Function
# ============================================================================

rebuild_project() {
    log_info "Rebuilding project..."

    cd "$BUILD_DIR" || exit 1

    # Run CMake and make
    if ! cmake .. > "$DATA_DIR/build.log" 2>&1; then
        log_error "CMake failed - see $DATA_DIR/build.log"
        cat "$DATA_DIR/build.log"
        return 1
    fi

    if ! make -j8 >> "$DATA_DIR/build.log" 2>&1; then
        log_error "Make failed - see $DATA_DIR/build.log"
        tail -50 "$DATA_DIR/build.log"
        return 1
    fi

    if [ ! -f "$EXECUTABLE" ]; then
        log_error "Executable not found after build: $EXECUTABLE"
        return 1
    fi

    log_success "Build completed successfully"
    return 0
}

# ============================================================================
# Test Execution Function
# ============================================================================

run_test() {
    local test_name=$1
    local num_steps=$2
    local output_file="$DATA_DIR/${test_name}_output.log"

    log_info "Running test: $test_name ($num_steps steps, timeout ${TEST_TIMEOUT}s)"

    cd "$BUILD_DIR" || return 1

    # Run with timeout
    if timeout $TEST_TIMEOUT "$EXECUTABLE" > "$output_file" 2>&1; then
        log_success "Test $test_name completed successfully"
        return 0
    else
        local exit_code=$?

        if [ $exit_code -eq 124 ]; then
            log_error "Test $test_name TIMED OUT after ${TEST_TIMEOUT}s"
            echo "TIMEOUT: Test exceeded ${TEST_TIMEOUT} seconds" >> "$output_file"
        else
            log_error "Test $test_name FAILED with exit code $exit_code"
        fi

        # Check for CUDA errors
        if grep -q "CUDA error" "$output_file"; then
            log_error "CUDA error detected in output"
            grep "CUDA error" "$output_file"
        fi

        # Check for NaN
        if grep -q "NaN" "$output_file"; then
            log_error "NaN detected in output"
        fi

        return 1
    fi
}

# ============================================================================
# Validation Function
# ============================================================================

validate_test() {
    local test_name=$1
    local log_file="$DATA_DIR/${test_name}_output.log"
    local report_file="$REPORT_DIR/${test_name}_validation.txt"

    log_info "Validating test results: $test_name"

    # Expected velocity ranges (mm/s) based on test
    local v_min=0
    local v_max=10000

    case $test_name in
        "BASELINE")
            v_min=0
            v_max=20  # Expect ~7 mm/s
            ;;
        "GRAD-2X")
            v_min=0
            v_max=50  # Expect modest increase
            ;;
        "GRAD-10X")
            v_min=0
            v_max=100  # Expect larger increase
            ;;
        "GRAD-REMOVED"|"CFL-REMOVED"|"LONG-STABILITY")
            v_min=50
            v_max=500  # Expect 50-500 mm/s (LPBF literature)
            ;;
    esac

    # Call Python validation script
    python3 "$SCRIPT_DIR/validate_results.py" \
        --log-file "$log_file" \
        --test-name "$test_name" \
        --v-min "$v_min" \
        --v-max "$v_max" \
        --output "$report_file"

    local validation_result=$?

    if [ $validation_result -eq 0 ]; then
        log_success "Validation PASSED for $test_name"
        return 0
    else
        log_error "Validation FAILED for $test_name"
        cat "$report_file"
        return 1
    fi
}

# ============================================================================
# Main Test Execution
# ============================================================================

run_test_stage() {
    local test_name=$1
    local modify_func=$2
    local num_steps=$3

    echo ""
    echo "========================================================================"
    echo "  TEST STAGE: $test_name"
    echo "========================================================================"
    echo ""

    # Restore to baseline first
    restore_sources

    # Apply modification (if not baseline)
    if [ "$modify_func" != "none" ]; then
        if ! $modify_func; then
            log_error "Failed to apply modifications for $test_name"
            return 1
        fi
    fi

    # Rebuild
    if ! rebuild_project; then
        log_error "Build failed for $test_name"
        return 1
    fi

    # Run test
    if ! run_test "$test_name" "$num_steps"; then
        log_error "Test execution failed for $test_name"
        return 1
    fi

    # Validate
    if ! validate_test "$test_name"; then
        log_warning "Validation failed for $test_name (continuing suite)"
        return 1
    fi

    log_success "Test stage $test_name completed"
    return 0
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo ""
    echo "========================================================================"
    echo "  Marangoni Limiter Removal - Automated Validation Suite"
    echo "========================================================================"
    echo "  Timestamp: $TIMESTAMP"
    echo "  Project: $PROJECT_ROOT"
    echo "  Build: $BUILD_DIR"
    echo "========================================================================"
    echo ""

    # Check prerequisites
    if [ ! -f "$MARANGONI_SRC" ]; then
        log_error "Source file not found: $MARANGONI_SRC"
        exit 1
    fi

    if [ ! -f "$MULTIPHYSICS_SRC" ]; then
        log_error "Source file not found: $MULTIPHYSICS_SRC"
        exit 1
    fi

    if [ ! -d "$BUILD_DIR" ]; then
        log_error "Build directory not found: $BUILD_DIR"
        exit 1
    fi

    # Create output directories
    mkdir -p "$DATA_DIR" "$REPORT_DIR" "$BACKUP_DIR"

    # Backup original sources
    backup_sources

    # Track results
    declare -A test_results

    # Run test stages
    run_test_stage "BASELINE" "none" $SHORT_STEPS
    test_results["BASELINE"]=$?

    run_test_stage "GRAD-2X" "apply_grad_2x" $SHORT_STEPS
    test_results["GRAD-2X"]=$?

    run_test_stage "GRAD-10X" "apply_grad_10x" $SHORT_STEPS
    test_results["GRAD-10X"]=$?

    run_test_stage "GRAD-REMOVED" "apply_grad_removed" $SHORT_STEPS
    test_results["GRAD-REMOVED"]=$?

    run_test_stage "CFL-REMOVED" "apply_cfl_removed" $SHORT_STEPS
    test_results["CFL-REMOVED"]=$?

    run_test_stage "LONG-STABILITY" "apply_cfl_removed" $LONG_STEPS
    test_results["LONG-STABILITY"]=$?

    # Restore sources
    restore_sources

    # Generate final report
    echo ""
    echo "========================================================================"
    log_info "Generating final report..."
    echo "========================================================================"

    python3 "$SCRIPT_DIR/generate_report.py" \
        --data-dir "$DATA_DIR" \
        --report-dir "$REPORT_DIR" \
        --output "$REPORT_DIR/validation_summary.md"

    # Display summary
    echo ""
    echo "========================================================================"
    echo "  VALIDATION SUITE SUMMARY"
    echo "========================================================================"
    echo ""

    local total_tests=0
    local passed_tests=0

    for test_name in BASELINE GRAD-2X GRAD-10X GRAD-REMOVED CFL-REMOVED LONG-STABILITY; do
        total_tests=$((total_tests + 1))
        if [ "${test_results[$test_name]}" -eq 0 ]; then
            echo -e "  ${GREEN}✓${NC} $test_name: PASSED"
            passed_tests=$((passed_tests + 1))
        else
            echo -e "  ${RED}✗${NC} $test_name: FAILED"
        fi
    done

    echo ""
    echo "  Total: $passed_tests/$total_tests tests passed"
    echo ""
    echo "  Reports: $REPORT_DIR/validation_summary.md"
    echo "  Data: $DATA_DIR"
    echo ""
    echo "========================================================================"

    # Exit with success if all tests passed
    if [ $passed_tests -eq $total_tests ]; then
        log_success "All validation tests PASSED"
        trap - INT TERM EXIT  # Disable trap before normal exit
        exit 0
    else
        log_warning "Some tests FAILED - review reports"
        trap - INT TERM EXIT  # Disable trap before normal exit
        exit 1
    fi
}

# Run main
main "$@"
