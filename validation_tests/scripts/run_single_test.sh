#!/bin/bash

################################################################################
# Run Single Validation Test
#
# Purpose: Execute a single test stage for debugging or manual testing
# Usage: ./run_single_test.sh TEST_NAME [NUM_STEPS]
#
# Examples:
#   ./run_single_test.sh BASELINE
#   ./run_single_test.sh GRAD-REMOVED 2000
################################################################################

set -e

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

MARANGONI_SRC="$SRC_DIR/physics/vof/marangoni.cu"
MULTIPHYSICS_SRC="$SRC_DIR/physics/multiphysics/multiphysics_solver.cu"
EXECUTABLE="$BUILD_DIR/visualize_lpbf_scanning"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_TIMEOUT=1800  # 30 minutes

# ============================================================================
# Color output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Source Modification Functions
# ============================================================================

backup_sources() {
    log_info "Creating backup..."
    cp "$MARANGONI_SRC" "$BACKUP_DIR/marangoni.cu.backup_$TIMESTAMP"
    cp "$MULTIPHYSICS_SRC" "$BACKUP_DIR/multiphysics_solver.cu.backup_$TIMESTAMP"
}

restore_sources() {
    log_info "Restoring from backup..."
    if [ -f "$BACKUP_DIR/marangoni.cu.backup_$TIMESTAMP" ]; then
        cp "$BACKUP_DIR/marangoni.cu.backup_$TIMESTAMP" "$MARANGONI_SRC"
    fi
    if [ -f "$BACKUP_DIR/multiphysics_solver.cu.backup_$TIMESTAMP" ]; then
        cp "$BACKUP_DIR/multiphysics_solver.cu.backup_$TIMESTAMP" "$MULTIPHYSICS_SRC"
    fi
}

apply_grad_2x() {
    log_info "Applying GRAD-2X modification..."
    sed -i '31s/constexpr float MAX_PHYSICAL_GRAD_T = 5\.0e8f;/constexpr float MAX_PHYSICAL_GRAD_T = 1.0e9f;/' "$MARANGONI_SRC"
}

apply_grad_10x() {
    log_info "Applying GRAD-10X modification..."
    sed -i '31s/constexpr float MAX_PHYSICAL_GRAD_T = 5\.0e8f;/constexpr float MAX_PHYSICAL_GRAD_T = 5.0e9f;/' "$MARANGONI_SRC"
}

apply_grad_removed() {
    log_info "Applying GRAD-REMOVED modification..."
    sed -i '109,114s/^/\/\/ /' "$MARANGONI_SRC"
    sed -i '199,204s/^/\/\/ /' "$MARANGONI_SRC"
}

apply_cfl_removed() {
    log_info "Applying CFL-REMOVED modification..."
    apply_grad_removed
    sed -i '624,636s/^/\/\/ /' "$MULTIPHYSICS_SRC"
}

# ============================================================================
# Build and Run
# ============================================================================

rebuild() {
    log_info "Rebuilding project..."
    cd "$BUILD_DIR" || exit 1

    if ! cmake .. > "$DATA_DIR/build_$TEST_NAME.log" 2>&1; then
        log_error "CMake failed"
        cat "$DATA_DIR/build_$TEST_NAME.log"
        return 1
    fi

    if ! make -j8 >> "$DATA_DIR/build_$TEST_NAME.log" 2>&1; then
        log_error "Make failed"
        tail -50 "$DATA_DIR/build_$TEST_NAME.log"
        return 1
    fi

    log_success "Build completed"
    return 0
}

run_simulation() {
    local output_file="$DATA_DIR/${TEST_NAME}_output.log"

    log_info "Running simulation: $TEST_NAME (timeout ${TEST_TIMEOUT}s)"
    cd "$BUILD_DIR" || return 1

    if timeout $TEST_TIMEOUT "$EXECUTABLE" > "$output_file" 2>&1; then
        log_success "Simulation completed"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_error "Simulation TIMED OUT"
        else
            log_error "Simulation FAILED (exit code $exit_code)"
        fi
        return 1
    fi
}

validate() {
    local v_min=0
    local v_max=1000

    case $TEST_NAME in
        "BASELINE") v_min=0; v_max=20 ;;
        "GRAD-2X") v_min=0; v_max=50 ;;
        "GRAD-10X") v_min=0; v_max=100 ;;
        "GRAD-REMOVED"|"CFL-REMOVED"|"LONG-STABILITY") v_min=50; v_max=500 ;;
    esac

    log_info "Validating results..."

    if python3 "$SCRIPT_DIR/validate_results.py" \
        --log-file "$DATA_DIR/${TEST_NAME}_output.log" \
        --test-name "$TEST_NAME" \
        --v-min "$v_min" \
        --v-max "$v_max" \
        --output "$REPORT_DIR/${TEST_NAME}_validation.txt"; then
        log_success "Validation PASSED"
        cat "$REPORT_DIR/${TEST_NAME}_validation.txt"
        return 0
    else
        log_error "Validation FAILED"
        cat "$REPORT_DIR/${TEST_NAME}_validation.txt"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

cleanup() {
    log_info "Cleaning up..."
    restore_sources
}

trap cleanup EXIT INT TERM

main() {
    # Parse arguments
    if [ $# -lt 1 ]; then
        echo "Usage: $0 TEST_NAME [NUM_STEPS]"
        echo ""
        echo "Available tests:"
        echo "  BASELINE      - Current code (no modifications)"
        echo "  GRAD-2X       - Gradient limit 5e8 -> 1e9 K/m"
        echo "  GRAD-10X      - Gradient limit 5e8 -> 5e9 K/m"
        echo "  GRAD-REMOVED  - Remove gradient limiter"
        echo "  CFL-REMOVED   - Remove both gradient and CFL limiters"
        echo ""
        echo "Example:"
        echo "  $0 GRAD-REMOVED 1000"
        exit 1
    fi

    TEST_NAME=$1
    NUM_STEPS=${2:-1000}

    # Create directories
    mkdir -p "$DATA_DIR" "$REPORT_DIR" "$BACKUP_DIR"

    echo ""
    echo "========================================================================"
    echo "  Single Test Execution: $TEST_NAME"
    echo "========================================================================"
    echo "  Steps: $NUM_STEPS"
    echo "  Timeout: $TEST_TIMEOUT s"
    echo "  Timestamp: $TIMESTAMP"
    echo "========================================================================"
    echo ""

    # Backup sources
    backup_sources

    # Apply modifications
    case $TEST_NAME in
        "BASELINE")
            log_info "Running baseline (no modifications)"
            ;;
        "GRAD-2X")
            apply_grad_2x || { log_error "Failed to apply modifications"; exit 1; }
            ;;
        "GRAD-10X")
            apply_grad_10x || { log_error "Failed to apply modifications"; exit 1; }
            ;;
        "GRAD-REMOVED")
            apply_grad_removed || { log_error "Failed to apply modifications"; exit 1; }
            ;;
        "CFL-REMOVED"|"LONG-STABILITY")
            apply_cfl_removed || { log_error "Failed to apply modifications"; exit 1; }
            ;;
        *)
            log_error "Unknown test: $TEST_NAME"
            exit 1
            ;;
    esac

    # Rebuild
    if ! rebuild; then
        log_error "Build failed"
        exit 1
    fi

    # Run simulation
    if ! run_simulation; then
        log_warning "Simulation did not complete successfully"
        # Continue to validation anyway
    fi

    # Validate
    if validate; then
        log_success "Test $TEST_NAME completed successfully"
        exit 0
    else
        log_error "Test $TEST_NAME failed validation"
        exit 1
    fi
}

main "$@"
