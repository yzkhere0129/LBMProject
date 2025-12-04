#!/bin/bash
#
# Disk Space Cleanup Script for Week 3 Preparation
# Frees ~64GB of obsolete test data
# Archives critical Week 2 validation data
#
# Usage: ./cleanup_disk_space.sh [--dry-run]
#
# Author: LBM-CFD Infrastructure Team
# Date: 2025-11-20

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/home/yzk/LBMProject"
ARCHIVE_ROOT="/home/yzk/LBM_Archives"
DRY_RUN=false

# Parse arguments
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE - No files will be deleted${NC}"
fi

# Function to print section headers
print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

# Function to print status
print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get directory size
get_size() {
    du -sh "$1" 2>/dev/null | awk '{print $1}'
}

# Function to execute or simulate command
execute() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would execute: $*"
    else
        "$@"
    fi
}

# Check if running from correct directory
cd "$PROJECT_ROOT" || {
    print_error "Cannot cd to $PROJECT_ROOT"
    exit 1
}

print_header "DISK SPACE CLEANUP - Week 3 Preparation"

# Initial disk usage
INITIAL_SIZE=$(get_size build/)
echo "Initial build/ size: $INITIAL_SIZE"
echo ""

# ============================================================================
# STEP 1: Create Archive Directory
# ============================================================================
print_header "STEP 1: Create Archive Directory"

if [ ! -d "$ARCHIVE_ROOT" ]; then
    execute mkdir -p "$ARCHIVE_ROOT/week2_validation"
    print_status "Created archive directory: $ARCHIVE_ROOT/week2_validation"
else
    print_status "Archive directory exists: $ARCHIVE_ROOT"
fi

# ============================================================================
# STEP 2: Archive Critical Week 2 Validation Data
# ============================================================================
print_header "STEP 2: Archive Critical Week 2 Validation Data"

ARCHIVE_FILE="$ARCHIVE_ROOT/week2_validation/convergence_data_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "Creating archive: $ARCHIVE_FILE"
echo "Archiving:"
echo "  - build/grid_study_*.log"
echo "  - build/timestep_*.log"
echo "  - build/test_2*.log"
echo "  - build/*BUG3*.md"
echo "  - calibration/verification/"

if [ "$DRY_RUN" = false ]; then
    tar -czf "$ARCHIVE_FILE" \
        build/grid_study_*.log \
        build/timestep_*.log \
        build/test_2*.log \
        build/*BUG3*.md \
        calibration/verification/ \
        2>/dev/null || print_warning "Some files may not exist (OK)"

    if [ -f "$ARCHIVE_FILE" ]; then
        ARCHIVE_SIZE=$(get_size "$ARCHIVE_FILE")
        print_status "Archive created: $ARCHIVE_FILE ($ARCHIVE_SIZE)"
    else
        print_warning "Archive creation skipped (no files found)"
    fi
else
    print_status "[DRY-RUN] Would create archive: $ARCHIVE_FILE"
fi

# ============================================================================
# STEP 3: Delete Obsolete Test Runs (~64GB)
# ============================================================================
print_header "STEP 3: Delete Obsolete Test Runs"

cd build || exit 1

# List of directories to delete
OBSOLETE_DIRS=(
    # Stage iterations (superseded by 195W)
    "lpbf_realistic_150W_stage3"

    # Test iterations (A through O, except current production)
    "lpbf_test_A_coupling"
    "lpbf_test_B_marangoni"
    "lpbf_test_C_extended"
    "lpbf_test_C_full"
    "lpbf_test_D_darcy500"
    "lpbf_test_E_vof_advection"
    "lpbf_test_F_vof_surface"
    "lpbf_test_G_"*
    "lpbf_test_H_"*
    "lpbf_test_I_"*
    "lpbf_test_J_"*
    "lpbf_test_K_"*
    "lpbf_test_M_"*
    "lpbf_test_O_"*

    # Version iterations (v2, v3, v4 variants - keep only latest)
    "lpbf_realistic_195W_v2"
    "lpbf_realistic_195W_v3"
    "lpbf_realistic_195W_v4"
    "lpbf_realistic_195W_v4a"
    "lpbf_realistic_195W_v4b"
    "lpbf_realistic_195W_v4c"

    # Presentation runs (one-time use)
    "lpbf_visual_showcase"
    "lpbf_extreme_visual"

    # Old emissivity sweeps
    "lpbf_*_emissivity_0p3"
    "lpbf_*_emissivity_0p5"
    "lpbf_*_emissivity_fix"
)

FREED_SPACE=0

for dir_pattern in "${OBSOLETE_DIRS[@]}"; do
    for dir in $dir_pattern; do
        if [ -d "$dir" ]; then
            SIZE=$(du -sm "$dir" 2>/dev/null | awk '{print $1}')
            FREED_SPACE=$((FREED_SPACE + SIZE))

            echo "Deleting: $dir (${SIZE}MB)"
            execute rm -rf "$dir"

            if [ "$DRY_RUN" = false ] && [ ! -d "$dir" ]; then
                print_status "Deleted: $dir"
            fi
        fi
    done
done

print_status "Step 3 complete. Freed: ~${FREED_SPACE}MB"

# ============================================================================
# STEP 4: Compress Calibration Data
# ============================================================================
print_header "STEP 4: Compress Calibration Data"

cd "$PROJECT_ROOT/calibration" || exit 1

CALIBRATION_DIRS=(
    "test_50W_dt_020us"
    "test_50W_dt_010us"
    "test_50W_dt_005us"
)

for dir in "${CALIBRATION_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        SIZE=$(get_size "$dir")
        echo "Compressing: $dir ($SIZE)"

        execute tar -czf "${dir}.tar.gz" "$dir"/

        if [ "$DRY_RUN" = false ] && [ -f "${dir}.tar.gz" ]; then
            COMPRESSED_SIZE=$(get_size "${dir}.tar.gz")
            execute rm -rf "$dir"
            print_status "Compressed: $dir → ${dir}.tar.gz ($SIZE → $COMPRESSED_SIZE)"
        fi
    else
        print_warning "Directory not found: $dir"
    fi
done

# ============================================================================
# STEP 5: Clean Build Artifacts
# ============================================================================
print_header "STEP 5: Clean Build Artifacts"

cd "$PROJECT_ROOT/build" || exit 1

echo "Cleaning CMake build artifacts (.o files, CMake cache)"
if [ "$DRY_RUN" = false ]; then
    make clean 2>/dev/null || print_warning "make clean failed (OK if no Makefile)"
    print_status "Build artifacts cleaned"
else
    print_status "[DRY-RUN] Would run: make clean"
fi

# ============================================================================
# STEP 6: Summary and Verification
# ============================================================================
print_header "STEP 6: Summary and Verification"

cd "$PROJECT_ROOT" || exit 1

# Final disk usage
FINAL_SIZE=$(get_size build/)
CALIBRATION_SIZE=$(get_size calibration/)

echo "Disk Usage Summary:"
echo "  Initial build/ size:      $INITIAL_SIZE"
echo "  Final build/ size:        $FINAL_SIZE"
echo "  calibration/ size:        $CALIBRATION_SIZE"
echo ""

if [ "$DRY_RUN" = false ]; then
    echo "Files in build/:"
    ls -lh build/ | grep "^d" | awk '{print $9, $5}' | head -20
    echo ""

    # Check if enough space freed
    CURRENT_GB=$(du -sm build/ | awk '{print $1/1024}')
    echo "Current build/ size: ${CURRENT_GB}GB"

    if (( $(echo "$CURRENT_GB < 80" | bc -l) )); then
        print_status "Disk space cleanup successful! (target: <80GB)"
    else
        print_warning "Disk space still high (${CURRENT_GB}GB). May need manual cleanup."
    fi
fi

# ============================================================================
# STEP 7: Recommendations
# ============================================================================
print_header "STEP 7: Next Steps"

echo "Cleanup complete. Next steps:"
echo ""
echo "1. Verify archive integrity:"
echo "   tar -tzf $ARCHIVE_FILE | head"
echo ""
echo "2. Check remaining disk space:"
echo "   df -h /home/yzk"
echo ""
echo "3. Ready for Week 3 convergence re-runs:"
echo "   cd build/"
echo "   ./visualize_lpbf_scanning ../configs/calibration/lpbf_50W_grid_4um.conf"
echo ""
echo "4. Monitor disk space during Week 3:"
echo "   watch -n 3600 'du -sh /home/yzk/LBMProject/build'"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN COMPLETE - No files were actually deleted${NC}"
    echo "Re-run without --dry-run to perform actual cleanup"
fi

print_header "CLEANUP SCRIPT COMPLETE"
