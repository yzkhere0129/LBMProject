#!/bin/bash
#
# Automated build and run script for Case 5: Laser Melting Validation
#
# This script automates the complete workflow:
# 1. Build the test
# 2. Run the simulation
# 3. Generate plots
# 4. Display results
#
# Usage:
#   ./run_laser_melting_validation.sh [options]
#
# Options:
#   --build-only      Only build, don't run
#   --run-only        Only run (assume already built)
#   --plot-only       Only generate plots (assume data exists)
#   --skip-plots      Skip plot generation
#   --help            Show this help message

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
PROJECT_ROOT="/home/yzk/LBMProject"
BUILD_DIR="${PROJECT_ROOT}/build"
VALIDATION_DIR="${PROJECT_ROOT}/tests/validation"
OUTPUT_DIR="${VALIDATION_DIR}/output_laser_melting_senior"
SCRIPTS_DIR="${VALIDATION_DIR}/scripts"

# Flags
BUILD_ONLY=0
RUN_ONLY=0
PLOT_ONLY=0
SKIP_PLOTS=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --run-only)
            RUN_ONLY=1
            shift
            ;;
        --plot-only)
            PLOT_ONLY=1
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=1
            shift
            ;;
        --help)
            echo "Case 5: Laser Melting Validation - Automated Build and Run Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --build-only      Only build, don't run"
            echo "  --run-only        Only run (assume already built)"
            echo "  --plot-only       Only generate plots (assume data exists)"
            echo "  --skip-plots      Skip plot generation"
            echo "  --help            Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Case 5: Laser Melting Validation${NC}"
echo -e "${BLUE}Automated Build and Run Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Build
if [[ $RUN_ONLY -eq 0 ]] && [[ $PLOT_ONLY -eq 0 ]]; then
    echo -e "${YELLOW}[Step 1/4] Building test...${NC}"

    # Check if build directory exists
    if [ ! -d "$BUILD_DIR" ]; then
        echo "Creating build directory: $BUILD_DIR"
        mkdir -p "$BUILD_DIR"
    fi

    cd "$BUILD_DIR"

    # Run cmake
    echo "Running cmake..."
    cmake .. > /dev/null 2>&1

    # Build test
    echo "Building test_laser_melting_senior..."
    make test_laser_melting_senior -j$(nproc)

    echo -e "${GREEN}✓ Build complete${NC}"
    echo ""

    if [[ $BUILD_ONLY -eq 1 ]]; then
        echo -e "${GREEN}Build-only mode: Exiting${NC}"
        exit 0
    fi
fi

# Step 2: Run simulation
if [[ $PLOT_ONLY -eq 0 ]]; then
    echo -e "${YELLOW}[Step 2/4] Running simulation...${NC}"
    echo "This will take approximately 20-30 minutes"
    echo ""

    cd "$BUILD_DIR"

    # Check if executable exists
    if [ ! -f "./tests/validation/test_laser_melting_senior" ]; then
        echo -e "${RED}Error: Executable not found. Build first with --build-only or without --run-only${NC}"
        exit 1
    fi

    # Run test
    echo "Starting test_laser_melting_senior..."
    echo "Output will appear below:"
    echo ""

    ./tests/validation/test_laser_melting_senior

    TEST_EXIT_CODE=$?

    echo ""
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Simulation complete (PASSED)${NC}"
    else
        echo -e "${RED}✗ Simulation failed (EXIT CODE: $TEST_EXIT_CODE)${NC}"
        exit $TEST_EXIT_CODE
    fi
    echo ""

    if [[ $RUN_ONLY -eq 1 ]]; then
        echo -e "${GREEN}Run-only mode: Exiting${NC}"
        exit 0
    fi
fi

# Step 3: Generate plots
if [[ $SKIP_PLOTS -eq 0 ]]; then
    echo -e "${YELLOW}[Step 3/4] Generating plots...${NC}"

    # Check if CSV file exists
    CSV_FILE="${OUTPUT_DIR}/melt_pool_depth.csv"
    if [ ! -f "$CSV_FILE" ]; then
        echo -e "${RED}Error: CSV file not found: $CSV_FILE${NC}"
        echo "Run simulation first (without --plot-only)"
        exit 1
    fi

    # Check if Python script exists
    PLOT_SCRIPT="${SCRIPTS_DIR}/plot_melt_pool_comparison.py"
    if [ ! -f "$PLOT_SCRIPT" ]; then
        echo -e "${RED}Error: Plot script not found: $PLOT_SCRIPT${NC}"
        exit 1
    fi

    # Check Python dependencies
    echo "Checking Python dependencies..."
    python3 -c "import numpy, matplotlib, pandas" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Python dependencies not installed${NC}"
        echo "Install with: pip install numpy matplotlib pandas vtk"
        echo "Skipping plot generation"
    else
        echo "Running plot script..."
        cd "$VALIDATION_DIR"
        python3 "$PLOT_SCRIPT" "$CSV_FILE"

        echo -e "${GREEN}✓ Plots generated${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}[Step 3/4] Skipping plot generation (--skip-plots)${NC}"
    echo ""
fi

# Step 4: Display results
echo -e "${YELLOW}[Step 4/4] Results Summary${NC}"
echo ""

# Check if output directory exists
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}Output directory:${NC} $OUTPUT_DIR"
    echo ""

    # Count VTK files
    VTK_COUNT=$(ls -1 "$OUTPUT_DIR"/*.vtk 2>/dev/null | wc -l)
    echo "VTK files: $VTK_COUNT"

    # Check CSV
    if [ -f "${OUTPUT_DIR}/melt_pool_depth.csv" ]; then
        CSV_LINES=$(wc -l < "${OUTPUT_DIR}/melt_pool_depth.csv")
        echo "CSV data: ${OUTPUT_DIR}/melt_pool_depth.csv ($CSV_LINES lines)"
    fi

    # Check plots
    PLOTS=("melt_pool_depth_vs_time.png" "temperature_evolution.png" "velocity_evolution.png" "combined_view.png")
    PLOT_COUNT=0
    for plot in "${PLOTS[@]}"; do
        if [ -f "${OUTPUT_DIR}/$plot" ]; then
            ((PLOT_COUNT++))
        fi
    done

    if [ $PLOT_COUNT -gt 0 ]; then
        echo "Plots: $PLOT_COUNT/4 generated"
        echo ""
        echo "Plot files:"
        for plot in "${PLOTS[@]}"; do
            if [ -f "${OUTPUT_DIR}/$plot" ]; then
                echo "  - ${OUTPUT_DIR}/$plot"
            fi
        done
    fi

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Validation complete!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Quick summary from CSV (if exists)
    if [ -f "${OUTPUT_DIR}/melt_pool_depth.csv" ]; then
        echo "Quick Summary (from CSV):"
        echo ""

        # Extract max depth (skip header)
        MAX_DEPTH=$(tail -n +2 "${OUTPUT_DIR}/melt_pool_depth.csv" | cut -d',' -f2 | sort -n | tail -1)
        echo "  Peak melt pool depth: ${MAX_DEPTH} μm"

        # Extract max velocity
        MAX_VEL=$(tail -n +2 "${OUTPUT_DIR}/melt_pool_depth.csv" | cut -d',' -f4 | sort -n | tail -1)
        echo "  Maximum Marangoni velocity: ${MAX_VEL} m/s"

        echo ""
    fi

    # Next steps
    echo "Next steps:"
    echo "  1. View plots: xdg-open ${OUTPUT_DIR}/*.png"
    echo "  2. Open in ParaView: paraview ${OUTPUT_DIR}/laser_melting_*.vtk"
    echo "  3. Read full documentation: ${VALIDATION_DIR}/LASER_MELTING_VALIDATION_README.md"
    echo ""
else
    echo -e "${RED}Error: Output directory not found${NC}"
    echo "Expected: $OUTPUT_DIR"
fi

echo -e "${BLUE}========================================${NC}"
