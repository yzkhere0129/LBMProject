#!/bin/bash
#
# Case 5: Laser Melting Validation Test Runner
#
# This script runs the laser melting validation test and provides
# convenient analysis and visualization of results.
#
# Usage:
#   ./run_case5_laser_melting.sh [OPTIONS]
#
# Options:
#   --quick      Run with reduced timesteps for quick testing
#   --full       Run full 100 μs simulation (default)
#   --visualize  Generate plots after simulation
#   --clean      Remove previous output before running
#   --help       Show this help message

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
MODE="full"
VISUALIZE=false
CLEAN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            grep '^#' "$0" | cut -c4-
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Project paths
PROJECT_ROOT="/home/yzk/LBMProject"
BUILD_DIR="${PROJECT_ROOT}/build"
TEST_EXECUTABLE="${BUILD_DIR}/tests/validation/test_laser_melting_senior"
OUTPUT_DIR="/home/yzk/LBMProject/tests/validation/output_laser_melting_senior"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Case 5: Laser Melting Validation${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}ERROR: Build directory not found: $BUILD_DIR${NC}"
    echo "Please run cmake to configure the project first"
    exit 1
fi

# Change to build directory
cd "$BUILD_DIR"

# Clean previous output if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning previous output...${NC}"
    if [ -d "$OUTPUT_DIR" ]; then
        rm -rf "$OUTPUT_DIR"
        echo -e "${GREEN}✓ Previous output removed${NC}"
    fi
    echo ""
fi

# Build the test
echo -e "${YELLOW}Building test executable...${NC}"
if ! cmake --build . --target test_laser_melting_senior 2>&1 | tail -5; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# Check if executable exists
if [ ! -f "$TEST_EXECUTABLE" ]; then
    echo -e "${RED}ERROR: Test executable not found: $TEST_EXECUTABLE${NC}"
    exit 1
fi

# Run the test
echo -e "${YELLOW}Running simulation...${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo "  Domain: 150 × 300 × 150 μm"
echo "  Grid: 40 × 80 × 80 cells"
echo "  Material: Ti6Al4V"
echo "  Laser: 200W, spot radius 30 μm"
echo "  Duration: 100 μs (50 μs ON, 50 μs OFF)"
echo "  Timestep: 1 ns"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the test and capture output
if ! "$TEST_EXECUTABLE" 2>&1 | tee "${OUTPUT_DIR}_run.log"; then
    echo ""
    echo -e "${RED}✗ Simulation failed${NC}"
    echo "Check log file: ${OUTPUT_DIR}_run.log"
    exit 1
fi

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${GREEN}✓ Simulation completed successfully${NC}"
echo -e "${GREEN}  Runtime: ${MINUTES}m ${SECONDS}s${NC}"
echo ""

# Check output directory
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}WARNING: Output directory not created${NC}"
    exit 0
fi

# Count output files
VTK_COUNT=$(find "$OUTPUT_DIR" -name "temperature_*.vtk" 2>/dev/null | wc -l)
CSV_EXISTS=$([ -f "$OUTPUT_DIR/melt_pool_depth.csv" ] && echo "yes" || echo "no")

echo -e "${BLUE}Output Summary:${NC}"
echo "  Location: $OUTPUT_DIR"
echo "  VTK files: $VTK_COUNT"
echo "  CSV data: $CSV_EXISTS"
echo ""

# Visualize if requested
if [ "$VISUALIZE" = true ]; then
    echo -e "${YELLOW}Generating plots...${NC}"

    # Check if CSV exists
    if [ "$CSV_EXISTS" = "yes" ]; then
        # Create a simple Python plotting script
        PLOT_SCRIPT="${OUTPUT_DIR}/plot_results.py"
        cat > "$PLOT_SCRIPT" << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Read data
csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Case 5: Laser Melting Validation Results', fontsize=16, fontweight='bold')

# Plot 1: Melt pool depth
ax = axes[0, 0]
ax.plot(df['time_us'], df['depth_um'], 'b-', linewidth=2, label='Melt pool depth')
ax.axvline(x=50, color='r', linestyle='--', linewidth=1.5, label='Laser shutoff')
ax.set_xlabel('Time (μs)', fontsize=12)
ax.set_ylabel('Melt Pool Depth (μm)', fontsize=12)
ax.set_title('Melt Pool Evolution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Maximum temperature
ax = axes[0, 1]
ax.plot(df['time_us'], df['max_temp_K'], 'r-', linewidth=2, label='Max temperature')
ax.axhline(y=1923, color='orange', linestyle='--', linewidth=1.5, label='T_liquidus (1923K)')
ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_xlabel('Time (μs)', fontsize=12)
ax.set_ylabel('Temperature (K)', fontsize=12)
ax.set_title('Temperature Evolution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Maximum velocity (Marangoni)
ax = axes[1, 0]
ax.plot(df['time_us'], df['max_velocity_m_s'], 'g-', linewidth=2, label='Max velocity')
ax.axvline(x=50, color='r', linestyle='--', linewidth=1.5, label='Laser shutoff')
ax.axhspan(0.5, 2.0, alpha=0.2, color='green', label='Literature range (0.5-2 m/s)')
ax.set_xlabel('Time (μs)', fontsize=12)
ax.set_ylabel('Velocity (m/s)', fontsize=12)
ax.set_title('Marangoni Flow Velocity', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Phase diagram (depth vs temperature)
ax = axes[1, 1]
scatter = ax.scatter(df['max_temp_K'], df['depth_um'], c=df['time_us'],
                     cmap='viridis', s=50, alpha=0.6)
cbar = plt.colorbar(scatter, ax=ax, label='Time (μs)')
ax.set_xlabel('Maximum Temperature (K)', fontsize=12)
ax.set_ylabel('Melt Pool Depth (μm)', fontsize=12)
ax.set_title('Melt Pool Depth vs Temperature', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add arrow annotations for laser shutoff
ax.annotate('Laser ON\n(heating)', xy=(2200, df['depth_um'].max() * 0.8),
            fontsize=11, ha='center', color='red', weight='bold')
ax.annotate('Laser OFF\n(cooling)', xy=(1500, df['depth_um'].max() * 0.3),
            fontsize=11, ha='center', color='blue', weight='bold')

plt.tight_layout()

# Save figure
output_dir = os.path.dirname(csv_file)
output_file = os.path.join(output_dir, 'case5_results.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")

# Show summary statistics
print("\n=== Summary Statistics ===")
print(f"Peak melt pool depth: {df['depth_um'].max():.2f} μm")
print(f"  (at t = {df.loc[df['depth_um'].idxmax(), 'time_us']:.2f} μs)")
print(f"Max temperature: {df['max_temp_K'].max():.1f} K")
print(f"Max Marangoni velocity: {df['max_velocity_m_s'].max():.3f} m/s")
print(f"Final melt pool depth: {df['depth_um'].iloc[-1]:.2f} μm")
print(f"Depth reduction: {df['depth_um'].max() - df['depth_um'].iloc[-1]:.2f} μm")
EOF

        # Run plotting script
        if python3 "$PLOT_SCRIPT" "$OUTPUT_DIR/melt_pool_depth.csv" 2>&1; then
            echo -e "${GREEN}✓ Plots generated successfully${NC}"
            echo "  Output: ${OUTPUT_DIR}/case5_results.png"
        else
            echo -e "${YELLOW}Warning: Plot generation failed${NC}"
        fi
    else
        echo -e "${YELLOW}Warning: CSV file not found, skipping plots${NC}"
    fi
    echo ""
fi

# Final summary
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}Case 5 validation complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Next steps:"
echo "  1. View VTK files in ParaView:"
echo "     paraview ${OUTPUT_DIR}/temperature_*.vtk"
echo ""
echo "  2. Analyze time series data:"
echo "     cat ${OUTPUT_DIR}/melt_pool_depth.csv"
echo ""
echo "  3. Generate plots (if not done):"
echo "     $0 --visualize"
echo ""
echo "  4. Compare with literature values (see README)"
echo ""

exit 0
