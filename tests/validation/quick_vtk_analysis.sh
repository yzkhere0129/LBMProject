#!/bin/bash
#
# Quick VTK Analysis Script for Case 5 Laser Melting
# Usage: ./quick_vtk_analysis.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VTK_TOOLS="/home/yzk/LBMProject/benchmark/vtk_comparison"
OUTPUT_DIR="/home/yzk/LBMProject/tests/validation/output_laser_melting_senior"
ANALYSIS_DIR="/home/yzk/LBMProject/tests/validation/analysis_case5"

echo "========================================================================"
echo "Case 5 Laser Melting - Quick VTK Analysis"
echo "========================================================================"
echo ""

# Check if VTK files exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

NUM_FILES=$(ls -1 "$OUTPUT_DIR"/temperature_*.vtk.vtk 2>/dev/null | wc -l)
echo "Found $NUM_FILES VTK files in $OUTPUT_DIR"
echo ""

# Activate virtual environment
if [ ! -d "$VTK_TOOLS/venv" ]; then
    echo "Setting up Python virtual environment..."
    cd "$VTK_TOOLS"
    python3 -m venv venv
    source venv/bin/activate
    pip install numpy scipy matplotlib pyvista
else
    echo "Activating existing virtual environment..."
    source "$VTK_TOOLS/venv/bin/activate"
fi

# Run timeseries analysis
echo ""
echo "========================================================================"
echo "Running timeseries analysis..."
echo "========================================================================"
cd "$VTK_TOOLS"
python3 vtk_compare.py timeseries "$OUTPUT_DIR" \
    --pattern "temperature_*.vtk.vtk" \
    --output "$ANALYSIS_DIR" \
    --solidus 1923.0 \
    --liquidus 1973.0

# Run detailed analysis
echo ""
echo "========================================================================"
echo "Running detailed analysis..."
echo "========================================================================"
cd "$SCRIPT_DIR"
python3 analyze_case5_detailed.py

# Run Rosenthal comparison (if needed)
echo ""
echo "========================================================================"
echo "Running Rosenthal comparison..."
echo "========================================================================"
python3 compare_with_rosenthal.py

# Display summary
echo ""
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $ANALYSIS_DIR"
echo ""
echo "Key files:"
echo "  - SUMMARY.txt                    : Quick reference"
echo "  - CASE5_ANALYSIS_REPORT.md       : Detailed report"
echo "  - detailed_analysis.png          : 4-panel visualization"
echo "  - timeseries_metrics.json        : Raw data"
echo ""
echo "View summary:"
echo "  cat $ANALYSIS_DIR/SUMMARY.txt"
echo ""
echo "View plots:"
echo "  xdg-open $ANALYSIS_DIR/detailed_analysis.png  # Linux"
echo "  open $ANALYSIS_DIR/detailed_analysis.png      # macOS"
echo ""
