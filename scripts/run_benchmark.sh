#!/bin/bash
#
# Run LBMProject thermal LPBF benchmark
#
# Usage:
#   ./run_benchmark.sh [mode] [size]
#
# Examples:
#   ./run_benchmark.sh A M      # Mode A, Medium size
#   ./run_benchmark.sh B L      # Mode B, Large size
#   ./run_benchmark.sh C S      # Mode C, Small size
#   ./run_benchmark.sh all M    # Run all modes
#

set -e

# Default values
MODE=${1:-A}
SIZE=${2:-M}
STEPS=${3:-10000}

# Paths
PROJECT_ROOT="/home/yzk/LBMProject"
BUILD_DIR="${PROJECT_ROOT}/build"
BENCHMARK_EXE="${BUILD_DIR}/benchmark_thermal_lpbf"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "     LBMProject Benchmark Runner"
echo "=============================================="

# Check if executable exists
if [ ! -f "$BENCHMARK_EXE" ]; then
    echo -e "${YELLOW}Benchmark executable not found. Building...${NC}"
    cd "$BUILD_DIR"
    make benchmark_thermal_lpbf -j4
    cd -
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a single benchmark
run_benchmark() {
    local mode=$1
    local size=$2
    local steps=$3

    echo ""
    echo -e "${GREEN}Running benchmark: Mode ${mode}, Size ${size}, Steps ${steps}${NC}"
    echo "----------------------------------------------"

    cd "$OUTPUT_DIR"

    # Run benchmark
    "${BENCHMARK_EXE}" --mode ${mode} --size ${size} --steps ${steps}

    # Copy results
    if [ -f "benchmark_results_mode${mode}.csv" ]; then
        mv "benchmark_results_mode${mode}.csv" \
           "benchmark_mode${mode}_size${size}_$(date +%Y%m%d_%H%M%S).csv"
    fi

    cd - > /dev/null
}

# Run benchmarks
if [ "$MODE" = "all" ]; then
    echo "Running all benchmark modes..."
    run_benchmark "A" "$SIZE" "$STEPS"
    run_benchmark "B" "$SIZE" "$STEPS"
    run_benchmark "C" "$SIZE" "$STEPS"
else
    run_benchmark "$MODE" "$SIZE" "$STEPS"
fi

echo ""
echo "=============================================="
echo "     Benchmark Complete"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

# List output files
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (no CSV files found)"

echo ""
echo "To compare with waLBerla:"
echo "  1. Run waLBerla benchmark:"
echo "     cd /home/yzk/walberla/build"
echo "     ./apps/showcases/LaserHeating/LaserHeating ../apps/showcases/LaserHeating/benchmark_comparison.cfg"
echo ""
echo "  2. Compare results:"
echo "     python ${PROJECT_ROOT}/scripts/compare_benchmark_results.py <lbm_csv> [walberla_csv]"
echo ""
