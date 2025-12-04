#!/bin/bash
# WalBerla Benchmark Comparison Script
# This script runs both WalBerla and LBMProject Poiseuille flow benchmarks
# and prepares results for comparison

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}WalBerla vs LBMProject Comparison${NC}"
echo -e "${BLUE}======================================${NC}"
echo

# Configuration
WALBERLA_DIR="/home/yzk/walberla"
LBMPROJECT_DIR="/home/yzk/LBMProject"
OUTPUT_DIR="${LBMPROJECT_DIR}/benchmark_comparison_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}Output directory: ${OUTPUT_DIR}${NC}"
echo

# ============================================================================
# Part 1: Run WalBerla Poiseuille Benchmark
# ============================================================================
echo -e "${YELLOW}[1/4] Running WalBerla Poiseuille Channel Benchmark...${NC}"
WALBERLA_OUTPUT="${OUTPUT_DIR}/walberla_poiseuille_${TIMESTAMP}.log"

cd "${WALBERLA_DIR}/build/apps/benchmarks/PoiseuilleChannel"
if [ ! -f "./PoiseuilleChannel" ]; then
    echo -e "${RED}ERROR: WalBerla executable not found!${NC}"
    echo "Expected: ${WALBERLA_DIR}/build/apps/benchmarks/PoiseuilleChannel/PoiseuilleChannel"
    exit 1
fi

echo "Running: ./PoiseuilleChannel TestParallelPlates0.dat"
./PoiseuilleChannel TestParallelPlates0.dat > "${WALBERLA_OUTPUT}" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ WalBerla benchmark completed successfully${NC}"
    echo "Output saved to: ${WALBERLA_OUTPUT}"
else
    echo -e "${RED}✗ WalBerla benchmark failed${NC}"
    exit 1
fi
echo

# ============================================================================
# Part 2: Extract WalBerla Results
# ============================================================================
echo -e "${YELLOW}[2/4] Extracting WalBerla Results...${NC}"

# Extract key metrics from WalBerla output
grep "L1:" "${WALBERLA_OUTPUT}" | tail -1 || true
grep "L2:" "${WALBERLA_OUTPUT}" | tail -1 || true
grep "Lmax:" "${WALBERLA_OUTPUT}" | tail -1 || true
grep "rel. error" "${WALBERLA_OUTPUT}" | tail -2 || true

echo -e "${GREEN}✓ WalBerla results extracted${NC}"
echo

# ============================================================================
# Part 3: Run LBMProject Poiseuille Test (if built)
# ============================================================================
echo -e "${YELLOW}[3/4] Running LBMProject Poiseuille Flow Test...${NC}"
LBMPROJECT_OUTPUT="${OUTPUT_DIR}/lbmproject_poiseuille_${TIMESTAMP}.log"

cd "${LBMPROJECT_DIR}/build"
if [ ! -f "./tests/integration/test_poiseuille_flow_fluidlbm" ]; then
    echo -e "${YELLOW}WARNING: LBMProject test not built. Skipping...${NC}"
    echo "To build: cd ${LBMPROJECT_DIR}/build && make test_poiseuille_flow_fluidlbm"
else
    echo "Running: ./tests/integration/test_poiseuille_flow_fluidlbm"
    ./tests/integration/test_poiseuille_flow_fluidlbm > "${LBMPROJECT_OUTPUT}" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ LBMProject test completed successfully${NC}"
        echo "Output saved to: ${LBMPROJECT_OUTPUT}"

        # Copy velocity profile if generated
        if [ -f "poiseuille_profile_fluidlbm.txt" ]; then
            cp poiseuille_profile_fluidlbm.txt "${OUTPUT_DIR}/lbmproject_profile_${TIMESTAMP}.txt"
            echo "Velocity profile saved to: ${OUTPUT_DIR}/lbmproject_profile_${TIMESTAMP}.txt"
        fi
    else
        echo -e "${YELLOW}WARNING: LBMProject test encountered issues${NC}"
    fi
fi
echo

# ============================================================================
# Part 4: Generate Comparison Summary
# ============================================================================
echo -e "${YELLOW}[4/4] Generating Comparison Summary...${NC}"
SUMMARY_FILE="${OUTPUT_DIR}/comparison_summary_${TIMESTAMP}.txt"

cat > "${SUMMARY_FILE}" << EOF
================================================================================
WalBerla vs LBMProject Poiseuille Flow Comparison
Generated: $(date)
================================================================================

Test Configuration:
-------------------
- Test Case: 2D Poiseuille Flow (pressure-driven channel flow)
- Reynolds Number: Re = 10
- Lattice Scheme: D3Q19 with BGK collision
- Analytical Solution: Parabolic velocity profile

WalBerla Results:
-----------------
Domain: 10 × 30 × 10 cells
Timesteps: 7000

EOF

# Extract WalBerla metrics
echo "Accuracy Metrics:" >> "${SUMMARY_FILE}"
grep "L1:" "${WALBERLA_OUTPUT}" | tail -1 >> "${SUMMARY_FILE}" 2>/dev/null || echo "  L1: Not found" >> "${SUMMARY_FILE}"
grep "L2:" "${WALBERLA_OUTPUT}" | tail -1 >> "${SUMMARY_FILE}" 2>/dev/null || echo "  L2: Not found" >> "${SUMMARY_FILE}"
grep "Lmax:" "${WALBERLA_OUTPUT}" | tail -1 >> "${SUMMARY_FILE}" 2>/dev/null || echo "  Lmax: Not found" >> "${SUMMARY_FILE}"

echo "" >> "${SUMMARY_FILE}"
echo "Flow Rate Error:" >> "${SUMMARY_FILE}"
grep "rel. error" "${WALBERLA_OUTPUT}" | tail -2 >> "${SUMMARY_FILE}" 2>/dev/null || echo "  Not found" >> "${SUMMARY_FILE}"

if [ -f "${LBMPROJECT_OUTPUT}" ]; then
    echo "" >> "${SUMMARY_FILE}"
    echo "LBMProject Results:" >> "${SUMMARY_FILE}"
    echo "-------------------" >> "${SUMMARY_FILE}"
    grep "Domain:" "${LBMPROJECT_OUTPUT}" | head -1 >> "${SUMMARY_FILE}" 2>/dev/null || true
    echo "" >> "${SUMMARY_FILE}"
    echo "Accuracy Metrics:" >> "${SUMMARY_FILE}"
    grep "L2 relative error:" "${LBMPROJECT_OUTPUT}" >> "${SUMMARY_FILE}" 2>/dev/null || echo "  L2: Not found" >> "${SUMMARY_FILE}"
    grep "Maximum point error:" "${LBMPROJECT_OUTPUT}" >> "${SUMMARY_FILE}" 2>/dev/null || echo "  Max error: Not found" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
    echo "Velocity Comparison:" >> "${SUMMARY_FILE}"
    grep -A 3 "Maximum velocity:" "${LBMPROJECT_OUTPUT}" | head -4 >> "${SUMMARY_FILE}" 2>/dev/null || echo "  Not found" >> "${SUMMARY_FILE}"
fi

cat >> "${SUMMARY_FILE}" << EOF

================================================================================
File Locations:
---------------
WalBerla Output:    ${WALBERLA_OUTPUT}
LBMProject Output:  ${LBMPROJECT_OUTPUT}
Summary:            ${SUMMARY_FILE}

Full Report:        ${LBMPROJECT_DIR}/WALBERLA_BENCHMARK_COMPARISON_REPORT.md
================================================================================
EOF

echo -e "${GREEN}✓ Comparison summary generated${NC}"
echo
cat "${SUMMARY_FILE}"
echo

# ============================================================================
# Completion
# ============================================================================
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}Comparison Complete!${NC}"
echo -e "${BLUE}======================================${NC}"
echo
echo "Results location: ${OUTPUT_DIR}"
echo
echo "Next steps:"
echo "1. Review: cat ${SUMMARY_FILE}"
echo "2. Compare velocity profiles (if available)"
echo "3. See full report: ${LBMPROJECT_DIR}/WALBERLA_BENCHMARK_COMPARISON_REPORT.md"
echo

exit 0
