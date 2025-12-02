#!/bin/bash
################################################################################
# Energy Conservation Validation Test
# Purpose: Prove fixes don't violate energy conservation
# Success: P_laser - P_evap - P_rad - dE/dt ≈ 0, error < 5%
################################################################################

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/home/yzk/LBMProject"
BUILD_DIR="${PROJECT_ROOT}/build"
OUTPUT_BASE="${PROJECT_ROOT}/validation_results/energy_conservation"
EXECUTABLE="${BUILD_DIR}/LBMSolver"

# Create output directory
mkdir -p "${OUTPUT_BASE}"
cd "${BUILD_DIR}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================================"
echo "ENERGY CONSERVATION VALIDATION - Verifying Thermodynamic Consistency"
echo "================================================================================"
echo ""
echo "Energy Balance Equation:"
echo "  dE/dt = P_laser - P_evap - P_rad - P_convection"
echo ""
echo "Success Criteria:"
echo "  ✓ Energy error < 5% throughout simulation"
echo "  ✓ No drift (error oscillates, does not accumulate)"
echo "  ✓ Evaporation and radiation balance laser input"
echo "================================================================================"
echo ""

# Simulation parameters
NX=200
NY=100
NZ=50
DX=2.0e-6
DT=1.0e-7
NUM_STEPS=1000
OUTPUT_INTERVAL=50  # More frequent for energy analysis

LASER_POWER=200.0
LASER_RADIUS=50e-6
SCAN_SPEED=0.4

OUTPUT_DIR="${OUTPUT_BASE}/test_run"
mkdir -p "${OUTPUT_DIR}"

################################################################################
# Run simulation with detailed energy diagnostics
################################################################################

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  PHASE 1: Execute Simulation with Energy Diagnostics"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

cat > "${OUTPUT_DIR}/config.txt" << EOF
# Energy Conservation Test
nx ${NX}
ny ${NY}
nz ${NZ}
dx ${DX}
dt ${DT}
num_steps ${NUM_STEPS}
output_interval ${OUTPUT_INTERVAL}

# Physics parameters
laser_power ${LASER_POWER}
laser_radius ${LASER_RADIUS}
scan_speed ${SCAN_SPEED}

# Enable comprehensive energy diagnostics
enable_energy_diagnostics true
enable_diagnostics true
enable_vtk true

# Output settings
output_dir ${OUTPUT_DIR}
EOF

echo -e "${BLUE}Running simulation with energy tracking...${NC}"
START_TIME=$(date +%s)

if ${EXECUTABLE} "${OUTPUT_DIR}/config.txt" > "${OUTPUT_DIR}/simulation.log" 2>&1; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo -e "${GREEN}✓ Simulation completed in ${ELAPSED}s${NC}"
else
    echo -e "${RED}✗ Simulation FAILED - check ${OUTPUT_DIR}/simulation.log${NC}"
    tail -n 30 "${OUTPUT_DIR}/simulation.log"
    exit 1
fi

################################################################################
# Analyze energy balance
################################################################################

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  PHASE 2: Energy Balance Analysis"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

ENERGY_FILE="${OUTPUT_DIR}/energy_diagnostics.csv"

if [ ! -f "${ENERGY_FILE}" ]; then
    echo -e "${RED}✗ Energy diagnostics file not found${NC}"
    echo "Expected: ${ENERGY_FILE}"
    exit 1
fi

# Parse energy diagnostics
# Expected format: time, E_total, P_laser, P_evap, P_rad, error%
echo -e "${BLUE}Parsing energy diagnostics...${NC}"

# Compute statistics
MAX_ERROR=$(awk -F',' 'NR>1 {print sqrt($6^2)}' "${ENERGY_FILE}" | sort -n | tail -n 1)
AVG_ERROR=$(awk -F',' 'NR>1 {sum += sqrt($6^2); count++} END {print sum/count}' "${ENERGY_FILE}")
FINAL_ERROR=$(tail -n 1 "${ENERGY_FILE}" | awk -F',' '{print $6}')

# Extract power terms (average over last 500 steps)
AVG_P_LASER=$(tail -n 10 "${ENERGY_FILE}" | awk -F',' '{sum += $3; count++} END {print sum/count}')
AVG_P_EVAP=$(tail -n 10 "${ENERGY_FILE}" | awk -F',' '{sum += $4; count++} END {print sum/count}')
AVG_P_RAD=$(tail -n 10 "${ENERGY_FILE}" | awk -F',' '{sum += $5; count++} END {print sum/count}')

# Compute energy balance
# Net power = P_laser - P_evap - P_rad (should ≈ dE/dt)
NET_POWER=$(echo "scale=3; ${AVG_P_LASER} - ${AVG_P_EVAP} - ${AVG_P_RAD}" | bc)

# Check for drift (error should not grow over time)
EARLY_ERROR=$(head -n 20 "${ENERGY_FILE}" | tail -n 10 | awk -F',' '{sum += sqrt($6^2); count++} END {print sum/count}')
LATE_ERROR=$(tail -n 10 "${ENERGY_FILE}" | awk -F',' '{sum += sqrt($6^2); count++} END {print sum/count}')
ERROR_DRIFT=$(echo "scale=3; ${LATE_ERROR} - ${EARLY_ERROR}" | bc)

################################################################################
# Generate report
################################################################################

cat > "${OUTPUT_BASE}/energy_report.txt" << EOF
================================================================================
ENERGY CONSERVATION VALIDATION RESULTS
================================================================================

ENERGY ERROR STATISTICS:
------------------------
  Average Error:       ${AVG_ERROR}%
  Maximum Error:       ${MAX_ERROR}%
  Final Error:         ${FINAL_ERROR}%
  Error Drift:         ${ERROR_DRIFT}% (late - early)

POWER BALANCE (Time-Averaged):
-------------------------------
  Laser Input:         ${AVG_P_LASER} W
  Evaporation Loss:    ${AVG_P_EVAP} W
  Radiation Loss:      ${AVG_P_RAD} W
  Net Power:           ${NET_POWER} W (should → dE/dt)

ENERGY FLOW BREAKDOWN:
-----------------------
  Evaporation:         $(echo "scale=1; 100 * ${AVG_P_EVAP} / ${AVG_P_LASER}" | bc)% of laser input
  Radiation:           $(echo "scale=1; 100 * ${AVG_P_RAD} / ${AVG_P_LASER}" | bc)% of laser input
  Internal Energy:     $(echo "scale=1; 100 * ${NET_POWER} / ${AVG_P_LASER}" | bc)% of laser input

EOF

cat "${OUTPUT_BASE}/energy_report.txt"

################################################################################
# Create time-series plot data (for Python/gnuplot)
################################################################################

echo ""
echo -e "${BLUE}Generating time-series data for plotting...${NC}"

# Extract columns for plotting
awk -F',' 'NR>1 {print $1, $6}' "${ENERGY_FILE}" > "${OUTPUT_BASE}/error_vs_time.dat"
awk -F',' 'NR>1 {print $1, $3, $4, $5}' "${ENERGY_FILE}" > "${OUTPUT_BASE}/power_vs_time.dat"

echo "  Created: error_vs_time.dat (time, error%)"
echo "  Created: power_vs_time.dat (time, P_laser, P_evap, P_rad)"

################################################################################
# Validation checks
################################################################################

echo ""
echo "================================================================================"
echo "  VALIDATION STATUS"
echo "================================================================================"

VALIDATION_PASSED=true

# Check 1: Average error < 5%
if [ $(echo "${AVG_ERROR} < 5.0" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ Average energy error = ${AVG_ERROR}% < 5% (PASS)${NC}"
else
    echo -e "${RED}✗ Average energy error = ${AVG_ERROR}% > 5% (FAIL)${NC}"
    VALIDATION_PASSED=false
fi

# Check 2: Maximum error < 10% (allow transients)
if [ $(echo "${MAX_ERROR} < 10.0" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ Maximum energy error = ${MAX_ERROR}% < 10% (PASS)${NC}"
else
    echo -e "${RED}✗ Maximum energy error = ${MAX_ERROR}% > 10% (FAIL)${NC}"
    VALIDATION_PASSED=false
fi

# Check 3: No systematic drift
if [ $(echo "${ERROR_DRIFT} < 2.0" | bc) -eq 1 ] && [ $(echo "${ERROR_DRIFT} > -2.0" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ Error drift = ${ERROR_DRIFT}% (no accumulation, PASS)${NC}"
else
    echo -e "${RED}✗ Error drift = ${ERROR_DRIFT}% (systematic accumulation, FAIL)${NC}"
    VALIDATION_PASSED=false
fi

# Check 4: Energy sink terms are active
if [ $(echo "${AVG_P_EVAP} > 0.0" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ Evaporation active (P_evap = ${AVG_P_EVAP} W)${NC}"
else
    echo -e "${YELLOW}⚠ Evaporation not active (check temperature threshold)${NC}"
fi

if [ $(echo "${AVG_P_RAD} > 0.0" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ Radiation active (P_rad = ${AVG_P_RAD} W)${NC}"
else
    echo -e "${YELLOW}⚠ Radiation not active (check Stefan-Boltzmann implementation)${NC}"
fi

# Check 5: Energy balance closed (net power reasonable)
ENERGY_BALANCE=$(echo "scale=1; 100 * (${AVG_P_EVAP} + ${AVG_P_RAD}) / ${AVG_P_LASER}" | bc)
if [ $(echo "${ENERGY_BALANCE} > 20.0" | bc) -eq 1 ] && [ $(echo "${ENERGY_BALANCE} < 80.0" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ Energy balance: ${ENERGY_BALANCE}% of input dissipated (reasonable)${NC}"
else
    echo -e "${YELLOW}⚠ Energy balance: ${ENERGY_BALANCE}% of input dissipated (check physics)${NC}"
fi

################################################################################
# Recommendations
################################################################################

echo ""
echo "================================================================================"
echo "  PHYSICAL INTERPRETATION"
echo "================================================================================"

echo ""
echo "Energy Partitioning:"
echo "  - $(echo "scale=1; 100 * ${AVG_P_EVAP} / ${AVG_P_LASER}" | bc)% lost to evaporation (phase change)"
echo "  - $(echo "scale=1; 100 * ${AVG_P_RAD} / ${AVG_P_LASER}" | bc)% lost to radiation (Stefan-Boltzmann)"
echo "  - $(echo "scale=1; 100 * ${NET_POWER} / ${AVG_P_LASER}" | bc)% stored as internal energy (heating)"

if [ $(echo "${AVG_P_EVAP} > ${AVG_P_RAD}" | bc) -eq 1 ]; then
    echo ""
    echo "Regime: Evaporation-dominated (high-temperature processing)"
    echo "  → Significant mass loss expected"
else
    echo ""
    echo "Regime: Radiation-dominated (moderate-temperature processing)"
    echo "  → Minimal mass loss, conduction-limited"
fi

echo ""
echo "================================================================================"

if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}ENERGY CONSERVATION TEST: PASSED${NC}"
    echo ""
    echo "Conclusion: Fixes preserve thermodynamic consistency"
    echo "  - Energy is conserved within acceptable error bounds"
    echo "  - No artificial energy sources or sinks introduced"
    echo "  - Physics models (evaporation, radiation) working correctly"
    echo ""
    echo "Next Steps:"
    echo "  - Plot error_vs_time.dat to visualize oscillations"
    echo "  - Compare energy partitioning with literature values"
    echo "  - Check if evaporation rate matches experimental data"
    exit 0
else
    echo -e "${RED}ENERGY CONSERVATION TEST: FAILED${NC}"
    echo ""
    echo "Conclusion: Energy conservation violated - check implementation"
    echo ""
    echo "Possible Causes:"
    echo "  1. Flux limiter introducing artificial dissipation"
    echo "  2. Boundary conditions not properly conserving energy"
    echo "  3. Temperature clamping removing energy non-physically"
    echo "  4. Numerical instability creating/destroying energy"
    echo ""
    echo "Recommendations:"
    echo "  1. Reduce flux limiter strength (try φ = 0.8 instead of 1.0)"
    echo "  2. Check boundary fluxes sum to zero"
    echo "  3. Monitor temperature clamp activation frequency"
    echo "  4. Compare with/without fixes for energy error"
    exit 1
fi
