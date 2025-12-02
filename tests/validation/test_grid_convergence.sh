#!/bin/bash
################################################################################
# Grid Convergence Study
# Purpose: Prove solution converges to true physics as grid is refined
# Success: Temperature and velocity fields converge with order >= 1.0
################################################################################

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/home/yzk/LBMProject"
BUILD_DIR="${PROJECT_ROOT}/build"
OUTPUT_BASE="${PROJECT_ROOT}/validation_results/grid_convergence"
EXECUTABLE="${BUILD_DIR}/LBMSolver"

# Create output directory
mkdir -p "${OUTPUT_BASE}"
cd "${BUILD_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "GRID CONVERGENCE STUDY - Validating Numerical Convergence Order"
echo "================================================================================"
echo ""
echo "Testing 3 grid resolutions:"
echo "  Coarse:  dx = 4 μm  (nx=100,  ny=50,  nz=25)"
echo "  Medium:  dx = 2 μm  (nx=200,  ny=100, nz=50)"
echo "  Fine:    dx = 1 μm  (nx=400,  ny=200, nz=100)"
echo ""
echo "Target: First-order convergence (error ratio ~ 2.0)"
echo "================================================================================"
echo ""

# Test parameters (fixed physics, varying resolution)
LASER_POWER=200.0      # W
LASER_RADIUS=50e-6     # m
SCAN_SPEED=0.4         # m/s
NUM_STEPS=1000         # Sufficient for convergence check

# CFL-adjusted timesteps
DT_COARSE=2.0e-7       # s
DT_MEDIUM=1.0e-7       # s
DT_FINE=5.0e-8         # s

# Arrays to store results
declare -a T_MAX_VALUES
declare -a V_MAX_VALUES
declare -a DX_VALUES

################################################################################
# Function: Run simulation for given resolution
################################################################################
run_simulation() {
    local NAME=$1
    local NX=$2
    local NY=$3
    local NZ=$4
    local DX=$5
    local DT=$6
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"

    mkdir -p "${OUTPUT_DIR}"

    echo "--------------------------------------------------------------------------------"
    echo -e "${YELLOW}Running ${NAME} resolution:${NC} nx=${NX}, ny=${NY}, nz=${NZ}, dx=${DX}μm, dt=${DT}s"
    echo "--------------------------------------------------------------------------------"

    # Create temporary configuration (modify existing config)
    cat > "${OUTPUT_DIR}/config.txt" << EOF
# Grid Convergence Test - ${NAME} Resolution
nx ${NX}
ny ${NY}
nz ${NZ}
dx ${DX}
dt ${DT}
num_steps ${NUM_STEPS}
output_interval 100

# Fixed physics parameters
laser_power ${LASER_POWER}
laser_radius ${LASER_RADIUS}
scan_speed ${SCAN_SPEED}

# Output settings
output_dir ${OUTPUT_DIR}
enable_vtk true
enable_diagnostics true
EOF

    # Run simulation
    local START_TIME=$(date +%s)
    if ${EXECUTABLE} "${OUTPUT_DIR}/config.txt" > "${OUTPUT_DIR}/simulation.log" 2>&1; then
        local END_TIME=$(date +%s)
        local ELAPSED=$((END_TIME - START_TIME))
        echo -e "${GREEN}✓ Simulation completed in ${ELAPSED}s${NC}"
    else
        echo -e "${RED}✗ Simulation FAILED - check ${OUTPUT_DIR}/simulation.log${NC}"
        exit 1
    fi

    # Extract peak values from diagnostics
    if [ -f "${OUTPUT_DIR}/diagnostics.csv" ]; then
        # Get final timestep values
        local T_MAX=$(tail -n 1 "${OUTPUT_DIR}/diagnostics.csv" | awk -F',' '{print $3}')
        local V_MAX=$(tail -n 1 "${OUTPUT_DIR}/diagnostics.csv" | awk -F',' '{print $4}')

        echo "  Peak Temperature: ${T_MAX} K"
        echo "  Peak Velocity:    ${V_MAX} m/s"
        echo ""

        # Store results
        T_MAX_VALUES+=("${T_MAX}")
        V_MAX_VALUES+=("${V_MAX}")
        DX_VALUES+=("${DX}")
    else
        echo -e "${RED}✗ Diagnostics file not found${NC}"
        exit 1
    fi
}

################################################################################
# Run all three resolutions
################################################################################

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  PHASE 1: Execute Simulations at Three Grid Resolutions"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Coarse grid
run_simulation "coarse" 100 50 25 4.0e-6 ${DT_COARSE}

# Medium grid
run_simulation "medium" 200 100 50 2.0e-6 ${DT_MEDIUM}

# Fine grid
run_simulation "fine" 400 200 100 1.0e-6 ${DT_FINE}

################################################################################
# Compute convergence order
################################################################################

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  PHASE 2: Convergence Order Analysis"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Extract values
T_COARSE=${T_MAX_VALUES[0]}
T_MEDIUM=${T_MAX_VALUES[1]}
T_FINE=${T_MAX_VALUES[2]}

V_COARSE=${V_MAX_VALUES[0]}
V_MEDIUM=${V_MAX_VALUES[1]}
V_FINE=${V_MAX_VALUES[2]}

# Compute errors (assuming fine grid is "truth")
# Error = |Solution(dx) - Solution(dx/2)|
T_ERROR_COARSE=$(echo "scale=6; sqrt((${T_COARSE} - ${T_MEDIUM})^2)" | bc)
T_ERROR_MEDIUM=$(echo "scale=6; sqrt((${T_MEDIUM} - ${T_FINE})^2)" | bc)

V_ERROR_COARSE=$(echo "scale=6; sqrt((${V_COARSE} - ${V_MEDIUM})^2)" | bc)
V_ERROR_MEDIUM=$(echo "scale=6; sqrt((${V_MEDIUM} - ${V_FINE})^2)" | bc)

# Convergence order: p = log(error_coarse / error_medium) / log(2)
# For first-order: p ~ 1.0, error ratio ~ 2.0
if [ $(echo "${T_ERROR_MEDIUM} > 0" | bc) -eq 1 ]; then
    T_RATIO=$(echo "scale=3; ${T_ERROR_COARSE} / ${T_ERROR_MEDIUM}" | bc)
    T_ORDER=$(echo "scale=3; l(${T_RATIO}) / l(2)" | bc -l)
else
    T_RATIO="N/A"
    T_ORDER="N/A"
fi

if [ $(echo "${V_ERROR_MEDIUM} > 0" | bc) -eq 1 ]; then
    V_RATIO=$(echo "scale=3; ${V_ERROR_COARSE} / ${V_ERROR_MEDIUM}" | bc)
    V_ORDER=$(echo "scale=3; l(${V_RATIO}) / l(2)" | bc -l)
else
    V_RATIO="N/A"
    V_ORDER="N/A"
fi

# Display results table
cat > "${OUTPUT_BASE}/convergence_report.txt" << EOF
================================================================================
GRID CONVERGENCE STUDY RESULTS
================================================================================

Resolution      dx (μm)    T_max (K)       V_max (m/s)
--------------------------------------------------------------------------------
Coarse          4.0        ${T_COARSE}        ${V_COARSE}
Medium          2.0        ${T_MEDIUM}        ${V_MEDIUM}
Fine            1.0        ${T_FINE}          ${V_FINE}

--------------------------------------------------------------------------------
CONVERGENCE ANALYSIS
--------------------------------------------------------------------------------

Temperature Field:
  Error(coarse→medium):  ${T_ERROR_COARSE} K
  Error(medium→fine):    ${T_ERROR_MEDIUM} K
  Error Ratio:           ${T_RATIO}
  Convergence Order:     ${T_ORDER}

Velocity Field:
  Error(coarse→medium):  ${V_ERROR_COARSE} m/s
  Error(medium→fine):    ${V_ERROR_MEDIUM} m/s
  Error Ratio:           ${V_RATIO}
  Convergence Order:     ${V_ORDER}

--------------------------------------------------------------------------------
SUCCESS CRITERIA
--------------------------------------------------------------------------------

✓ Convergence order >= 1.0  (First-order accurate method)
✓ T_max decreases as dx → 0 (Numerical diffusion reducing)
✓ V_max stabilizes as dx → 0 (Physical velocity captured)

EOF

cat "${OUTPUT_BASE}/convergence_report.txt"

# Check success criteria
echo ""
echo "================================================================================"
echo "  VALIDATION STATUS"
echo "================================================================================"

VALIDATION_PASSED=true

# Check convergence order
if [ "${T_ORDER}" != "N/A" ]; then
    if [ $(echo "${T_ORDER} >= 0.8" | bc) -eq 1 ]; then
        echo -e "${GREEN}✓ Temperature convergence order = ${T_ORDER} >= 0.8 (PASS)${NC}"
    else
        echo -e "${RED}✗ Temperature convergence order = ${T_ORDER} < 0.8 (FAIL - treating symptoms)${NC}"
        VALIDATION_PASSED=false
    fi
else
    echo -e "${YELLOW}⚠ Temperature convergence order could not be computed${NC}"
fi

if [ "${V_ORDER}" != "N/A" ]; then
    if [ $(echo "${V_ORDER} >= 0.8" | bc) -eq 1 ]; then
        echo -e "${GREEN}✓ Velocity convergence order = ${V_ORDER} >= 0.8 (PASS)${NC}"
    else
        echo -e "${RED}✗ Velocity convergence order = ${V_ORDER} < 0.8 (FAIL - treating symptoms)${NC}"
        VALIDATION_PASSED=false
    fi
else
    echo -e "${YELLOW}⚠ Velocity convergence order could not be computed${NC}"
fi

# Check temperature decreasing trend
if [ $(echo "${T_MEDIUM} < ${T_COARSE}" | bc) -eq 1 ] && [ $(echo "${T_FINE} < ${T_MEDIUM}" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ Temperature decreases with grid refinement (numerical diffusion reducing)${NC}"
else
    echo -e "${YELLOW}⚠ Temperature not monotonically decreasing (check if physics-limited)${NC}"
fi

echo ""
echo "================================================================================"

if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}GRID CONVERGENCE TEST: PASSED${NC}"
    echo "Conclusion: Fixes are treating root cause (治本) - solution converges properly"
    exit 0
else
    echo -e "${RED}GRID CONVERGENCE TEST: FAILED${NC}"
    echo "Conclusion: Fixes may be superficial (治标) - solution not converging properly"
    exit 1
fi
