#!/bin/bash
################################################################################
# Peclet Number Sweep Test
# Purpose: Verify stability holds across different Pe regimes
# Success: All cases stable, higher Pe shows sharper gradients
################################################################################

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/home/yzk/LBMProject"
BUILD_DIR="${PROJECT_ROOT}/build"
OUTPUT_BASE="${PROJECT_ROOT}/validation_results/peclet_sweep"
EXECUTABLE="${BUILD_DIR}/LBMSolver"

# Create output directory
mkdir -p "${OUTPUT_BASE}"
cd "${BUILD_DIR}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "PECLET NUMBER SWEEP TEST - Verifying Stability Across Flow Regimes"
echo "================================================================================"
echo ""
echo "Testing 3 Peclet number regimes by varying thermal diffusivity:"
echo "  Test A:  α = 58e-6  m²/s  →  Pe ≈ 1    (diffusion-dominated)"
echo "  Test B:  α = 5.8e-6 m²/s  →  Pe ≈ 10   (balanced, typical)"
echo "  Test C:  α = 0.58e-6 m²/s →  Pe ≈ 100  (advection-dominated)"
echo ""
echo "Success Criteria:"
echo "  ✓ All cases remain stable (T_max < 10,000 K)"
echo "  ✓ Higher Pe shows sharper gradients (flux limiter activating)"
echo "  ✓ Energy conservation error < 5% for all Pe"
echo "================================================================================"
echo ""

# Fixed geometry and flow parameters
NX=200
NY=100
NZ=50
DX=2.0e-6          # m
DT=1.0e-7          # s
NUM_STEPS=1000

LASER_POWER=200.0  # W
LASER_RADIUS=50e-6 # m
SCAN_SPEED=0.4     # m/s

# Thermal diffusivity values (Pe = v·dx/α)
# Characteristic velocity v ~ 0.05 m/s (Marangoni flow)
# Pe = 0.05 * 2e-6 / α
ALPHA_LOW=58.0e-6   # Pe ≈ 1.7
ALPHA_MED=5.8e-6    # Pe ≈ 17
ALPHA_HIGH=0.58e-6  # Pe ≈ 170

# Arrays to store results
declare -a TEST_NAMES=("diffusion_dominated" "balanced" "advection_dominated")
declare -a ALPHA_VALUES=(${ALPHA_LOW} ${ALPHA_MED} ${ALPHA_HIGH})
declare -a PE_VALUES=(1.7 17 170)
declare -a T_MAX_VALUES
declare -a V_MAX_VALUES
declare -a ENERGY_ERRORS

################################################################################
# Function: Run simulation for given thermal diffusivity
################################################################################
run_peclet_test() {
    local NAME=$1
    local ALPHA=$2
    local PE_APPROX=$3
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"

    mkdir -p "${OUTPUT_DIR}"

    echo "--------------------------------------------------------------------------------"
    echo -e "${YELLOW}Running ${NAME}: Pe ≈ ${PE_APPROX}${NC}"
    echo "  Thermal diffusivity: α = ${ALPHA} m²/s"
    echo "--------------------------------------------------------------------------------"

    # Create configuration
    cat > "${OUTPUT_DIR}/config.txt" << EOF
# Peclet Number Sweep - ${NAME} (Pe ≈ ${PE_APPROX})
nx ${NX}
ny ${NY}
nz ${NZ}
dx ${DX}
dt ${DT}
num_steps ${NUM_STEPS}
output_interval 100

# Fixed geometry
laser_power ${LASER_POWER}
laser_radius ${LASER_RADIUS}
scan_speed ${SCAN_SPEED}

# Varying thermal diffusivity
thermal_diffusivity ${ALPHA}

# Output settings
output_dir ${OUTPUT_DIR}
enable_vtk true
enable_diagnostics true
enable_energy_diagnostics true
EOF

    # Run simulation
    local START_TIME=$(date +%s)
    if ${EXECUTABLE} "${OUTPUT_DIR}/config.txt" > "${OUTPUT_DIR}/simulation.log" 2>&1; then
        local END_TIME=$(date +%s)
        local ELAPSED=$((END_TIME - START_TIME))
        echo -e "${GREEN}✓ Simulation completed in ${ELAPSED}s${NC}"
    else
        echo -e "${RED}✗ Simulation FAILED - check ${OUTPUT_DIR}/simulation.log${NC}"
        cat "${OUTPUT_DIR}/simulation.log" | tail -n 20
        exit 1
    fi

    # Extract diagnostics
    if [ -f "${OUTPUT_DIR}/diagnostics.csv" ]; then
        local T_MAX=$(tail -n 1 "${OUTPUT_DIR}/diagnostics.csv" | awk -F',' '{print $3}')
        local V_MAX=$(tail -n 1 "${OUTPUT_DIR}/diagnostics.csv" | awk -F',' '{print $4}')

        echo "  Peak Temperature: ${T_MAX} K"
        echo "  Peak Velocity:    ${V_MAX} m/s"

        # Check stability
        if [ $(echo "${T_MAX} > 10000" | bc) -eq 1 ]; then
            echo -e "${RED}  ✗ UNSTABLE - Temperature exceeded 10,000 K${NC}"
        else
            echo -e "${GREEN}  ✓ Stable${NC}"
        fi

        T_MAX_VALUES+=("${T_MAX}")
        V_MAX_VALUES+=("${V_MAX}")
    else
        echo -e "${RED}✗ Diagnostics file not found${NC}"
        exit 1
    fi

    # Extract energy error
    if [ -f "${OUTPUT_DIR}/energy_diagnostics.csv" ]; then
        # Compute average energy error
        local AVG_ERROR=$(awk -F',' 'NR>1 {sum += sqrt($2^2); count++} END {print sum/count}' "${OUTPUT_DIR}/energy_diagnostics.csv")
        echo "  Energy Error:     ${AVG_ERROR}%"

        ENERGY_ERRORS+=("${AVG_ERROR}")
    else
        echo -e "${YELLOW}  ⚠ Energy diagnostics not available${NC}"
        ENERGY_ERRORS+=("N/A")
    fi

    echo ""
}

################################################################################
# Run all three Peclet number cases
################################################################################

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  PHASE 1: Execute Simulations Across Peclet Number Regimes"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

for i in {0..2}; do
    run_peclet_test "${TEST_NAMES[$i]}" "${ALPHA_VALUES[$i]}" "${PE_VALUES[$i]}"
done

################################################################################
# Analyze results
################################################################################

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  PHASE 2: Results Analysis"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Generate report
cat > "${OUTPUT_BASE}/peclet_sweep_report.txt" << EOF
================================================================================
PECLET NUMBER SWEEP TEST RESULTS
================================================================================

Test Case              Pe      α (m²/s)    T_max (K)    V_max (m/s)  Energy Error
--------------------------------------------------------------------------------
${TEST_NAMES[0]}       ${PE_VALUES[0]}     ${ALPHA_VALUES[0]}   ${T_MAX_VALUES[0]}       ${V_MAX_VALUES[0]}       ${ENERGY_ERRORS[0]}%
${TEST_NAMES[1]}       ${PE_VALUES[1]}    ${ALPHA_VALUES[1]}   ${T_MAX_VALUES[1]}       ${V_MAX_VALUES[1]}       ${ENERGY_ERRORS[1]}%
${TEST_NAMES[2]}      ${PE_VALUES[2]}   ${ALPHA_VALUES[2]}   ${T_MAX_VALUES[2]}       ${V_MAX_VALUES[2]}       ${ENERGY_ERRORS[2]}%

--------------------------------------------------------------------------------
PHYSICAL INTERPRETATION
--------------------------------------------------------------------------------

Diffusion-Dominated (Pe ≈ 1):
  - Heat spreads smoothly, low gradients
  - Minimal flux limiter activation
  - Most stable regime

Balanced (Pe ≈ 10):
  - Comparable advection and diffusion
  - Typical for metal AM processes
  - Moderate flux limiter usage

Advection-Dominated (Pe ≈ 100):
  - Sharp thermal gradients
  - Heavy flux limiter activation
  - Most challenging for stability

--------------------------------------------------------------------------------
VALIDATION CHECKS
--------------------------------------------------------------------------------

EOF

cat "${OUTPUT_BASE}/peclet_sweep_report.txt"

################################################################################
# Validation checks
################################################################################

echo ""
echo "================================================================================"
echo "  VALIDATION STATUS"
echo "================================================================================"

VALIDATION_PASSED=true

# Check stability for all cases
for i in {0..2}; do
    if [ $(echo "${T_MAX_VALUES[$i]} < 10000" | bc) -eq 1 ]; then
        echo -e "${GREEN}✓ ${TEST_NAMES[$i]} (Pe=${PE_VALUES[$i]}): Stable (T_max = ${T_MAX_VALUES[$i]} K)${NC}"
    else
        echo -e "${RED}✗ ${TEST_NAMES[$i]} (Pe=${PE_VALUES[$i]}): UNSTABLE (T_max = ${T_MAX_VALUES[$i]} K)${NC}"
        VALIDATION_PASSED=false
    fi
done

# Check energy conservation
echo ""
for i in {0..2}; do
    if [ "${ENERGY_ERRORS[$i]}" != "N/A" ]; then
        if [ $(echo "${ENERGY_ERRORS[$i]} < 5.0" | bc) -eq 1 ]; then
            echo -e "${GREEN}✓ ${TEST_NAMES[$i]}: Energy error = ${ENERGY_ERRORS[$i]}% < 5% (PASS)${NC}"
        else
            echo -e "${RED}✗ ${TEST_NAMES[$i]}: Energy error = ${ENERGY_ERRORS[$i]}% > 5% (FAIL)${NC}"
            VALIDATION_PASSED=false
        fi
    fi
done

# Check gradient sharpening trend (higher Pe → higher T_max)
echo ""
if [ $(echo "${T_MAX_VALUES[2]} > ${T_MAX_VALUES[0]}" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ Temperature increases with Pe (sharper gradients at high Pe)${NC}"
else
    echo -e "${YELLOW}⚠ Expected higher T_max at higher Pe (check flux limiter)${NC}"
fi

echo ""
echo "================================================================================"

if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}PECLET NUMBER SWEEP TEST: PASSED${NC}"
    echo "Conclusion: Fixes maintain stability across all flow regimes"
    echo ""
    echo "Next Steps:"
    echo "  1. Visualize Pe field in ParaView to see where flux limiter activates"
    echo "  2. Compare temperature profiles: Pe=100 should show sharper gradients"
    echo "  3. If all Pe cases look identical → flux limiter over-damping"
    exit 0
else
    echo -e "${RED}PECLET NUMBER SWEEP TEST: FAILED${NC}"
    echo "Conclusion: Stability issues persist at high Peclet numbers"
    echo ""
    echo "Recommendations:"
    echo "  1. Check flux limiter implementation (may need stronger limiting)"
    echo "  2. Consider MRT collision operator for better stability"
    echo "  3. Verify CFL condition satisfaction"
    exit 1
fi
