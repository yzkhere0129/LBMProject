#!/bin/bash
###############################################################################
# Week 3 Certification Report Generator
#
# This script automatically generates the Week 3 certification report by:
# 1. Running all validation tests
# 2. Analyzing convergence data
# 3. Computing scores
# 4. Filling the report template
# 5. Making GO/NO-GO decision
#
# Usage:
#   cd /home/yzk/LBMProject/build
#   bash ../scripts/generate_week3_report.sh
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
BUILD_DIR="/home/yzk/LBMProject/build"
SCRIPTS_DIR="/home/yzk/LBMProject/scripts"
REPORT_TEMPLATE="/home/yzk/LBMProject/WEEK3_FINAL_CERTIFICATION.md.template"
FINAL_REPORT="${BUILD_DIR}/WEEK3_FINAL_CERTIFICATION.md"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}WEEK 3 CERTIFICATION REPORT GENERATOR${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Change to build directory
cd "${BUILD_DIR}" || exit 1

# Step 1: Run validation tests
echo -e "${YELLOW}[Step 1] Running Week 3 Validation Tests...${NC}"
if [ -f "./test_week3_readiness" ]; then
    ./test_week3_readiness > validation_results.txt 2>&1
    TEST_RESULT=$?

    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Validation tests PASSED${NC}"
    else
        echo -e "${RED}✗ Some validation tests FAILED (see validation_results.txt)${NC}"
    fi
else
    echo -e "${RED}ERROR: test_week3_readiness executable not found!${NC}"
    echo "Build the test first: cmake --build . --target test_week3_readiness"
    exit 1
fi
echo ""

# Step 2: Analyze convergence
echo -e "${YELLOW}[Step 2] Analyzing Timestep Convergence...${NC}"
if [ -f "${SCRIPTS_DIR}/analyze_convergence_final.py" ]; then
    python3 "${SCRIPTS_DIR}/analyze_convergence_final.py" > convergence_results.txt 2>&1
    CONV_RESULT=$?

    if [ $CONV_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Convergence analysis PASSED${NC}"
    else
        echo -e "${YELLOW}⚠ Convergence analysis reported issues (see convergence_results.txt)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Convergence analysis script not found - skipping${NC}"
fi
echo ""

# Step 3: Check steady state
echo -e "${YELLOW}[Step 3] Checking Steady-State Achievement...${NC}"
if [ -f "steady_state_verification.log" ]; then
    # Extract final dE/dt value
    LAST_DEDT=$(grep "dE/dt" steady_state_verification.log | tail -1 | grep -oP '[-+]?[0-9]*\.?[0-9]+' | tail -1 || echo "N/A")
    echo "Final |dE/dt|: ${LAST_DEDT} W"

    # Check if < 0.5 W
    if (( $(echo "$LAST_DEDT < 0.5" | bc -l) )); then
        echo -e "${GREEN}✓ Steady state achieved (|dE/dt| < 0.5 W)${NC}"
    else
        echo -e "${YELLOW}⚠ Steady state not fully achieved (|dE/dt| = ${LAST_DEDT} W)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Steady-state log file not found - run simulation first${NC}"
fi
echo ""

# Step 4: Compute overall score
echo -e "${YELLOW}[Step 4] Computing Overall Score...${NC}"

# Initialize scores
THERMAL_TAU_SCORE=0
FLUID_TAU_SCORE=0
STEADY_STATE_SCORE=0
ENERGY_SCORE=0
STABILITY_SCORE=0
CONVERGENCE_SCORE=0
TEMP_SCORE=0
CFL_SCORE=0

# Parse test results (simplified - would need actual parsing logic)
echo "Parsing test results from validation_results.txt..."

# Check for test passes
if grep -q "ThermalTauScaling.*PASSED" validation_results.txt; then
    THERMAL_TAU_SCORE=20
    echo -e "${GREEN}  Test 1 (Thermal Tau): 20/20${NC}"
else
    echo -e "${RED}  Test 1 (Thermal Tau): 0/20${NC}"
fi

if grep -q "FluidTauScaling.*PASSED" validation_results.txt; then
    FLUID_TAU_SCORE=20
    echo -e "${GREEN}  Test 2 (Fluid Tau): 20/20${NC}"
else
    echo -e "${RED}  Test 2 (Fluid Tau): 0/20${NC}"
fi

if grep -q "SteadyStateAchievement.*PASSED" validation_results.txt; then
    STEADY_STATE_SCORE=15
    echo -e "${GREEN}  Test 3 (Steady State): 15/15${NC}"
else
    echo -e "${RED}  Test 3 (Steady State): 0/15${NC}"
fi

if grep -q "EnergyConservation.*PASSED" validation_results.txt; then
    ENERGY_SCORE=15
    echo -e "${GREEN}  Test 4 (Energy): 15/15${NC}"
else
    echo -e "${RED}  Test 4 (Energy): 0/15${NC}"
fi

if grep -q "NumericalStability.*PASSED" validation_results.txt; then
    STABILITY_SCORE=10
    echo -e "${GREEN}  Test 5 (Stability): 10/10${NC}"
else
    echo -e "${RED}  Test 5 (Stability): 0/10${NC}"
fi

if grep -q "TimestepConvergence.*PASSED" validation_results.txt; then
    CONVERGENCE_SCORE=10
    echo -e "${GREEN}  Test 6 (Convergence): 10/10${NC}"
else
    echo -e "${YELLOW}  Test 6 (Convergence): 0/10${NC}"
fi

if grep -q "TemperatureValidation.*PASSED" validation_results.txt; then
    TEMP_SCORE=5
    echo -e "${GREEN}  Test 7 (Temperature): 5/5${NC}"
else
    echo -e "${YELLOW}  Test 7 (Temperature): 0/5${NC}"
fi

if grep -q "CFLStability.*PASSED" validation_results.txt; then
    CFL_SCORE=5
    echo -e "${GREEN}  Test 8 (CFL): 5/5${NC}"
else
    echo -e "${YELLOW}  Test 8 (CFL): 0/5${NC}"
fi

# Compute totals
P0_SCORE=$((THERMAL_TAU_SCORE + FLUID_TAU_SCORE + STEADY_STATE_SCORE + ENERGY_SCORE + STABILITY_SCORE))
P1_SCORE=$((CONVERGENCE_SCORE + TEMP_SCORE + CFL_SCORE))
TOTAL_SCORE=$((P0_SCORE + P1_SCORE))

echo ""
echo -e "${BLUE}Scorecard:${NC}"
echo "  P0 (Critical):     ${P0_SCORE}/80"
echo "  P1 (High Priority): ${P1_SCORE}/20"
echo "  --------------------------------"
echo "  TOTAL:             ${TOTAL_SCORE}/100"
echo ""

# Step 5: Make GO/NO-GO decision
echo -e "${YELLOW}[Step 5] Making GO/NO-GO Decision...${NC}"

if [ $TOTAL_SCORE -ge 85 ]; then
    VERDICT="FULL GO"
    COLOR="${GREEN}"
    SYMBOL="✓"
elif [ $TOTAL_SCORE -ge 70 ]; then
    VERDICT="CONDITIONAL GO"
    COLOR="${YELLOW}"
    SYMBOL="⚠"
else
    VERDICT="NO GO"
    COLOR="${RED}"
    SYMBOL="✗"
fi

echo -e "${COLOR}${SYMBOL} ${VERDICT}${NC}"
echo "Score: ${TOTAL_SCORE}/100"
echo ""

# Step 6: Generate report
echo -e "${YELLOW}[Step 6] Generating Certification Report...${NC}"

# Simple template substitution (in practice, use a proper template engine)
REPORT_DATE=$(date +"%Y-%m-%d")

cat > "${FINAL_REPORT}" <<EOF
# Week 3 Platform Certification - Final Report

**Date**: ${REPORT_DATE}
**Status**: ${VERDICT}
**Overall Score**: ${TOTAL_SCORE}/100

---

## Executive Summary

The Week 3 readiness validation suite has been completed with the following results:

**Overall Verdict**: ${VERDICT}
**Total Score**: ${TOTAL_SCORE}/100 points
- P0 (Critical): ${P0_SCORE}/80 points
- P1 (High Priority): ${P1_SCORE}/20 points

**Key Findings**:
- Thermal tau scaling: $([ $THERMAL_TAU_SCORE -eq 20 ] && echo "PASS" || echo "FAIL")
- Fluid tau scaling: $([ $FLUID_TAU_SCORE -eq 20 ] && echo "PASS" || echo "FAIL")
- Steady state: $([ $STEADY_STATE_SCORE -eq 15 ] && echo "PASS" || echo "FAIL")
- Energy conservation: $([ $ENERGY_SCORE -eq 15 ] && echo "PASS" || echo "FAIL")
- Numerical stability: $([ $STABILITY_SCORE -eq 10 ] && echo "PASS" || echo "FAIL")

---

## Validation Test Results

### P0 (Critical) Tests

1. **Thermal Tau Scaling**: ${THERMAL_TAU_SCORE}/20 points
2. **Fluid Tau Scaling**: ${FLUID_TAU_SCORE}/20 points
3. **Steady State**: ${STEADY_STATE_SCORE}/15 points
4. **Energy Conservation**: ${ENERGY_SCORE}/15 points
5. **Numerical Stability**: ${STABILITY_SCORE}/10 points

**P0 Subtotal**: ${P0_SCORE}/80 points

### P1 (High Priority) Tests

6. **Timestep Convergence**: ${CONVERGENCE_SCORE}/10 points
7. **Temperature Validation**: ${TEMP_SCORE}/5 points
8. **CFL Stability**: ${CFL_SCORE}/5 points

**P1 Subtotal**: ${P1_SCORE}/20 points

---

## Decision

**VERDICT**: ${VERDICT}

$(if [ "${VERDICT}" = "FULL GO" ]; then
    echo "**Recommendation**: Proceed immediately with Week 3 vapor phase implementation."
    echo "All critical requirements passed. Use any validated timestep (recommend dt=0.10 μs)."
elif [ "${VERDICT}" = "CONDITIONAL GO" ]; then
    echo "**Recommendation**: Proceed with Week 3 with caution."
    echo "Use baseline dt=0.10 μs only. Monitor closely and address P1 failures in parallel."
else
    echo "**Recommendation**: DO NOT proceed with Week 3."
    echo "Critical requirements not met. Additional debugging and validation required."
fi)

---

## Detailed Results

See attached files:
- Validation test output: validation_results.txt
- Convergence analysis: convergence_results.txt
- Steady-state log: steady_state_verification.log

---

**Report Generated**: ${REPORT_DATE}
**By**: Automated Week 3 Certification System

EOF

echo -e "${GREEN}✓ Report generated: ${FINAL_REPORT}${NC}"
echo ""

# Step 7: Summary
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}SUMMARY${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Overall Score: ${TOTAL_SCORE}/100"
echo "Verdict: ${VERDICT}"
echo ""
echo "Generated Files:"
echo "  - ${FINAL_REPORT}"
echo "  - validation_results.txt"
echo "  - convergence_results.txt"
echo ""
echo -e "${BLUE}================================================================================${NC}"

exit 0
