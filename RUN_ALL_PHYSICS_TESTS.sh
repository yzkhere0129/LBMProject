#!/bin/bash
# =============================================================================
# RUN_ALL_PHYSICS_TESTS.sh
# Comprehensive physics test suite runner for LBM multiphysics simulation
# =============================================================================

set -e  # Exit on first error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="/home/yzk/LBMProject/build"
REPORT_DIR="/home/yzk/LBMProject/test_reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="${REPORT_DIR}/test_summary_${TIMESTAMP}.txt"
FAILED_TESTS_FILE="${REPORT_DIR}/failed_tests_${TIMESTAMP}.txt"

# Create report directory if it doesn't exist
mkdir -p "${REPORT_DIR}"

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Test categories
declare -a TEST_CATEGORIES=(
    "VOF_SOLVER"
    "MARANGONI"
    "THERMAL"
    "FLUID_LBM"
    "PHASE_CHANGE"
    "MULTIPHYSICS_COUPLING"
    "ENERGY_CONSERVATION"
    "STABILITY"
)

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

print_section() {
    echo -e "\n${BLUE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Run test category and capture results
run_test_category() {
    local category_name=$1
    local test_pattern=$2
    local timeout=${3:-300}  # Default 5 min timeout

    print_section "Running ${category_name} Tests"

    local start_time=$(date +%s)
    local category_passed=0
    local category_failed=0

    # Run tests and capture output
    if timeout ${timeout} ctest --test-dir "${BUILD_DIR}" -R "${test_pattern}" --output-on-failure 2>&1 | tee -a "${REPORT_FILE}"; then
        local exit_code=${PIPESTATUS[0]}

        # Parse ctest output for test counts
        local test_count=$(grep -o "[0-9]* tests passed" "${REPORT_FILE}" | tail -1 | awk '{print $1}')
        local fail_count=$(grep -o "[0-9]* tests failed" "${REPORT_FILE}" | tail -1 | awk '{print $1}')

        category_passed=${test_count:-0}
        category_failed=${fail_count:-0}

        PASSED_TESTS=$((PASSED_TESTS + category_passed))
        FAILED_TESTS=$((FAILED_TESTS + category_failed))
        TOTAL_TESTS=$((TOTAL_TESTS + category_passed + category_failed))

        if [ $exit_code -eq 0 ]; then
            print_success "${category_name}: All tests passed (${category_passed})"
        else
            print_failure "${category_name}: ${category_failed} test(s) failed"
            echo "${category_name}: ${category_failed} failed" >> "${FAILED_TESTS_FILE}"
        fi
    else
        print_failure "${category_name}: Timeout or error"
        echo "${category_name}: TIMEOUT" >> "${FAILED_TESTS_FILE}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_info "Duration: ${duration}s"
}

# =============================================================================
# Main Test Execution
# =============================================================================

print_header "LBM MULTIPHYSICS TEST SUITE"
print_info "Build directory: ${BUILD_DIR}"
print_info "Report file: ${REPORT_FILE}"
print_info "Started: $(date)"

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo -e "${RED}ERROR: Build directory not found: ${BUILD_DIR}${NC}"
    echo "Please run: mkdir build && cd build && cmake .. && make -j"
    exit 1
fi

# Initialize report file
{
    echo "=============================================="
    echo "LBM MULTIPHYSICS TEST SUITE REPORT"
    echo "=============================================="
    echo "Date: $(date)"
    echo "Build: ${BUILD_DIR}"
    echo "=============================================="
    echo ""
} > "${REPORT_FILE}"

# =============================================================================
# 1. VOF SOLVER TESTS
# =============================================================================
print_header "1. VOF SOLVER TESTS"

print_section "1.1 VOF Advection Tests"
run_test_category "VOF Advection" "test_vof_advection" 300

print_section "1.2 VOF Reconstruction & Curvature"
run_test_category "VOF Reconstruction" "test_vof_reconstruction|test_vof_curvature" 300

print_section "1.3 VOF Mass Conservation"
run_test_category "VOF Mass Conservation" "test_vof_mass_conservation" 180

print_section "1.4 VOF Surface Tension"
run_test_category "VOF Surface Tension" "test_vof_surface_tension|test_vof_contact_angle" 180

print_section "1.5 VOF Evaporation"
run_test_category "VOF Evaporation" "test_vof_evaporation|test_evaporation_temperature_check" 180

print_section "1.6 VOF Recoil Pressure"
run_test_category "VOF Recoil Pressure" "test_recoil" 180

# =============================================================================
# 2. MARANGONI EFFECT TESTS
# =============================================================================
print_header "2. MARANGONI EFFECT TESTS"

print_section "2.1 Marangoni Force Calculation"
run_test_category "Marangoni Force" "test_marangoni_force|test_interface_geometry" 180

print_section "2.2 Marangoni Gradient & Stability"
run_test_category "Marangoni Gradients" "test_marangoni_gradient|test_marangoni_stability" 180

print_section "2.3 Marangoni Integration Tests"
run_test_category "Marangoni Integration" "test_marangoni_flow|test_marangoni_system|test_marangoni_velocity" 300

# =============================================================================
# 3. THERMAL SOLVER TESTS
# =============================================================================
print_header "3. THERMAL SOLVER TESTS"

print_section "3.1 Thermal LBM Core"
run_test_category "Thermal LBM" "test_thermal_lbm|test_lattice_d3q7" 180

print_section "3.2 Thermal Stability Tests"
run_test_category "Thermal Stability" "test_flux_limiter|test_temperature_bounds|test_omega_reduction" 180

print_section "3.3 Thermal Validation Tests"
run_test_category "Thermal Validation" "test_pure_conduction|test_stefan_problem|test_high_pe_stability" 600

print_section "3.4 Laser Heating Tests"
run_test_category "Laser Heating" "test_laser_heating|test_laser_melting|test_laser_source" 300

# =============================================================================
# 4. FLUID LBM TESTS
# =============================================================================
print_header "4. FLUID LBM TESTS"

print_section "4.1 Fluid LBM Core"
run_test_category "Fluid Core" "test_fluid_lbm|test_d3q19|test_bgk|test_streaming" 180

print_section "4.2 Fluid Boundaries"
run_test_category "Fluid Boundaries" "test_fluid_boundaries|test_boundary|test_no_slip" 180

print_section "4.3 Fluid Validation Tests"
run_test_category "Fluid Validation" "test_poiseuille|test_uniform_flow" 300

# =============================================================================
# 5. PHASE CHANGE TESTS
# =============================================================================
print_header "5. PHASE CHANGE TESTS"

print_section "5.1 Phase Change Core"
run_test_category "Phase Change" "test_liquid_fraction|test_enthalpy|test_phase_properties" 180

print_section "5.2 Phase Change Validation"
run_test_category "Phase Change Validation" "test_stefan|test_phase_change_robustness" 300

print_section "5.3 Material Properties"
run_test_category "Materials" "test_materials" 60

# =============================================================================
# 6. MULTIPHYSICS COUPLING TESTS
# =============================================================================
print_header "6. MULTIPHYSICS COUPLING TESTS"

print_section "6.1 Thermal-Fluid Coupling"
run_test_category "Thermal-Fluid" "test_thermal_fluid_coupling|test_mp_thermal_fluid" 300

print_section "6.2 VOF-Fluid Coupling"
run_test_category "VOF-Fluid" "test_vof_fluid_coupling" 300

print_section "6.3 Thermal-VOF Coupling"
run_test_category "Thermal-VOF" "test_thermal_vof_coupling" 300

print_section "6.4 Phase-Fluid Coupling"
run_test_category "Phase-Fluid" "test_phase_fluid_coupling|test_darcy" 300

print_section "6.5 Force Balance Tests"
run_test_category "Force Balance" "test_force_balance|test_force_magnitude|test_force_direction" 180

# =============================================================================
# 7. ENERGY CONSERVATION TESTS
# =============================================================================
print_header "7. ENERGY CONSERVATION TESTS"

print_section "7.1 Energy Conservation Tests"
run_test_category "Energy Conservation" "test_energy_conservation" 360

print_section "7.2 Evaporation Energy Balance"
run_test_category "Evaporation Energy" "test_evaporation_energy_balance|test_evaporation_hertz_knudsen" 300

# =============================================================================
# 8. STABILITY & VALIDATION TESTS
# =============================================================================
print_header "8. STABILITY & VALIDATION TESTS"

print_section "8.1 CFL Stability"
run_test_category "CFL Stability" "test_cfl|test_extreme_gradients" 300

print_section "8.2 Subcycling Tests"
run_test_category "Subcycling" "test_subcycling|test_vof_subcycling" 300

print_section "8.3 Unit Conversion Tests"
run_test_category "Unit Conversion" "test_unit_conversion" 120

print_section "8.4 Regression Tests"
run_test_category "Regression" "test_regression|test_substrate_bc_stability" 300

print_section "8.5 Validation Suite"
run_test_category "Validation Suite" "test_week3_readiness|test_divergence_free|test_realistic_lpbf" 300

# =============================================================================
# Generate Summary Report
# =============================================================================

print_header "TEST SUITE SUMMARY"

{
    echo ""
    echo "=============================================="
    echo "TEST EXECUTION SUMMARY"
    echo "=============================================="
    echo "Total Tests:   ${TOTAL_TESTS}"
    echo "Passed:        ${PASSED_TESTS}"
    echo "Failed:        ${FAILED_TESTS}"
    echo "Skipped:       ${SKIPPED_TESTS}"
    echo ""

    if [ ${FAILED_TESTS} -eq 0 ]; then
        echo "STATUS: ALL TESTS PASSED"
        echo ""
    else
        echo "STATUS: FAILURES DETECTED"
        echo ""
        echo "Failed test categories:"
        if [ -f "${FAILED_TESTS_FILE}" ]; then
            cat "${FAILED_TESTS_FILE}"
        fi
        echo ""
    fi

    echo "Completion time: $(date)"
    echo "=============================================="
} | tee -a "${REPORT_FILE}"

# Print summary to console with colors
echo ""
echo -e "${CYAN}=============================================${NC}"
echo -e "${CYAN}TEST RESULTS${NC}"
echo -e "${CYAN}=============================================${NC}"
echo -e "Total Tests:   ${TOTAL_TESTS}"
echo -e "${GREEN}Passed:        ${PASSED_TESTS}${NC}"
if [ ${FAILED_TESTS} -gt 0 ]; then
    echo -e "${RED}Failed:        ${FAILED_TESTS}${NC}"
else
    echo -e "Failed:        ${FAILED_TESTS}"
fi
echo -e "Skipped:       ${SKIPPED_TESTS}"
echo -e "${CYAN}=============================================${NC}"

if [ ${FAILED_TESTS} -eq 0 ]; then
    echo -e "${GREEN}SUCCESS: All physics tests passed!${NC}"
    print_info "Full report: ${REPORT_FILE}"
    exit 0
else
    echo -e "${RED}FAILURE: ${FAILED_TESTS} test(s) failed${NC}"
    print_info "Full report: ${REPORT_FILE}"
    print_info "Failed tests: ${FAILED_TESTS_FILE}"
    exit 1
fi
