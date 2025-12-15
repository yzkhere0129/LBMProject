#!/bin/bash
#
# Stability Test Suite Runner
# Comprehensive testing for thermal LBM stability fixes
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BUILD_DIR="/home/yzk/LBMProject/build"

# Print header
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  STABILITY REGRESSION TEST SUITE${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory not found at $BUILD_DIR${NC}"
    echo "Please run: mkdir build && cd build && cmake .. && make"
    exit 1
fi

cd "$BUILD_DIR"

# Parse command line arguments
TEST_LEVEL="${1:-quick}"

case "$TEST_LEVEL" in
    quick|fast|pre-commit)
        echo -e "${YELLOW}Running QUICK stability tests (pre-commit)${NC}"
        echo "Expected time: < 30 seconds"
        echo ""

        make run_stability_quick
        ;;

    medium|pre-push)
        echo -e "${YELLOW}Running MEDIUM stability tests (pre-push)${NC}"
        echo "Expected time: < 2 minutes"
        echo ""

        make run_stability_medium
        ;;

    full|all)
        echo -e "${YELLOW}Running FULL stability test suite${NC}"
        echo "Expected time: < 5 minutes"
        echo ""

        make run_stability_full
        ;;

    unit)
        echo -e "${YELLOW}Running UNIT tests only${NC}"
        echo ""

        ctest -R "test_flux_limiter|test_temperature_bounds" --output-on-failure
        ;;

    regression)
        echo -e "${YELLOW}Running REGRESSION tests only${NC}"
        echo ""

        ctest -R "test_omega_reduction" --output-on-failure
        ;;

    integration)
        echo -e "${YELLOW}Running INTEGRATION tests only${NC}"
        echo ""

        ctest -R "test_high_pe_stability" --output-on-failure
        ;;

    performance)
        echo -e "${YELLOW}Running PERFORMANCE benchmarks${NC}"
        echo ""

        ctest -R "test_flux_limiter_overhead" --output-on-failure
        ./tests/performance/test_flux_limiter_overhead --gtest_filter="*Throughput*" || true
        ;;

    build)
        echo -e "${YELLOW}Building stability tests${NC}"
        echo ""

        make test_flux_limiter test_temperature_bounds test_omega_reduction test_high_pe_stability test_flux_limiter_overhead
        ;;

    help|--help|-h)
        echo "Usage: $0 [TEST_LEVEL]"
        echo ""
        echo "TEST_LEVEL options:"
        echo "  quick, fast, pre-commit    - Quick unit tests (< 30s, default)"
        echo "  medium, pre-push           - Unit + integration tests (< 2min)"
        echo "  full, all                  - All tests including performance (< 5min)"
        echo "  unit                       - Unit tests only"
        echo "  regression                 - Regression tests only"
        echo "  integration                - Integration tests only"
        echo "  performance                - Performance benchmarks only"
        echo "  build                      - Build tests without running"
        echo "  help, --help, -h           - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                         # Run quick tests (default)"
        echo "  $0 pre-push                # Run before pushing to repo"
        echo "  $0 full                    # Run complete test suite"
        exit 0
        ;;

    *)
        echo -e "${RED}Error: Unknown test level '$TEST_LEVEL'${NC}"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}======================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}  TESTS FAILED!${NC}"
    echo -e "${RED}======================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  1. Check test output above for failure details"
    echo "  2. Review tests/STABILITY_TESTS_README.md"
    echo "  3. Verify stability fixes are in place:"
    echo "     - Flux limiter in lattice_d3q7.cu"
    echo "     - Temperature bounds in thermal_lbm.cu"
    echo "     - Omega capping in ThermalLBM constructor"
    exit 1
fi
