#!/bin/bash

################################################################################
# Verify Validation Framework Installation
#
# Purpose: Check that all components are installed and ready to run
################################################################################

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

check_pass() { echo -e "${GREEN}✓${NC} $1"; }
check_fail() { echo -e "${RED}✗${NC} $1"; }
check_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
check_info() { echo -e "${BLUE}ℹ${NC} $1"; }

echo ""
echo "========================================================================"
echo "  Validation Framework - Installation Verification"
echo "========================================================================"
echo ""

FAILED=0

# ============================================================================
# Check Directory Structure
# ============================================================================

echo "Checking directory structure..."
echo ""

if [ -d "/home/yzk/LBMProject/validation_tests" ]; then
    check_pass "validation_tests/ directory exists"
else
    check_fail "validation_tests/ directory NOT FOUND"
    FAILED=1
fi

if [ -d "/home/yzk/LBMProject/validation_tests/scripts" ]; then
    check_pass "scripts/ directory exists"
else
    check_fail "scripts/ directory NOT FOUND"
    FAILED=1
fi

for dir in data reports backups; do
    if [ -d "/home/yzk/LBMProject/validation_tests/$dir" ]; then
        check_pass "$dir/ directory exists"
    else
        check_warn "$dir/ directory missing (will be created)"
        mkdir -p "/home/yzk/LBMProject/validation_tests/$dir"
    fi
done

echo ""

# ============================================================================
# Check Scripts
# ============================================================================

echo "Checking scripts..."
echo ""

SCRIPTS=(
    "run_validation_suite.sh"
    "run_single_test.sh"
    "validate_results.py"
    "extract_metrics.py"
    "compare_baseline.py"
    "generate_report.py"
)

for script in "${SCRIPTS[@]}"; do
    script_path="/home/yzk/LBMProject/validation_tests/scripts/$script"
    if [ -f "$script_path" ]; then
        if [ -x "$script_path" ]; then
            check_pass "$script (exists and executable)"
        else
            check_warn "$script (exists but not executable - fixing...)"
            chmod +x "$script_path"
            check_pass "  Made executable"
        fi
    else
        check_fail "$script NOT FOUND"
        FAILED=1
    fi
done

echo ""

# ============================================================================
# Check Documentation
# ============================================================================

echo "Checking documentation..."
echo ""

if [ -f "/home/yzk/LBMProject/validation_tests/README.md" ]; then
    check_pass "README.md exists"
else
    check_fail "README.md NOT FOUND"
    FAILED=1
fi

if [ -f "/home/yzk/LBMProject/validation_tests/QUICKSTART.md" ]; then
    check_pass "QUICKSTART.md exists"
else
    check_warn "QUICKSTART.md not found (optional)"
fi

echo ""

# ============================================================================
# Check Prerequisites
# ============================================================================

echo "Checking prerequisites..."
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    check_pass "Python 3 installed (version $PYTHON_VERSION)"
else
    check_fail "Python 3 NOT FOUND"
    FAILED=1
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    check_pass "NVCC installed (CUDA $CUDA_VERSION)"
else
    check_fail "NVCC NOT FOUND (CUDA required)"
    FAILED=1
fi

# Check CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    check_pass "CMake installed (version $CMAKE_VERSION)"
else
    check_fail "CMake NOT FOUND"
    FAILED=1
fi

# Check Make
if command -v make &> /dev/null; then
    MAKE_VERSION=$(make --version | head -1 | awk '{print $3}')
    check_pass "Make installed (version $MAKE_VERSION)"
else
    check_fail "Make NOT FOUND"
    FAILED=1
fi

echo ""

# ============================================================================
# Check Project Structure
# ============================================================================

echo "Checking project structure..."
echo ""

if [ -d "/home/yzk/LBMProject" ]; then
    check_pass "Project root exists"
else
    check_fail "Project root NOT FOUND at /home/yzk/LBMProject"
    FAILED=1
fi

if [ -d "/home/yzk/LBMProject/build" ]; then
    check_pass "Build directory exists"
else
    check_warn "Build directory not found (will be created during tests)"
fi

if [ -d "/home/yzk/LBMProject/src" ]; then
    check_pass "Source directory exists"
else
    check_fail "Source directory NOT FOUND"
    FAILED=1
fi

# Check source files
if [ -f "/home/yzk/LBMProject/src/physics/vof/marangoni.cu" ]; then
    check_pass "marangoni.cu exists"
else
    check_fail "marangoni.cu NOT FOUND"
    FAILED=1
fi

if [ -f "/home/yzk/LBMProject/src/physics/multiphysics/multiphysics_solver.cu" ]; then
    check_pass "multiphysics_solver.cu exists"
else
    check_fail "multiphysics_solver.cu NOT FOUND"
    FAILED=1
fi

echo ""

# ============================================================================
# Check Executable
# ============================================================================

echo "Checking executable..."
echo ""

if [ -f "/home/yzk/LBMProject/build/visualize_lpbf_scanning" ]; then
    check_pass "visualize_lpbf_scanning executable exists"

    # Check if it's actually executable
    if [ -x "/home/yzk/LBMProject/build/visualize_lpbf_scanning" ]; then
        check_pass "Executable has correct permissions"
    else
        check_warn "Executable lacks execute permission"
    fi
else
    check_warn "visualize_lpbf_scanning NOT FOUND (will be built during tests)"
fi

echo ""

# ============================================================================
# Test Script Syntax
# ============================================================================

echo "Testing script syntax..."
echo ""

for script in "${SCRIPTS[@]}"; do
    if [[ $script == *.py ]]; then
        if python3 -m py_compile "/home/yzk/LBMProject/validation_tests/scripts/$script" 2>/dev/null; then
            check_pass "$script syntax OK"
        else
            check_fail "$script has syntax errors"
            FAILED=1
        fi
    elif [[ $script == *.sh ]]; then
        if bash -n "/home/yzk/LBMProject/validation_tests/scripts/$script" 2>/dev/null; then
            check_pass "$script syntax OK"
        else
            check_fail "$script has syntax errors"
            FAILED=1
        fi
    fi
done

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "========================================================================"
echo "  VERIFICATION SUMMARY"
echo "========================================================================"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "Framework is ready to use!"
    echo ""
    echo "Quick start:"
    echo "  cd /home/yzk/LBMProject/validation_tests/scripts"
    echo "  ./run_validation_suite.sh"
    echo ""
    echo "Documentation:"
    echo "  cat /home/yzk/LBMProject/validation_tests/QUICKSTART.md"
    echo "  cat /home/yzk/LBMProject/validation_tests/README.md"
    echo ""
else
    echo -e "${RED}✗ VERIFICATION FAILED${NC}"
    echo ""
    echo "Some checks failed. Please fix the issues above before running tests."
    echo ""
fi

echo "========================================================================"
echo ""

exit $FAILED
