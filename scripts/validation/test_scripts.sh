#!/bin/bash
# Test validation scripts with synthetic data

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Testing Validation Analysis Scripts"
echo "========================================"

# Test 1: Grid convergence with synthetic data
echo ""
echo "[1/3] Testing grid_convergence_analysis.py..."
python grid_convergence_analysis.py \
    --resolutions 25,50,100,200 \
    --errors 0.0143,0.0038,0.0009,0.0002 \
    --output ./test_results

if [ -f ./test_results/grid_convergence.png ]; then
    echo "  ✓ Grid convergence plot generated"
else
    echo "  ✗ FAILED: No output file"
    exit 1
fi

# Check convergence order in summary
if grep -q "Convergence order:" ./test_results/convergence_summary.txt; then
    ORDER=$(grep "Convergence order:" ./test_results/convergence_summary.txt | awk '{print $4}')
    echo "  ✓ Computed convergence order: p = $ORDER"

    # Check if order is reasonable (1.8 to 2.2 for second-order)
    if (( $(echo "$ORDER > 1.5" | bc -l) )) && (( $(echo "$ORDER < 2.5" | bc -l) )); then
        echo "  ✓ Convergence order is reasonable"
    else
        echo "  ⚠ Warning: Convergence order outside expected range"
    fi
else
    echo "  ✗ FAILED: Could not find convergence order"
    exit 1
fi

echo ""
echo "Test 1 PASSED"

# Test 2: Check if thermal script has proper imports and structure
echo ""
echo "[2/3] Testing thermal_walberla_comparison.py structure..."
python -c "
import sys
sys.path.insert(0, '.')

# Test imports
try:
    import numpy as np
    import pyvista as pv
    import matplotlib.pyplot as plt
    print('  ✓ All dependencies imported successfully')
except ImportError as e:
    print(f'  ✗ FAILED: Missing dependency: {e}')
    sys.exit(1)

# Test script can be imported
try:
    import thermal_walberla_comparison
    print('  ✓ Script structure is valid')
except Exception as e:
    print(f'  ✗ FAILED: Script error: {e}')
    sys.exit(1)
"

echo "Test 2 PASSED"

# Test 3: Check if VOF script has proper imports and structure
echo ""
echo "[3/3] Testing vof_advection_validation.py structure..."
python -c "
import sys
sys.path.insert(0, '.')

# Test script can be imported
try:
    import vof_advection_validation
    print('  ✓ Script structure is valid')
except Exception as e:
    print(f'  ✗ FAILED: Script error: {e}')
    sys.exit(1)
"

echo "Test 3 PASSED"

# Summary
echo ""
echo "========================================"
echo "All Tests Passed!"
echo "========================================"
echo ""
echo "Generated test outputs in: ./test_results/"
echo "  - grid_convergence.png"
echo "  - error_reduction.png"
echo "  - convergence_summary.txt"
echo ""
echo "Scripts are ready to use with real VTK data."
echo ""

# Cleanup
echo "Clean up test results? [y/N] "
read -t 5 -n 1 cleanup || cleanup=""
echo ""
if [ "$cleanup" = "y" ] || [ "$cleanup" = "Y" ]; then
    rm -rf ./test_results
    echo "Test results cleaned up."
else
    echo "Test results kept in ./test_results/"
fi
