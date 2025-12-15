#!/bin/bash
# ============================================================================
# TEST C EXECUTION SCRIPT
# ============================================================================
# Automated execution of Test C with monitoring
# ============================================================================

set -e  # Exit on error

cd /home/yzk/LBMProject/build

echo "════════════════════════════════════════════════════════════════"
echo "  TEST C: Full Physics (Coupling + Marangoni + Radiation)"
echo "  Automated Execution & Monitoring"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if executable exists
if [ ! -f "./visualize_lpbf_scanning" ]; then
    echo "❌ ERROR: visualize_lpbf_scanning executable not found!"
    echo "Build it with: cmake --build . --target visualize_lpbf_scanning"
    exit 1
fi

# Check if config exists
if [ ! -f "../configs/lpbf_195W_test_C_full_coupling.conf" ]; then
    echo "❌ ERROR: Test C config file not found!"
    echo "Expected: /home/yzk/LBMProject/configs/lpbf_195W_test_C_full_coupling.conf"
    exit 1
fi

# Create output directory
mkdir -p test_C_full_coupling

# Display configuration summary
echo "Configuration:"
echo "  Executable:    ./visualize_lpbf_scanning"
echo "  Config:        ../configs/lpbf_195W_test_C_full_coupling.conf"
echo "  Steps:         5000 (500 μs simulation)"
echo "  Output:        test_C_full_coupling/"
echo "  Log:           test_C_execution.log"
echo ""
echo "Expected Runtime: 6-10 minutes (15 min max)"
echo "GPU Memory:       ~140 MB"
echo ""

# Confirm execution
read -p "Proceed with Test C execution? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Execution cancelled."
    exit 0
fi

echo ""
echo "Starting Test C..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# Launch simulation in background
./visualize_lpbf_scanning \
    --config ../configs/lpbf_195W_test_C_full_coupling.conf \
    --steps 5000 \
    --output test_C_full_coupling \
    > test_C_execution.log 2>&1 &

SIM_PID=$!
echo "✅ Test C launched (PID: $SIM_PID)"
echo ""

# Wait a moment for log file to be created
sleep 2

# Check if process started successfully
if ! ps -p $SIM_PID > /dev/null; then
    echo "❌ ERROR: Simulation failed to start!"
    echo ""
    cat test_C_execution.log
    exit 1
fi

echo "Real-time monitoring options:"
echo ""
echo "  [1] Automatic monitor: ./monitor_test_C.sh"
echo "  [2] Manual tail:       tail -f test_C_execution.log"
echo "  [3] Background mode:   (monitor later)"
echo ""
read -p "Choose monitoring mode (1/2/3): " -n 1 -r MONITOR_CHOICE
echo ""

case $MONITOR_CHOICE in
    1)
        echo "Launching real-time monitor..."
        sleep 1
        ./monitor_test_C.sh
        ;;
    2)
        echo "Following log file (Ctrl+C to exit)..."
        sleep 1
        tail -f test_C_execution.log
        ;;
    3)
        echo "Running in background."
        echo "Monitor later with: ./monitor_test_C.sh"
        echo "Or check log: tail -f test_C_execution.log"
        ;;
    *)
        echo "Invalid choice. Running in background."
        echo "Monitor with: ./monitor_test_C.sh"
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Test C execution script completed."
echo "════════════════════════════════════════════════════════════════"
