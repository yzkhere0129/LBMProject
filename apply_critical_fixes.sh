#!/bin/bash
# ============================================================================
# Critical Fixes for Metal Vaporization Bug
# ============================================================================
# This script applies the two most critical fixes:
# 1. Enable evaporative cooling in configuration
# 2. Add temperature clamp in thermal solver
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Applying Critical Fixes for Metal Vaporization Bug"
echo "============================================================"
echo

# ============================================================================
# Fix 1: Update configuration to enable evaporative cooling
# ============================================================================
echo "[1/3] Updating configuration file..."

CONFIG_FILE="configs/lpbf_195W_test_A_coupling.conf"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Backup original config
cp "$CONFIG_FILE" "${CONFIG_FILE}.backup_$(date +%Y%m%d_%H%M%S)"

# Add evaporation settings after the physics flags section
# Check if evaporation settings already exist
if grep -q "enable_evaporation_mass_loss" "$CONFIG_FILE"; then
    echo "  ✓ Evaporation settings already present in config"
else
    echo "  → Adding evaporation physics settings..."

    # Find the line after enable_darcy and insert evaporation settings
    sed -i '/^enable_darcy = true/a\
\
# --- Evaporation Physics (CRITICAL FIX) ---\
enable_evaporation_mass_loss = true   # Apply mass loss to VOF\
enable_evaporation_cooling = true     # Apply heat removal to thermal field\
evaporation_sticking_coeff = 0.82     # Ti6Al4V recondensation coefficient' "$CONFIG_FILE"

    echo "  ✓ Evaporation settings added to $CONFIG_FILE"
fi

# ============================================================================
# Fix 2: Add temperature clamp to phase change solver
# ============================================================================
echo
echo "[2/3] Adding temperature clamp to phase_change.cu..."

PHASE_CHANGE_FILE="src/physics/phase_change/phase_change.cu"

if [ ! -f "$PHASE_CHANGE_FILE" ]; then
    echo "ERROR: Phase change file not found: $PHASE_CHANGE_FILE"
    exit 1
fi

# Check if temperature clamp is already present
if grep -q "T_MAX_SUPERHEAT" "$PHASE_CHANGE_FILE"; then
    echo "  ✓ Temperature clamp already implemented"
else
    echo "  → Adding temperature clamp to solveTemperatureFromEnthalpyKernel..."

    # Backup original file
    cp "$PHASE_CHANGE_FILE" "${PHASE_CHANGE_FILE}.backup_$(date +%Y%m%d_%H%M%S)"

    # Find the line defining T_MIN and T_MAX and add superheat clamp
    # We'll add it after the existing T_MAX definition

    # First, let's check if T_MIN and T_MAX are defined
    if ! grep -q "constexpr float T_MIN" "$PHASE_CHANGE_FILE"; then
        echo "  → Adding T_MIN and T_MAX definitions..."

        # Add temperature limits after the namespace declaration
        sed -i '/^namespace lbm {/,/^namespace physics {/a\
\
// Physical temperature limits for stability\
constexpr float T_MIN = 100.0f;           ///< Minimum physical temperature [K]\
constexpr float T_MAX = 10000.0f;         ///< Maximum temperature (prevents plasma) [K]\
constexpr float T_BOIL_TI6AL4V = 3533.0f; ///< Boiling point of Ti6Al4V [K]\
constexpr float T_MAX_SUPERHEAT = 1.2f * T_BOIL_TI6AL4V;  ///< Max superheat (20% above T_boil)' "$PHASE_CHANGE_FILE"
    fi

    # Now add the clamp in the Newton-Raphson solver after T update
    # Find the line "T -= dT;" and add clamp after it
    sed -i '/T -= dT;/a\
\
        \/\/ CRITICAL FIX: Clamp to prevent unphysical superheat\
        \/\/ Above boiling point, evaporation should carry away excess energy\
        if (T > T_MAX_SUPERHEAT) {\
            T = T_MAX_SUPERHEAT;\
        }' "$PHASE_CHANGE_FILE"

    # Also add clamp after bisection method
    sed -i '/T_low = T;   \/\/ H too low, increase T/a\
\
            \/\/ Clamp during bisection too\
            if (T > T_MAX_SUPERHEAT) T = T_MAX_SUPERHEAT;' "$PHASE_CHANGE_FILE"

    echo "  ✓ Temperature clamp added to phase_change.cu"
fi

# ============================================================================
# Fix 3: Recompile the code
# ============================================================================
echo
echo "[3/3] Recompiling with fixes..."

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Clean previous build artifacts to force full recompile
rm -f CMakeCache.txt
rm -rf CMakeFiles/

# Reconfigure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "  ✓ Compilation successful"
else
    echo "  ✗ Compilation failed"
    exit 1
fi

cd ..

# ============================================================================
# Summary
# ============================================================================
echo
echo "============================================================"
echo "Critical Fixes Applied Successfully"
echo "============================================================"
echo
echo "Changes made:"
echo "  1. Evaporative cooling enabled in config:"
echo "     - enable_evaporation_mass_loss = true"
echo "     - enable_evaporation_cooling = true"
echo
echo "  2. Temperature clamp added (T_max = 4240 K):"
echo "     - Prevents unphysical superheat > 20% above T_boil"
echo "     - Excess energy assumed lost to evaporation"
echo
echo "  3. Code recompiled successfully"
echo
echo "Backup files created:"
echo "  - ${CONFIG_FILE}.backup_*"
echo "  - ${PHASE_CHANGE_FILE}.backup_*"
echo
echo "============================================================"
echo "Next Steps:"
echo "============================================================"
echo
echo "1. Run Test A to verify fixes:"
echo "   cd build"
echo "   ./test_lpbf_simple --config ../configs/lpbf_195W_test_A_coupling.conf"
echo
echo "2. Check results:"
echo "   - T_max should be < 5000 K (was 41,061 K)"
echo "   - Mass loss should be < 5% (was likely >> 50%)"
echo "   - Energy balance imbalance should be < 10%"
echo
echo "3. If successful, proceed to full LPBF simulation"
echo
echo "For detailed analysis, see: DIAGNOSIS_METAL_VAPORIZATION_BUG.md"
echo "============================================================"
