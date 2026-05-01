#!/usr/bin/env bash
# run_plic_calibration.sh
#
# Runs the legacy (algebraic VOF) and PLIC-VOF line-scan apps back-to-back
# on a single GPU, then prints a summary header pointing to the VTK output
# directories ready for scripts/compare_plic_vs_f3d.py.
#
# Usage (from repo root):
#   bash scripts/run_plic_calibration.sh [--skip-legacy] [--skip-plic]
#
# Options:
#   --skip-legacy   Skip the legacy run (re-use existing output_line_scan_316L/)
#   --skip-plic     Skip the PLIC run (re-use existing output_line_scan_316L_plic/)
#
# Prerequisites:
#   cmake --build build --target sim_line_scan_316L sim_line_scan_316L_plic -j8

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
LEGACY_BIN="${BUILD_DIR}/sim_line_scan_316L"
PLIC_BIN="${BUILD_DIR}/sim_line_scan_316L_plic"
LEGACY_OUT="${REPO_ROOT}/output_line_scan_316L"
PLIC_OUT="${REPO_ROOT}/output_line_scan_316L_plic"

RUN_LEGACY=1
RUN_PLIC=1

for arg in "$@"; do
    case "$arg" in
        --skip-legacy) RUN_LEGACY=0 ;;
        --skip-plic)   RUN_PLIC=0   ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

echo "========================================================"
echo "  PLIC Calibration Run: 316L line scan"
echo "  F3D reference: 150 W / 800 mm/s / r0=50 μm"
echo "  NOTE: both apps use absorptivity=0.40 (F3D uses 0.70)"
echo "========================================================"
echo ""

# --- Verify binaries exist ---
if [[ $RUN_LEGACY -eq 1 && ! -x "$LEGACY_BIN" ]]; then
    echo "ERROR: binary not found: $LEGACY_BIN"
    echo "  Build with: cmake --build ${BUILD_DIR} --target sim_line_scan_316L -j8"
    exit 1
fi
if [[ $RUN_PLIC -eq 1 && ! -x "$PLIC_BIN" ]]; then
    echo "ERROR: binary not found: $PLIC_BIN"
    echo "  Build with: cmake --build ${BUILD_DIR} --target sim_line_scan_316L_plic -j8"
    exit 1
fi

# --- Legacy run ---
if [[ $RUN_LEGACY -eq 1 ]]; then
    echo ">>> [1/2] Legacy (algebraic VOF) — output: ${LEGACY_OUT}"
    echo "    Start: $(date)"
    mkdir -p "${LEGACY_OUT}"
    cd "${REPO_ROOT}"
    "${LEGACY_BIN}" 2>&1 | tee "${LEGACY_OUT}/run_legacy.log"
    echo "    Done:  $(date)"
    echo ""
else
    echo ">>> [1/2] Legacy skipped (--skip-legacy). Using existing: ${LEGACY_OUT}"
    echo ""
fi

# --- PLIC run ---
if [[ $RUN_PLIC -eq 1 ]]; then
    echo ">>> [2/2] PLIC full-stack — output: ${PLIC_OUT}"
    echo "    Start: $(date)"
    mkdir -p "${PLIC_OUT}"
    cd "${REPO_ROOT}"
    "${PLIC_BIN}" 2>&1 | tee "${PLIC_OUT}/run_plic.log"
    echo "    Done:  $(date)"
    echo ""
else
    echo ">>> [2/2] PLIC skipped (--skip-plic). Using existing: ${PLIC_OUT}"
    echo ""
fi

# --- Summary ---
echo "========================================================"
echo "  Both runs complete. VTK output ready for comparison."
echo "========================================================"
echo ""
echo "  Legacy output : ${LEGACY_OUT}/"
LEGACY_VTK_COUNT=$(ls "${LEGACY_OUT}"/line_scan_*.vtk 2>/dev/null | wc -l || echo 0)
echo "    VTK files   : ${LEGACY_VTK_COUNT}"

echo "  PLIC output   : ${PLIC_OUT}/"
PLIC_VTK_COUNT=$(ls "${PLIC_OUT}"/line_scan_plic_*.vtk 2>/dev/null | wc -l || echo 0)
echo "    VTK files   : ${PLIC_VTK_COUNT}"

echo ""
echo "  F3D reference : /home/yzk/LBMProject/vtk-316L-150W-50um-V800mms/"
echo ""
echo "  Next step — run the comparison script:"
echo "    python3 ${REPO_ROOT}/scripts/compare_plic_vs_f3d.py"
echo ""
echo "  Or with custom output directories:"
echo "    python3 ${REPO_ROOT}/scripts/compare_plic_vs_f3d.py \\"
echo "        --legacy ${LEGACY_OUT} \\"
echo "        --plic   ${PLIC_OUT} \\"
echo "        --f3d    /home/yzk/LBMProject/vtk-316L-150W-50um-V800mms"
echo "========================================================"
