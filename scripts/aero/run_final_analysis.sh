#!/bin/bash
# Final post-overnight analysis.
# Runs after overnight_30k_amr.sh completes. Generates plots + verdict.

set -u
ROOT=/home/yzk/CompressibleCFD

echo "=========================================="
echo "Final AMR analysis $(date)"
echo "=========================================="

# Phase 1: settled mean Cl/Cd comparison
echo ""
echo "--- 30k settled mean comparison ---"
python3 $ROOT/scripts/aero/plot_amr_30k_comparison.py

# Phase 2: flow field comparison at LE
echo ""
echo "--- Flow field comparison (LE zoom) ---"
python3 $ROOT/scripts/aero/plot_amr_flowfield_compare.py

echo ""
echo "=========================================="
echo "Final analysis done $(date)"
echo "Outputs:"
echo "  images/amr_30k_cl_cd_trajectory.png"
echo "  images/amr_30k_settled_summary.png"
echo "  images/amr_30k_flowfield_compare.png"
echo "=========================================="
