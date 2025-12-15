#!/bin/bash
# Quick VTK Comparison Commands Reference
# ========================================
# Collection of commonly used VTK comparison and visualization commands
# Author: LBMProject Team
# Date: 2025-12-04

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# Set project directories
export LBM_DIR="/home/yzk/LBMProject"
export WALBERLA_DIR="/home/yzk/walberla"
export LBM_BUILD="${LBM_DIR}/build"
export WALBERLA_OUTPUT="${WALBERLA_DIR}/sim_output"

# Create results directory
mkdir -p ${LBM_DIR}/results/vtk_comparison

# ==============================================================================
# BASIC COMPARISONS
# ==============================================================================

# Compare two VTK files (basic)
compare_basic() {
    python ${LBM_DIR}/scripts/compare_vtk_files.py \
        "${1}" "${2}"
}

# Compare with all visualizations
compare_full() {
    python ${LBM_DIR}/scripts/compare_vtk_files.py \
        "${1}" "${2}" \
        --plot-velocity \
        --plot-temperature \
        --output-dir "${LBM_DIR}/results/vtk_comparison/$(basename ${1} .vtk)_vs_$(basename ${2} .vtk)"
}

# Compare specific fields
compare_fields() {
    python ${LBM_DIR}/scripts/compare_vtk_files.py \
        "${1}" "${2}" \
        --fields Velocity Temperature \
        --output-dir "${LBM_DIR}/results/vtk_comparison/fields"
}

# ==============================================================================
# LBMPROJECT SPECIFIC
# ==============================================================================

# Compare LBMProject config parser test
compare_lbm_config_test() {
    compare_full \
        "${LBM_BUILD}/config_parser_test/output_000000.vtk" \
        "${LBM_BUILD}/config_parser_test/output_010000.vtk"
}

# Visualize Marangoni flow (single timestep)
visualize_marangoni_single() {
    python ${LBM_DIR}/scripts/visualize_marangoni.py \
        "${LBM_BUILD}/tests/validation/phase6_test2c_visualization/marangoni_flow_010000.vtk" \
        --output-dir "${LBM_DIR}/results/vtk_comparison/marangoni_single"
}

# Visualize Marangoni flow evolution
visualize_marangoni_evolution() {
    python ${LBM_DIR}/scripts/visualize_marangoni.py \
        ${LBM_BUILD}/tests/validation/phase6_test2c_visualization/marangoni_flow_*.vtk \
        --time-evolution \
        --output-dir "${LBM_DIR}/results/vtk_comparison/marangoni_evolution"
}

# ==============================================================================
# LBMPROJECT VS WALBERLA
# ==============================================================================

# Compare LBMProject vs WalBerla (timestep 0)
compare_lbm_vs_walberla_t0() {
    compare_full \
        "${LBM_BUILD}/config_parser_test/output_000000.vtk" \
        "${WALBERLA_OUTPUT}/sim_output_00000000.vtk"
}

# Compare LBMProject vs WalBerla (final timestep)
compare_lbm_vs_walberla_final() {
    compare_full \
        "${LBM_BUILD}/config_parser_test/output_010000.vtk" \
        "${WALBERLA_OUTPUT}/sim_output_00000450.vtk"
}

# Poiseuille flow comparison
compare_poiseuille() {
    python ${LBM_DIR}/scripts/compare_poiseuille.py \
        "${1}" "${2}" \
        --output-dir "${LBM_DIR}/results/vtk_comparison/poiseuille"
}

# ==============================================================================
# SLICE COMPARISONS
# ==============================================================================

# Compare Z-slice at middle
compare_z_slice() {
    python ${LBM_DIR}/scripts/compare_vtk_files.py \
        "${1}" "${2}" \
        --slice-field Velocity \
        --slice-axis 2 \
        --slice-index 25 \
        --output-dir "${LBM_DIR}/results/vtk_comparison/z_slice"
}

# Compare Y-slice at middle
compare_y_slice() {
    python ${LBM_DIR}/scripts/compare_vtk_files.py \
        "${1}" "${2}" \
        --slice-field Temperature \
        --slice-axis 1 \
        --slice-index 50 \
        --output-dir "${LBM_DIR}/results/vtk_comparison/y_slice"
}

# Compare X-slice at middle
compare_x_slice() {
    python ${LBM_DIR}/scripts/compare_vtk_files.py \
        "${1}" "${2}" \
        --slice-field Velocity \
        --slice-axis 0 \
        --slice-index 50 \
        --output-dir "${LBM_DIR}/results/vtk_comparison/x_slice"
}

# ==============================================================================
# PARAVIEW COMMANDS
# ==============================================================================

# Open single VTK in ParaView
paraview_open() {
    paraview "${1}" &
}

# Open LBMProject Marangoni time series
paraview_marangoni_series() {
    paraview --data="${LBM_BUILD}/tests/validation/phase6_test2c_visualization/marangoni_flow_*.vtk" &
}

# Open WalBerla time series
paraview_walberla_series() {
    paraview --data="${WALBERLA_OUTPUT}/sim_output_*.vtk" &
}

# Open ParaView with state file
paraview_state() {
    paraview --state="${1}" &
}

# ==============================================================================
# BATCH OPERATIONS
# ==============================================================================

# Compare all Marangoni timesteps
batch_compare_marangoni() {
    OUTPUT_DIR="${LBM_DIR}/results/vtk_comparison/batch_marangoni"
    mkdir -p ${OUTPUT_DIR}

    for file in ${LBM_BUILD}/tests/validation/phase6_test2c_visualization/marangoni_flow_*.vtk; do
        echo "Processing $(basename $file)..."
        python ${LBM_DIR}/scripts/visualize_marangoni.py \
            "${file}" \
            --output-dir "${OUTPUT_DIR}/$(basename ${file} .vtk)"
    done

    echo "Batch processing complete. Results in ${OUTPUT_DIR}"
}

# Compare all WalBerla outputs
batch_compare_walberla() {
    OUTPUT_DIR="${LBM_DIR}/results/vtk_comparison/batch_walberla"
    mkdir -p ${OUTPUT_DIR}

    i=0
    for file in ${WALBERLA_OUTPUT}/sim_output_*.vtk; do
        echo "Processing $(basename $file)..."
        python ${LBM_DIR}/scripts/compare_vtk_files.py \
            "${file}" \
            "${file}" \
            --plot-velocity \
            --plot-temperature \
            --output-dir "${OUTPUT_DIR}/timestep_$(printf %05d $i)"
        ((i++))
    done

    echo "Batch processing complete. Results in ${OUTPUT_DIR}"
}

# Compare corresponding LBM and WalBerla timesteps
batch_compare_lbm_walberla() {
    OUTPUT_DIR="${LBM_DIR}/results/vtk_comparison/batch_lbm_vs_walberla"
    mkdir -p ${OUTPUT_DIR}

    # Map LBM files to WalBerla files
    LBM_FILES=(${LBM_BUILD}/config_parser_test/output_*.vtk)
    WALBERLA_FILES=(${WALBERLA_OUTPUT}/sim_output_*.vtk)

    for i in "${!LBM_FILES[@]}"; do
        if [ $i -lt ${#WALBERLA_FILES[@]} ]; then
            echo "Comparing timestep $i..."
            python ${LBM_DIR}/scripts/compare_vtk_files.py \
                "${LBM_FILES[$i]}" \
                "${WALBERLA_FILES[$i]}" \
                --plot-velocity \
                --plot-temperature \
                --output-dir "${OUTPUT_DIR}/timestep_$(printf %05d $i)"
        fi
    done

    echo "Batch comparison complete. Results in ${OUTPUT_DIR}"
}

# ==============================================================================
# FILE INSPECTION
# ==============================================================================

# List all LBMProject VTK files
list_lbm_vtk() {
    echo "=== LBMProject VTK Files ==="
    find ${LBM_BUILD} -name "*.vtk" -type f | sort
}

# List all WalBerla VTK files
list_walberla_vtk() {
    echo "=== WalBerla VTK Files ==="
    find ${WALBERLA_DIR} -name "*.vtk" -type f | sort
}

# Show VTK file header
show_vtk_header() {
    echo "=== VTK Header: ${1} ==="
    head -30 "${1}"
}

# Show VTK fields
show_vtk_fields() {
    echo "=== VTK Fields: ${1} ==="
    grep -E "VECTORS|SCALARS" "${1}"
}

# Show VTK dimensions
show_vtk_dimensions() {
    echo "=== VTK Dimensions: ${1} ==="
    grep "DIMENSIONS" "${1}"
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

# Clean results directory
clean_results() {
    rm -rf ${LBM_DIR}/results/vtk_comparison/*
    echo "Results directory cleaned"
}

# Create results directory structure
setup_results_dirs() {
    mkdir -p ${LBM_DIR}/results/vtk_comparison/{basic,full,fields,slices,marangoni,poiseuille,batch}
    echo "Results directory structure created"
}

# Quick help
show_help() {
    cat << EOF
=============================================================================
VTK Comparison Quick Commands
=============================================================================

BASIC COMPARISONS:
  compare_basic FILE1 FILE2              - Basic comparison with console output
  compare_full FILE1 FILE2               - Full comparison with all plots
  compare_fields FILE1 FILE2             - Compare specific fields

LBMPROJECT SPECIFIC:
  compare_lbm_config_test                - Compare config parser test outputs
  visualize_marangoni_single             - Visualize single Marangoni timestep
  visualize_marangoni_evolution          - Visualize Marangoni time evolution

LBMPROJECT VS WALBERLA:
  compare_lbm_vs_walberla_t0             - Compare initial timestep
  compare_lbm_vs_walberla_final          - Compare final timestep
  compare_poiseuille FILE1 FILE2         - Poiseuille flow comparison

SLICE COMPARISONS:
  compare_z_slice FILE1 FILE2            - Compare Z-slice
  compare_y_slice FILE1 FILE2            - Compare Y-slice
  compare_x_slice FILE1 FILE2            - Compare X-slice

PARAVIEW:
  paraview_open FILE                     - Open single file in ParaView
  paraview_marangoni_series              - Open Marangoni time series
  paraview_walberla_series               - Open WalBerla time series
  paraview_state STATEFILE               - Open with state file

BATCH OPERATIONS:
  batch_compare_marangoni                - Compare all Marangoni timesteps
  batch_compare_walberla                 - Compare all WalBerla outputs
  batch_compare_lbm_walberla             - Compare LBM vs WalBerla timesteps

FILE INSPECTION:
  list_lbm_vtk                           - List LBMProject VTK files
  list_walberla_vtk                      - List WalBerla VTK files
  show_vtk_header FILE                   - Show VTK header
  show_vtk_fields FILE                   - Show VTK fields
  show_vtk_dimensions FILE               - Show VTK dimensions

UTILITY:
  clean_results                          - Clean results directory
  setup_results_dirs                     - Create results directory structure
  show_help                              - Show this help message

USAGE EXAMPLES:
  # Source this file to load functions
  source ${LBM_DIR}/scripts/QUICK_VTK_COMMANDS.sh

  # Compare two files
  compare_full file1.vtk file2.vtk

  # Visualize Marangoni flow
  visualize_marangoni_single

  # Open in ParaView
  paraview_marangoni_series

=============================================================================
EOF
}

# ==============================================================================
# MAIN
# ==============================================================================

# If script is sourced, just load functions
# If executed, show help
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    show_help
else
    echo "VTK comparison functions loaded. Type 'show_help' for usage."
fi
