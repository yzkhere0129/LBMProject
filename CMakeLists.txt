cmake_minimum_required(VERSION 3.18)
project(LBM_Metal_Melting LANGUAGES CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust for your GPU)
# Common architectures: 75 (RTX 20xx), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------
# Source collections
# ---------------------------------------------------------------------------

file(GLOB_RECURSE LBM_CORE_SOURCES
    src/core/lattice/*.cu
    src/core/collision/*.cu
    src/core/streaming/*.cu
    src/core/boundary/*.cu
)

file(GLOB_RECURSE LBM_PHYSICS_SOURCES
    src/physics/*.cu
    src/physics/thermal/*.cu
    src/physics/laser/*.cu
    src/physics/materials/*.cu
    src/physics/phase_change/*.cu
    src/physics/fluid/*.cu
    src/physics/vof/*.cu
    src/physics/multiphysics/*.cu
    src/physics/powder/*.cu
    src/cuda/*.cu
)

file(GLOB_RECURSE LBM_DIAGNOSTICS_SOURCES
    src/diagnostics/*.cu
)

file(GLOB_RECURSE LBM_IO_SOURCES
    src/io/*.cu
    src/io/*.cpp
)

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------

add_library(lbm_diagnostics STATIC ${LBM_DIAGNOSTICS_SOURCES})
set_target_properties(lbm_diagnostics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lbm_diagnostics PUBLIC CUDA::cudart)

add_library(lbm_core STATIC ${LBM_CORE_SOURCES})
set_target_properties(lbm_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lbm_core PUBLIC CUDA::cudart)

add_library(lbm_physics STATIC ${LBM_PHYSICS_SOURCES})
set_target_properties(lbm_physics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lbm_physics PUBLIC CUDA::cudart lbm_core lbm_diagnostics lbm_io)

add_library(lbm_io STATIC ${LBM_IO_SOURCES})
set_target_properties(lbm_io PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lbm_io PUBLIC CUDA::cudart)

add_library(lbm_config STATIC
    src/config/simulation_config.cpp
)
target_include_directories(lbm_config PUBLIC include)

# ---------------------------------------------------------------------------
# Application executables
# ---------------------------------------------------------------------------

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps)

    # Generic simulation runner with config file support
    add_executable(run_simulation apps/run_simulation.cu)
    set_target_properties(run_simulation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(run_simulation lbm_physics lbm_io lbm_config CUDA::cudart)

    # VTK visualization application
    add_executable(visualize_laser_heating apps/visualize_laser_heating.cu)
    set_target_properties(visualize_laser_heating PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(visualize_laser_heating lbm_physics lbm_io CUDA::cudart)

    # Real-time monitor application
    add_executable(realtime_monitor apps/realtime_monitor.cu)
    set_target_properties(realtime_monitor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(realtime_monitor lbm_physics lbm_io CUDA::cudart)

    # Quick test/visualization application
    add_executable(test_visualization apps/test_visualization.cu)
    set_target_properties(test_visualization PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_visualization lbm_physics lbm_io CUDA::cudart)

    # Phase 5: Laser melting with melt pool convection
    add_executable(visualize_laser_melting_with_flow apps/visualize_laser_melting_with_flow.cu)
    set_target_properties(visualize_laser_melting_with_flow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(visualize_laser_melting_with_flow lbm_physics lbm_io CUDA::cudart)

    # Phase 6: Marangoni Effect Visualization
    add_executable(visualize_phase6_marangoni apps/visualize_phase6_marangoni_simple.cu)
    set_target_properties(visualize_phase6_marangoni PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(visualize_phase6_marangoni lbm_physics lbm_io CUDA::cudart)

    # Realistic LPBF simulation with Marangoni convection
    add_executable(visualize_lpbf_marangoni_realistic apps/visualize_lpbf_marangoni_realistic.cu)
    set_target_properties(visualize_lpbf_marangoni_realistic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(visualize_lpbf_marangoni_realistic lbm_physics lbm_io CUDA::cudart)

    # LPBF laser scanning simulation
    add_executable(visualize_lpbf_scanning apps/visualize_lpbf_scanning.cu)
    set_target_properties(visualize_lpbf_scanning PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(visualize_lpbf_scanning lbm_physics lbm_io CUDA::cudart)

    # Energy balance diagnostic tool
    add_executable(diagnose_energy_balance apps/diagnose_energy_balance.cu)
    set_target_properties(diagnose_energy_balance PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(diagnose_energy_balance lbm_physics lbm_io CUDA::cudart)

    # Thermal boundary condition tool
    add_executable(test_thermal_boundary apps/test_thermal_boundary.cu)
    set_target_properties(test_thermal_boundary PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_thermal_boundary lbm_physics lbm_io CUDA::cudart)

    # Stability diagnostic tool
    add_executable(stability_diagnostic apps/stability_diagnostic.cu)
    set_target_properties(stability_diagnostic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(stability_diagnostic lbm_physics lbm_io CUDA::cudart)

    # Spot melting simulation (316L, Marangoni convection)
    add_executable(sim_spot_melt_316L apps/sim_spot_melt_316L.cu)
    set_target_properties(sim_spot_melt_316L PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(sim_spot_melt_316L lbm_physics lbm_io CUDA::cudart)

    # LPBF line scan simulation (316L, moving laser)
    add_executable(sim_line_scan_316L apps/sim_line_scan_316L.cu)
    set_target_properties(sim_line_scan_316L PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(sim_line_scan_316L lbm_physics lbm_io CUDA::cudart)

    # LPBF powder bed simulation (316L, realistic powder layer)
    add_executable(sim_powder_bed_316L apps/sim_powder_bed_316L.cu)
    set_target_properties(sim_powder_bed_316L PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(sim_powder_bed_316L lbm_physics lbm_io CUDA::cudart)

    # Powder bed LPBF demo
    add_executable(demo_powder_bed_lpbf apps/demo_powder_bed_lpbf.cu)
    set_target_properties(demo_powder_bed_lpbf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(demo_powder_bed_lpbf lbm_physics lbm_io CUDA::cudart)

    # Performance benchmark (waLBerla comparison)
    add_executable(benchmark_thermal_lpbf apps/benchmark_thermal_lpbf.cu)
    set_target_properties(benchmark_thermal_lpbf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(benchmark_thermal_lpbf lbm_physics lbm_io CUDA::cudart)

    # Visualization: rising bubble
    add_executable(viz_bubble scripts/viz/viz_bubble.cu)
    set_target_properties(viz_bubble PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_bubble lbm_physics lbm_core CUDA::cudart)

    # Visualization: Marangoni 1D return flow benchmark
    add_executable(viz_marangoni_returnflow scripts/viz/viz_marangoni_returnflow.cu)
    set_target_properties(viz_marangoni_returnflow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_marangoni_returnflow lbm_physics lbm_core CUDA::cudart)

    # Visualization: Marangoni 2D thermocapillary cavity
    add_executable(viz_marangoni_cavity scripts/viz/viz_marangoni_cavity.cu)
    set_target_properties(viz_marangoni_cavity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_marangoni_cavity lbm_physics lbm_core CUDA::cudart)

    # Phase inversion diagnostic (oil rising in water, PLIC-VOF + TRT)
    add_executable(viz_phase_inversion scripts/viz/viz_phase_inversion.cu)
    set_target_properties(viz_phase_inversion PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_phase_inversion lbm_physics lbm_core CUDA::cudart)

    # Visualization: Rayleigh-Taylor instability (PLIC-VOF + TRT + Guo)
    add_executable(viz_rt_plic scripts/viz/viz_rt_plic.cu)
    set_target_properties(viz_rt_plic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_rt_plic lbm_physics lbm_core CUDA::cudart)

    # Visualization: Darcy semi-implicit diagnostics
    add_executable(viz_darcy_diagnostics scripts/viz/viz_darcy_diagnostics.cu)
    set_target_properties(viz_darcy_diagnostics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_darcy_diagnostics lbm_physics lbm_core CUDA::cudart)

    # LaserMeltingIron post-mortem VTK dump
    add_executable(viz_laser_melting_postmortem scripts/viz/viz_laser_melting_postmortem.cu)
    set_target_properties(viz_laser_melting_postmortem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_laser_melting_postmortem lbm_physics lbm_io CUDA::cudart)

    # Surface force isolation tests (Marangoni vs ST vs neither)
    add_executable(viz_surface_isolation scripts/viz/viz_surface_isolation.cu)
    set_target_properties(viz_surface_isolation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_surface_isolation lbm_physics lbm_io CUDA::cudart)

    # Marangoni fix validation (CSF smearing + Δu cap evidence)
    add_executable(viz_marangoni_validation scripts/viz/viz_marangoni_validation.cu)
    set_target_properties(viz_marangoni_validation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_marangoni_validation lbm_physics lbm_io CUDA::cudart)

    # Khairallah NaN diagnostic (crime scene at step 7-8)
    add_executable(viz_khairallah_diagnostic scripts/viz/viz_khairallah_diagnostic.cu)
    set_target_properties(viz_khairallah_diagnostic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(viz_khairallah_diagnostic lbm_physics lbm_io CUDA::cudart)

    # Stefan problem benchmark (pure metal, enthalpy source term validation)
    # Uses a minimal thermal-only library to avoid fluid solver dependencies
    add_library(lbm_thermal_only STATIC
        src/physics/thermal/thermal_lbm.cu
        src/physics/thermal/lattice_d3q7.cu
        src/physics/phase_change/phase_change.cu
        src/physics/materials/material_database.cu
    )
    set_target_properties(lbm_thermal_only PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(lbm_thermal_only PUBLIC CUDA::cudart)

    add_executable(stefan_benchmark apps/stefan_benchmark.cu)
    set_target_properties(stefan_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(stefan_benchmark lbm_thermal_only CUDA::cudart)

    # Pure conduction + phase change benchmark (316L spot melting, OpenFOAM cross-verification)
    add_executable(benchmark_conduction_316L apps/benchmark_conduction_316L.cu)
    set_target_properties(benchmark_conduction_316L PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(benchmark_conduction_316L lbm_thermal_only CUDA::cudart)

    # Spot melting benchmark with Marangoni convection (full multiphysics)
    add_executable(benchmark_spot_melt_316L apps/benchmark_spot_melt_316L.cu)
    set_target_properties(benchmark_spot_melt_316L PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(benchmark_spot_melt_316L lbm_physics lbm_io CUDA::cudart)

    # Diagnostic: Marangoni isolation test + probes
    add_executable(diag_marangoni_isolated apps/diag_marangoni_isolated.cu)
    set_target_properties(diag_marangoni_isolated PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(diag_marangoni_isolated lbm_physics lbm_io CUDA::cudart)

    # LPBF powder bed single-track (laserMeltFoam tutorial reproduction)
    add_executable(benchmark_powder_bed_316L apps/benchmark_powder_bed_316L.cu)
    set_target_properties(benchmark_powder_bed_316L PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(benchmark_powder_bed_316L lbm_physics lbm_io CUDA::cudart)

    # Phase 5: Full LPBF keyhole simulation
    add_executable(benchmark_keyhole_316L apps/benchmark_keyhole_316L.cu)
    set_target_properties(benchmark_keyhole_316L PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(benchmark_keyhole_316L lbm_physics lbm_io CUDA::cudart)

    # ThermalFDM validation tests
    add_executable(test_thermal_fdm apps/test_thermal_fdm.cu)
    set_target_properties(test_thermal_fdm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_thermal_fdm lbm_physics lbm_io CUDA::cudart)

    # Enthalpy budget closure diagnostic (316L phase transition, <2% residual)
    add_executable(test_enthalpy_diag apps/test_enthalpy_diag.cu)
    set_target_properties(test_enthalpy_diag PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_enthalpy_diag lbm_physics lbm_io CUDA::cudart)

    # Diagnostic: VOF mass conservation closed-box audit
    add_executable(diag_vof_mass_audit apps/diag_vof_mass_audit.cu)
    set_target_properties(diag_vof_mass_audit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(diag_vof_mass_audit lbm_physics CUDA::cudart)

    # Diagnostic: Thermal bridge fix for powder-substrate contact
    add_executable(diag_thermal_bridge apps/diag_thermal_bridge.cu)
    set_target_properties(diag_thermal_bridge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(diag_thermal_bridge lbm_physics lbm_io CUDA::cudart)

    # Diagnostic: Mass loss destination tracking
    add_executable(diag_mass_loss_tracking apps/diag_mass_loss_tracking.cu)
    set_target_properties(diag_mass_loss_tracking PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(diag_mass_loss_tracking lbm_physics lbm_io CUDA::cudart)

    # Diagnostic: Static spot melt + solidification recoil
    add_executable(diag_spot_melt_solidify apps/diag_spot_melt_solidify.cu)
    set_target_properties(diag_spot_melt_solidify PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(diag_spot_melt_solidify lbm_physics lbm_io CUDA::cudart)

    # Diagnostic: Flat plate baseline (no powder)
    add_executable(diag_flat_plate_scan apps/diag_flat_plate_scan.cu)
    set_target_properties(diag_flat_plate_scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(diag_flat_plate_scan lbm_physics lbm_io CUDA::cudart)

    # Diagnostic: Isothermal capillary relaxation test
    add_executable(diag_capillary_relax apps/diag_capillary_relax.cu)
    set_target_properties(diag_capillary_relax PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(diag_capillary_relax lbm_physics lbm_io CUDA::cudart)

    # Unit test: Marangoni force kernel — isolated analytical comparison
    add_executable(test_marangoni_unit apps/test_marangoni_unit.cu)
    set_target_properties(test_marangoni_unit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_marangoni_unit lbm_physics CUDA::cudart)

    # Unit test: Marangoni force scale — isolated analytical comparison (no fluid/thermal/laser)
    add_executable(test_marangoni_scale apps/test_marangoni_scale.cu)
    set_target_properties(test_marangoni_scale PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_marangoni_scale lbm_physics CUDA::cudart)

    # Unit test: HKL evaporation cooling — isolated analytical comparison
    add_executable(test_evaporation_unit apps/test_evaporation_unit.cu)
    set_target_properties(test_evaporation_unit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_evaporation_unit lbm_physics CUDA::cudart)

    # Unit test: Recoil-pressure scale isolation (flat plate, T=3500 K)
    add_executable(test_recoil_scale apps/test_recoil_scale.cu)
    set_target_properties(test_recoil_scale PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_recoil_scale lbm_physics CUDA::cudart)

    # Unit test: Recoil pressure force — isolated analytical comparison
    add_executable(test_recoil_unit apps/test_recoil_unit.cu)
    set_target_properties(test_recoil_unit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_recoil_unit lbm_physics CUDA::cudart)

    # Phase 2 verification tests
    add_executable(audit_conduction overnight_audit/tests/test_pure_conduction.cu)
    set_target_properties(audit_conduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(audit_conduction lbm_physics CUDA::cudart)

    add_executable(audit_low_power overnight_audit/tests/test_low_power.cu)
    set_target_properties(audit_low_power PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(audit_low_power lbm_physics lbm_io CUDA::cudart)

    # Round 2 audit tests
    add_executable(r2_ma_evap round2_audit/tests/test_ma_with_evap.cu)
    set_target_properties(r2_ma_evap PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r2_ma_evap lbm_physics lbm_io CUDA::cudart)

    add_executable(r2_energy round2_audit/tests/test_energy_conservation.cu)
    set_target_properties(r2_energy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r2_energy lbm_physics lbm_io CUDA::cudart)

    add_executable(r2_energy_v2 round2_audit/tests/test_energy_v2.cu)
    set_target_properties(r2_energy_v2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r2_energy_v2 lbm_physics CUDA::cudart)

    add_executable(r2_iface_vel round2_audit/tests/test_interface_velocity.cu)
    set_target_properties(r2_iface_vel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r2_iface_vel lbm_physics lbm_io CUDA::cudart)

    add_executable(r2_iface_150w round2_audit/tests/test_iface_vel_150w.cu)
    set_target_properties(r2_iface_150w PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r2_iface_150w lbm_physics lbm_io CUDA::cudart)

    add_executable(r2_iface_150w_relax round2_audit/tests/test_iface_vel_150w_relaxed.cu)
    set_target_properties(r2_iface_150w_relax PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r2_iface_150w_relax lbm_physics lbm_io CUDA::cudart)

    add_executable(r2_meltpool_state round2_audit/tests/test_meltpool_state.cu)
    set_target_properties(r2_meltpool_state PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r2_meltpool_state lbm_physics lbm_io CUDA::cudart)

    add_executable(r2_maran_vcheck round2_audit/tests/test_marangoni_velocity_check.cu)
    set_target_properties(r2_maran_vcheck PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r2_maran_vcheck lbm_physics lbm_io CUDA::cudart)

    # Round 3
    add_executable(r3_cfl_compare round3_audit/tests/test_cfl_comparison.cu)
    set_target_properties(r3_cfl_compare PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r3_cfl_compare lbm_physics lbm_io CUDA::cudart)

    # Round 4
    add_executable(r4_trt_vs_reg round4/tests/test_trt_vs_reg.cu)
    set_target_properties(r4_trt_vs_reg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(r4_trt_vs_reg lbm_physics lbm_io CUDA::cudart)

endif()

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_subdirectory(tests)

# Marangoni benchmark — source file deleted, target disabled
# add_executable(test_marangoni_benchmark tests/visualization/test_marangoni_benchmark.cu)
# set_target_properties(test_marangoni_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(test_marangoni_benchmark lbm_physics lbm_io CUDA::cudart gtest gtest_main)

# LBM performance benchmark (comprehensive metrics for waLBerla comparison)
add_executable(test_lbm_benchmark tests/performance/test_lbm_benchmark.cu)
set_target_properties(test_lbm_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_lbm_benchmark CUDA::cudart gtest gtest_main)


# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------

message(STATUS "")
message(STATUS "=============== Configuration Summary ===============")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard:        ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Toolkit Path:    ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "====================================================")
message(STATUS "")
