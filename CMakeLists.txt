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

# Marangoni benchmark (visualization output for comparison with waLBerla)
add_executable(test_marangoni_benchmark tests/visualization/test_marangoni_benchmark.cu)
set_target_properties(test_marangoni_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_marangoni_benchmark lbm_physics lbm_io CUDA::cudart gtest gtest_main)

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
