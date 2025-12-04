cmake_minimum_required(VERSION 3.18)
project(LBM_Metal_Melting LANGUAGES CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
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

# Collect source files
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
)

# Diagnostics library
file(GLOB_RECURSE LBM_DIAGNOSTICS_SOURCES
    src/diagnostics/*.cu
)
add_library(lbm_diagnostics STATIC ${LBM_DIAGNOSTICS_SOURCES})
set_target_properties(lbm_diagnostics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lbm_diagnostics PUBLIC CUDA::cudart)

# Create LBM core library
add_library(lbm_core STATIC ${LBM_CORE_SOURCES})

# Create LBM physics library
add_library(lbm_physics STATIC ${LBM_PHYSICS_SOURCES})

# Create I/O library
file(GLOB_RECURSE LBM_IO_SOURCES
    src/io/*.cu
)
add_library(lbm_io STATIC ${LBM_IO_SOURCES})

set_target_properties(lbm_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lbm_core PUBLIC CUDA::cudart)

set_target_properties(lbm_physics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lbm_physics PUBLIC CUDA::cudart lbm_core lbm_diagnostics)

set_target_properties(lbm_io PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lbm_io PUBLIC CUDA::cudart)

# Add visualization applications
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps)
    # VTK visualization application
    add_executable(visualize_laser_heating apps/visualize_laser_heating.cu)
    set_target_properties(visualize_laser_heating PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(visualize_laser_heating
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Real-time monitor application
    add_executable(realtime_monitor apps/realtime_monitor.cu)
    set_target_properties(realtime_monitor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(realtime_monitor
        lbm_physics
        CUDA::cudart
    )

    # Quick test application
    add_executable(test_visualization apps/test_visualization.cu)
    set_target_properties(test_visualization PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_visualization
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Phase 5: Laser melting with melt pool convection
    add_executable(visualize_laser_melting_with_flow apps/visualize_laser_melting_with_flow.cu)
    set_target_properties(visualize_laser_melting_with_flow PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(visualize_laser_melting_with_flow
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Diagnostic test for force propagation bug
    add_executable(diagnostic_force_propagation tests/diagnostic_force_propagation.cu)
    set_target_properties(diagnostic_force_propagation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(diagnostic_force_propagation
        lbm_physics
        lbm_core
        CUDA::cudart
    )

    # Diagnostic test for Phase 5 force magnitudes
    add_executable(diagnostic_phase5_forces tests/diagnostic_phase5_forces.cu)
    set_target_properties(diagnostic_phase5_forces PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(diagnostic_phase5_forces
        lbm_physics
        lbm_core
        CUDA::cudart
    )

    # Diagnostic test for Phase 5 actual force values
    add_executable(diagnostic_phase5_actual_run tests/diagnostic_phase5_actual_run.cu)
    set_target_properties(diagnostic_phase5_actual_run PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(diagnostic_phase5_actual_run
        lbm_physics
        lbm_core
        CUDA::cudart
    )

    # Phase 6: Marangoni Effect Visualization (Simple Version)
    add_executable(visualize_phase6_marangoni apps/visualize_phase6_marangoni_simple.cu)
    set_target_properties(visualize_phase6_marangoni PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(visualize_phase6_marangoni
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Realistic LPBF simulation with Marangoni convection
    add_executable(visualize_lpbf_marangoni_realistic apps/visualize_lpbf_marangoni_realistic.cu)
    set_target_properties(visualize_lpbf_marangoni_realistic PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(visualize_lpbf_marangoni_realistic
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # LPBF laser scanning simulation
    add_executable(visualize_lpbf_scanning apps/visualize_lpbf_scanning.cu)
    set_target_properties(visualize_lpbf_scanning PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(visualize_lpbf_scanning
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Energy balance diagnostic tool (Week 1 Monday: 33% error investigation)
    add_executable(diagnose_energy_balance apps/diagnose_energy_balance.cu)
    set_target_properties(diagnose_energy_balance PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(diagnose_energy_balance
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Thermal boundary condition test (debug surface temperature problem)
    add_executable(test_thermal_boundary apps/test_thermal_boundary.cu)
    set_target_properties(test_thermal_boundary PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_thermal_boundary
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Stability diagnostic tool (comprehensive stability analysis)
    add_executable(stability_diagnostic apps/stability_diagnostic.cu)
    set_target_properties(stability_diagnostic PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(stability_diagnostic
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # TEST SUITE: Fix validation tests
    # Test 1: Newton-Raphson bisection fallback stress test
    add_executable(test_newton_bisection_stress apps/test_newton_bisection_stress.cu)
    set_target_properties(test_newton_bisection_stress PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_newton_bisection_stress
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Test 2: Laser shutoff time configuration test
    add_executable(test_laser_shutoff apps/test_laser_shutoff.cu)
    set_target_properties(test_laser_shutoff PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_laser_shutoff
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Test 3: Marangoni gradient limiter stress test (CRITICAL)
    add_executable(test_marangoni_gradient_limiter apps/test_marangoni_gradient_limiter.cu)
    set_target_properties(test_marangoni_gradient_limiter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_marangoni_gradient_limiter
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Test 4: v5 Thermal-Fluid Coupling Fix (Incremental Testing)
    add_executable(test_thermal_coupling_v5 apps/test_thermal_coupling_v5.cu)
    set_target_properties(test_thermal_coupling_v5 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_thermal_coupling_v5
        lbm_physics
        lbm_io
        CUDA::cudart
    )

    # Test 5: Buoyancy Force Integration Test (Unit Test)
    add_executable(test_buoyancy_integration tests/test_buoyancy_integration.cu)
    set_target_properties(test_buoyancy_integration PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_buoyancy_integration
        lbm_physics
        lbm_core
        CUDA::cudart
    )

    # Test 6: Marangoni Velocity Scale Validation (CRITICAL - P1 Bug Fix)
    add_executable(test_marangoni_velocity_scale tests/validation/test_marangoni_velocity_scale.cu)
    set_target_properties(test_marangoni_velocity_scale PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(test_marangoni_velocity_scale
        lbm_physics
        lbm_io
        CUDA::cudart
    )
endif()

# Enable testing
enable_testing()

# Find or fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Include Google Test
include(GoogleTest)

# Add test subdirectory
add_subdirectory(tests)

# Print configuration summary
message(STATUS "")
message(STATUS "=============== Configuration Summary ===============")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard:        ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Toolkit Path:    ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "====================================================")
message(STATUS "")
# Config system library
add_library(lbm_config STATIC
    src/config/simulation_config.cpp
)
target_include_directories(lbm_config PUBLIC include)

# Generic simulation runner with config file support
add_executable(run_simulation apps/run_simulation.cu)
set_target_properties(run_simulation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(run_simulation
    lbm_physics
    lbm_io
    lbm_config
    CUDA::cudart
)

# Validation case: Single-spot laser melting (currently disabled)
# add_executable(validation_case_single_spot apps/validation_case_single_spot.cu)
# set_target_properties(validation_case_single_spot PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )
# target_link_libraries(validation_case_single_spot
#     lbm_physics
#     lbm_io
#     lbm_core
#     CUDA::cudart
# )


# Radiation BC unit test
add_executable(test_radiation_bc test_radiation_bc.cu)
set_target_properties(test_radiation_bc PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_radiation_bc
    lbm_physics
    CUDA::cudart
)

# Evaporation cooling test
add_executable(test_evaporation_cooling test_evaporation_cooling.cu)
set_target_properties(test_evaporation_cooling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_evaporation_cooling
    lbm_physics
    CUDA::cudart
)

# Simple evaporation physics validation
add_executable(test_evaporation_simple test_evaporation_simple.cu)
set_target_properties(test_evaporation_simple PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_evaporation_simple
    lbm_physics
    CUDA::cudart
)

# Powder bed generation test
add_executable(test_powder_bed_generation tests/validation/test_powder_bed_generation.cu)
set_target_properties(test_powder_bed_generation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_powder_bed_generation
    lbm_physics
    lbm_io
    CUDA::cudart
)

# Powder bed LPBF demo
add_executable(demo_powder_bed_lpbf apps/demo_powder_bed_lpbf.cu)
set_target_properties(demo_powder_bed_lpbf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(demo_powder_bed_lpbf
    lbm_physics
    lbm_io
    CUDA::cudart
)

# Performance benchmark application (waLBerla comparison)
add_executable(benchmark_thermal_lpbf apps/benchmark_thermal_lpbf.cu)
set_target_properties(benchmark_thermal_lpbf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(benchmark_thermal_lpbf
    lbm_physics
    lbm_io
    CUDA::cudart
)
add_executable(test_evaporation_diagnostic test_evaporation_diagnostic.cu)
set_target_properties(test_evaporation_diagnostic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_evaporation_diagnostic lbm_physics CUDA::cudart)

# Marangoni benchmark test (visualization output for comparison with waLBerla)
add_executable(test_marangoni_benchmark tests/visualization/test_marangoni_benchmark.cu)
set_target_properties(test_marangoni_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_marangoni_benchmark
    lbm_physics
    lbm_io
    CUDA::cudart
    gtest
    gtest_main
)

# LBM Performance Benchmark (comprehensive performance metrics for waLBerla comparison)
add_executable(test_lbm_benchmark tests/performance/test_lbm_benchmark.cu)
set_target_properties(test_lbm_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_lbm_benchmark
    CUDA::cudart
    gtest
    gtest_main
)
