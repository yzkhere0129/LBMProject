cmake_minimum_required(VERSION 3.18)
project(LBM_CUDA LANGUAGES CUDA CXX)

# C++ / CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Default to common GPU architectures: 75 = RTX 20xx, 80 = A100, 86 = RTX 30xx,
# 89 = RTX 40xx. Override with -DCMAKE_CUDA_ARCHITECTURES=<arch>.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

set(CMAKE_CXX_FLAGS_DEBUG    "-g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native")
set(CMAKE_CUDA_FLAGS_DEBUG   "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------
# Static libraries
# ---------------------------------------------------------------------------
file(GLOB_RECURSE LBM_CORE_SOURCES
    src/core/lattice/*.cu
    src/core/collision/*.cu
    src/core/streaming/*.cu
    src/core/boundary/*.cu)

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
    src/cuda/*.cu)

file(GLOB_RECURSE LBM_DIAGNOSTICS_SOURCES src/diagnostics/*.cu)
file(GLOB_RECURSE LBM_IO_SOURCES          src/io/*.cu src/io/*.cpp)

add_library(lbm_diagnostics STATIC ${LBM_DIAGNOSTICS_SOURCES})
set_target_properties(lbm_diagnostics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lbm_diagnostics PUBLIC CUDA::cudart)

add_library(lbm_io STATIC ${LBM_IO_SOURCES})
set_target_properties(lbm_io PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lbm_io PUBLIC CUDA::cudart)

add_library(lbm_core STATIC ${LBM_CORE_SOURCES})
set_target_properties(lbm_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lbm_core PUBLIC CUDA::cudart)

add_library(lbm_physics STATIC ${LBM_PHYSICS_SOURCES})
set_target_properties(lbm_physics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lbm_physics PUBLIC CUDA::cudart lbm_core lbm_diagnostics lbm_io)

add_library(lbm_config STATIC src/config/simulation_config.cpp)
target_include_directories(lbm_config PUBLIC include)

# Common link set for all executables
set(LBM_LINK_LIBS lbm_physics lbm_core lbm_io lbm_diagnostics lbm_config CUDA::cudart)

# Helper: declare an executable with the standard properties.
function(lbm_add_exe target source)
    add_executable(${target} ${source})
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${target} PRIVATE ${LBM_LINK_LIBS})
endfunction()

# ---------------------------------------------------------------------------
# Showcase / benchmark applications
# ---------------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps)
    # LPBF benchmark suite
    lbm_add_exe(benchmark_keyhole_316L         apps/benchmark_keyhole_316L.cu)
    lbm_add_exe(benchmark_conduction_316L      apps/benchmark_conduction_316L.cu)
    lbm_add_exe(benchmark_powder_bed_316L      apps/benchmark_powder_bed_316L.cu)
    lbm_add_exe(benchmark_spot_melt_316L       apps/benchmark_spot_melt_316L.cu)
    lbm_add_exe(benchmark_thermal_lpbf         apps/benchmark_thermal_lpbf.cu)
    lbm_add_exe(sim_line_scan_316L_plic        apps/sim_line_scan_316L_plic.cu)
    lbm_add_exe(demo_powder_bed_lpbf           apps/demo_powder_bed_lpbf.cu)

    # Classic CFD benchmarks (single-physics)
    lbm_add_exe(benchmark_1d_stefan            apps/benchmark_1d_stefan.cu)
    lbm_add_exe(stefan_benchmark               apps/stefan_benchmark.cu)
    lbm_add_exe(benchmark_2d_marangoni_cavity  apps/benchmark_2d_marangoni_cavity.cu)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/viz)
    lbm_add_exe(viz_cavity              scripts/viz/viz_cavity.cu)
    lbm_add_exe(viz_taylor_green        scripts/viz/viz_taylor_green.cu)
    lbm_add_exe(viz_natural_convection  scripts/viz/viz_natural_convection.cu)
    lbm_add_exe(viz_marangoni_cavity    scripts/viz/viz_marangoni_cavity.cu)

    set_target_properties(viz_cavity viz_taylor_green viz_natural_convection
                          viz_marangoni_cavity
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/viz)
endif()
