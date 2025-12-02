# CMakeLists.txt for Marangoni diagnostic tests

cmake_minimum_required(VERSION 3.18)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Diagnostic test for velocity analysis
add_executable(test_marangoni_velocity_diagnosis test_marangoni_velocity_diagnosis.cu)
target_link_libraries(test_marangoni_velocity_diagnosis PRIVATE lbm_marangoni CUDA::cudart)
set_target_properties(test_marangoni_velocity_diagnosis PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

# Note: Diagnostic tests are not automatically run as part of test suite
# They are meant to be run manually for analysis
