# CMakeLists.txt for Marangoni unit tests

cmake_minimum_required(VERSION 3.18)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Helper function to create test executable
function(add_marangoni_test test_name)
    add_executable(${test_name} ${test_name}.cu)
    target_link_libraries(${test_name} PRIVATE lbm_marangoni CUDA::cudart)
    set_target_properties(${test_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86"
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add all Marangoni tests
add_marangoni_test(test_marangoni_force_direction)
add_marangoni_test(test_marangoni_force_magnitude)
add_marangoni_test(test_marangoni_velocity_scale)
add_marangoni_test(test_marangoni_tangential_projection)
add_marangoni_test(test_marangoni_interface_localization)
add_marangoni_test(test_marangoni_gradient_limiter)
add_marangoni_test(test_marangoni_material_properties)
add_marangoni_test(test_marangoni_stability)
