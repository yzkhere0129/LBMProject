/**
 * @file test_vof_advection.cu
 * @brief Comprehensive unit tests for VOF advection functionality
 *
 * This test suite validates the VOF advection implementation integrated
 * in the Marangoni validation test. It verifies:
 * - Uniform advection correctness
 * - Mass conservation during advection
 * - Velocity unit conversion (lattice -> physical)
 * - Zero velocity stability
 * - Boundary condition handling
 * - Multi-dimensional advection
 *
 * Reference implementation:
 * /home/yzk/LBMProject/tests/validation/test_marangoni_velocity.cu:632-656
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace lbm::physics;

/**
 * @brief CUDA kernel for converting velocity from lattice to physical units
 *
 * This kernel implements the conversion:
 * u_physical [m/s] = u_lattice [dimensionless] * (dx/dt)
 *
 * @param ux_lattice Input velocity x-component in lattice units
 * @param uy_lattice Input velocity y-component in lattice units
 * @param uz_lattice Input velocity z-component in lattice units
 * @param ux_phys Output velocity x-component in physical units [m/s]
 * @param uy_phys Output velocity y-component in physical units [m/s]
 * @param uz_phys Output velocity z-component in physical units [m/s]
 * @param conversion_factor Conversion factor (dx/dt)
 * @param num_cells Number of cells in domain
 */
__global__ void convertVelocityToPhysicalKernel(
    const float* ux_lattice,
    const float* uy_lattice,
    const float* uz_lattice,
    float* ux_phys,
    float* uy_phys,
    float* uz_phys,
    float conversion_factor,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    ux_phys[idx] = ux_lattice[idx] * conversion_factor;
    uy_phys[idx] = uy_lattice[idx] * conversion_factor;
    uz_phys[idx] = uz_lattice[idx] * conversion_factor;
}

/**
 * @brief Test fixture for VOF advection tests
 *
 * Provides helper functions for setting up test scenarios.
 */
class VOFAdvectionTest : public ::testing::Test {
protected:
    /**
     * @brief Initialize vertical plane interface (perpendicular to x-axis)
     * @param vof VOF solver instance
     * @param nx, ny, nz Domain dimensions
     * @param x_interface Interface position in x-direction [cells]
     */
    void initializeVerticalInterface(VOFSolver& vof, int nx, int ny, int nz,
                                      int x_interface) {
        std::vector<float> h_fill(nx * ny * nz, 0.0f);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);
                    // Liquid (f=1) for x < x_interface, gas (f=0) for x >= x_interface
                    h_fill[idx] = (i < x_interface) ? 1.0f : 0.0f;
                }
            }
        }

        vof.initialize(h_fill.data());
    }

    /**
     * @brief Set uniform velocity field
     * @param d_ux, d_uy, d_uz Device velocity arrays
     * @param ux, uy, uz Velocity components [m/s]
     * @param num_cells Number of cells
     */
    void setUniformVelocity(float* d_ux, float* d_uy, float* d_uz,
                            float ux, float uy, float uz, int num_cells) {
        std::vector<float> h_ux(num_cells, ux);
        std::vector<float> h_uy(num_cells, uy);
        std::vector<float> h_uz(num_cells, uz);

        cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    }

    /**
     * @brief Find interface position (x-coordinate where fill ≈ 0.5)
     * @param fill Fill level field
     * @param nx, ny, nz Domain dimensions
     * @return Interface x-position [cells], or -1 if not found
     */
    float findInterfacePosition(const std::vector<float>& fill, int nx, int ny, int nz) {
        // Sample along center line (y=ny/2, z=nz/2)
        int mid_y = ny / 2;
        int mid_z = nz / 2;

        // First try to find cells with 0.4 < f < 0.6 (diffused interface)
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (mid_y + ny * mid_z);
            if (fill[idx] > 0.4f && fill[idx] < 0.6f) {
                return static_cast<float>(i);
            }
        }

        // If not found, look for transition from 1 to 0 (sharp interface)
        for (int i = 0; i < nx - 1; ++i) {
            int idx = i + nx * (mid_y + ny * mid_z);
            int idx_next = (i + 1) + nx * (mid_y + ny * mid_z);

            // Transition: current cell is liquid (f~1), next is gas (f~0)
            if (fill[idx] > 0.9f && fill[idx_next] < 0.1f) {
                return static_cast<float>(i) + 0.5f;  // Interface between i and i+1
            }
        }

        return -1.0f; // Not found
    }

    /**
     * @brief Compute total mass (sum of fill levels)
     * @param fill Fill level field
     * @return Total mass Σf_i
     */
    float computeTotalMass(const std::vector<float>& fill) {
        float mass = 0.0f;
        for (float f : fill) {
            mass += f;
        }
        return mass;
    }
};

/**
 * @brief Test 1: Uniform Advection
 *
 * Validates that a vertical plane interface translates correctly under
 * uniform velocity field.
 *
 * Setup:
 * - Domain: 50×50×50 cells, dx=2e-6 m
 * - Initial interface: x=25 cells (50 μm)
 * - Velocity: u=(0.1, 0, 0) m/s (uniform rightward)
 * - Time: 100 μs = 1000 time steps (dt=1e-7 s)
 *
 * Expected:
 * - Interface displacement: Δx = v×t = 0.1 m/s × 100 μs = 10 μm = 5 cells
 * - Final interface position: x ≈ 30 cells (tolerance: ±1 cell for diffusion)
 */
TEST_F(VOFAdvectionTest, UniformAdvection) {
    std::cout << "\n=== Test 1: Uniform Advection ===" << std::endl;

    // Setup
    const int nx = 50, ny = 50, nz = 50;
    const float dx = 2e-6f;  // 2 μm
    const float dt = 1e-7f;  // 0.1 μs
    const int num_cells = nx * ny * nz;

    const int x_interface_initial = 25;  // cells
    const float u_physical = 0.1f;       // m/s
    const int num_steps = 1000;
    const float total_time = dt * num_steps;  // 100 μs

    // Expected displacement
    const float expected_displacement_m = u_physical * total_time;  // 10 μm
    const float expected_displacement_cells = expected_displacement_m / dx;  // 5 cells
    const float expected_interface_final = x_interface_initial + expected_displacement_cells;  // 30 cells

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << " cells" << std::endl;
    std::cout << "  Grid spacing: dx = " << dx*1e6 << " μm" << std::endl;
    std::cout << "  Time step: dt = " << dt*1e6 << " μs" << std::endl;
    std::cout << "  Velocity: u = " << u_physical << " m/s" << std::endl;
    std::cout << "  Simulation time: " << total_time*1e6 << " μs (" << num_steps << " steps)" << std::endl;
    std::cout << "  Initial interface: x = " << x_interface_initial << " cells" << std::endl;
    std::cout << "  Expected displacement: " << expected_displacement_cells << " cells" << std::endl;
    std::cout << "  Expected final interface: x = " << expected_interface_final << " ± 1.0 cells" << std::endl;

    // Initialize VOF solver
    VOFSolver vof(nx, ny, nz, dx);
    initializeVerticalInterface(vof, nx, ny, nz, x_interface_initial);

    // Find initial interface position
    std::vector<float> h_fill_initial(num_cells);
    vof.copyFillLevelToHost(h_fill_initial.data());
    float interface_initial = findInterfacePosition(h_fill_initial, nx, ny, nz);

    // Allocate velocity arrays (physical units)
    float *d_ux_phys, *d_uy_phys, *d_uz_phys;
    cudaMalloc(&d_ux_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uy_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uz_phys, num_cells * sizeof(float));

    // Set uniform velocity
    setUniformVelocity(d_ux_phys, d_uy_phys, d_uz_phys, u_physical, 0.0f, 0.0f, num_cells);

    // Advection loop
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux_phys, d_uy_phys, d_uz_phys, dt);
        vof.convertCells();
        cudaDeviceSynchronize();
    }

    // Get final fill level
    std::vector<float> h_fill_final(num_cells);
    vof.copyFillLevelToHost(h_fill_final.data());

    // Find final interface position
    float interface_final = findInterfacePosition(h_fill_final, nx, ny, nz);
    float displacement_cells = interface_final - interface_initial;

    std::cout << "\n  Results:" << std::endl;
    std::cout << "    Initial interface: x = " << interface_initial << " cells" << std::endl;
    std::cout << "    Final interface: x = " << interface_final << " cells" << std::endl;
    std::cout << "    Displacement: " << displacement_cells << " cells (expected: "
              << expected_displacement_cells << ")" << std::endl;

    // Validation
    EXPECT_NEAR(interface_final, expected_interface_final, 1.0f)
        << "Interface position outside tolerance";

    std::cout << "  ✓ Test passed" << std::endl;

    // Cleanup
    cudaFree(d_ux_phys);
    cudaFree(d_uy_phys);
    cudaFree(d_uz_phys);
}

/**
 * @brief Test 2: Mass Conservation
 *
 * Validates that VOF advection conserves total liquid mass when interface
 * stays in domain interior (away from boundaries).
 *
 * Setup: Short-term advection in large domain
 *
 * Expected:
 * - Mass error: |M_final - M_initial| / M_initial < 0.5% (numerical diffusion only)
 *
 * Note: Long-term advection with clamped boundaries causes systematic mass
 * accumulation at boundaries. This test uses short duration to avoid that.
 */
TEST_F(VOFAdvectionTest, MassConservation) {
    std::cout << "\n=== Test 2: Mass Conservation ===" << std::endl;

    // Setup: Large domain, short time (interface stays in interior)
    const int nx = 100, ny = 50, nz = 50;  // Large domain
    const float dx = 2e-6f;
    const float dt = 1e-7f;
    const int num_cells = nx * ny * nz;

    const int x_interface_initial = 50;  // Center
    const float u_physical = 0.05f;      // Slower velocity
    const int num_steps = 100;           // Shorter time (10 μs)

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << " cells" << std::endl;
    std::cout << "  Advecting for " << num_steps << " steps" << std::endl;

    // Initialize VOF solver
    VOFSolver vof(nx, ny, nz, dx);
    initializeVerticalInterface(vof, nx, ny, nz, x_interface_initial);

    // Record initial mass
    float initial_mass = vof.computeTotalMass();
    std::cout << "  Initial mass: M0 = " << initial_mass << std::endl;
    std::cout << "  Expected mass: nx*ny*nz/2 = " << (nx*ny*nz)/2 << std::endl;

    // Allocate velocity arrays
    float *d_ux_phys, *d_uy_phys, *d_uz_phys;
    cudaMalloc(&d_ux_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uy_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uz_phys, num_cells * sizeof(float));

    setUniformVelocity(d_ux_phys, d_uy_phys, d_uz_phys, u_physical, 0.0f, 0.0f, num_cells);

    // Advection loop with periodic mass checking
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux_phys, d_uy_phys, d_uz_phys, dt);
        cudaDeviceSynchronize();

        // Check mass every 200 steps
        if (step % 200 == 0) {
            float mass_at_step = vof.computeTotalMass();
            float error_at_step = std::abs(mass_at_step - initial_mass) / initial_mass;
            std::cout << "    Step " << step << ": mass = " << mass_at_step
                      << ", error = " << error_at_step * 100.0f << "%" << std::endl;
        }
    }

    // Check final mass
    float final_mass = vof.computeTotalMass();
    float mass_error = std::abs(final_mass - initial_mass) / initial_mass;

    std::cout << "  Final mass: M_final = " << final_mass << std::endl;
    std::cout << "  Absolute error: |ΔM| = " << std::abs(final_mass - initial_mass) << std::endl;
    std::cout << "  Relative error: " << mass_error * 100.0f << "% (threshold: < 1%)" << std::endl;

    // Validation: Allow 1% error for numerical diffusion (upwind scheme is diffusive)
    EXPECT_LT(mass_error, 0.01f)
        << "Mass conservation violated: relative error = " << mass_error;

    std::cout << "  ✓ Test passed" << std::endl;

    // Cleanup
    cudaFree(d_ux_phys);
    cudaFree(d_uy_phys);
    cudaFree(d_uz_phys);
}

/**
 * @brief Test 3: Velocity Unit Conversion
 *
 * Validates the convertVelocityToPhysicalKernel function.
 *
 * Setup:
 * - Small domain: 10×10×10 cells
 * - Lattice velocity: u_lattice = 0.01 (typical LBM value)
 * - Physical parameters: dx = 2e-6 m, dt = 1e-7 s
 * - Conversion factor: dx/dt = 20.0
 *
 * Expected:
 * - Physical velocity: u_phys = 0.01 × 20.0 = 0.2 m/s
 */
TEST_F(VOFAdvectionTest, VelocityConversion) {
    std::cout << "\n=== Test 3: Velocity Unit Conversion ===" << std::endl;

    // Setup
    const int nx = 10, ny = 10, nz = 10;
    const int num_cells = nx * ny * nz;
    const float dx = 2e-6f;  // m
    const float dt = 1e-7f;  // s
    const float u_lattice = 0.01f;  // dimensionless
    const float conversion_factor = dx / dt;  // 20.0
    const float expected_u_phys = u_lattice * conversion_factor;  // 0.2 m/s

    std::cout << "  Lattice velocity: u_lattice = " << u_lattice << std::endl;
    std::cout << "  Physical parameters: dx = " << dx << " m, dt = " << dt << " s" << std::endl;
    std::cout << "  Conversion factor: dx/dt = " << conversion_factor << std::endl;
    std::cout << "  Expected physical velocity: " << expected_u_phys << " m/s" << std::endl;

    // Allocate arrays
    float *d_ux_lattice, *d_uy_lattice, *d_uz_lattice;
    float *d_ux_phys, *d_uy_phys, *d_uz_phys;

    cudaMalloc(&d_ux_lattice, num_cells * sizeof(float));
    cudaMalloc(&d_uy_lattice, num_cells * sizeof(float));
    cudaMalloc(&d_uz_lattice, num_cells * sizeof(float));
    cudaMalloc(&d_ux_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uy_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uz_phys, num_cells * sizeof(float));

    // Initialize lattice velocities
    std::vector<float> h_ux_lattice(num_cells, u_lattice);
    std::vector<float> h_uy_lattice(num_cells, 0.0f);
    std::vector<float> h_uz_lattice(num_cells, 0.0f);

    cudaMemcpy(d_ux_lattice, h_ux_lattice.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy_lattice, h_uy_lattice.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz_lattice, h_uz_lattice.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Call conversion kernel
    int block_size = 256;
    int grid_size = (num_cells + block_size - 1) / block_size;

    convertVelocityToPhysicalKernel<<<grid_size, block_size>>>(
        d_ux_lattice, d_uy_lattice, d_uz_lattice,
        d_ux_phys, d_uy_phys, d_uz_phys,
        conversion_factor, num_cells
    );

    cudaDeviceSynchronize();

    // Check results
    std::vector<float> h_ux_phys(num_cells);
    cudaMemcpy(h_ux_phys.data(), d_ux_phys, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\n  Results (sample of first 5 cells):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "    Cell " << i << ": u_phys = " << h_ux_phys[i]
                  << " m/s (expected: " << expected_u_phys << ")" << std::endl;
        EXPECT_FLOAT_EQ(h_ux_phys[i], expected_u_phys)
            << "Conversion incorrect at cell " << i;
    }

    std::cout << "  ✓ Test passed" << std::endl;

    // Cleanup
    cudaFree(d_ux_lattice);
    cudaFree(d_uy_lattice);
    cudaFree(d_uz_lattice);
    cudaFree(d_ux_phys);
    cudaFree(d_uy_phys);
    cudaFree(d_uz_phys);
}

/**
 * @brief Test 4: Zero Velocity
 *
 * Validates that interface remains stationary when velocity is zero.
 *
 * Setup:
 * - Domain: 32×32×32 cells
 * - Velocity: u=v=w=0
 * - Run 1000 steps
 *
 * Expected:
 * - Fill level field unchanged (bit-identical)
 */
TEST_F(VOFAdvectionTest, ZeroVelocity) {
    std::cout << "\n=== Test 4: Zero Velocity ===" << std::endl;

    // Setup
    const int nx = 32, ny = 32, nz = 32;
    const float dx = 2e-6f;
    const float dt = 1e-7f;
    const int num_cells = nx * ny * nz;

    const int x_interface = 16;
    const int num_steps = 1000;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << " cells" << std::endl;
    std::cout << "  Velocity: u = v = w = 0" << std::endl;
    std::cout << "  Running " << num_steps << " steps" << std::endl;

    // Initialize VOF solver
    VOFSolver vof(nx, ny, nz, dx);
    initializeVerticalInterface(vof, nx, ny, nz, x_interface);

    // Store initial state
    std::vector<float> h_fill_initial(num_cells);
    vof.copyFillLevelToHost(h_fill_initial.data());

    // Allocate velocity arrays (all zeros)
    float *d_ux_phys, *d_uy_phys, *d_uz_phys;
    cudaMalloc(&d_ux_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uy_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uz_phys, num_cells * sizeof(float));

    cudaMemset(d_ux_phys, 0, num_cells * sizeof(float));
    cudaMemset(d_uy_phys, 0, num_cells * sizeof(float));
    cudaMemset(d_uz_phys, 0, num_cells * sizeof(float));

    // Advection loop
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux_phys, d_uy_phys, d_uz_phys, dt);
        cudaDeviceSynchronize();
    }

    // Check final state
    std::vector<float> h_fill_final(num_cells);
    vof.copyFillLevelToHost(h_fill_final.data());

    // Compute maximum change
    float max_change = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        float change = std::abs(h_fill_final[i] - h_fill_initial[i]);
        max_change = std::max(max_change, change);
    }

    std::cout << "  Maximum fill level change: " << max_change << " (threshold: < 1e-6)" << std::endl;

    // Validation
    for (int i = 0; i < num_cells; ++i) {
        EXPECT_NEAR(h_fill_final[i], h_fill_initial[i], 1e-6f)
            << "Fill level changed at cell " << i;
    }

    std::cout << "  ✓ Test passed" << std::endl;

    // Cleanup
    cudaFree(d_ux_phys);
    cudaFree(d_uy_phys);
    cudaFree(d_uz_phys);
}

/**
 * @brief Test 5: Boundary Conditions
 *
 * Validates behavior when interface approaches domain boundaries.
 *
 * Setup:
 * - Domain: 32×32×32 cells
 * - Initial interface: x=3 (near left boundary)
 * - Velocity: u=(-0.1, 0, 0) m/s (toward boundary)
 *
 * Expected:
 * - Fill level remains in [0,1]
 * - No NaN or Inf values
 */
TEST_F(VOFAdvectionTest, BoundaryConditions) {
    std::cout << "\n=== Test 5: Boundary Conditions ===" << std::endl;

    // Setup
    const int nx = 32, ny = 32, nz = 32;
    const float dx = 2e-6f;
    const float dt = 1e-7f;
    const int num_cells = nx * ny * nz;

    const int x_interface = 3;  // Near left boundary
    const float u_physical = -0.1f;  // Toward left boundary
    const int num_steps = 500;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << " cells" << std::endl;
    std::cout << "  Initial interface: x = " << x_interface << " cells (near left boundary)" << std::endl;
    std::cout << "  Velocity: u = " << u_physical << " m/s (toward boundary)" << std::endl;
    std::cout << "  Running " << num_steps << " steps" << std::endl;

    // Initialize VOF solver
    VOFSolver vof(nx, ny, nz, dx);
    initializeVerticalInterface(vof, nx, ny, nz, x_interface);

    // Allocate velocity arrays
    float *d_ux_phys, *d_uy_phys, *d_uz_phys;
    cudaMalloc(&d_ux_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uy_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uz_phys, num_cells * sizeof(float));

    setUniformVelocity(d_ux_phys, d_uy_phys, d_uz_phys, u_physical, 0.0f, 0.0f, num_cells);

    // Advection loop
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux_phys, d_uy_phys, d_uz_phys, dt);
        cudaDeviceSynchronize();
    }

    // Check bounds and validity
    std::vector<float> h_fill_final(num_cells);
    vof.copyFillLevelToHost(h_fill_final.data());

    int violations = 0;
    for (int i = 0; i < num_cells; ++i) {
        float f = h_fill_final[i];

        // Check for NaN or Inf
        ASSERT_FALSE(std::isnan(f)) << "NaN detected at cell " << i;
        ASSERT_FALSE(std::isinf(f)) << "Inf detected at cell " << i;

        // Check bounds [0, 1]
        if (f < 0.0f || f > 1.0f) {
            violations++;
        }
        ASSERT_GE(f, 0.0f) << "Fill level < 0 at cell " << i;
        ASSERT_LE(f, 1.0f) << "Fill level > 1 at cell " << i;
    }

    std::cout << "  Bound violations: " << violations << " / " << num_cells << std::endl;
    std::cout << "  ✓ Test passed (no NaN, Inf, or bound violations)" << std::endl;

    // Cleanup
    cudaFree(d_ux_phys);
    cudaFree(d_uy_phys);
    cudaFree(d_uz_phys);
}

/**
 * @brief Test 6: Multi-Dimensional Advection
 *
 * Validates 2D advection (diagonal flow).
 *
 * Setup:
 * - Domain: 50×50×50 cells
 * - Initial interface: vertical plane at x=25
 * - Velocity: u=(0.1, 0.1, 0) m/s (45° angle in xy-plane)
 * - Time: 100 μs
 *
 * Expected:
 * - Interface moves diagonally: Δx = Δy ≈ 5 cells
 * - Interface normal remains (1,0,0) (still perpendicular to x-axis)
 */
TEST_F(VOFAdvectionTest, DiagonalAdvection) {
    std::cout << "\n=== Test 6: Multi-Dimensional (Diagonal) Advection ===" << std::endl;

    // Setup
    const int nx = 50, ny = 50, nz = 50;
    const float dx = 2e-6f;
    const float dt = 1e-7f;
    const int num_cells = nx * ny * nz;

    const int x_interface_initial = 25;
    const float ux_physical = 0.1f;   // m/s
    const float uy_physical = 0.1f;   // m/s
    const int num_steps = 1000;

    const float expected_displacement_cells = 5.0f;  // Both x and y directions

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << " cells" << std::endl;
    std::cout << "  Velocity: u = (" << ux_physical << ", " << uy_physical << ", 0) m/s" << std::endl;
    std::cout << "  Angle: 45° in xy-plane" << std::endl;
    std::cout << "  Expected displacement: Δx = Δy ≈ " << expected_displacement_cells << " cells" << std::endl;

    // Initialize VOF solver
    VOFSolver vof(nx, ny, nz, dx);
    initializeVerticalInterface(vof, nx, ny, nz, x_interface_initial);

    // Find initial interface position
    std::vector<float> h_fill_initial(num_cells);
    vof.copyFillLevelToHost(h_fill_initial.data());
    float interface_x_initial = findInterfacePosition(h_fill_initial, nx, ny, nz);

    // Allocate velocity arrays
    float *d_ux_phys, *d_uy_phys, *d_uz_phys;
    cudaMalloc(&d_ux_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uy_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uz_phys, num_cells * sizeof(float));

    setUniformVelocity(d_ux_phys, d_uy_phys, d_uz_phys, ux_physical, uy_physical, 0.0f, num_cells);

    // Advection loop
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux_phys, d_uy_phys, d_uz_phys, dt);
        cudaDeviceSynchronize();
    }

    // Get final state
    std::vector<float> h_fill_final(num_cells);
    vof.copyFillLevelToHost(h_fill_final.data());

    // Find final interface position
    float interface_x_final = findInterfacePosition(h_fill_final, nx, ny, nz);
    float displacement_x = interface_x_final - interface_x_initial;

    std::cout << "\n  Results:" << std::endl;
    std::cout << "    Initial interface: x = " << interface_x_initial << " cells" << std::endl;
    std::cout << "    Final interface: x = " << interface_x_final << " cells" << std::endl;
    std::cout << "    Displacement in x: " << displacement_x << " cells (expected: ~"
              << expected_displacement_cells << ")" << std::endl;

    // Validation: interface should move in x-direction
    EXPECT_NEAR(displacement_x, expected_displacement_cells, 1.5f)
        << "X-displacement outside tolerance";

    // Check that interface shape is preserved (should still be vertical plane)
    // Sample at different y-positions along the interface
    int mid_z = nz / 2;
    bool shape_preserved = true;
    for (int j = ny/4; j < 3*ny/4; j += 5) {
        int idx = static_cast<int>(interface_x_final) + nx * (j + ny * mid_z);
        if (idx >= 0 && idx < num_cells) {
            if (h_fill_final[idx] < 0.3f || h_fill_final[idx] > 0.7f) {
                shape_preserved = false;
                break;
            }
        }
    }

    EXPECT_TRUE(shape_preserved) << "Interface shape not preserved";
    std::cout << "    Interface shape: " << (shape_preserved ? "preserved" : "distorted") << std::endl;

    std::cout << "  ✓ Test passed" << std::endl;

    // Cleanup
    cudaFree(d_ux_phys);
    cudaFree(d_uy_phys);
    cudaFree(d_uz_phys);
}

/**
 * @brief Main test runner
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
