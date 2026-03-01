/**
 * @file test_fluid_lbm.cu
 * @brief Unit tests for FluidLBM solver
 *
 * Tests include:
 * - Initialization and memory allocation
 * - Collision operator with and without forces
 * - Streaming correctness
 * - Macroscopic quantity computation
 * - Pressure recovery
 * - Shear flow validation
 * - Force response validation
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm::physics;
using namespace lbm::core;

class FluidLBMTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        D3Q19::initializeDevice();
    }

    void TearDown() override {
        // Use cudaDeviceSynchronize() instead of cudaDeviceReset()
        // cudaDeviceReset() destroys all GPU state including constant memory,
        // but D3Q19::initialized flag remains true, causing subsequent tests
        // to skip re-initialization and use invalid constant memory.
        cudaDeviceSynchronize();
    }
};

// Test 1: Constructor and initialization
TEST_F(FluidLBMTest, ConstructorAndInitialization) {
    const int nx = 10, ny = 10, nz = 10;
    float nu = 0.1f;
    float rho0 = 1.0f;

    FluidLBM solver(nx, ny, nz, nu, rho0, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, 1.0f, 1.0f);

    EXPECT_EQ(solver.getNx(), nx);
    EXPECT_EQ(solver.getNy(), ny);
    EXPECT_EQ(solver.getNz(), nz);
    EXPECT_FLOAT_EQ(solver.getViscosity(), nu);
    EXPECT_FLOAT_EQ(solver.getReferenceDensity(), rho0);
    EXPECT_GT(solver.getOmega(), 0.0f);
    EXPECT_LT(solver.getOmega(), 2.0f);

    // Initialize with uniform conditions
    solver.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    // Copy back and check
    int n_cells = nx * ny * nz;
    std::vector<float> rho(n_cells), ux(n_cells), uy(n_cells), uz(n_cells);

    solver.copyDensityToHost(rho.data());
    solver.copyVelocityToHost(ux.data(), uy.data(), uz.data());

    for (int i = 0; i < n_cells; ++i) {
        EXPECT_NEAR(rho[i], 1.0f, 1e-5f);
        EXPECT_NEAR(ux[i], 0.0f, 1e-5f);
        EXPECT_NEAR(uy[i], 0.0f, 1e-5f);
        EXPECT_NEAR(uz[i], 0.0f, 1e-5f);
    }
}

// Test 2: Uniform flow preservation (no forces, no gradients)
TEST_F(FluidLBMTest, UniformFlowPreservation) {
    const int nx = 8, ny = 8, nz = 8;
    float nu = 0.1f;
    float rho0 = 1.0f;
    float u0 = 0.05f;  // Small velocity for stability

    FluidLBM solver(nx, ny, nz, nu, rho0, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, 1.0f, 1.0f);
    solver.initialize(rho0, u0, 0.0f, 0.0f);

    // Run simulation without forces
    const int n_steps = 100;
    for (int step = 0; step < n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(0.0f, 0.0f, 0.0f);
        solver.streaming();
    }

    solver.computeMacroscopic();

    // Check that uniform flow is preserved
    int n_cells = nx * ny * nz;
    std::vector<float> ux(n_cells);
    solver.copyVelocityToHost(ux.data(), nullptr, nullptr);

    for (int i = 0; i < n_cells; ++i) {
        EXPECT_NEAR(ux[i], u0, 1e-3f) << "Uniform flow not preserved at cell " << i;
    }
}

// Test 3: Pressure computation
TEST_F(FluidLBMTest, PressureComputation) {
    const int nx = 5, ny = 5, nz = 5;
    float nu = 0.1f;
    float rho0 = 1.0f;

    FluidLBM solver(nx, ny, nz, nu, rho0, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, 1.0f, 1.0f);

    // Initialize with varying density
    int n_cells = nx * ny * nz;
    std::vector<float> rho_init(n_cells);
    std::vector<float> ux_init(n_cells, 0.0f);
    std::vector<float> uy_init(n_cells, 0.0f);
    std::vector<float> uz_init(n_cells, 0.0f);

    for (int i = 0; i < n_cells; ++i) {
        rho_init[i] = 1.0f + 0.1f * static_cast<float>(i) / n_cells;
    }

    // Upload to device
    float *d_rho, *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_rho, n_cells * sizeof(float));
    cudaMalloc(&d_ux, n_cells * sizeof(float));
    cudaMalloc(&d_uy, n_cells * sizeof(float));
    cudaMalloc(&d_uz, n_cells * sizeof(float));

    cudaMemcpy(d_rho, rho_init.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ux, ux_init.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, uy_init.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, uz_init.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);

    solver.initialize(d_rho, d_ux, d_uy, d_uz);
    solver.computeMacroscopic();

    // Check pressure: p = c_s²(ρ - ρ₀)
    std::vector<float> pressure(n_cells);
    solver.copyPressureToHost(pressure.data());

    for (int i = 0; i < n_cells; ++i) {
        float expected_p = D3Q19::CS2 * (rho_init[i] - rho0);
        EXPECT_NEAR(pressure[i], expected_p, 1e-5f);
    }

    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

// Test 4: Body force response (uniform acceleration)
TEST_F(FluidLBMTest, UniformBodyForceResponse) {
    const int nx = 8, ny = 8, nz = 8;
    float nu = 0.1f;
    float rho0 = 1.0f;
    float force_x = 1e-4f;  // Small force

    FluidLBM solver(nx, ny, nz, nu, rho0, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, 1.0f, 1.0f);
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Apply uniform force for several steps
    const int n_steps = 100;
    for (int step = 0; step < n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(force_x, 0.0f, 0.0f);
        solver.streaming();
    }

    solver.computeMacroscopic();

    // Check that velocity increased in x-direction
    int n_cells = nx * ny * nz;
    std::vector<float> ux(n_cells);
    solver.copyVelocityToHost(ux.data(), nullptr, nullptr);

    // Average velocity should be positive and roughly uniform
    float avg_ux = 0.0f;
    for (int i = 0; i < n_cells; ++i) {
        avg_ux += ux[i];
    }
    avg_ux /= n_cells;

    EXPECT_GT(avg_ux, 0.0f) << "Force did not accelerate fluid";
    EXPECT_LT(avg_ux, 0.1f) << "Velocity too large, possible instability";

    // Check uniformity
    for (int i = 0; i < n_cells; ++i) {
        EXPECT_NEAR(ux[i], avg_ux, 1e-3f) << "Non-uniform response to uniform force";
    }
}

// Test 5: Shear flow stability
TEST_F(FluidLBMTest, ShearFlowStability) {
    const int nx = 8, ny = 16, nz = 8;
    float nu = 0.1f;
    float rho0 = 1.0f;

    FluidLBM solver(nx, ny, nz, nu, rho0, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, 1.0f, 1.0f);

    // Initialize with linear shear profile in y-direction
    int n_cells = nx * ny * nz;
    std::vector<float> rho_init(n_cells, rho0);
    std::vector<float> ux_init(n_cells);
    std::vector<float> uy_init(n_cells, 0.0f);
    std::vector<float> uz_init(n_cells, 0.0f);

    float u_max = 0.05f;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;
                ux_init[id] = u_max * static_cast<float>(iy) / (ny - 1);
            }
        }
    }

    // Upload to device
    float *d_rho, *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_rho, n_cells * sizeof(float));
    cudaMalloc(&d_ux, n_cells * sizeof(float));
    cudaMalloc(&d_uy, n_cells * sizeof(float));
    cudaMalloc(&d_uz, n_cells * sizeof(float));

    cudaMemcpy(d_rho, rho_init.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ux, ux_init.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, uy_init.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, uz_init.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);

    solver.initialize(d_rho, d_ux, d_uy, d_uz);

    // Run simulation
    const int n_steps = 100;
    for (int step = 0; step < n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(0.0f, 0.0f, 0.0f);
        solver.streaming();
    }

    solver.computeMacroscopic();

    // Check stability (no NaN or excessive values)
    std::vector<float> ux(n_cells);
    solver.copyVelocityToHost(ux.data(), nullptr, nullptr);

    for (int i = 0; i < n_cells; ++i) {
        EXPECT_FALSE(std::isnan(ux[i])) << "NaN detected in velocity field";
        EXPECT_LT(std::abs(ux[i]), 0.2f) << "Excessive velocity, instability detected";
    }

    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

// Test 6: Buoyancy force computation
TEST_F(FluidLBMTest, BuoyancyForceComputation) {
    const int nx = 5, ny = 5, nz = 5;
    float nu = 0.1f;
    float rho0 = 1000.0f;

    FluidLBM solver(nx, ny, nz, nu, rho0, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, 1.0f, 1.0f);
    solver.initialize();

    // Create temperature field
    int n_cells = nx * ny * nz;
    float *d_temperature, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_temperature, n_cells * sizeof(float));
    cudaMalloc(&d_fx, n_cells * sizeof(float));
    cudaMalloc(&d_fy, n_cells * sizeof(float));
    cudaMalloc(&d_fz, n_cells * sizeof(float));

    // Initialize temperature field (linear gradient)
    std::vector<float> temp(n_cells);
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;
                temp[id] = 300.0f + 100.0f * static_cast<float>(iy) / (ny - 1);
            }
        }
    }
    cudaMemcpy(d_temperature, temp.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Compute buoyancy force
    float T_ref = 300.0f;
    float beta = 1e-4f;  // Thermal expansion coefficient
    float g = 9.81f;

    solver.computeBuoyancyForce(d_temperature, T_ref, beta,
                                0.0f, -g, 0.0f,  // Gravity in -y direction
                                d_fx, d_fy, d_fz);

    // Copy back and verify
    std::vector<float> fx(n_cells), fy(n_cells), fz(n_cells);
    cudaMemcpy(fx.data(), d_fx, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), d_fy, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), d_fz, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;

                // Expected: F_y = ρ₀·β·(T - T_ref)·g
                float dT = temp[id] - T_ref;
                float expected_fy = rho0 * beta * dT * (-g);

                EXPECT_NEAR(fx[id], 0.0f, 1e-5f);
                EXPECT_NEAR(fy[id], expected_fy, 1e-2f);
                EXPECT_NEAR(fz[id], 0.0f, 1e-5f);
            }
        }
    }

    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

// Test 7: Darcy damping application
TEST_F(FluidLBMTest, DarcyDampingApplication) {
    const int nx = 5, ny = 5, nz = 5;
    float nu = 0.1f;
    float rho0 = 1000.0f;

    FluidLBM solver(nx, ny, nz, nu, rho0, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, 1.0f, 1.0f);

    // Initialize with uniform velocity
    solver.initialize(rho0, 0.1f, 0.0f, 0.0f);

    // Create liquid fraction field
    int n_cells = nx * ny * nz;
    float *d_liquid_fraction, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_liquid_fraction, n_cells * sizeof(float));
    cudaMalloc(&d_fx, n_cells * sizeof(float));
    cudaMalloc(&d_fy, n_cells * sizeof(float));
    cudaMalloc(&d_fz, n_cells * sizeof(float));

    // Initialize liquid fraction (varying from solid to liquid)
    std::vector<float> fl(n_cells);
    for (int i = 0; i < n_cells; ++i) {
        fl[i] = static_cast<float>(i) / (n_cells - 1);  // 0 to 1
    }
    cudaMemcpy(d_liquid_fraction, fl.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize forces to zero
    cudaMemset(d_fx, 0, n_cells * sizeof(float));
    cudaMemset(d_fy, 0, n_cells * sizeof(float));
    cudaMemset(d_fz, 0, n_cells * sizeof(float));

    // Apply Darcy damping
    float darcy_const = 1e5f;
    solver.applyDarcyDamping(d_liquid_fraction, darcy_const, d_fx, d_fy, d_fz);

    // Copy back and verify
    std::vector<float> fx(n_cells);
    cudaMemcpy(fx.data(), d_fx, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Get velocities
    std::vector<float> ux(n_cells);
    solver.copyVelocityToHost(ux.data(), nullptr, nullptr);

    for (int i = 0; i < n_cells; ++i) {
        // Solid (fl=0) should have maximum damping
        // Liquid (fl=1) should have minimal/zero damping
        // Carman-Kozeny: F_damping = -C·(1-fl)²/(fl³+ε)·u
        // With C=1e5, u=0.1, damping decreases as fl increases
        if (fl[i] < 0.1f) {
            // Near solid: strong negative force (opposing flow)
            EXPECT_LT(fx[i], -1000.0f) << "Insufficient damping in solid region at fl=" << fl[i];
        } else if (fl[i] > 0.98f) {
            // Near liquid: damping should be much weaker
            // At fl=0.99: damping ~ -1e5·0.0001/(0.97+0.001)·0.1 ~ -1
            // At fl=1.0: damping = 0 (fully liquid, no resistance)
            EXPECT_GT(fx[i], -100.0f) << "Excessive damping in liquid region at fl=" << fl[i];
        }
        // Verify damping force opposes flow or is zero (for fl=1)
        EXPECT_LE(fx[i], 0.0f) << "Damping force should oppose flow or be zero at fl=" << fl[i];
    }

    cudaFree(d_liquid_fraction);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

// Test 8: Reynolds number computation
TEST_F(FluidLBMTest, ReynoldsNumberComputation) {
    const int nx = 10, ny = 10, nz = 10;
    float nu = 0.1f;
    float rho0 = 1.0f;

    FluidLBM solver(nx, ny, nz, nu, rho0, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, 1.0f, 1.0f);

    // Compute Re for typical values
    float U = 0.1f;
    float L = 10.0f;
    float Re = solver.computeReynoldsNumber(U, L);

    // Expected: Re = U·L/ν = 0.1 * 10 / 0.1 = 10
    EXPECT_NEAR(Re, 10.0f, 1e-5f);
}

// Test 9: Conservation of mass
TEST_F(FluidLBMTest, MassConservation) {
    const int nx = 8, ny = 8, nz = 8;
    float nu = 0.1f;
    float rho0 = 1.0f;

    FluidLBM solver(nx, ny, nz, nu, rho0, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, lbm::physics::BoundaryType::PERIODIC, 1.0f, 1.0f);
    solver.initialize(rho0, 0.01f, 0.0f, 0.0f);

    // Compute initial total mass
    int n_cells = nx * ny * nz;
    std::vector<float> rho_initial(n_cells);
    solver.copyDensityToHost(rho_initial.data());
    float mass_initial = 0.0f;
    for (int i = 0; i < n_cells; ++i) {
        mass_initial += rho_initial[i];
    }

    // Run simulation
    const int n_steps = 100;
    for (int step = 0; step < n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(0.0f, 0.0f, 0.0f);
        solver.streaming();
    }

    solver.computeMacroscopic();

    // Compute final total mass
    std::vector<float> rho_final(n_cells);
    solver.copyDensityToHost(rho_final.data());
    float mass_final = 0.0f;
    for (int i = 0; i < n_cells; ++i) {
        mass_final += rho_final[i];
    }

    // Check mass conservation
    float relative_error = std::abs(mass_final - mass_initial) / mass_initial;
    EXPECT_LT(relative_error, 1e-5f) << "Mass not conserved";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
