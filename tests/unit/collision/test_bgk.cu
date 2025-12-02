/**
 * @file test_bgk.cu
 * @brief Unit tests for BGK collision operator
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "core/collision_bgk.h"
#include "core/lattice_d3q19.h"

using namespace lbm::core;

class BGKCollisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        D3Q19::initializeDevice();
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

// Test omega calculation from viscosity
TEST_F(BGKCollisionTest, OmegaCalculation) {
    float nu = 0.1f;  // Kinematic viscosity
    float dt = 1.0f;  // Time step

    float omega = BGKCollision::computeOmega(nu, dt);

    // Expected: omega = dt / (nu/cs^2 + dt/2) = 1.0 / (0.1*3 + 0.5) = 1.0/0.8 = 1.25
    float expected_tau = nu / D3Q19::CS2 + 0.5f * dt;
    float expected_omega = dt / expected_tau;

    EXPECT_NEAR(omega, expected_omega, 1e-6);
    EXPECT_NEAR(omega, 1.25f, 1e-6);
}

// Test viscosity calculation from omega
TEST_F(BGKCollisionTest, ViscosityCalculation) {
    float omega = 1.0f;
    float dt = 1.0f;

    float nu = BGKCollision::computeViscosity(omega, dt);

    // Expected: nu = cs^2 * (dt/omega - dt/2) = (1/3) * (1.0 - 0.5) = 1/6
    EXPECT_NEAR(nu, 1.0f/6.0f, 1e-6);
}

// Test tau calculation
TEST_F(BGKCollisionTest, TauCalculation) {
    float omega = 1.5f;
    float dt = 1.0f;

    float tau = BGKCollision::computeTau(omega, dt);
    EXPECT_NEAR(tau, dt/omega, 1e-6);
}

// Test stability criterion
TEST_F(BGKCollisionTest, StabilityCriterion) {
    // Test stable values
    EXPECT_TRUE(BGKCollision::isStable(1.0f));
    EXPECT_TRUE(BGKCollision::isStable(1.5f));
    EXPECT_TRUE(BGKCollision::isStable(0.8f));

    // Test unstable values
    EXPECT_FALSE(BGKCollision::isStable(0.4f));
    EXPECT_FALSE(BGKCollision::isStable(2.1f));
    EXPECT_FALSE(BGKCollision::isStable(0.0f));
}

// Test maximum stable velocity
TEST_F(BGKCollisionTest, MaxStableVelocity) {
    float omega = 1.0f;
    float u_max = BGKCollision::maxStableVelocity(omega);

    // Should be around 0.1 * sqrt(1/3)
    float expected = 0.1f * std::sqrt(1.0f/3.0f);
    EXPECT_NEAR(u_max, expected, 1e-6);
}

// Test single collision operation
TEST_F(BGKCollisionTest, SingleCollision) {
    float f = 0.05f;
    float feq = 0.06f;
    float omega = 1.2f;

    float f_new = BGKCollision::collide(f, feq, omega);

    // Expected: f_new = f + omega * (feq - f)
    float expected = f + omega * (feq - f);
    EXPECT_NEAR(f_new, expected, 1e-7);
}

// Test node collision
TEST_F(BGKCollisionTest, NodeCollision) {
    float f_in[D3Q19::Q];
    float f_out[D3Q19::Q];

    float rho = 1.0f;
    float ux = 0.1f, uy = 0.05f, uz = -0.02f;
    float omega = 1.0f;

    // Initialize with equilibrium
    for (int q = 0; q < D3Q19::Q; ++q) {
        f_in[q] = D3Q19::computeEquilibrium(q, rho, ux, uy, uz);
    }

    // Add small perturbation
    f_in[0] += 0.01f;
    f_in[1] -= 0.005f;

    BGKCollision::collideNode(f_in, f_out, rho, ux, uy, uz, omega);

    // Check mass conservation
    float total_mass = 0.0f;
    for (int q = 0; q < D3Q19::Q; ++q) {
        total_mass += f_out[q];
    }
    EXPECT_NEAR(total_mass, rho, 1e-5);
}

// Test mass and momentum conservation after collision
TEST_F(BGKCollisionTest, ConservationLaws) {
    float f_in[D3Q19::Q];
    float f_out[D3Q19::Q];

    // Initial state with some non-equilibrium
    float rho = 1.5f;
    float ux = 0.08f, uy = -0.04f, uz = 0.02f;

    for (int q = 0; q < D3Q19::Q; ++q) {
        f_in[q] = D3Q19::computeEquilibrium(q, rho, ux, uy, uz);
    }

    // Add non-equilibrium perturbation
    f_in[1] += 0.02f;
    f_in[2] -= 0.02f;
    f_in[3] += 0.01f;
    f_in[4] -= 0.01f;

    // Compute actual macroscopic values
    float actual_rho = D3Q19::computeDensity(f_in);
    float actual_ux, actual_uy, actual_uz;
    D3Q19::computeVelocity(f_in, actual_rho, actual_ux, actual_uy, actual_uz);

    // Apply collision
    float omega = 1.2f;
    BGKCollision::collideNode(f_in, f_out, actual_rho, actual_ux, actual_uy, actual_uz, omega);

    // Check conservation after collision
    float post_rho = D3Q19::computeDensity(f_out);
    float post_ux, post_uy, post_uz;
    D3Q19::computeVelocity(f_out, post_rho, post_ux, post_uy, post_uz);

    // Mass and momentum should be conserved
    EXPECT_NEAR(post_rho, actual_rho, 1e-6);
    EXPECT_NEAR(post_ux, actual_ux, 1e-6);
    EXPECT_NEAR(post_uy, actual_uy, 1e-6);
    EXPECT_NEAR(post_uz, actual_uz, 1e-6);
}

// Note: We cannot call a __global__ function from another __global__ function
// So we'll test the kernel directly

TEST_F(BGKCollisionTest, GPUCollisionKernel) {
    const int nx = 4, ny = 4, nz = 4;
    const int n_cells = nx * ny * nz;
    const int n_total = n_cells * D3Q19::Q;

    // Allocate device memory
    float *d_f_src, *d_f_dst, *d_rho, *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_f_src, n_total * sizeof(float));
    cudaMalloc(&d_f_dst, n_total * sizeof(float));
    cudaMalloc(&d_rho, n_cells * sizeof(float));
    cudaMalloc(&d_ux, n_cells * sizeof(float));
    cudaMalloc(&d_uy, n_cells * sizeof(float));
    cudaMalloc(&d_uz, n_cells * sizeof(float));

    // Initialize with equilibrium distribution on host
    float* h_f_src = new float[n_total];
    for (int id = 0; id < n_cells; ++id) {
        float rho = 1.0f + 0.01f * id;
        float ux = 0.05f;
        float uy = -0.02f;
        float uz = 0.01f;

        for (int q = 0; q < D3Q19::Q; ++q) {
            h_f_src[id + q * n_cells] = D3Q19::computeEquilibrium(q, rho, ux, uy, uz);
        }
    }

    // Copy to device
    cudaMemcpy(d_f_src, h_f_src, n_total * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(2, 2, 2);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    float omega = 1.0f;
    bgkCollisionKernel<<<grid, block>>>(d_f_src, d_f_dst, d_rho, d_ux, d_uy, d_uz, omega, nx, ny, nz);
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

    // Copy results back
    float* h_rho = new float[n_cells];
    cudaMemcpy(h_rho, d_rho, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Basic sanity check - density should be positive
    for (int id = 0; id < n_cells; ++id) {
        EXPECT_GT(h_rho[id], 0.0f);
    }

    // Clean up
    delete[] h_f_src;
    delete[] h_rho;
    cudaFree(d_f_src);
    cudaFree(d_f_dst);
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

// Test fused collision-stream kernel
TEST_F(BGKCollisionTest, GPUCollisionStreamKernel) {
    const int nx = 8, ny = 8, nz = 8;
    const int n_cells = nx * ny * nz;
    const int n_total = n_cells * D3Q19::Q;

    // Allocate device memory
    float *d_f_src, *d_f_dst, *d_rho, *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_f_src, n_total * sizeof(float));
    cudaMalloc(&d_f_dst, n_total * sizeof(float));
    cudaMalloc(&d_rho, n_cells * sizeof(float));
    cudaMalloc(&d_ux, n_cells * sizeof(float));
    cudaMalloc(&d_uy, n_cells * sizeof(float));
    cudaMalloc(&d_uz, n_cells * sizeof(float));

    // Initialize with uniform flow
    float* h_f_src = new float[n_total];
    float init_rho = 1.0f;
    float init_ux = 0.05f;
    float init_uy = 0.0f;
    float init_uz = 0.0f;

    for (int id = 0; id < n_cells; ++id) {
        for (int q = 0; q < D3Q19::Q; ++q) {
            h_f_src[id + q * n_cells] = D3Q19::computeEquilibrium(q, init_rho, init_ux, init_uy, init_uz);
        }
    }

    cudaMemcpy(d_f_src, h_f_src, n_total * sizeof(float), cudaMemcpyHostToDevice);

    // Launch fused kernel
    dim3 block(4, 4, 4);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    float omega = 1.2f;
    bgkCollisionStreamKernel<<<grid, block>>>(d_f_src, d_f_dst, d_rho, d_ux, d_uy, d_uz, omega, nx, ny, nz);
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

    // Copy results back and verify
    float* h_f_dst = new float[n_total];
    float* h_rho = new float[n_cells];
    cudaMemcpy(h_f_dst, d_f_dst, n_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rho, d_rho, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check mass conservation globally
    float total_mass = 0.0f;
    for (int id = 0; id < n_cells; ++id) {
        total_mass += h_rho[id];
    }
    float expected_mass = init_rho * n_cells;
    EXPECT_NEAR(total_mass, expected_mass, 1e-4);

    // Clean up
    delete[] h_f_src;
    delete[] h_f_dst;
    delete[] h_rho;
    cudaFree(d_f_src);
    cudaFree(d_f_dst);
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}