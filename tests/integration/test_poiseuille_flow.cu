/**
 * @file test_poiseuille_flow.cu
 * @brief Integration test for Poiseuille flow (channel flow between parallel plates)
 *
 * This test validates the complete LBM implementation by simulating
 * pressure-driven flow between two parallel plates and comparing the
 * velocity profile with the analytical solution.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include "core/lattice_d3q19.h"
#include "core/collision_bgk.h"
#include "core/streaming.h"
#include "core/boundary_conditions.h"

using namespace lbm::core;

class PoiseuilleFlowTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        D3Q19::initializeDevice();
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    /**
     * Analytical solution for Poiseuille flow
     * u(y) = (dp/dx) * y * (H - y) / (2 * mu)
     * where H is channel height, dp/dx is pressure gradient, mu is dynamic viscosity
     */
    float analyticalVelocity(float y, float H, float dp_dx, float nu, float rho) {
        float mu = nu * rho;  // Dynamic viscosity
        return -dp_dx * y * (H - y) / (2.0f * mu);
    }

    /**
     * Maximum velocity at channel center
     */
    float maxAnalyticalVelocity(float H, float dp_dx, float nu, float rho) {
        return analyticalVelocity(H / 2.0f, H, dp_dx, nu, rho);
    }
};

/**
 * Complete Poiseuille flow simulation kernel
 */
__global__ void poiseuilleLBMStep(
    float* f_src, float* f_dst,
    float* rho, float* ux, float* uy, float* uz,
    const BoundaryNode* boundary_nodes, int n_boundary,
    float omega, float body_force_x,
    int nx, int ny, int nz) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    // Pull streaming first: read from neighbors
    float f[D3Q19::Q];
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        int src_x = (idx - ex[q] + nx) % nx;
        int src_y = idy - ey[q];
        int src_z = (idz - ez[q] + nz) % nz;

        // Handle y boundaries (walls) - bounce back
        if (idy == 0 && ey[q] < 0) {
            // Bottom wall - bounce back incoming distributions
            f[q] = f_src[id + opposite[q] * n_cells];
        } else if (idy == ny - 1 && ey[q] > 0) {
            // Top wall - bounce back incoming distributions
            f[q] = f_src[id + opposite[q] * n_cells];
        } else if (src_y < 0 || src_y >= ny) {
            // Shouldn't reach here with correct logic, but handle edge case
            f[q] = f_src[id + q * n_cells];
        } else {
            // Normal streaming from neighbor
            int src_id = src_x + src_y * nx + src_z * nx * ny;
            f[q] = f_src[src_id + q * n_cells];
        }
    }

    // Compute macroscopic quantities
    float m_rho = D3Q19::computeDensity(f);
    float m_ux, m_uy, m_uz;
    D3Q19::computeVelocity(f, m_rho, m_ux, m_uy, m_uz);

    // Apply body force (Guo forcing scheme - correct formulation)
    float m_ux_force = m_ux + 0.5f * body_force_x / m_rho;
    float m_uy_force = m_uy;
    float m_uz_force = m_uz;

    // Store macroscopic values (with force correction)
    rho[id] = m_rho;
    ux[id] = m_ux_force;
    uy[id] = m_uy_force;
    uz[id] = m_uz_force;

    // BGK collision with forcing term
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        float feq = D3Q19::computeEquilibrium(q, m_rho, m_ux_force, m_uy_force, m_uz_force);

        // Guo forcing term
        float force_term = (1.0f - 0.5f * omega) * 3.0f * w[q] * ex[q] * body_force_x;

        f_dst[id + q * n_cells] = f[q] - omega * (f[q] - feq) + force_term;
    }
}

TEST_F(PoiseuilleFlowTest, ChannelFlowConvergence) {
    // Simulation parameters
    const int nx = 3;   // Thin in x (periodic)
    const int ny = 32;  // Channel height
    const int nz = 3;   // Thin in z (periodic)
    const int n_cells = nx * ny * nz;
    const int n_total = n_cells * D3Q19::Q;

    // Physical parameters
    float nu = 0.1f;           // Kinematic viscosity
    float rho0 = 1.0f;         // Initial density
    float dp_dx = -1e-4f;      // Pressure gradient (body force)
    float body_force_x = -dp_dx / rho0;  // Body force per unit mass

    // LBM parameters
    float omega = BGKCollision::computeOmega(nu);
    ASSERT_TRUE(BGKCollision::isStable(omega)) << "Omega not stable: " << omega;

    // Allocate device memory
    float *d_f_src, *d_f_dst;
    float *d_rho, *d_ux, *d_uy, *d_uz;

    cudaMalloc(&d_f_src, n_total * sizeof(float));
    cudaMalloc(&d_f_dst, n_total * sizeof(float));
    cudaMalloc(&d_rho, n_cells * sizeof(float));
    cudaMalloc(&d_ux, n_cells * sizeof(float));
    cudaMalloc(&d_uy, n_cells * sizeof(float));
    cudaMalloc(&d_uz, n_cells * sizeof(float));

    // Initialize with equilibrium distribution at rest
    float* h_f = new float[n_total];
    for (int id = 0; id < n_cells; ++id) {
        for (int q = 0; q < D3Q19::Q; ++q) {
            h_f[id + q * n_cells] = D3Q19::computeEquilibrium(q, rho0, 0.0f, 0.0f, 0.0f);
        }
    }
    cudaMemcpy(d_f_src, h_f, n_total * sizeof(float), cudaMemcpyHostToDevice);

    // No explicit boundary nodes needed (handled in kernel)
    BoundaryNode* d_boundary_nodes = nullptr;
    int n_boundary = 0;

    // Simulation parameters - run longer for better convergence
    const int n_steps = 10000;
    const int check_interval = 1000;

    // Launch configuration
    dim3 block(nx, 4, nz);  // Small domain, use small blocks
    dim3 grid(1, (ny + block.y - 1) / block.y, 1);

    // Time evolution
    for (int step = 0; step < n_steps; ++step) {
        poiseuilleLBMStep<<<grid, block>>>(
            d_f_src, d_f_dst,
            d_rho, d_ux, d_uy, d_uz,
            d_boundary_nodes, n_boundary,
            omega, body_force_x,
            nx, ny, nz
        );

        // Swap buffers
        std::swap(d_f_src, d_f_dst);

        // Check convergence periodically
        if (step % check_interval == 0) {
            cudaDeviceSynchronize();
            std::cout << "Step " << step << " / " << n_steps << std::endl;
        }
    }

    // Final synchronization
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

    // Copy results back
    float* h_ux = new float[n_cells];
    float* h_rho = new float[n_cells];
    cudaMemcpy(h_ux, d_ux, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rho, d_rho, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Extract velocity profile at channel center
    std::vector<float> velocity_profile(ny);
    std::vector<float> analytical_profile(ny);
    float channel_height = static_cast<float>(ny - 1);

    for (int y = 0; y < ny; ++y) {
        // Average over x and z
        float avg_ux = 0.0f;
        for (int x = 0; x < nx; ++x) {
            for (int z = 0; z < nz; ++z) {
                int id = x + y * nx + z * nx * ny;
                avg_ux += h_ux[id];
            }
        }
        velocity_profile[y] = avg_ux / (nx * nz);

        // Analytical solution
        analytical_profile[y] = analyticalVelocity(
            static_cast<float>(y), channel_height, dp_dx, nu, rho0);
    }

    // Compute error metrics
    float max_velocity = maxAnalyticalVelocity(channel_height, dp_dx, nu, rho0);

    // L2 relative error (skip boundary nodes)
    float sum_sq_error = 0.0f;
    float sum_sq_analytical = 0.0f;
    float total_relative_error = 0.0f;
    float max_absolute_error = 0.0f;
    int n_interior = 0;

    for (int y = 1; y < ny - 1; ++y) {
        float error = velocity_profile[y] - analytical_profile[y];
        float abs_error = std::abs(error);

        sum_sq_error += error * error;
        sum_sq_analytical += analytical_profile[y] * analytical_profile[y];
        total_relative_error += abs_error / (std::abs(analytical_profile[y]) + 1e-10f);
        max_absolute_error = std::max(max_absolute_error, abs_error);
        n_interior++;

        // Debug output for central nodes
        if (y >= ny/2 - 2 && y <= ny/2 + 2) {
            std::cout << "y=" << y << ": LBM=" << velocity_profile[y]
                     << ", Analytical=" << analytical_profile[y]
                     << ", Error=" << error << std::endl;
        }
    }

    float L2_error = std::sqrt(sum_sq_error / (sum_sq_analytical + 1e-15f));
    float avg_relative_error = total_relative_error / n_interior;

    // Velocity comparison
    float u_max_numerical = velocity_profile[ny/2];
    float u_max_analytical = max_velocity;
    float u_max_error_percent = std::abs(u_max_numerical - u_max_analytical) / u_max_analytical * 100.0f;

    // Average velocity (excluding boundaries)
    float u_avg_numerical = 0.0f;
    for (int y = 1; y < ny - 1; ++y) {
        u_avg_numerical += velocity_profile[y];
    }
    u_avg_numerical /= n_interior;
    float u_avg_analytical = (2.0f / 3.0f) * u_max_analytical;
    float u_avg_error_percent = std::abs(u_avg_numerical - u_avg_analytical) / u_avg_analytical * 100.0f;

    std::cout << "\n=== Poiseuille Flow Validation Results ===" << std::endl;
    std::cout << "Maximum velocity:" << std::endl;
    std::cout << "  Analytical:  " << u_max_analytical << std::endl;
    std::cout << "  Numerical:   " << u_max_numerical << std::endl;
    std::cout << "  Error:       " << u_max_error_percent << " %" << std::endl;
    std::cout << "Average velocity:" << std::endl;
    std::cout << "  Analytical:  " << u_avg_analytical << std::endl;
    std::cout << "  Numerical:   " << u_avg_numerical << std::endl;
    std::cout << "  Error:       " << u_avg_error_percent << " %" << std::endl;
    std::cout << "Error metrics:" << std::endl;
    std::cout << "  L2 relative error:    " << (L2_error * 100.0f) << " %" << std::endl;
    std::cout << "  Average relative err: " << (avg_relative_error * 100.0f) << " %" << std::endl;
    std::cout << "  Maximum absolute err: " << max_absolute_error << std::endl;
    std::cout << "==========================================\n" << std::endl;

    // Check validation criteria
    EXPECT_LT(L2_error, 0.05f) << "L2 relative error exceeds 5%";
    EXPECT_LT(u_max_error_percent, 3.5f) << "Maximum velocity error exceeds 3.5%";
    EXPECT_NEAR(u_avg_numerical / u_max_numerical, 2.0f / 3.0f, 0.03f)
        << "Average/max velocity ratio should be 2/3 for Poiseuille flow";

    // Check parabolic profile shape (velocity should be maximum at center)
    float center_velocity = velocity_profile[ny/2];
    EXPECT_GT(center_velocity, velocity_profile[1]);
    EXPECT_GT(center_velocity, velocity_profile[ny-2]);

    // Check symmetry
    for (int y = 1; y < ny/2; ++y) {
        float v1 = velocity_profile[y];
        float v2 = velocity_profile[ny - 1 - y];
        float symmetry_error = std::abs(v1 - v2) / (std::abs(v1) + std::abs(v2) + 1e-10f);
        EXPECT_LT(symmetry_error, 0.1f) << "Profile not symmetric at y=" << y;
    }

    // Optional: Save velocity profile to file
    std::ofstream profile_file("poiseuille_profile.txt");
    if (profile_file.is_open()) {
        profile_file << "# y\tLBM_velocity\tAnalytical_velocity\n";
        for (int y = 0; y < ny; ++y) {
            profile_file << y << "\t" << velocity_profile[y] << "\t"
                        << analytical_profile[y] << "\n";
        }
        profile_file.close();
        std::cout << "Velocity profile saved to poiseuille_profile.txt" << std::endl;
    }

    // Clean up
    delete[] h_f;
    delete[] h_ux;
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