/**
 * @file test_bounce_back_kernel_direct.cu
 * @brief Direct test of the bounce-back kernel
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "core/boundary_conditions.h"
#include "core/lattice_d3q19.h"
#include "core/streaming.h"

using namespace lbm::core;

TEST(BounceBackKernel, DirectTest) {
    std::cout << "\n=== Direct Bounce-Back Kernel Test ===" << std::endl;

    // Initialize D3Q19 lattice
    D3Q19::initializeDevice();

    // Simple 1x1x1 domain (single cell)
    const int nx = 1, ny = 1, nz = 1;
    const int n_cells = nx * ny * nz;
    const int Q = D3Q19::Q;

    // Allocate device memory for distributions
    float* d_f;
    cudaMalloc(&d_f, Q * n_cells * sizeof(float));

    // Initialize distributions with test pattern
    // Set all distributions to have some momentum in +x direction
    std::vector<float> h_f(Q * n_cells, 0.0f);

    // Equilibrium-like distribution with ux = 0.1, uy = 0, uz = 0, rho = 1.0
    // Simplified: just set f[1] (ex=+1) to be larger than f[2] (ex=-1)
    h_f[0] = 0.3f;  // rest
    h_f[1] = 0.2f;  // +x
    h_f[2] = 0.1f;  // -x
    h_f[3] = 0.1f;  // +y
    h_f[4] = 0.1f;  // -y
    h_f[5] = 0.1f;  // +z
    h_f[6] = 0.05f; // -z (outgoing for Z_MIN boundary)

    // Copy to device
    cudaMemcpy(d_f, h_f.data(), Q * n_cells * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "  Initial distributions:" << std::endl;
    std::cout << "    f[5] (incoming, ez=+1): " << h_f[5] << std::endl;
    std::cout << "    f[6] (outgoing, ez=-1): " << h_f[6] << std::endl;

    // Create boundary node for Z_MIN
    BoundaryNode h_node;
    h_node.x = 0;
    h_node.y = 0;
    h_node.z = 0;
    h_node.type = BoundaryType::BOUNCE_BACK;
    h_node.ux = 0.0f;
    h_node.uy = 0.0f;
    h_node.uz = 0.0f;
    h_node.pressure = 0.0f;
    h_node.directions = Streaming::BOUNDARY_Z_MIN;

    BoundaryNode* d_boundary_nodes;
    cudaMalloc(&d_boundary_nodes, sizeof(BoundaryNode));
    cudaMemcpy(d_boundary_nodes, &h_node, sizeof(BoundaryNode), cudaMemcpyHostToDevice);

    // Launch kernel
    applyBounceBackKernel<<<1, 1>>>(d_f, d_boundary_nodes, 1, nx, ny, nz);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(error) << std::endl;
        FAIL() << "Kernel launch failed";
    }

    // Copy result back
    cudaMemcpy(h_f.data(), d_f, Q * n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "  After bounce-back:" << std::endl;
    std::cout << "    f[5] (should be " << 0.05f << "): " << h_f[5] << std::endl;
    std::cout << "    f[6] (should be " << 0.1f << "): " << h_f[6] << std::endl;

    // After bounce-back:
    // f[5] should get f_old[6] = 0.05
    // f[6] should get f_old[5] = 0.1
    EXPECT_NEAR(h_f[5], 0.05f, 1e-6f) << "Incoming direction should get outgoing value";
    EXPECT_NEAR(h_f[6], 0.1f, 1e-6f) << "Outgoing direction should get incoming value";

    // Cleanup
    cudaFree(d_f);
    cudaFree(d_boundary_nodes);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
