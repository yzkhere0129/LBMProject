/**
 * @file boundary_conditions.cu
 * @brief Implementation of boundary conditions
 */

#include "core/boundary_conditions.h"
#include "core/lattice_d3q19.h"
#include "core/streaming.h"
#include <cstdio>
#include "utils/cuda_check.h"

namespace lbm {
namespace core {

__host__ __device__ float BoundaryConditions::bounceBack(const float* f, int q) {
    // Simple bounce-back: return the opposite direction
#ifdef __CUDA_ARCH__
    return f[opposite[q]];
#else
    // Host version - use hardcoded opposites (must match D3Q19::h_opposite in d3q19.cu)
    const int h_opposite[19] = {
        0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15
    };
    return f[h_opposite[q]];
#endif
}

__host__ __device__ void BoundaryConditions::bounceBackNode(const float* f_in, float* f_out) {
    // Apply bounce-back to all directions
#ifdef __CUDA_ARCH__
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        f_out[q] = f_in[opposite[q]];
    }
#else
    // Host version (must match D3Q19::h_opposite in d3q19.cu)
    const int h_opposite[19] = {
        0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15
    };
    for (int q = 0; q < D3Q19::Q; ++q) {
        f_out[q] = f_in[h_opposite[q]];
    }
#endif
}

__device__ void BoundaryConditions::velocityBoundaryZouHe(
    float* f, float rho_unused,
    float ux_wall, float uy_wall, float uz_wall,
    int normal, int sign) {

    // Zou-He velocity boundary condition implementation
    // Based on Zou & He (1997) equilibrium extrapolation scheme
    //
    // CRITICAL FIX: Compute rho from current distributions using mass conservation
    // The formula is: rho = (sum of known) + 2*(sum of outgoing directions)
    // where outgoing directions have velocity pointing away from domain
    //
    // For velocity BC with u_normal = 0:
    //   rho = 1/(1 - u_normal) * [f_known_sum + 2*f_outgoing_sum]
    //   which simplifies to: rho = f_known_sum + 2*f_outgoing_sum for u_normal = 0

    float rho;  // Will be computed from distributions

    if (normal == 0) {  // X boundaries
        if (sign == -1) {  // x_min boundary (x=0, flow enters from left)
            // Unknown distributions: f1, f7, f9, f11, f13 (pointing into domain in +x direction)
            // Outgoing (known) distributions: f2, f8, f10, f12, f14 (pointing out in -x direction)
            // Other known: f0, f3, f4, f5, f6, f15, f16, f17, f18

            // Compute rho from mass conservation for ux_wall BC
            // rho = 1/(1-ux_wall) * (f_known + 2*f_outgoing)
            float f_known = f[0] + f[3] + f[4] + f[5] + f[6] + f[15] + f[16] + f[17] + f[18];
            float f_outgoing = f[2] + f[8] + f[10] + f[12] + f[14];
            rho = (f_known + 2.0f * f_outgoing) / (1.0f - ux_wall);

            float rho_ux = rho * ux_wall;
            float rho_uy = rho * uy_wall;
            float rho_uz = rho * uz_wall;

            // Main direction
            f[1] = f[2] + (2.0f / 3.0f) * rho_ux;

            // Diagonal distributions
            f[7] = f[10] + (1.0f / 6.0f) * rho_ux + 0.5f * (f[4] - f[3]) + 0.5f * rho_uy;
            f[9] = f[8] + (1.0f / 6.0f) * rho_ux + 0.5f * (f[3] - f[4]) - 0.5f * rho_uy;
            f[11] = f[14] + (1.0f / 6.0f) * rho_ux + 0.5f * (f[6] - f[5]) + 0.5f * rho_uz;
            f[13] = f[12] + (1.0f / 6.0f) * rho_ux + 0.5f * (f[5] - f[6]) - 0.5f * rho_uz;
        }
        else if (sign == 1) {  // x_max boundary (x=nx-1, flow exits to right)
            // Unknown distributions: f2, f8, f10, f12, f14 (pointing into domain in -x direction)
            // Outgoing (known): f1, f7, f9, f11, f13 (pointing out in +x direction)

            float f_known = f[0] + f[3] + f[4] + f[5] + f[6] + f[15] + f[16] + f[17] + f[18];
            float f_outgoing = f[1] + f[7] + f[9] + f[11] + f[13];
            rho = (f_known + 2.0f * f_outgoing) / (1.0f + ux_wall);

            float rho_ux = rho * ux_wall;
            float rho_uy = rho * uy_wall;
            float rho_uz = rho * uz_wall;

            // Main direction
            f[2] = f[1] - (2.0f / 3.0f) * rho_ux;

            // Diagonal distributions
            f[8] = f[9] - (1.0f / 6.0f) * rho_ux + 0.5f * (f[4] - f[3]) + 0.5f * rho_uy;
            f[10] = f[7] - (1.0f / 6.0f) * rho_ux + 0.5f * (f[3] - f[4]) - 0.5f * rho_uy;
            f[12] = f[13] - (1.0f / 6.0f) * rho_ux + 0.5f * (f[6] - f[5]) + 0.5f * rho_uz;
            f[14] = f[11] - (1.0f / 6.0f) * rho_ux + 0.5f * (f[5] - f[6]) - 0.5f * rho_uz;
        }
    }
    else if (normal == 1) {  // Y boundaries
        if (sign == -1) {  // y_min boundary
            // Unknown: f3, f7, f8, f15, f17 (ey > 0, pointing into domain)
            // Outgoing: f4, f9, f10, f16, f18 (ey < 0, pointing out)

            float f_known = f[0] + f[1] + f[2] + f[5] + f[6] + f[11] + f[12] + f[13] + f[14];
            float f_outgoing = f[4] + f[9] + f[10] + f[16] + f[18];
            rho = (f_known + 2.0f * f_outgoing) / (1.0f - uy_wall);

            float rho_ux = rho * ux_wall;
            float rho_uy = rho * uy_wall;
            float rho_uz = rho * uz_wall;

            f[3] = f[4] + (2.0f / 3.0f) * rho_uy;
            f[7] = f[10] + (1.0f / 6.0f) * rho_uy + 0.5f * (f[2] - f[1]) + 0.5f * rho_ux;
            f[8] = f[9] + (1.0f / 6.0f) * rho_uy + 0.5f * (f[1] - f[2]) - 0.5f * rho_ux;
            f[15] = f[18] + (1.0f / 6.0f) * rho_uy + 0.5f * (f[6] - f[5]) + 0.5f * rho_uz;
            f[17] = f[16] + (1.0f / 6.0f) * rho_uy + 0.5f * (f[5] - f[6]) - 0.5f * rho_uz;
        }
        else if (sign == 1) {  // y_max boundary (LID-DRIVEN CAVITY TOP WALL)
            // Unknown: f4, f9, f10, f16, f18 (ey < 0, pointing into domain)
            // Outgoing: f3, f7, f8, f15, f17 (ey > 0, pointing out)

            float f_known = f[0] + f[1] + f[2] + f[5] + f[6] + f[11] + f[12] + f[13] + f[14];
            float f_outgoing = f[3] + f[7] + f[8] + f[15] + f[17];
            rho = (f_known + 2.0f * f_outgoing) / (1.0f + uy_wall);

            float rho_ux = rho * ux_wall;
            float rho_uy = rho * uy_wall;
            float rho_uz = rho * uz_wall;

            f[4] = f[3] - (2.0f / 3.0f) * rho_uy;
            f[9] = f[8] - (1.0f / 6.0f) * rho_uy + 0.5f * (f[2] - f[1]) + 0.5f * rho_ux;
            f[10] = f[7] - (1.0f / 6.0f) * rho_uy + 0.5f * (f[1] - f[2]) - 0.5f * rho_ux;
            f[16] = f[17] - (1.0f / 6.0f) * rho_uy + 0.5f * (f[6] - f[5]) + 0.5f * rho_uz;
            f[18] = f[15] - (1.0f / 6.0f) * rho_uy + 0.5f * (f[5] - f[6]) - 0.5f * rho_uz;
        }
    }
    else if (normal == 2) {  // Z boundaries
        if (sign == -1) {  // z_min boundary
            // Unknown: f5, f11, f12, f15, f16 (ez > 0, pointing into domain)
            // Outgoing: f6, f13, f14, f17, f18 (ez < 0, pointing out)

            float f_known = f[0] + f[1] + f[2] + f[3] + f[4] + f[7] + f[8] + f[9] + f[10];
            float f_outgoing = f[6] + f[13] + f[14] + f[17] + f[18];
            rho = (f_known + 2.0f * f_outgoing) / (1.0f - uz_wall);

            float rho_ux = rho * ux_wall;
            float rho_uy = rho * uy_wall;
            float rho_uz = rho * uz_wall;

            f[5] = f[6] + (2.0f / 3.0f) * rho_uz;
            f[11] = f[14] + (1.0f / 6.0f) * rho_uz + 0.5f * (f[2] - f[1]) + 0.5f * rho_ux;
            f[12] = f[13] + (1.0f / 6.0f) * rho_uz + 0.5f * (f[1] - f[2]) - 0.5f * rho_ux;
            f[15] = f[18] + (1.0f / 6.0f) * rho_uz + 0.5f * (f[4] - f[3]) + 0.5f * rho_uy;
            f[16] = f[17] + (1.0f / 6.0f) * rho_uz + 0.5f * (f[3] - f[4]) - 0.5f * rho_uy;
        }
        else if (sign == 1) {  // z_max boundary
            // Unknown: f6, f13, f14, f17, f18 (ez < 0, pointing into domain)
            // Outgoing: f5, f11, f12, f15, f16 (ez > 0, pointing out)

            float f_known = f[0] + f[1] + f[2] + f[3] + f[4] + f[7] + f[8] + f[9] + f[10];
            float f_outgoing = f[5] + f[11] + f[12] + f[15] + f[16];
            rho = (f_known + 2.0f * f_outgoing) / (1.0f + uz_wall);

            float rho_ux = rho * ux_wall;
            float rho_uy = rho * uy_wall;
            float rho_uz = rho * uz_wall;

            f[6] = f[5] - (2.0f / 3.0f) * rho_uz;
            f[13] = f[12] - (1.0f / 6.0f) * rho_uz + 0.5f * (f[2] - f[1]) + 0.5f * rho_ux;
            f[14] = f[11] - (1.0f / 6.0f) * rho_uz + 0.5f * (f[1] - f[2]) - 0.5f * rho_ux;
            f[17] = f[16] - (1.0f / 6.0f) * rho_uz + 0.5f * (f[4] - f[3]) + 0.5f * rho_uy;
            f[18] = f[15] - (1.0f / 6.0f) * rho_uz + 0.5f * (f[3] - f[4]) - 0.5f * rho_uy;
        }
    }
}

__device__ void BoundaryConditions::pressureBoundaryZouHe(
    float* f, float p_boundary,
    int normal, int sign) {

    // Zou-He pressure boundary condition
    // Convert pressure to density (assuming cs^2 = 1/3)
    float rho_boundary = p_boundary * 3.0f;

    if (normal == 0 && sign == -1) {  // x_min boundary
        // Extrapolate velocity from interior
        float ux = 1.0f - (f[0] + f[3] + f[4] + f[5] + f[6] +
                           2.0f * (f[2] + f[8] + f[10] + f[12] + f[14])) / rho_boundary;

        // Set unknown distributions
        f[1] = f[2] + 2.0f * rho_boundary * ux / 3.0f;

        // Corner distributions (simplified)
        f[7] = f[8] + rho_boundary * ux / 6.0f;
        f[9] = f[10] + rho_boundary * ux / 6.0f;
        f[11] = f[12] + rho_boundary * ux / 6.0f;
        f[13] = f[14] + rho_boundary * ux / 6.0f;
    }
    // Similar for other boundaries...
}

__host__ __device__ bool BoundaryConditions::isIncomingDirection(int q, unsigned int boundary_type) {
    // Check if direction q points into the domain from boundary

#ifdef __CUDA_ARCH__
    // Device code - use constant memory
    if (boundary_type & Streaming::BOUNDARY_X_MIN) {
        if (ex[q] > 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_X_MAX) {
        if (ex[q] < 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_Y_MIN) {
        if (ey[q] > 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_Y_MAX) {
        if (ey[q] < 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_Z_MIN) {
        if (ez[q] > 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_Z_MAX) {
        if (ez[q] < 0) return true;
    }
#else
    // Host code - use hardcoded arrays
    const int h_ex[19] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
    const int h_ey[19] = {0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1};
    const int h_ez[19] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1};

    if (boundary_type & Streaming::BOUNDARY_X_MIN) {
        if (h_ex[q] > 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_X_MAX) {
        if (h_ex[q] < 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_Y_MIN) {
        if (h_ey[q] > 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_Y_MAX) {
        if (h_ey[q] < 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_Z_MIN) {
        if (h_ez[q] > 0) return true;
    }
    if (boundary_type & Streaming::BOUNDARY_Z_MAX) {
        if (h_ez[q] < 0) return true;
    }
#endif

    return false;
}

__device__ float BoundaryConditions::movingWallBounceBack(
    const float* f, int q,
    float ux_wall, float uy_wall, float uz_wall) {

    // Moving wall bounce-back (Ladd's method)
    int q_opp = opposite[q];

    // Wall velocity contribution
    float eu_wall = ex[q] * ux_wall +
                   ey[q] * uy_wall +
                   ez[q] * uz_wall;

    // Apply moving wall correction
    return f[q_opp] + 2.0f * w[q] * 3.0f * eu_wall;
}

__global__ void applyBounceBackKernel(
    float* f,
    const BoundaryNode* boundary_nodes,
    int n_boundary,
    int nx, int ny, int nz) {

    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_boundary) return;

    BoundaryNode node = boundary_nodes[bid];
    if (node.type != BoundaryType::BOUNCE_BACK) return;

    int id = node.x + node.y * nx + node.z * nx * ny;
    int n_cells = nx * ny * nz;


    // Bounce-back boundary condition for no-slip walls
    //
    // Physical interpretation:
    // - Particles hitting the wall bounce back in the opposite direction
    // - This enforces zero velocity at the wall
    //
    // Implementation after streaming (post-streaming bounce-back):
    // - After streaming, incoming distributions have values from interior
    // - Outgoing distributions were not written during streaming (undefined)
    // - Set outgoing = incoming (bounce back the particles)
    //
    // CRITICAL FIX: Don't swap incoming/outgoing, just copy incoming→outgoing
    // Because outgoing values are undefined after streaming

    float f_temp[D3Q19::Q];

    // Step 1: Load all distributions into temporary storage
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        f_temp[q] = f[id + q * n_cells];
    }

    // Step 2: Apply bounce-back by copying incoming to outgoing
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        bool is_incoming = false;

        // Check if this direction points INTO the domain at this boundary
        if ((node.directions & Streaming::BOUNDARY_X_MIN) && ex[q] > 0) is_incoming = true;
        if ((node.directions & Streaming::BOUNDARY_X_MAX) && ex[q] < 0) is_incoming = true;
        if ((node.directions & Streaming::BOUNDARY_Y_MIN) && ey[q] > 0) is_incoming = true;
        if ((node.directions & Streaming::BOUNDARY_Y_MAX) && ey[q] < 0) is_incoming = true;
        if ((node.directions & Streaming::BOUNDARY_Z_MIN) && ez[q] > 0) is_incoming = true;
        if ((node.directions & Streaming::BOUNDARY_Z_MAX) && ez[q] < 0) is_incoming = true;

        // For each incoming direction, copy its value to the opposite outgoing direction
        if (is_incoming) {
            int q_out = opposite[q];  // q is incoming, q_out is outgoing
            // Bounce back: outgoing gets incoming value
            f[id + q_out * n_cells] = f_temp[q];
        }
    }
}

__global__ void applyVelocityBoundaryKernel(
    float* f,
    const float* rho,
    const BoundaryNode* boundary_nodes,
    int n_boundary,
    int nx, int ny, int nz) {

    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_boundary) return;

    BoundaryNode node = boundary_nodes[bid];
    if (node.type != BoundaryType::VELOCITY) return;

    int id = node.x + node.y * nx + node.z * nx * ny;
    int n_cells = nx * ny * nz;

    // Load distributions into local array
    float f_local[D3Q19::Q];
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        f_local[q] = f[id + q * n_cells];
    }

    // Determine boundary normal
    int normal = -1;
    int sign = 0;

    if (node.directions & Streaming::BOUNDARY_X_MIN) { normal = 0; sign = -1; }
    else if (node.directions & Streaming::BOUNDARY_X_MAX) { normal = 0; sign = 1; }
    else if (node.directions & Streaming::BOUNDARY_Y_MIN) { normal = 1; sign = -1; }
    else if (node.directions & Streaming::BOUNDARY_Y_MAX) { normal = 1; sign = 1; }
    else if (node.directions & Streaming::BOUNDARY_Z_MIN) { normal = 2; sign = -1; }
    else if (node.directions & Streaming::BOUNDARY_Z_MAX) { normal = 2; sign = 1; }

    if (normal >= 0) {
        BoundaryConditions::velocityBoundaryZouHe(
            f_local, rho[id], node.ux, node.uy, node.uz, normal, sign);

        // Write back
        #pragma unroll
        for (int q = 0; q < D3Q19::Q; ++q) {
            f[id + q * n_cells] = f_local[q];
        }
    }
}

__global__ void applyBoundaryConditionsKernel(
    float* f,
    const float* rho,
    const BoundaryNode* boundary_nodes,
    int n_boundary,
    int nx, int ny, int nz) {

    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_boundary) return;

    BoundaryNode node = boundary_nodes[bid];
    int id = node.x + node.y * nx + node.z * nx * ny;
    int n_cells = nx * ny * nz;

    // Load distributions
    float f_local[D3Q19::Q];
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        f_local[q] = f[id + q * n_cells];
    }

    // Apply appropriate boundary condition
    switch (node.type) {
        case BoundaryType::BOUNCE_BACK:
            {
                // Bounce-back for distributions pointing INTO the fluid (away from wall)
                // After PUSH streaming at a boundary node:
                // - Distributions pointing INTO the domain (toward interior) need to be set
                // - They will propagate back into the fluid on the next streaming step
                // - Source is the opposite direction (which came from interior, is valid)
                //
                // At Y_MAX boundary (top wall): distributions with ey < 0 point into fluid
                // At Y_MIN boundary (bot wall): distributions with ey > 0 point into fluid

                #pragma unroll
                for (int q = 0; q < D3Q19::Q; ++q) {
                    bool needs_bc = false;

                    // Check if direction q points INTO the fluid domain at this wall
                    // These are the distributions that need bounce-back treatment
                    if ((node.directions & Streaming::BOUNDARY_X_MIN) && ex[q] > 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_X_MAX) && ex[q] < 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_Y_MIN) && ey[q] > 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_Y_MAX) && ey[q] < 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_Z_MIN) && ez[q] > 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_Z_MAX) && ez[q] < 0) needs_bc = true;

                    if (needs_bc) {
                        // Set distribution pointing into fluid = opposite direction (came from interior)
                        f_local[q] = f[id + opposite[q] * n_cells];
                    }
                }
            }
            break;

        case BoundaryType::VELOCITY:
            {
                // Moving-wall bounce-back (Ladd's method)
                // For distributions pointing INTO the fluid, set:
                //   f[q] = f[opposite[q]] - 2 * w[q] * 3 * (c_q · u_wall)
                //
                // Note: The sign is NEGATIVE because c_q points INTO the fluid,
                // so (c_q · u_wall) has the opposite sign compared to when c_q
                // points toward the wall.

                #pragma unroll
                for (int q = 0; q < D3Q19::Q; ++q) {
                    bool needs_bc = false;

                    // Check if direction q points INTO the fluid domain at this wall
                    if ((node.directions & Streaming::BOUNDARY_X_MIN) && ex[q] > 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_X_MAX) && ex[q] < 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_Y_MIN) && ey[q] > 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_Y_MAX) && ey[q] < 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_Z_MIN) && ez[q] > 0) needs_bc = true;
                    if ((node.directions & Streaming::BOUNDARY_Z_MAX) && ez[q] < 0) needs_bc = true;

                    if (needs_bc) {
                        // Moving wall bounce-back (matches walberla SimpleUBB)
                        // f[q] = f[opposite[q]] - 6 * w[q] * rho * (c_opposite · u_wall)
                        // where c_opposite points TOWARD the wall (from fluid to wall)
                        int q_opp = opposite[q];
                        // c_opposite points toward wall, so (c_opposite · u_wall) gives correct sign
                        float eu_wall = ex[q_opp] * node.ux + ey[q_opp] * node.uy + ez[q_opp] * node.uz;
                        // ACCURACY FIX: Include local density in SimpleUBB formula
                        // This is important when using Guo forcing scheme
                        float rho_local = rho[id];
                        f_local[q] = f_local[q_opp] - 6.0f * w[q] * rho_local * eu_wall;
                    }
                }
            }
            break;

        case BoundaryType::PRESSURE:
            {
                int normal = -1, sign = 0;
                if (node.directions & Streaming::BOUNDARY_X_MIN) { normal = 0; sign = -1; }
                else if (node.directions & Streaming::BOUNDARY_X_MAX) { normal = 0; sign = 1; }
                else if (node.directions & Streaming::BOUNDARY_Y_MIN) { normal = 1; sign = -1; }
                else if (node.directions & Streaming::BOUNDARY_Y_MAX) { normal = 1; sign = 1; }
                else if (node.directions & Streaming::BOUNDARY_Z_MIN) { normal = 2; sign = -1; }
                else if (node.directions & Streaming::BOUNDARY_Z_MAX) { normal = 2; sign = 1; }

                if (normal >= 0) {
                    BoundaryConditions::pressureBoundaryZouHe(f_local, node.pressure, normal, sign);
                }
            }
            break;

        default:
            // No action for other types
            break;
    }

    // Write back
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        f[id + q * n_cells] = f_local[q];
    }
}

} // namespace core
} // namespace lbm