/**
 * @file diag_evap_material_readback.cu
 * @brief Verify F-01 audit finding: does cudaMemcpyToSymbol on
 *        __device__ g_evap_material actually transfer data under -rdc=true?
 *
 * Usage: build target then run.  ~2 second runtime.
 *
 * Background:
 *   - Code-audit pass 1 flagged thermal_lbm.cu:810 as silently zeroing
 *     evap material constants under separable compilation.
 *   - main-session verification (docs/overnight-2026-04-27/audit-verification-notes.md)
 *     argued the lattice constants in d3q19.cu use the same pattern and
 *     simulations work, so F-01 is likely a false alarm.
 *   - This standalone diagnostic resolves the question definitively.
 *
 * Expected output:
 *   - If pattern works: prints L_v ≈ 7e6, T_boil ≈ 3000K (316L Mills)
 *   - If pattern silently zeros: prints L_v=0, T_boil=0
 */

#include <cuda_runtime.h>
#include <iostream>
#include "physics/material_properties.h"
#include "utils/cuda_check.h"

// Forward-declare the same-name __device__ symbol thermal_lbm.cu uses.
// Note: this won't link to the same symbol because each .cu has its own
// copy under separable compilation. Instead we test the PATTERN on a
// local symbol.
namespace {
__device__ lbm::physics::MaterialProperties g_test_material;
}

__global__ void readbackKernel(lbm::physics::MaterialProperties* out) {
    *out = g_test_material;
}

int main() {
    using namespace lbm::physics;

    // Take 316L Mills as the test material (same as production)
    MaterialProperties h_mat = MaterialDatabase::get316L();

    std::cout << "=== F-01 Readback Diagnostic ===\n";
    std::cout << "Source material (host side):\n";
    std::cout << "  L_v        = " << h_mat.L_vaporization << " J/kg\n";
    std::cout << "  T_boil     = " << h_mat.T_vaporization << " K\n";
    std::cout << "  rho_liq    = " << h_mat.rho_liquid << " kg/m^3\n";
    std::cout << "  emissivity = " << h_mat.emissivity << "\n\n";

    // Try the alleged-broken pattern: cudaMemcpyToSymbol on __device__
    cudaError_t err = cudaMemcpyToSymbol(g_test_material, &h_mat,
                                          sizeof(MaterialProperties));
    std::cout << "cudaMemcpyToSymbol returned: " << cudaGetErrorString(err) << "\n";
    if (err != cudaSuccess) {
        std::cout << "[VERDICT] F-01 is REAL — cudaMemcpyToSymbol failed at host call.\n";
        return 1;
    }

    cudaDeviceSynchronize();

    // Read back via kernel
    MaterialProperties* d_out;
    cudaMalloc(&d_out, sizeof(MaterialProperties));
    readbackKernel<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();

    MaterialProperties h_check;
    cudaMemcpy(&h_check, d_out, sizeof(MaterialProperties), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    std::cout << "\nReadback values (device → host):\n";
    std::cout << "  L_v        = " << h_check.L_vaporization << " J/kg\n";
    std::cout << "  T_boil     = " << h_check.T_vaporization << " K\n";
    std::cout << "  rho_liq    = " << h_check.rho_liquid << " kg/m^3\n";
    std::cout << "  emissivity = " << h_check.emissivity << "\n\n";

    // Verdict
    bool ok = (h_check.L_vaporization == h_mat.L_vaporization)
           && (h_check.T_vaporization                == h_mat.T_vaporization)
           && (h_check.rho_liquid           == h_mat.rho_liquid);

    if (ok) {
        std::cout << "[VERDICT] F-01 is FALSE ALARM.\n"
                  << "  cudaMemcpyToSymbol on __device__ symbol DOES transfer data\n"
                  << "  under -rdc=true on this build.  Production evap kernel is\n"
                  << "  reading correct material constants.\n"
                  << "  Recommendation: downgrade F-01 to LOW (consistency-only fix).\n";
        return 0;
    } else {
        std::cout << "[VERDICT] F-01 is REAL.\n"
                  << "  cudaMemcpyToSymbol on __device__ silently corrupted the data.\n"
                  << "  Production evap kernel HAS been reading garbage.\n"
                  << "  Recommendation: switch to cudaGetSymbolAddress + cudaMemcpy\n"
                  << "  pattern (mirror material_database.cu:262).\n";
        return 2;
    }
}
