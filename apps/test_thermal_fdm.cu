/**
 * @file test_thermal_fdm.cu
 * @brief ThermalFDM 独立验证: 科目一(纯传导) + 科目二(强制对流耗散)
 *
 * 科目一: 纯传导 benchmark (u=0), 与 ThermalLBM 对标
 * 科目二: 恒定 vz=-50 m/s 对流高斯温度包, 评估迎风耗散
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include "physics/thermal_fdm.h"
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

static MaterialProperties createMat() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L_FDM_Test", sizeof(mat.name) - 1);
    mat.rho_solid=7900; mat.rho_liquid=7900;
    mat.cp_solid=700; mat.cp_liquid=700;
    mat.k_solid=20; mat.k_liquid=20;
    mat.mu_liquid=0.005f;
    mat.T_solidus=1650; mat.T_liquidus=1700; mat.T_vaporization=3500;
    mat.L_fusion=260000; mat.L_vaporization=7e6; mat.molar_mass=0.0558f;
    mat.surface_tension=1.75f; mat.dsigma_dT=+1e-4f;
    mat.absorptivity_solid=0.35f; mat.absorptivity_liquid=0.35f; mat.emissivity=0.3f;
    return mat;
}

// ============================================================================
// Gaussian heat source kernel (same as benchmark_conduction_316L)
// ============================================================================
__global__ void gaussianHeatSourceKernel(
    float* Q, float q0, float r0, float dx,
    int cx, int cy, int iz_top,
    int nx, int ny, int nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    int idx = ix + iy * nx + iz_top * nx * ny;
    float x = (ix - cx) * dx;
    float y = (iy - cy) * dx;
    float r2 = x*x + y*y;
    Q[idx] = q0 * expf(-2.0f * r2 / (r0*r0)) / dx;  // W/m^3
}

int main() {
    auto wall_t0 = std::chrono::high_resolution_clock::now();
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k=20;
    const float alpha = k/(rho*cp);
    const int NX=100, NY=100, NZ=50;
    const float dx=2e-6f;
    const int num_cells = NX*NY*NZ;
    const int cx=NX/2, cy=NY/2;

    // Laser params
    const float P=150, eta=0.35f, r0=25e-6f;
    const float q0 = 2*eta*P / (static_cast<float>(M_PI)*r0*r0);
    const float t_on = 50e-6f;

    // Use tau=1.0 for LBM (same as benchmark_conduction_316L)
    const float tau_lbm = 1.0f;
    const float alpha_LU = (tau_lbm - 0.5f) * 0.25f;
    const float dt = alpha_LU * dx * dx / alpha;  // 1.383e-7 s

    printf("============================================================\n");
    printf("  ThermalFDM Validation Tests\n");
    printf("============================================================\n");
    printf("dx=%.0f um, dt=%.2f ns, alpha=%.4e m^2/s\n\n", dx*1e6, dt*1e9, alpha);

    // ==================================================================
    // Prepare laser heat source (device)
    // ==================================================================
    float* d_Q;
    cudaMalloc(&d_Q, num_cells * sizeof(float));
    cudaMemset(d_Q, 0, num_cells * sizeof(float));
    {
        dim3 b(16,16), g((NX+15)/16,(NY+15)/16);
        gaussianHeatSourceKernel<<<g,b>>>(d_Q, q0, r0, dx, cx, cy, NZ-1, NX, NY, NZ);
        cudaDeviceSynchronize();
    }

    // ==================================================================
    // 科目一: 纯传导对标 (FDM vs LBM, u=0)
    // ==================================================================
    printf("========== TEST 1: Pure Conduction (FDM vs LBM) ==========\n\n");

    // --- LBM solver (reference) ---
    if (!D3Q7::isInitialized()) D3Q7::initializeDevice();
    ThermalLBM lbm(NX, NY, NZ, mat, alpha, true, dt, dx);
    lbm.initialize(300.0f);
    lbm.setSkipTemperatureCap(true);

    // --- FDM solver ---
    ThermalFDM fdm(NX, NY, NZ, mat, alpha, true, dt, dx);
    fdm.initialize(300.0f);
    fdm.setSkipTemperatureCap(true);

    // Snapshot times
    float snap_times[] = {25e-6f, 50e-6f, 75e-6f};
    const int n_snap = 3;
    int total_steps = static_cast<int>(100e-6f / dt + 0.5f);
    int laser_off = static_cast<int>(t_on / dt + 0.5f);

    printf("Total steps: %d, Laser off at step %d\n\n", total_steps, laser_off);

    // Output files
    FILE* f_out = fopen("test_fdm_conduction.csv", "w");
    fprintf(f_out, "snapshot,solver,z_idx,depth_um,T_K\n");

    std::vector<float> h_T_lbm(num_cells), h_T_fdm(num_cells);
    int snap_idx = 0;

    for (int step = 0; step <= total_steps; step++) {
        bool laser_on = (step < laser_off);

        // --- LBM step ---
        if (laser_on) lbm.addHeatSource(d_Q, dt);
        lbm.collisionBGK();
        lbm.streaming();
        lbm.computeTemperature();
        lbm.applyFaceThermalBC(4, 2, dt, dx, 300.0f);
        lbm.applyFaceThermalBC(0, 2, dt, dx, 300.0f);
        lbm.applyFaceThermalBC(1, 2, dt, dx, 300.0f);
        lbm.applyFaceThermalBC(2, 2, dt, dx, 300.0f);
        lbm.applyFaceThermalBC(3, 2, dt, dx, 300.0f);
        lbm.applyFaceThermalBC(5, 1, dt, dx);

        // --- FDM step ---
        if (laser_on) fdm.addHeatSource(d_Q, dt);
        fdm.applyFaceThermalBC(4, 2, dt, dx, 300.0f);
        fdm.applyFaceThermalBC(0, 2, dt, dx, 300.0f);
        fdm.applyFaceThermalBC(1, 2, dt, dx, 300.0f);
        fdm.applyFaceThermalBC(2, 2, dt, dx, 300.0f);
        fdm.applyFaceThermalBC(3, 2, dt, dx, 300.0f);
        fdm.collisionBGK();  // advDiff (u=nullptr → pure diffusion)
        fdm.streaming();      // buffer swap
        fdm.computeTemperature();  // ESM
        fdm.applyFaceThermalBC(4, 2, dt, dx, 300.0f);
        fdm.applyFaceThermalBC(0, 2, dt, dx, 300.0f);
        fdm.applyFaceThermalBC(1, 2, dt, dx, 300.0f);
        fdm.applyFaceThermalBC(2, 2, dt, dx, 300.0f);
        fdm.applyFaceThermalBC(3, 2, dt, dx, 300.0f);
        fdm.applyFaceThermalBC(5, 1, dt, dx);

        // Check snapshot
        if (snap_idx < n_snap) {
            int snap_step = static_cast<int>(snap_times[snap_idx] / dt + 0.5f);
            if (step == snap_step) {
                lbm.copyTemperatureToHost(h_T_lbm.data());
                fdm.copyTemperatureToHost(h_T_fdm.data());

                float T_max_lbm = *std::max_element(h_T_lbm.begin(), h_T_lbm.end());
                float T_max_fdm = *std::max_element(h_T_fdm.begin(), h_T_fdm.end());

                // Centerline profile comparison
                float max_err = 0, rms_err = 0;
                int n_compare = 0;
                for (int iz = 0; iz < NZ; iz++) {
                    int idx = cx + cy*NX + iz*NX*NY;
                    float T_l = h_T_lbm[idx], T_f = h_T_fdm[idx];
                    float depth = (NZ-1-iz) * dx * 1e6f;

                    fprintf(f_out, "%.0fus,LBM,%d,%.1f,%.4f\n",
                            snap_times[snap_idx]*1e6, iz, depth, T_l);
                    fprintf(f_out, "%.0fus,FDM,%d,%.1f,%.4f\n",
                            snap_times[snap_idx]*1e6, iz, depth, T_f);

                    float err = fabsf(T_l - T_f);
                    if (T_l > 310.0f) {  // only count heated region
                        if (err > max_err) max_err = err;
                        rms_err += err * err;
                        n_compare++;
                    }
                }
                rms_err = (n_compare > 0) ? sqrtf(rms_err / n_compare) : 0;

                printf("Snapshot t=%.0f us:\n", snap_times[snap_idx]*1e6);
                printf("  LBM T_max = %.1f K,  FDM T_max = %.1f K  (diff = %.1f K)\n",
                       T_max_lbm, T_max_fdm, T_max_fdm - T_max_lbm);
                printf("  Centerline: max_err = %.1f K, rms_err = %.1f K\n\n",
                       max_err, rms_err);
                snap_idx++;
            }
        }
    }
    fclose(f_out);
    printf("CSV: test_fdm_conduction.csv\n\n");

    // ==================================================================
    // 科目二: 强制对流耗散测试 (vz = -50 m/s, 高斯温度包)
    // ==================================================================
    printf("========== TEST 2: Forced Convection Smearing ==========\n\n");

    // Velocity for forced convection test
    const float vz_phys = -5.0f;  // m/s (realistic Marangoni-scale jet)
    const float vz_lu = vz_phys * dt / dx;

    // Reinitialize FDM
    ThermalFDM fdm2(NX, NY, NZ, mat, alpha, false, dt, dx);

    // Subcycling: combined stability 6·Fo + |Cz| ≤ 1 per sub-step
    float Cz_full = fabsf(vz_phys) * dt / dx;
    float combined = 6.0f * alpha * dt / (dx*dx) + Cz_full;
    int n_sub_conv = std::max(1, static_cast<int>(std::ceil(combined)));
    fdm2.setSubcycleCount(n_sub_conv);
    float dt_sub_c = dt / n_sub_conv;
    printf("vz = %.0f m/s (%.4f LU)\n", vz_phys, vz_lu);
    printf("Subcycles: %d (Fo_sub=%.3f, Cz_sub=%.3f, combined=%.3f)\n",
           n_sub_conv, alpha*dt_sub_c/(dx*dx), fabsf(vz_phys)*dt_sub_c/dx,
           6*alpha*dt_sub_c/(dx*dx) + fabsf(vz_phys)*dt_sub_c/dx);

    std::vector<float> h_T_init(num_cells, 300.0f);
    const float T_peak = 2000.0f;
    const float r0_gauss = 15.0f;  // cells
    const float z0_gauss = 10.0f;  // cells
    const int z_center = NZ/2;     // Gaussian center at mid-depth

    for (int iz=0; iz<NZ; iz++)
        for (int iy=0; iy<NY; iy++)
            for (int ix=0; ix<NX; ix++) {
                float rx = ix - cx, ry = iy - cy;
                float rr = sqrtf(rx*rx + ry*ry);
                float dz = iz - z_center;
                float T = 300.0f + (T_peak - 300.0f)
                          * expf(-rr*rr/(r0_gauss*r0_gauss))
                          * expf(-dz*dz/(z0_gauss*z0_gauss));
                h_T_init[ix + iy*NX + iz*NX*NY] = T;
            }

    fdm2.initialize(h_T_init.data());

    // vz_phys and vz_lu already defined above
    printf("Displacement in 10us: %.0f um (%.0f cells)\n\n",
           fabsf(vz_phys)*10e-6f*1e6f, fabsf(vz_phys)*10e-6f/dx);

    // Actually CFL for FDM is |v_phys|*dt_sub/dx. With v_conv = dx/dt:
    // The kernel receives vz_lu and multiplies by v_conv → v_phys. Then Co = dt_sub/dx.
    // |Cz| = |v_phys| * dt_sub / dx = |vz| * dt_sub / dx
    // For single step (n_sub=1): |Cz| = 50 * 1.383e-7 / 2e-6 = 3.46 → UNSTABLE!
    // FDM will auto-subcycle.

    // Allocate device velocity arrays (only vz is nonzero)
    float* d_vz;
    cudaMalloc(&d_vz, num_cells * sizeof(float));
    {
        std::vector<float> h_vz(num_cells, vz_lu);
        cudaMemcpy(d_vz, h_vz.data(), num_cells*sizeof(float), cudaMemcpyHostToDevice);
    }

    // Run 10 us of advection
    int steps_10us = static_cast<int>(10e-6f / dt + 0.5f);
    printf("Running %d steps (10 us) with forced convection...\n", steps_10us);

    // Output initial + final centerline
    FILE* f_conv = fopen("test_fdm_convection.csv", "w");
    fprintf(f_conv, "state,z_idx,z_um,T_K\n");

    // Initial state
    fdm2.copyTemperatureToHost(h_T_fdm.data());
    for (int iz = 0; iz < NZ; iz++) {
        int idx = cx + cy*NX + iz*NX*NY;
        fprintf(f_conv, "initial,%d,%.1f,%.4f\n", iz, iz*dx*1e6, h_T_fdm[idx]);
    }

    for (int step = 0; step < steps_10us; step++) {
        // Adiabatic BCs (default stencil clamping)
        fdm2.collisionBGK(nullptr, nullptr, d_vz);  // only vz active
        fdm2.streaming();
        fdm2.computeTemperature();
    }

    // Final state
    fdm2.copyTemperatureToHost(h_T_fdm.data());
    float T_max_final = *std::max_element(h_T_fdm.begin(), h_T_fdm.end());
    for (int iz = 0; iz < NZ; iz++) {
        int idx = cx + cy*NX + iz*NX*NY;
        fprintf(f_conv, "final,%d,%.1f,%.4f\n", iz, iz*dx*1e6, h_T_fdm[idx]);
    }

    // Analytical: the Gaussian should have advected downward by vz*t = 50*10e-6 = 500um
    // But domain is only 100um, so it wraps/hits boundary. The key metric is peak smearing.
    float analytical_shift = fabsf(vz_phys) * 10e-6f;
    printf("\nResults:\n");
    printf("  Analytical displacement: %.0f um (%.0f cells)\n",
           analytical_shift*1e6, analytical_shift/dx);
    printf("  Peak T (initial): %.1f K\n", T_peak);
    printf("  Peak T (final):   %.1f K\n", T_max_final);
    printf("  Peak reduction:   %.1f%%\n", (1.0f - (T_max_final-300)/(T_peak-300))*100);

    fclose(f_conv);
    printf("CSV: test_fdm_convection.csv\n");

    cudaFree(d_Q);
    cudaFree(d_vz);

    auto wall_t1 = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(wall_t1 - wall_t0).count();
    printf("\nTotal wall time: %.1f s\n", elapsed);

    return 0;
}
