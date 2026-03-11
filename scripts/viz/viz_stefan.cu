/**
 * @file viz_stefan.cu
 * @brief Visualization program for 1D Stefan problem (melting front propagation)
 *
 * Runs ThermalLBM + PhaseChangeSolver on a 1D domain with Ti6Al4V material,
 * dumps temperature and liquid fraction profiles at multiple time snapshots
 * to CSV for plotting with matplotlib.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "physics/thermal_lbm.h"
#include "physics/lattice_d3q7.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// --- Neumann analytical solution helpers ---

static float computeStefanNumber(const MaterialProperties& mat) {
    float dT = mat.T_liquidus - mat.T_solidus;
    return mat.cp_solid * dT / mat.L_fusion;
}

static float computeLambda(float St) {
    const float sqrt_pi = sqrtf(M_PI);
    float target = St / sqrt_pi;
    float lam = 0.15f;
    for (int i = 0; i < 50; ++i) {
        float exp_term = expf(lam * lam);
        float erf_term = erf(lam);
        float f = lam * exp_term * erf_term - target;
        float df = exp_term * erf_term +
                   lam * exp_term * (2.0f * lam * erf_term + 2.0f / sqrt_pi);
        float lam_new = lam - f / df;
        if (fabsf(lam_new - lam) < 1e-8f) return lam_new;
        lam = lam_new;
    }
    return lam;
}

static float analyticalTemperature(float x, float t, float lam, float alpha,
                                    float T_liquidus, float T_solidus) {
    if (t <= 0.0f) return T_solidus;
    float eta = x / (2.0f * sqrtf(alpha * t));
    return T_liquidus - (T_liquidus - T_solidus) * erf(eta) / erf(lam);
}

static float analyticalFrontPosition(float t, float lam, float alpha) {
    return 2.0f * lam * sqrtf(alpha * t);
}

int main() {
    // --- Material and domain setup ---
    MaterialProperties material = MaterialDatabase::getTi6Al4V();

    const int NX = 400;
    const int NY = 1;
    const int NZ = 1;
    const float DOMAIN_LENGTH = 2000.0e-6f;  // 2 mm
    const float DX = DOMAIN_LENGTH / (NX - 1);

    float St = computeStefanNumber(material);
    float lam = computeLambda(St);
    float alpha = material.getThermalDiffusivity(material.T_solidus);

    float dt = 0.05f * DX * DX / alpha;

    printf("Stefan Problem Visualization\n");
    printf("  NX=%d, DX=%.2f um, dt=%.4f us\n", NX, DX * 1e6, dt * 1e6);
    printf("  St=%.4f, lambda=%.4f, alpha=%.3e m^2/s\n", St, lam, alpha);

    // --- Init lattice and solver ---
    if (!D3Q7::isInitialized()) D3Q7::initializeDevice();

    ThermalLBM solver(NX, NY, NZ, material, alpha, true, dt, DX);
    solver.initialize(material.T_solidus);

    // --- Snapshot times ---
    const int N_SNAPSHOTS = 5;
    float snap_times[N_SNAPSHOTS] = {0.2e-3f, 0.5e-3f, 1.0e-3f, 1.5e-3f, 2.0e-3f};
    int snap_steps[N_SNAPSHOTS];
    for (int s = 0; s < N_SNAPSHOTS; ++s)
        snap_steps[s] = static_cast<int>(snap_times[s] / dt);

    int total_steps = snap_steps[N_SNAPSHOTS - 1];

    // --- Host buffers ---
    std::vector<float> h_temp(NX);
    std::vector<float> h_fl(NX);

    // --- Output CSV ---
    std::ofstream csv("/home/yzk/LBMProject/scripts/viz/stefan_data.csv");
    csv << "x_um,T_K,fl,T_analytical,front_analytical_um,time_ms,snapshot\n";

    int snap_idx = 0;

    // --- Time integration ---
    for (int step = 1; step <= total_steps; ++step) {
        solver.collisionBGK();
        solver.streaming();
        solver.computeTemperature();

        // Apply melting BC: x=0 -> T_liquidus
        float* d_temp = solver.getTemperature();
        float T_liq = material.T_liquidus;
        cudaMemcpy(d_temp, &T_liq, sizeof(float), cudaMemcpyHostToDevice);

        // Check snapshot
        if (snap_idx < N_SNAPSHOTS && step == snap_steps[snap_idx]) {
            float t = snap_times[snap_idx];
            float front_analytical = analyticalFrontPosition(t, lam, alpha) * 1e6f;

            solver.copyTemperatureToHost(h_temp.data());
            solver.copyLiquidFractionToHost(h_fl.data());

            printf("  Snapshot %d: t=%.2f ms, analytical front=%.1f um\n",
                   snap_idx, t * 1e3, front_analytical);

            for (int i = 0; i < NX; ++i) {
                float x_um = i * DX * 1e6f;
                float T_ana = analyticalTemperature(i * DX, t, lam, alpha,
                                                     material.T_liquidus,
                                                     material.T_solidus);
                // Clamp analytical T to solidus for x beyond front
                float s_ana = analyticalFrontPosition(t, lam, alpha);
                if (i * DX > s_ana) T_ana = material.T_solidus;

                csv << x_um << ","
                    << h_temp[i] << ","
                    << h_fl[i] << ","
                    << T_ana << ","
                    << front_analytical << ","
                    << t * 1e3f << ","
                    << snap_idx << "\n";
            }

            snap_idx++;
        }

        if (step % 500 == 0) {
            printf("  step %d / %d\n", step, total_steps);
        }
    }

    csv.close();
    printf("Done. Data written to scripts/viz/stefan_data.csv\n");
    return 0;
}
