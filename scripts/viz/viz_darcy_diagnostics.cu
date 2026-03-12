/**
 * @file viz_darcy_diagnostics.cu
 * @brief Standalone diagnostic program for semi-implicit Darcy damping validation
 *
 * Generates data artifacts for visual verification:
 *   1. MushyZoneSharp:   4-cell sharp fl gradient (realistic LPBF sharpness)
 *   2. PressureImpact:   Flow driven PERPENDICULAR into solid wall
 *   3. CoefficientValues: K_darcy vs fl sweep (analytical Carman-Kozeny)
 *   4. ExtremeBraking:   C=1e15 time history
 *
 * Usage:  ./viz_darcy_diagnostics
 * Output: scripts/viz/darcy_*.csv, darcy_*.vtk
 */

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "physics/fluid_lbm.h"
#include "physics/force_accumulator.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

static const std::string OUT_DIR = "/home/yzk/LBMProject/scripts/viz/";

// Helper: write VTK structured points
static void writeVTK(const std::string& path, const std::string& title,
                     int NX, int NY,
                     const std::vector<std::pair<std::string, std::vector<float>>>& scalars,
                     const std::vector<float>* vx = nullptr,
                     const std::vector<float>* vy = nullptr) {
    std::ofstream vtk(path);
    vtk << "# vtk DataFile Version 3.0\n" << title << "\nASCII\n";
    vtk << "DATASET STRUCTURED_POINTS\n";
    vtk << "DIMENSIONS " << NX << " " << NY << " 1\n";
    vtk << "ORIGIN 0 0 0\nSPACING 1 1 1\n";
    vtk << "POINT_DATA " << NX * NY << "\n";
    for (auto& [name, data] : scalars) {
        vtk << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                vtk << std::scientific << std::setprecision(8)
                    << data[i + NX * j] << "\n";
    }
    if (vx && vy) {
        vtk << "VECTORS velocity float\n";
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * j;
                vtk << (*vx)[idx] << " " << (*vy)[idx] << " 0.0\n";
            }
    }
    vtk.close();
}

// ============================================================================
// Scenario 1: SHARP mushy zone (4-cell gradient, realistic LPBF)
// ============================================================================
void runSharpMushyZone() {
    std::cout << "\n=== Scenario 1: Sharp 4-Cell Mushy Zone ===" << std::endl;

    // Domain: liquid region → 4-cell mushy zone → solid region
    const int NX = 64, NY = 8, NZ = 1;
    const int NC = NX * NY * NZ;
    const float nu = 0.1f, rho = 1.0f, dx = 1.0f, dt = 1.0f;
    const float C_darcy = 1e7f;  // Same as LaserMeltingIron

    FluidLBM fluid(NX, NY, NZ, nu, rho,
                   BoundaryType::PERIODIC, BoundaryType::WALL,
                   BoundaryType::PERIODIC, dt, dx);
    fluid.initialize(rho, 0.05f, 0.0f, 0.0f);

    // Sharp fl: liquid (fl=1.0) for i<28, 4-cell transition, solid (fl=0.001) for i>=32
    const int i_start = 28;  // Mushy zone starts here
    const int i_end = 32;    // Mushy zone ends here (4 cells wide)
    std::vector<float> h_lf(NC);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            float fl;
            if (i < i_start) {
                fl = 1.0f;
            } else if (i >= i_end) {
                fl = 0.001f;
            } else {
                // Linear ramp over 4 cells: 1.0 → 0.001
                float t = (float)(i - i_start) / (float)(i_end - i_start);
                fl = 1.0f - t * (1.0f - 0.001f);
            }
            h_lf[i + NX * j] = fl;
        }

    float* d_lf;
    CUDA_CHECK(cudaMalloc(&d_lf, NC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lf, h_lf.data(), NC * sizeof(float), cudaMemcpyHostToDevice));

    ForceAccumulator forces(NX, NY, NZ);
    forces.computeDarcyCoefficientField(d_lf, C_darcy, rho, dx, dt);
    forces.reset();
    const float* darcy_K = forces.getDarcyCoefficient();

    // Run 500 steps
    const int STEPS = 500;
    for (int step = 0; step < STEPS; ++step) {
        fluid.collisionBGK(forces.getFx(), forces.getFy(), forces.getFz());
        fluid.streaming();
        fluid.computeMacroscopic(forces.getFx(), forces.getFy(),
                                  forces.getFz(), darcy_K);
    }

    // Extract fields
    std::vector<float> h_ux(NC), h_uy(NC), h_uz(NC), h_rho(NC);
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
    fluid.copyDensityToHost(h_rho.data());
    std::vector<float> h_K(NC);
    CUDA_CHECK(cudaMemcpy(h_K.data(), darcy_K, NC * sizeof(float), cudaMemcpyDeviceToHost));

    // Dump 1D line at j = NY/2
    int j_mid = NY / 2;
    std::string path = OUT_DIR + "darcy_sharp_mushy_line.csv";
    std::ofstream ofs(path);
    ofs << "i,fl,ux,uy,vmag,K_darcy,rho" << std::endl;

    bool monotonic = true;
    float prev_vmag = 1e10f;
    int violations = 0;

    for (int i = 0; i < NX; ++i) {
        int idx = i + NX * j_mid;
        float vmag = std::sqrt(h_ux[idx] * h_ux[idx] + h_uy[idx] * h_uy[idx]);
        ofs << i << ","
            << std::scientific << std::setprecision(8)
            << h_lf[idx] << "," << h_ux[idx] << "," << h_uy[idx] << ","
            << vmag << "," << h_K[idx] << "," << h_rho[idx] << std::endl;

        // Check monotonic decrease in mushy zone + solid region (i >= i_start-2)
        if (i >= i_start - 2 && i < NX - 1) {
            if (vmag > prev_vmag * 1.05f) {  // Allow 5% noise tolerance
                violations++;
                std::cout << "  WARNING: Non-monotonic at i=" << i
                          << " vmag=" << vmag << " prev=" << prev_vmag << std::endl;
            }
        }
        prev_vmag = vmag;
    }
    ofs.close();
    std::cout << "  Written: " << path << std::endl;

    // Print profile through mushy zone
    std::cout << "  Velocity profile through 4-cell mushy zone:" << std::endl;
    for (int i = i_start - 4; i < i_end + 8 && i < NX; ++i) {
        int idx = i + NX * j_mid;
        float vmag = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx]);
        std::cout << "    i=" << std::setw(3) << i
                  << "  fl=" << std::fixed << std::setprecision(4) << h_lf[idx]
                  << "  |u|=" << std::scientific << std::setprecision(6) << vmag;
        if (i >= i_start && i < i_end)
            std::cout << "  <-- MUSHY ZONE";
        std::cout << std::endl;
    }

    float v_liq = std::sqrt(h_ux[10 + NX*j_mid]*h_ux[10 + NX*j_mid] +
                            h_uy[10 + NX*j_mid]*h_uy[10 + NX*j_mid]);
    float v_sol = std::sqrt(h_ux[50 + NX*j_mid]*h_ux[50 + NX*j_mid] +
                            h_uy[50 + NX*j_mid]*h_uy[50 + NX*j_mid]);
    std::cout << "  v_liquid (i=10):  " << std::scientific << v_liq << std::endl;
    std::cout << "  v_solid (i=50):   " << std::scientific << v_sol << std::endl;
    std::cout << "  Monotonic violations: " << violations << std::endl;

    CUDA_CHECK(cudaFree(d_lf));
}

// ============================================================================
// Scenario 2: Body-force-driven flow PERPENDICULAR into Darcy solid
//
// Physics: constant body force F_x drives fluid in +x into a Darcy zone.
// Steady state has a Poiseuille-like profile in the liquid, with pressure
// rising at the liquid-solid interface.  This tests real pressure-velocity
// coupling under normal impact — NOT a parallel/unforced scenario.
//
// Parameters chosen so:
//   u_max ≈ F_body * L² / (8ν) ≈ 0.036  (safe Ma)
//   Δp ≈ F_body * L ≈ 2.4e-3  (easily visible)
// ============================================================================
void runPerpendicularImpact() {
    std::cout << "\n=== Scenario 2: Body-Force Driven Impact Into Solid ===" << std::endl;

    const int NX = 48, NY = 24, NZ = 1;
    const int NC = NX * NY * NZ;
    const float nu = 0.1f, rho = 1.0f, dx = 1.0f, dt = 1.0f;

    // Moderate Darcy — decelerates over a few cells, creating visible pressure
    const float C_darcy = 1e5f;
    const float fl_solid = 0.05f;  // Moderately porous, not totally rigid

    // Constant body force in +x direction (only in liquid zone)
    const float F_body = 1e-4f;

    // WALL on x (impermeable ends), WALL on y (no-slip channel), periodic z
    FluidLBM fluid(NX, NY, NZ, nu, rho,
                   BoundaryType::WALL, BoundaryType::WALL,
                   BoundaryType::PERIODIC, dt, dx);
    fluid.initialize(rho, 0.0f, 0.0f, 0.0f);  // Start from rest

    // Liquid left half, solid right half
    std::vector<float> h_lf(NC);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            h_lf[i + NX * j] = (i < NX / 2) ? 1.0f : fl_solid;

    float* d_lf;
    CUDA_CHECK(cudaMalloc(&d_lf, NC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lf, h_lf.data(), NC * sizeof(float), cudaMemcpyHostToDevice));

    // Darcy coefficient field
    ForceAccumulator forces(NX, NY, NZ);
    forces.computeDarcyCoefficientField(d_lf, C_darcy, rho, dx, dt);
    const float* darcy_K = forces.getDarcyCoefficient();

    // Body force arrays: F_x = F_body in liquid zone, 0 in solid zone
    // (In reality, body forces also act in solid, but we zero them to isolate
    //  the impact effect — any residual pressure buildup is purely from
    //  momentum flux impacting the Darcy interface.)
    std::vector<float> h_fx(NC, 0.0f);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            if (i > 0 && i < NX/2 && j > 0 && j < NY-1)  // Interior liquid only
                h_fx[i + NX * j] = F_body;

    float *d_fx, *d_fy, *d_fz;
    CUDA_CHECK(cudaMalloc(&d_fx, NC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fy, NC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fz, NC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fx, h_fx.data(), NC * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_fy, 0, NC * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fz, 0, NC * sizeof(float)));

    // Run to steady state (viscous timescale ~ L²/ν = 24²/0.1 = 5760 steps)
    const int STEPS = 6000;
    int j_mid = NY / 2;

    // Time history
    std::string hist_path = OUT_DIR + "darcy_impact_pressure_history.csv";
    std::ofstream hofs(hist_path);
    hofs << "step,p_min,p_max,p_range,p_at_interface,max_ux,has_nan" << std::endl;

    for (int step = 0; step <= STEPS; ++step) {
        if (step > 0) {
            fluid.collisionBGK(d_fx, d_fy, d_fz);
            fluid.streaming();
            fluid.computeMacroscopic(d_fx, d_fy, d_fz, darcy_K);
        }

        if (step % 200 == 0 || step == STEPS) {
            std::vector<float> h_p(NC), h_ux(NC), h_uy(NC), h_uz(NC);
            fluid.copyPressureToHost(h_p.data());
            fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

            float p_min = *std::min_element(h_p.begin(), h_p.end());
            float p_max = *std::max_element(h_p.begin(), h_p.end());
            float p_iface = h_p[(NX/2) + NX * j_mid];
            float max_ux = 0;
            bool has_nan = false;
            for (int idx = 0; idx < NC; ++idx) {
                if (std::isnan(h_ux[idx])) has_nan = true;
                max_ux = std::max(max_ux, std::abs(h_ux[idx]));
            }

            hofs << step << ","
                 << std::scientific << std::setprecision(8)
                 << p_min << "," << p_max << "," << (p_max - p_min) << ","
                 << p_iface << "," << max_ux << ","
                 << (has_nan ? 1 : 0) << std::endl;

            if (step % 1000 == 0) {
                std::cout << "  Step " << step << ": p_range=" << (p_max-p_min)
                          << "  max_ux=" << max_ux
                          << (has_nan ? " NaN!" : "") << std::endl;
            }
        }
    }
    hofs.close();
    std::cout << "  Written: " << hist_path << std::endl;

    // Extract final fields
    std::vector<float> h_p(NC), h_ux(NC), h_uy(NC), h_uz(NC), h_rho(NC);
    fluid.copyPressureToHost(h_p.data());
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
    fluid.copyDensityToHost(h_rho.data());

    // CSV
    std::string csv_path = OUT_DIR + "darcy_impact_fields.csv";
    std::ofstream cofs(csv_path);
    cofs << "i,j,fl,pressure,ux,uy,vmag" << std::endl;
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            int idx = i + NX * j;
            float vmag = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx]);
            cofs << i << "," << j << ","
                 << std::scientific << std::setprecision(8)
                 << h_lf[idx] << "," << h_p[idx] << ","
                 << h_ux[idx] << "," << h_uy[idx] << "," << vmag << std::endl;
        }
    cofs.close();
    std::cout << "  Written: " << csv_path << std::endl;

    // VTK
    std::vector<float> h_vmag(NC);
    for (int idx = 0; idx < NC; ++idx)
        h_vmag[idx] = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx]);

    writeVTK(OUT_DIR + "darcy_impact_steady.vtk",
             "Body-force driven impact into Darcy solid (steady state)",
             NX, NY,
             {{"pressure", h_p}, {"liquid_fraction", h_lf},
              {"velocity_magnitude", h_vmag}, {"density", h_rho}},
             &h_ux, &h_uy);
    std::cout << "  VTK: " << OUT_DIR << "darcy_impact_steady.vtk" << std::endl;

    // Print mid-line profile
    std::cout << "  Pressure & velocity profile at j=" << j_mid << ":" << std::endl;
    float p_at_wall = h_p[(NX/2 - 1) + NX * j_mid];
    float p_at_inlet = h_p[1 + NX * j_mid];
    float dp_total = p_at_wall - p_at_inlet;
    std::cout << "  Expected Δp ≈ F_body × L = " << F_body * (NX/2)
              << "  Actual Δp = " << dp_total << std::endl;

    for (int i = 0; i < NX; ++i) {
        int idx = i + NX * j_mid;
        if (i % 3 == 0 || (i >= NX/2 - 3 && i <= NX/2 + 3)) {
            std::cout << "    i=" << std::setw(3) << i
                      << "  fl=" << std::fixed << std::setprecision(2) << h_lf[idx]
                      << "  p=" << std::scientific << std::setprecision(6) << h_p[idx]
                      << "  ux=" << h_ux[idx] << std::endl;
        }
    }

    // Checkerboard metric
    float p_even_sum = 0, p_odd_sum = 0;
    int n_even = 0, n_odd = 0;
    for (int j = 2; j < NY - 2; ++j) {
        for (int i = NX/2 + 2; i < NX - 2; ++i) {
            int idx = i + NX * j;
            if ((i + j) % 2 == 0) { p_even_sum += h_p[idx]; n_even++; }
            else                   { p_odd_sum += h_p[idx]; n_odd++; }
        }
    }
    float checkerboard = std::abs(p_even_sum/n_even - p_odd_sum/n_odd);
    std::cout << "  Checkerboard metric: " << std::scientific << checkerboard << std::endl;

    CUDA_CHECK(cudaFree(d_lf));
    CUDA_CHECK(cudaFree(d_fx));
    CUDA_CHECK(cudaFree(d_fy));
    CUDA_CHECK(cudaFree(d_fz));
}

// ============================================================================
// Scenario 3: K_darcy vs fl scatter plot (analytical Carman-Kozeny)
// (unchanged from previous version)
// ============================================================================
void runCoefficientSweep() {
    std::cout << "\n=== Scenario 3: Darcy Coefficient Sweep ===" << std::endl;

    const int N_POINTS = 200;
    const float C = 1e6f, rho = 1.0f, dx = 1.0f, dt = 1.0f;
    const float eps = 1e-3f;

    std::vector<float> h_lf(N_POINTS);
    for (int i = 0; i < N_POINTS; ++i) {
        float t = (float)i / (float)(N_POINTS - 1);
        h_lf[i] = std::pow(10.0f, -3.0f + 3.0f * t);
    }

    float* d_lf;
    CUDA_CHECK(cudaMalloc(&d_lf, N_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lf, h_lf.data(), N_POINTS * sizeof(float), cudaMemcpyHostToDevice));

    ForceAccumulator forces(N_POINTS, 1, 1);
    forces.computeDarcyCoefficientField(d_lf, C, rho, dx, dt);

    std::vector<float> h_K(N_POINTS);
    CUDA_CHECK(cudaMemcpy(h_K.data(), forces.getDarcyCoefficient(),
                           N_POINTS * sizeof(float), cudaMemcpyDeviceToHost));

    std::string path = OUT_DIR + "darcy_coefficient_sweep.csv";
    std::ofstream ofs(path);
    ofs << "fl,K_numerical,K_analytical,relative_error" << std::endl;
    for (int i = 0; i < N_POINTS; ++i) {
        float fl = h_lf[i];
        float one_minus_fl = 1.0f - fl;
        float K_analytical = C * one_minus_fl * one_minus_fl / (fl * fl * fl + eps) * rho * dt;
        float rel_err = (K_analytical > 0) ?
            std::abs(h_K[i] - K_analytical) / K_analytical : 0.0f;
        ofs << std::scientific << std::setprecision(10)
            << fl << "," << h_K[i] << "," << K_analytical << "," << rel_err << std::endl;
    }
    ofs.close();
    std::cout << "  Written: " << path << std::endl;

    float max_err = 0.0f;
    for (int i = 0; i < N_POINTS; ++i) {
        float fl = h_lf[i];
        float one_minus_fl = 1.0f - fl;
        float K_analytical = C * one_minus_fl * one_minus_fl / (fl * fl * fl + eps) * rho * dt;
        if (K_analytical > 0)
            max_err = std::max(max_err, std::abs(h_K[i] - K_analytical) / K_analytical);
    }
    std::cout << "  Max relative error: " << std::scientific << max_err << std::endl;

    CUDA_CHECK(cudaFree(d_lf));
}

// ============================================================================
// Scenario 4: Extreme Darcy braking with time history
// (unchanged from previous version)
// ============================================================================
void runExtremeBraking() {
    std::cout << "\n=== Scenario 4: Extreme Darcy Braking Time History ===" << std::endl;

    const int NX = 64, NY = 16, NZ = 1;
    const int NC = NX * NY * NZ;
    const float nu = 0.1f, rho = 1.0f, dx = 1.0f, dt = 1.0f;
    const float C_darcy = 1e15f;
    const float fl_solid = 1e-5f;

    FluidLBM fluid(NX, NY, NZ, nu, rho,
                   BoundaryType::PERIODIC, BoundaryType::WALL,
                   BoundaryType::PERIODIC, dt, dx);
    fluid.initialize(rho, 0.05f, 0.0f, 0.0f);

    std::vector<float> h_lf(NC);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            h_lf[i + NX * j] = (i < NX / 2) ? 1.0f : fl_solid;

    float* d_lf;
    CUDA_CHECK(cudaMalloc(&d_lf, NC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lf, h_lf.data(), NC * sizeof(float), cudaMemcpyHostToDevice));

    ForceAccumulator forces(NX, NY, NZ);
    forces.computeDarcyCoefficientField(d_lf, C_darcy, rho, dx, dt);
    forces.reset();
    const float* darcy_K = forces.getDarcyCoefficient();

    std::string time_path = OUT_DIR + "darcy_extreme_braking_history.csv";
    std::ofstream tofs(time_path);
    tofs << "step,max_v_liquid,max_v_solid,max_v_transition,max_rho_dev,has_nan" << std::endl;

    const int STEPS = 300;
    int j_mid = NY / 2;

    for (int step = 0; step <= STEPS; ++step) {
        if (step > 0) {
            fluid.collisionBGK(forces.getFx(), forces.getFy(), forces.getFz());
            fluid.streaming();
            fluid.computeMacroscopic(forces.getFx(), forces.getFy(),
                                      forces.getFz(), darcy_K);
        }
        if (step % 10 == 0 || step <= 5) {
            std::vector<float> h_ux(NC), h_uy(NC), h_uz(NC), h_rho(NC);
            fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
            fluid.copyDensityToHost(h_rho.data());

            bool has_nan = false;
            float max_v_liquid = 0, max_v_solid = 0, max_v_trans = 0, max_rho_dev = 0;
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * j_mid;
                float v = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx]);
                if (std::isnan(v) || std::isinf(v)) has_nan = true;
                if (i < NX/2 - 2) max_v_liquid = std::max(max_v_liquid, v);
                else if (i >= NX/2 + 2) max_v_solid = std::max(max_v_solid, v);
                else max_v_trans = std::max(max_v_trans, v);
            }
            for (int idx = 0; idx < NC; ++idx)
                max_rho_dev = std::max(max_rho_dev, std::abs(h_rho[idx] - rho));

            tofs << step << ","
                 << std::scientific << std::setprecision(8)
                 << max_v_liquid << "," << max_v_solid << ","
                 << max_v_trans << "," << max_rho_dev << ","
                 << (has_nan ? 1 : 0) << std::endl;
        }
    }
    tofs.close();
    std::cout << "  Written: " << time_path << std::endl;

    std::vector<float> h_ux(NC), h_uy(NC), h_uz(NC), h_rho(NC);
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
    fluid.copyDensityToHost(h_rho.data());
    std::vector<float> h_K(NC);
    CUDA_CHECK(cudaMemcpy(h_K.data(), darcy_K, NC * sizeof(float), cudaMemcpyDeviceToHost));

    std::string line_path = OUT_DIR + "darcy_extreme_braking_line.csv";
    std::ofstream lofs(line_path);
    lofs << "i,fl,ux,vmag,K_darcy,rho" << std::endl;
    for (int i = 0; i < NX; ++i) {
        int idx = i + NX * j_mid;
        float vmag = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx]);
        lofs << i << ","
             << std::scientific << std::setprecision(8)
             << h_lf[idx] << "," << h_ux[idx] << "," << vmag << ","
             << h_K[idx] << "," << h_rho[idx] << std::endl;
    }
    lofs.close();
    std::cout << "  Written: " << line_path << std::endl;

    CUDA_CHECK(cudaFree(d_lf));
}

// ============================================================================
int main() {
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "=== Darcy Semi-Implicit Diagnostics (v2) ===" << std::endl;

    runSharpMushyZone();
    runPerpendicularImpact();
    runCoefficientSweep();
    runExtremeBraking();

    std::cout << "\n=== All diagnostics complete ===" << std::endl;
    return 0;
}
