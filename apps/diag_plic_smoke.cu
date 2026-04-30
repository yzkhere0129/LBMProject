/**
 * @file diag_plic_smoke.cu
 * @brief PLIC vs legacy diagnostic — runs a minimal LPBF setup in both
 *        modes and prints a side-by-side comparison.
 *
 * This is a developer diagnostic, not a calibration benchmark. The grid
 * (32 × 32 × 24, dx=4 μm, dt=5 ns) is intentionally small so the binary
 * runs in seconds. Output goes to stdout; no VTK is dumped.
 *
 * Usage:
 *   ./diag_plic_smoke                  # run both legacy and PLIC, compare
 *   ./diag_plic_smoke --plic           # run only PLIC mode
 *   ./diag_plic_smoke --legacy         # run only legacy mode
 *   ./diag_plic_smoke --steps N        # override step count (default 200)
 *
 * Reports (per mode):
 *   - Initial / final / Δm/m₀ VOF mass
 *   - peak v_max and T_max over the run
 *   - completion time
 *   - whether NaN was hit (pass/fail)
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include "physics/vof_solver.h"

using namespace lbm::physics;

namespace {

struct Result {
    bool   ok           = true;
    float  m0           = 0.0f;
    float  m_final      = 0.0f;
    float  v_peak       = 0.0f;
    float  T_peak       = 0.0f;
    float  T_initial    = 0.0f;
    double wall_seconds = 0.0;
};

void configureCommon(MultiphysicsConfig& cfg) {
    cfg.nx = 32; cfg.ny = 32; cfg.nz = 24;
    cfg.dx = 4e-6f;
    cfg.dt = 5e-9f;
    cfg.material = MaterialDatabase::get316L();

    cfg.enable_thermal             = true;
    cfg.enable_thermal_advection   = true;
    cfg.enable_phase_change        = true;
    cfg.enable_fluid               = true;
    cfg.enable_vof                 = true;
    cfg.enable_vof_advection       = true;
    cfg.enable_surface_tension     = true;
    cfg.enable_marangoni           = true;
    cfg.enable_laser               = true;
    cfg.enable_recoil_pressure     = true;
    cfg.enable_evaporation_mass_loss = true;
    cfg.enable_darcy               = true;

    cfg.laser_power = 60.0f;
    cfg.laser_spot_radius = 25e-6f;
    cfg.laser_absorptivity = 0.35f;
    cfg.laser_penetration_depth = 5e-6f;
    cfg.laser_scan_vx = 0.4f;
    cfg.laser_start_x = -1.0f;
    cfg.laser_start_y = -1.0f;
    cfg.surface_tension_coeff = cfg.material.surface_tension;
    cfg.dsigma_dT             = cfg.material.dsigma_dT;
}

Result runOnce(const MultiphysicsConfig& cfg, bool use_plic, int n_steps) {
    Result r;
    MultiphysicsConfig active = cfg;

    if (use_plic) {
        active.enableFullPLICStack();
    }

    MultiphysicsSolver solver(active);

    if (use_plic) {
        if (auto* vof = solver.getVOFSolver()) {
            vof->setNormalReconstructionMethod(NormalReconstructionMethod::HEIGHT_FUNCTION);
            vof->setCurvatureMethod(CurvatureMethod::PLIC_DIVERGENCE);
        }
    }

    int N = active.nx * active.ny * active.nz;
    std::vector<float> fill(N);
    for (int k = 0; k < active.nz; ++k) {
        for (int j = 0; j < active.ny; ++j) {
            for (int i = 0; i < active.nx; ++i) {
                float z_surf = (active.nz - 6) + 0.05f * (i - active.nx * 0.5f);
                float v = z_surf - k;
                fill[i + active.nx * (j + active.ny * k)] =
                    std::max(0.0f, std::min(1.0f, 0.5f + 0.5f * v));
            }
        }
    }
    r.T_initial = 1500.0f;
    solver.initialize(r.T_initial, 0.5f);
    if (auto* vof = solver.getVOFSolver()) {
        vof->initialize(fill.data());
        r.m0 = vof->computeTotalMass();
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < n_steps; ++step) {
        solver.step();
        if ((step + 1) % 10 == 0) {
            if (solver.checkNaN()) { r.ok = false; break; }
            float v = solver.getMaxVelocity();
            float T = solver.getMaxTemperature();
            r.v_peak = std::max(r.v_peak, v);
            r.T_peak = std::max(r.T_peak, T);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    r.wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    if (auto* vof = solver.getVOFSolver()) {
        r.m_final = vof->computeTotalMass();
    }
    return r;
}

void printRow(const char* label, const Result& r) {
    if (!r.ok) {
        printf("  %-12s  FAILED (NaN detected)\n", label);
        return;
    }
    float dm_rel = std::fabs(r.m_final - r.m0) / r.m0;
    printf("  %-12s  m0=%.3f  m_final=%.3f  |Δm/m₀|=%.2e  "
           "v_peak=%.3e m/s  T_peak=%.1f K  wall=%.2f s\n",
           label, r.m0, r.m_final, dm_rel, r.v_peak, r.T_peak, r.wall_seconds);
}

}  // namespace

int main(int argc, char** argv) {
    bool run_plic   = true;
    bool run_legacy = true;
    int  n_steps    = 200;
    for (int a = 1; a < argc; ++a) {
        if (!std::strcmp(argv[a], "--plic"))      run_legacy = false;
        else if (!std::strcmp(argv[a], "--legacy")) run_plic = false;
        else if (!std::strcmp(argv[a], "--steps") && a + 1 < argc)
            n_steps = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "-h") || !std::strcmp(argv[a], "--help")) {
            printf("Usage: %s [--plic|--legacy] [--steps N]\n", argv[0]);
            return 0;
        }
    }

    MultiphysicsConfig cfg;
    configureCommon(cfg);

    printf("PLIC smoke diagnostic — %d steps, grid %d×%d×%d, dx=%.0f μm, dt=%.0f ns\n",
           n_steps, cfg.nx, cfg.ny, cfg.nz, cfg.dx * 1e6f, cfg.dt * 1e9f);
    printf("Laser: P=%.0f W, spot=%.0f μm\n", cfg.laser_power, cfg.laser_spot_radius * 1e6f);
    printf("Material: %s, T_init=1500 K\n\n", cfg.material.name);

    if (run_legacy) {
        printf("Running LEGACY mode (default kernels)...\n");
        Result r = runOnce(cfg, /*use_plic=*/false, n_steps);
        printRow("legacy", r);
    }
    if (run_plic) {
        printf("Running PLIC mode (Phase 2/3/4 sharp-delta paths)...\n");
        Result r = runOnce(cfg, /*use_plic=*/true, n_steps);
        printRow("plic", r);
    }

    return 0;
}
