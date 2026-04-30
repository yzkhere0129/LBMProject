/**
 * @file test_plic_full_stack.cu
 * @brief Phase 6 — end-to-end stability test for the full PLIC stack.
 *
 * Spins up MultiphysicsSolver with EVERY PLIC opt-in flag enabled
 * (Phase 1 reconstruction API + Phase 1.5 HEIGHT_FUNCTION normals + Phase
 * 2 laser column-march + Phase 3a HF curvature + Phase 3b/c/d sharp-delta
 * CSF/Marangoni/recoil + Phase 4a sharp-delta evaporation + Phase 5
 * MultiphysicsSolver dispatch) and runs the integration step loop on a
 * compact LPBF-like setup.
 *
 * This is a PROPERTY test, not a calibration test: we verify the full
 * PLIC pipeline executes without NaN, with bounded velocities, and
 * conserves mass within tolerance. Calibration to F3D / experiment is
 * out of scope for unit testing.
 *
 * Acceptance:
 *   - 50 steps complete without NaN
 *   - v_max remains bounded (< 100 m/s — sanity, not physical limit)
 *   - VOF mass conservation: |Δm/m₀| < 5% (sharp delta + non-conservative
 *     surface forces inevitably perturb the discrete VOF mass; 5% is the
 *     bound on a small grid and short run)
 *   - Total fluid energy stays finite
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "physics/multiphysics_solver.h"
#include "physics/vof_solver.h"

using namespace lbm::physics;

namespace {

void configurePLICStack(MultiphysicsConfig& cfg) {
    cfg.nx = 32; cfg.ny = 32; cfg.nz = 24;
    cfg.dx = 4e-6f;
    cfg.dt = 5e-9f;

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
    cfg.enable_buoyancy            = false;
    cfg.enable_solidification_shrinkage = false;

    // ---------------- Phase 2 / 3 / 4 PLIC opt-ins -----------------
    // (Preset method bundles all five flags + h_smooth_lu.)
    cfg.enableFullPLICStack(1.5f);

    // Reasonable LPBF-ish parameters (compact grid → small power).
    cfg.laser_power = 30.0f;
    cfg.laser_spot_radius = 25e-6f;
    cfg.laser_absorptivity = 0.35f;
    cfg.laser_penetration_depth = 5e-6f;
    cfg.laser_scan_vx = 0.4f;
    cfg.laser_start_x = -1.0f;     // auto-center
    cfg.laser_start_y = -1.0f;
}

}  // namespace

// ---------------------------------------------------------------------------
// Full PLIC stack: NaN-free + bounded run.
// ---------------------------------------------------------------------------
TEST(PLICFullStack, RunsWithoutNaNFor50Steps) {
    MultiphysicsConfig cfg;
    configurePLICStack(cfg);

    MultiphysicsSolver solver(cfg);

    // Caller is responsible for activating PLIC normal + curvature methods
    // on the underlying VOFSolver — MultiphysicsSolver does not auto-pick
    // them when the surface flags flip on.
    if (auto* vof = solver.getVOFSolver()) {
        vof->setNormalReconstructionMethod(NormalReconstructionMethod::HEIGHT_FUNCTION);
        vof->setCurvatureMethod(CurvatureMethod::PLIC_DIVERGENCE);
    }

    // Initialise a half-filled domain with a tilted top surface; gives the
    // PLIC plane some non-axis-aligned cells to exercise.
    int N = cfg.nx * cfg.ny * cfg.nz;
    std::vector<float> fill(N);
    for (int k = 0; k < cfg.nz; ++k) {
        for (int j = 0; j < cfg.ny; ++j) {
            for (int i = 0; i < cfg.nx; ++i) {
                // Surface at z = (cfg.nz - 6) + 0.05 * (i - cfg.nx/2)
                // gives a slight tilt in x direction.
                float z_surf = (cfg.nz - 6) + 0.05f * (i - cfg.nx * 0.5f);
                float v = z_surf - k;
                fill[i + cfg.nx * (j + cfg.ny * k)] =
                    std::max(0.0f, std::min(1.0f, 0.5f + 0.5f * v));
            }
        }
    }
    solver.initialize(/*T_init=*/1500.0f, /*fill=*/0.5f);   // T uniform
    if (auto* vof = solver.getVOFSolver()) {
        vof->initialize(fill.data());
    }

    float m0 = 0.0f;
    if (auto* vof = solver.getVOFSolver()) {
        m0 = vof->computeTotalMass();
    }
    ASSERT_GT(m0, 0.0f) << "Initial mass should be positive";

    const int n_steps = 50;
    for (int step = 0; step < n_steps; ++step) {
        solver.step();

        // NaN check every 10 steps
        if ((step + 1) % 10 == 0) {
            ASSERT_FALSE(solver.checkNaN()) << "NaN at step " << step + 1;
            float vmax = solver.getMaxVelocity();
            float Tmax = solver.getMaxTemperature();
            EXPECT_LT(vmax, 100.0f) << "v_max sanity bound at step " << step + 1
                                     << " (got " << vmax << " m/s)";
            EXPECT_LT(Tmax, 50000.0f) << "T_max sanity bound at step " << step + 1;
            EXPECT_GT(Tmax, 0.0f) << "T_max positivity at step " << step + 1;
        }
    }

    // Mass conservation (loose: 5% — see header for justification)
    float m_final = 0.0f;
    if (auto* vof = solver.getVOFSolver()) {
        m_final = vof->computeTotalMass();
    }
    float dm_rel = std::fabs(m_final - m0) / m0;
    printf("[PLIC FULL STACK] m0=%.4f, m_final=%.4f, |Δm|/m₀=%.3e, v_max(end)=%.3f m/s, T_max(end)=%.1f K\n",
           m0, m_final, dm_rel, solver.getMaxVelocity(), solver.getMaxTemperature());
    EXPECT_LT(dm_rel, 0.05f)
        << "Mass conservation: |Δm/m₀| < 5% over 50 steps";
}

// ---------------------------------------------------------------------------
// Long run: 200 steps gives the fluid time to respond to the laser deposit
// and confirms the PLIC stack is stable beyond initial transients.
// ---------------------------------------------------------------------------
TEST(PLICFullStack, LongRunStable) {
    MultiphysicsConfig cfg;
    configurePLICStack(cfg);
    cfg.laser_power = 60.0f;          // bump power so heating is visible

    MultiphysicsSolver solver(cfg);
    if (auto* vof = solver.getVOFSolver()) {
        vof->setNormalReconstructionMethod(NormalReconstructionMethod::HEIGHT_FUNCTION);
        vof->setCurvatureMethod(CurvatureMethod::PLIC_DIVERGENCE);
    }

    int N = cfg.nx * cfg.ny * cfg.nz;
    std::vector<float> fill(N);
    for (int k = 0; k < cfg.nz; ++k) {
        for (int j = 0; j < cfg.ny; ++j) {
            for (int i = 0; i < cfg.nx; ++i) {
                float z_surf = (cfg.nz - 6) + 0.05f * (i - cfg.nx * 0.5f);
                float v = z_surf - k;
                fill[i + cfg.nx * (j + cfg.ny * k)] =
                    std::max(0.0f, std::min(1.0f, 0.5f + 0.5f * v));
            }
        }
    }
    solver.initialize(1500.0f, 0.5f);
    if (auto* vof = solver.getVOFSolver()) {
        vof->initialize(fill.data());
    }

    float m0 = solver.getVOFSolver()->computeTotalMass();

    const int n_steps = 200;
    float v_peak = 0.0f, T_peak = 0.0f;
    for (int step = 0; step < n_steps; ++step) {
        solver.step();
        if ((step + 1) % 20 == 0) {
            ASSERT_FALSE(solver.checkNaN()) << "NaN at step " << step + 1;
            float v = solver.getMaxVelocity();
            float T = solver.getMaxTemperature();
            v_peak = std::max(v_peak, v);
            T_peak = std::max(T_peak, T);
        }
    }
    float m_final = solver.getVOFSolver()->computeTotalMass();
    float dm_rel = std::fabs(m_final - m0) / m0;
    printf("[PLIC LONG RUN] m0=%.4f, m_final=%.4f, |Δm/m₀|=%.3e, v_peak=%.3e m/s, T_peak=%.1f K\n",
           m0, m_final, dm_rel, v_peak, T_peak);

    EXPECT_LT(dm_rel, 0.10f) << "Mass conservation < 10% over 200 steps";
    EXPECT_LT(v_peak, 200.0f) << "v_peak sanity bound";
    EXPECT_LT(T_peak, 50000.0f) << "T_peak sanity bound";
    EXPECT_GT(T_peak, 1500.0f) << "Laser should have raised T above initial 1500 K";
}

// ---------------------------------------------------------------------------
// Default (legacy) path: same setup with all PLIC flags off — should
// run identically clean. Confirms the PLIC opt-ins don't break the
// default path through the same MultiphysicsSolver dispatch code.
// ---------------------------------------------------------------------------
TEST(PLICFullStack, LegacyPathStillStable) {
    MultiphysicsConfig cfg;
    configurePLICStack(cfg);
    cfg.laser.plic_aware_column_march    = false;
    cfg.surface.csf_use_plic_delta       = false;
    cfg.surface.marangoni_use_plic_delta = false;
    cfg.surface.recoil_use_plic_delta    = false;
    cfg.surface.evap_use_plic_delta      = false;

    MultiphysicsSolver solver(cfg);
    int N = cfg.nx * cfg.ny * cfg.nz;
    std::vector<float> fill(N);
    for (int k = 0; k < cfg.nz; ++k) {
        for (int j = 0; j < cfg.ny; ++j) {
            for (int i = 0; i < cfg.nx; ++i) {
                float z_surf = (cfg.nz - 6) + 0.05f * (i - cfg.nx * 0.5f);
                float v = z_surf - k;
                fill[i + cfg.nx * (j + cfg.ny * k)] =
                    std::max(0.0f, std::min(1.0f, 0.5f + 0.5f * v));
            }
        }
    }
    solver.initialize(1500.0f, 0.5f);
    if (auto* vof = solver.getVOFSolver()) {
        vof->initialize(fill.data());
    }

    for (int step = 0; step < 50; ++step) {
        solver.step();
        if ((step + 1) % 10 == 0) {
            ASSERT_FALSE(solver.checkNaN()) << "Legacy NaN at step " << step + 1;
        }
    }
}
