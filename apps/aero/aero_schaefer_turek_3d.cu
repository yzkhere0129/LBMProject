/**
 * @file aero_schaefer_turek_3d.cu
 * @brief Schäfer-Turek 2D-2 cylinder benchmark, run as 3D thin slab + z-periodic
 *
 * DFG benchmark (Schäfer & Turek 1996):
 *   Domain: 2.2 m × 0.41 m × Lz (z thin, z-periodic)
 *   Cylinder D=0.1 m at (0.2, 0.2)
 *   Inlet: parabolic u_x(y) = 4·U·y(H-y)/H², U=0.3 m/s (2D-1) or 1.5 m/s (2D-2)
 *   Outlet: constant pressure (rho=1)
 *   Walls: no-slip top/bottom (y), periodic z, cylinder no-slip
 *   Re = U_avg · D / ν = 100 (2D-2)
 *
 * Reference values (DFG, 2D-2):
 *   Cd_max ∈ [3.22, 3.24]
 *   Cl_max ∈ [0.99, 1.01]
 *   St     ∈ [0.295, 0.305]
 *
 * Quick configurable parameters via CLI:
 *   --resolution N   cells per cylinder diameter (default 20 for smoke,
 *                    50 for production)
 *   --re R           Reynolds number (default 100)
 *   --steps N        total steps (default scaled to ~10 shedding cycles)
 *   --case 2d-1|2d-2 (default 2d-2)
 *   --vtk-every N    VTK snapshot interval (default 0 = no VTK)
 *   --probe-every N  Cd/Cl sample interval (default 50)
 *   --output-dir D   output directory (default output_aero_schaefer_turek/)
 */

#include "physics/fluid_lbm.h"
#include "physics/aero/obstacle_geometry.h"
#include "physics/aero/momentum_exchange_force.h"
#include "io/vtk_writer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <filesystem>

using namespace lbm;
namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// CLI parsing (tiny, no dependencies)
// -----------------------------------------------------------------------------
struct Args {
    int   resolution = 20;
    float re         = 100.0f;
    int   steps      = -1;            // -1 = auto from re
    std::string preset = "2d-2";       // 2d-1 or 2d-2
    int   vtk_every  = 0;
    int   probe_every = 50;
    std::string output_dir = "output_aero_schaefer_turek";
    int   nz_thin    = 4;             // slab depth
    std::string bc   = "qbb-quad";    // stair / qbb / qbb-snode / qbb-quad / qbb-half
    std::string wall_bc = "halfway";  // fullway / halfway   (Y top/bottom walls)
    float ly_phys    = 0.41f;         // channel height [m]; default = ST spec
    float u_max_lu   = 0.05f;         // target u_max in lattice units (Mach control)
    float lx_phys    = 2.2f;          // domain length [m]; default = ST spec
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << s << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };
        if (s == "--resolution") a.resolution = std::stoi(next());
        else if (s == "--re")    a.re = std::stof(next());
        else if (s == "--steps") a.steps = std::stoi(next());
        else if (s == "--case")  a.preset = next();
        else if (s == "--vtk-every")   a.vtk_every = std::stoi(next());
        else if (s == "--probe-every") a.probe_every = std::stoi(next());
        else if (s == "--output-dir")  a.output_dir = next();
        else if (s == "--nz-thin")     a.nz_thin = std::stoi(next());
        else if (s == "--bc")          a.bc = next();
        else if (s == "--wall-bc")     a.wall_bc = next();
        else if (s == "--lx")          a.lx_phys = std::stof(next());
        else if (s == "--ly")          a.ly_phys = std::stof(next());
        else if (s == "--u-max-lu")    a.u_max_lu = std::stof(next());
        else if (s == "-h" || s == "--help") {
            std::cout <<
              "Usage: aero_schaefer_turek_3d [opts]\n"
              "  --resolution N   cells per cylinder diameter (default 20)\n"
              "  --re R           Reynolds number (default 100)\n"
              "  --steps N        total LBM steps (default auto from Re)\n"
              "  --case 2d-1|2d-2 inlet preset (default 2d-2)\n"
              "  --vtk-every N    VTK interval (default 0 = none)\n"
              "  --probe-every N  Cd/Cl sample interval (default 50)\n"
              "  --output-dir D   output directory\n"
              "  --nz-thin N      z-slab depth (default 4)\n"
              "  --bc stair|qbb|qbb-snode|qbb-quad|qbb-half   solid BC\n"
              "         stair    = halfway BB only (Phase 1 baseline)\n"
              "         qbb      = Bouzidi linear, O(dx) at wall (Phase 1b)\n"
              "         qbb-snode= single-node 2nd-order QBB (lbmpy formula)\n"
              "         qbb-quad = 3-cell quadratic Bouzidi (BFL 2001 Eq.12),\n"
              "                    O(dx²) at wall — strict-band path (default)\n"
              "         qbb-half = diagnostic (qfrac forced 0.5)\n"
              "  --lx X           domain length [m] (default 2.2 = ST spec; "
              "use 4.4+ to test outlet reflection)\n"
              "  --wall-bc fullway|halfway   Y top/bottom wall convention\n"
              "         halfway = wall at link midpoint (default, ST literature)\n"
              "         fullway = wall at cell centre (legacy, +5-7% Cd bias)\n";
            std::exit(0);
        }
        else {
            std::cerr << "Unknown arg: " << s << std::endl;
            std::exit(1);
        }
    }
    return a;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    // ---- Schäfer-Turek geometry (physical units) --------------------------
    const float Lx_phys = args.lx_phys;
    const float Ly_phys = args.ly_phys;
    const float D_phys  = 0.1f;       // cylinder diameter
    const float cx_phys = 0.2f;
    // ST spec puts cylinder slightly off mid-channel (cy=0.20 vs midline=0.205)
    // to trigger shedding. When Ly≠0.41 (e.g., wide-channel diagnostic), keep
    // a 5mm offset from mid-channel to retain asymmetry.
    const float cy_phys = 0.5f * Ly_phys - 0.005f;

    // Inlet velocity (m/s): 2D-1 uses U=0.3 (Re=20), 2D-2 uses U=1.5 (Re=100)
    // U here is centreline (u_max), not mean. u_avg = 2/3 * u_max for parabolic
    // profile (Schäfer-Turek convention).
    float U_max_phys = (args.preset == "2d-1") ? 0.3f : 1.5f;
    const float U_avg_phys = (2.0f / 3.0f) * U_max_phys;

    // Choose viscosity from desired Re. Re = U_avg * D / nu.
    const float nu_phys = U_avg_phys * D_phys / args.re;

    // ---- Mesh -------------------------------------------------------------
    const float dx = D_phys / args.resolution;
    const int nx = static_cast<int>(std::round(Lx_phys / dx));
    const int ny = static_cast<int>(std::round(Ly_phys / dx)) + 1; // +1 so walls sit at j=0 and j=ny-1
    const int nz = args.nz_thin;

    // ---- Time step: target u_max_LU configurable (default 0.05; Ma ≈ 0.087)
    const float u_max_lu_target = args.u_max_lu;
    const float dt = u_max_lu_target * dx / U_max_phys;

    // ---- Total steps: ~10 shedding cycles (T_shed = D / (St * U_avg)) ----
    int total_steps = args.steps;
    if (total_steps < 0) {
        const float St_guess = 0.30f;
        const float T_shed = D_phys / (St_guess * U_avg_phys);
        const float T_total = 12.0f * T_shed;   // 12 shedding cycles
        total_steps = static_cast<int>(std::ceil(T_total / dt));
    }

    fs::create_directories(args.output_dir);

    // ---- Banner -----------------------------------------------------------
    std::cout << "================================================================\n"
              << " Schäfer-Turek " << args.preset << " (3D thin slab, z-periodic)\n"
              << "================================================================\n"
              << " Domain (phys):  " << Lx_phys << " m × " << Ly_phys
              << " m × " << (nz * dx) << " m\n"
              << " Mesh:           " << nx << " × " << ny << " × " << nz
              << " (= " << (long long)nx * ny * nz << " cells)\n"
              << " dx:             " << dx << " m  (D/dx = " << args.resolution << ")\n"
              << " dt:             " << dt << " s  (u_max_LU = "
              << (U_max_phys * dt / dx) << ")\n"
              << " Cylinder:       D=" << D_phys << " m at ("
              << cx_phys << ", " << cy_phys << ")\n"
              << " U_max / U_avg:  " << U_max_phys << " / "
              << U_avg_phys << " m/s\n"
              << " nu:             " << nu_phys << " m²/s\n"
              << " Re:             " << args.re << "\n"
              << " Total steps:    " << total_steps << "\n"
              << " Probe every:    " << args.probe_every << "\n"
              << " VTK every:      " << args.vtk_every << "\n"
              << " Solid BC:       " << args.bc << "\n"
              << " Y wall BC:      " << args.wall_bc << "\n"
              << "================================================================\n";

    // ---- Build solver -----------------------------------------------------
    // boundary_x = WALL → face nodes created at x=0 and x=nx-1 (we then convert
    //    them to VELOCITY (parabolic inlet) and PRESSURE (outlet)).
    // boundary_y = WALL → face nodes at y=0/ny-1 (no-slip top/bottom).
    // boundary_z = PERIODIC → no z-face nodes; z-direction wraps in streaming.
    physics::FluidLBM fluid(nx, ny, nz, nu_phys, /*rho0*/ 1.0f,
                            physics::BoundaryType::WALL,
                            physics::BoundaryType::WALL,
                            physics::BoundaryType::PERIODIC,
                            dt, dx);
    // setTRT() prepares the omega_minus value, but we use collisionTRT() below
    // which takes lambda directly. Both paths hit the same stable TRT kernel.
    fluid.setTRT();

    // Init at rest with reference density
    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    // Inlet / outlet conversion
    fluid.setParabolicInletX(U_max_phys);
    fluid.setPressureOutletX(1.0f);

    // Y-wall convention: halfway BB (wall at link midpoint, ST literature
    // standard) drops Cd bias by ~5-7% vs the fullway BB (wall at cell centre)
    // we used by default. Use --wall-bc fullway to revert.
    if (args.wall_bc == "halfway") {
        const unsigned int y_face_mask = lbm::core::Streaming::BOUNDARY_Y_MIN
                                       | lbm::core::Streaming::BOUNDARY_Y_MAX;
        fluid.setHalfwayWallFaces(y_face_mask);
    } else if (args.wall_bc != "fullway") {
        std::cerr << "Unknown --wall-bc value: " << args.wall_bc << std::endl;
        return 1;
    }

    // ---- Stamp cylinder ---------------------------------------------------
    auto mask = physics::aero::makeFluidMask(nx, ny, nz);
    physics::aero::stampCylinderZ(mask, nx, ny, nz, dx,
                                  cx_phys, cy_phys, 0.5f * D_phys);
    fluid.setSolidMask(mask.data());

    if (args.bc == "qbb" || args.bc == "qbb-snode" || args.bc == "qbb-quad") {
        // Per-link q-fractions for any curved BC variant.
        auto qfrac = physics::aero::makeUnitQFraction(nx, ny, nz);
        physics::aero::computeCylinderZQ(qfrac, mask, nx, ny, nz, dx,
                                         cx_phys, cy_phys, 0.5f * D_phys);
        fluid.setObstacleQ(qfrac.data());

        if (args.bc == "qbb-snode") {
            const float omega_qbb = fluid.getOmega();
            fluid.setSingleNodeQBB(omega_qbb);
        } else if (args.bc == "qbb-quad") {
            fluid.setQuadBouzidi(true);
        }
    } else if (args.bc == "qbb-half") {
        // DIAGNOSTIC: force qfrac=0.5. With BFL kernel this matches halfway BB.
        auto qfrac = physics::aero::makeUnitQFraction(nx, ny, nz);
        for (auto& v : qfrac) v = 0.5f;
        fluid.setObstacleQ(qfrac.data());
    } else if (args.bc != "stair") {
        std::cerr << "Unknown --bc value: " << args.bc
                  << ". Use 'stair' / 'qbb' / 'qbb-snode' / 'qbb-half'."
                  << std::endl;
        return 1;
    }

    // ---- Open Cd/Cl CSV ---------------------------------------------------
    std::ofstream csv(args.output_dir + "/forces.csv");
    csv << "step,t,Fx_LU,Fy_LU,Fz_LU,Fx_phys_per_m,Fy_phys_per_m,Cd,Cl\n";
    csv.precision(8);

    // For 2D slab benchmark, the standard Schäfer-Turek normalization is
    //   Cd = 2 * F_x / (rho * U_avg² * D * L_z)
    // where L_z is the slab depth in physical units. We use rho_phys = 1 kg/m³
    // (matches lattice rho0=1 and standard non-dimensional reference).
    const float rho_phys = 1.0f;
    const float Lz_phys  = nz * dx;
    // Conversion of LU force to physical Newtons:
    //   F_phys = F_LU · rho_phys · dx⁴ / dt²
    const float force_lu_to_N = rho_phys * dx * dx * dx * dx / (dt * dt);
    // For 2D-equivalent comparison, divide by Lz to get force per unit depth
    // in N/m, then non-dim by 0.5 * rho * U_avg² * D.
    const float cd_denom_per_m = 0.5f * rho_phys * U_avg_phys * U_avg_phys * D_phys;

    // ---- Time loop --------------------------------------------------------
    auto t_start = std::chrono::steady_clock::now();
    int next_log = 1000;

    for (int step = 0; step < total_steps; ++step) {
        // 1. TRT collision (Λ=3/16). Schäfer-Turek tau ~ 0.52 is too low for
        //    plain BGK; TRT splits even/odd relaxation and removes the
        //    checkerboard mode without changing physical viscosity.
        fluid.collisionTRT(0.0f, 0.0f, 0.0f, 3.0f / 16.0f);

        // 2. Sample MEM force AFTER collision, BEFORE streaming. This captures
        //    the post-collision (pre-stream) outgoing populations f_q^pre that
        //    are about to push into the solid. Halfway BB then bounces them.
        if (args.probe_every > 0 && (step % args.probe_every == 0)) {
            float Fx_lu = 0, Fy_lu = 0, Fz_lu = 0;
            // f_src after collision is what's about to be streamed
            if (fluid.hasQuadBouzidi()) {
                physics::aero::computeObstacleForceLU_QuadBouzidi(
                    fluid.getDistributionSrc(), fluid.getSolidMask(),
                    fluid.getObstacleQ(),
                    nx, ny, nz, 0, 0, 1,
                    Fx_lu, Fy_lu, Fz_lu);
            } else if (fluid.getQBBOmega() > 0.0f) {
                physics::aero::computeObstacleForceLU_SingleNodeQBB(
                    fluid.getDistributionSrc(), fluid.getSolidMask(),
                    fluid.getObstacleQ(), fluid.getQBBOmega(),
                    nx, ny, nz, 0, 0, 1,
                    Fx_lu, Fy_lu, Fz_lu);
            } else {
                physics::aero::computeObstacleForceLU(
                    fluid.getDistributionSrc(), fluid.getSolidMask(),
                    fluid.getObstacleQ(),
                    nx, ny, nz, 0, 0, 1,
                    Fx_lu, Fy_lu, Fz_lu);
            }
            const float Fx_N = Fx_lu * force_lu_to_N;
            const float Fy_N = Fy_lu * force_lu_to_N;
            const float Fx_per_m = Fx_N / Lz_phys;
            const float Fy_per_m = Fy_N / Lz_phys;
            const float Cd = Fx_per_m / cd_denom_per_m;
            const float Cl = Fy_per_m / cd_denom_per_m;
            csv << step << "," << (step * dt) << ","
                << Fx_lu << "," << Fy_lu << "," << Fz_lu << ","
                << Fx_per_m << "," << Fy_per_m << ","
                << Cd << "," << Cl << "\n";
        }

        // 3. Streaming (uses the solid-aware kernel because mask is set).
        fluid.streaming();

        // 4. Apply face BCs (Ladd UBB at inlet, Zou-He pressure at outlet,
        //    bounce-back at top/bot walls).
        fluid.applyBoundaryConditions(1);

        // 5. Macroscopic update.
        fluid.computeMacroscopic();

        // ---- Logging / VTK ----------------------------------------------
        if (step + 1 == next_log) {
            csv.flush();
            auto t_now = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t_now - t_start).count();
            double mlups = (double)nx * ny * nz * (step + 1) / 1e6 / secs;
            std::cout << "[" << (step + 1) << "/" << total_steps << "] t="
                      << ((step + 1) * dt) << " s,  wall=" << secs
                      << " s,  " << mlups << " MLUPS" << std::endl;
            next_log = std::min(total_steps, next_log * 2);
        }
        if (args.vtk_every > 0 && ((step + 1) % args.vtk_every == 0)) {
            std::vector<float> ux(nx * ny * nz), uy(nx * ny * nz), uz(nx * ny * nz);
            fluid.copyVelocityToHost(ux.data(), uy.data(), uz.data());
            char buf[64];
            std::snprintf(buf, sizeof(buf), "/snap_%07d", step + 1);
            io::VTKWriter::writeVectorField(
                args.output_dir + buf,
                ux.data(), uy.data(), uz.data(),
                nx, ny, nz, dx, dx, dx, "velocity");
        }
    }

    csv.close();
    auto t_end = std::chrono::steady_clock::now();
    double tot_secs = std::chrono::duration<double>(t_end - t_start).count();
    double mlups_final = (double)nx * ny * nz * total_steps / 1e6 / tot_secs;
    std::cout << "Done. Wall time " << tot_secs << " s, "
              << mlups_final << " MLUPS." << std::endl;
    std::cout << "Forces written to " << args.output_dir << "/forces.csv\n";
    std::cout << "Run scripts/aero/strouhal_fft.py for Cl FFT and DFG check.\n";

    return 0;
}
