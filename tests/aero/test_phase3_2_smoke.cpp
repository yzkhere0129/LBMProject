/**
 * @file test_phase3_2_smoke.cpp
 * @brief CPU-only smoke tests for Phase 3.2 code paths:
 *   - stampNacaAirfoil4Digit span-limit (k_z_lo / k_z_hi parameters)
 *   - makeNacaQFractionSparse correctness on span-limited mask
 *
 * No GPU needed — catches obvious bugs before transferring to a GPU machine.
 * Compiled as a small standalone executable.
 */

#include "physics/aero/obstacle_geometry.h"
#include "core/streaming.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

using namespace lbm::physics::aero;
using lbm::core::Streaming;

static int n_failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL " << msg << " at line " << __LINE__ << "\n"; ++n_failed; } \
    else        { std::cout << "  ok  " << msg << "\n"; } \
} while (0)

int main() {
    // Small CPU grid: 30 cells per chord, 5c × 3c × 2c domain
    const int   res  = 30;
    const float dx   = 1.0f / res;
    const float chord = 1.0f;
    const float lx_over_c = 5.0f, ly_over_c = 3.0f, lz_over_c = 2.0f;
    const int   nx = (int)(lx_over_c * res);
    const int   ny = (int)(ly_over_c * res) + 1;
    const int   nz = (int)(lz_over_c * res);
    const float xLE = 2.0f, yLE = 1.5f;
    const float alpha_rad = 8.0f * 3.14159265f / 180.0f;

    std::cout << "Grid: " << nx << "×" << ny << "×" << nz
              << " = " << (long long)nx*ny*nz << " cells\n";
    std::cout << "Wing in z fraction [0.25, 0.75] = z ∈ ["
              << nz/4 << ", " << 3*nz/4 << ")\n";

    // === Test 1: full-span stamp (no span limit) ===
    {
        auto mask = makeFluidMask(nx, ny, nz);
        stampNacaAirfoil4Digit(mask, nx, ny, nz, dx,
                               xLE, yLE, chord, 12.0f, alpha_rad);
        long long n_solid = countSolid(mask);
        std::cout << "\nFull span: " << n_solid << " solid cells\n";
        CHECK(n_solid > 0, "full stamp non-empty");
        // Verify solid distribution is z-uniform: count solid per z-slice
        long long s_z0 = 0, s_zhalf = 0, s_zlast = 0;
        const size_t plane = (size_t)nx * ny;
        for (size_t id = 0; id < plane; ++id) if (mask[id] == Streaming::CELL_SOLID) ++s_z0;
        for (size_t id = (nz/2)*plane; id < (nz/2)*plane + plane; ++id) if (mask[id] == Streaming::CELL_SOLID) ++s_zhalf;
        for (size_t id = (nz-1)*plane; id < (nz-1)*plane + plane; ++id) if (mask[id] == Streaming::CELL_SOLID) ++s_zlast;
        CHECK(s_z0 == s_zhalf && s_zhalf == s_zlast, "full stamp z-uniform (every z-slab equal)");
        std::cout << "  per-slice: z=0 has " << s_z0 << ", z=mid " << s_zhalf << ", z=last " << s_zlast << "\n";
    }

    // === Test 2: span-limited stamp (z 25-75%) ===
    {
        auto mask = makeFluidMask(nx, ny, nz);
        const int k_z_lo = nz / 4;
        const int k_z_hi = 3 * nz / 4;
        stampNacaAirfoil4Digit(mask, nx, ny, nz, dx,
                               xLE, yLE, chord, 12.0f, alpha_rad,
                               k_z_lo, k_z_hi);
        long long n_solid = countSolid(mask);
        std::cout << "\nSpan-limited [" << k_z_lo << ", " << k_z_hi << "): "
                  << n_solid << " solid cells\n";

        // Check: z<k_z_lo and z>=k_z_hi should be ALL fluid
        const size_t plane = (size_t)nx * ny;
        long long solid_outside = 0;
        for (int k = 0; k < nz; ++k) {
            if (k >= k_z_lo && k < k_z_hi) continue;
            for (size_t id = k*plane; id < (k+1)*plane; ++id) {
                if (mask[id] == Streaming::CELL_SOLID) ++solid_outside;
            }
        }
        CHECK(solid_outside == 0, "no solid cells outside span limit");

        // Check: z inside [k_z_lo, k_z_hi) has the same solid count per slice
        long long s_inside = 0;
        long long s_first  = 0;
        for (size_t id = k_z_lo*plane; id < (size_t)(k_z_lo+1)*plane; ++id) {
            if (mask[id] == Streaming::CELL_SOLID) ++s_first;
        }
        for (int k = k_z_lo; k < k_z_hi; ++k) {
            long long slc = 0;
            for (size_t id = k*plane; id < (size_t)(k+1)*plane; ++id) {
                if (mask[id] == Streaming::CELL_SOLID) ++slc;
            }
            s_inside += slc;
        }
        const int n_z_solid_layers = k_z_hi - k_z_lo;
        CHECK(s_inside == s_first * n_z_solid_layers,
              "stamped z-slabs identical to each other");
        CHECK((long long)s_first * n_z_solid_layers == n_solid,
              "total solid = per-slice × n_solid_z_slabs");
        std::cout << "  s_first_slab=" << s_first << "  n_solid_layers=" << n_z_solid_layers
                  << "  product=" << s_first * n_z_solid_layers << " (= n_solid)\n";
    }

    // === Test 3: qfrac on span-limited mask produces correct CSR ===
    {
        auto mask = makeFluidMask(nx, ny, nz);
        const int k_z_lo = nz / 4;
        const int k_z_hi = 3 * nz / 4;
        stampNacaAirfoil4Digit(mask, nx, ny, nz, dx,
                               xLE, yLE, chord, 12.0f, alpha_rad,
                               k_z_lo, k_z_hi);
        auto sq = makeNacaQFractionSparse(mask, nx, ny, nz, dx,
                                          xLE, yLE, chord, 12.0f, alpha_rad);
        std::cout << "\nQfrac on span-limited mask:\n";
        std::cout << "  offset.size = " << sq.offset.size() << " (expect " << (long long)nx*ny*nz + 1 << ")\n";
        std::cout << "  K (n_links)  = " << sq.link_q.size() << "\n";
        CHECK(sq.offset.size() == (size_t)(nx*ny*nz + 1), "offset has n_cells+1 entries");
        CHECK(sq.link_q.size() > 0, "qfrac has at least 1 wall-link");
        CHECK(sq.link_q.size() == sq.link_qfrac.size(), "link_q matches link_qfrac size");

        // Check qfrac values are in valid range
        long long bad_qf = 0;
        for (auto q : sq.link_qfrac) {
            if (q < 0.0f || q > 1.0f) ++bad_qf;
        }
        CHECK(bad_qf == 0, "all qfrac values in [0, 1]");
    }

    std::cout << "\n=== Phase 3.2 smoke result: ";
    if (n_failed == 0) {
        std::cout << "PASS ===\n";
        return 0;
    } else {
        std::cout << n_failed << " FAILED ===\n";
        return 1;
    }
}
