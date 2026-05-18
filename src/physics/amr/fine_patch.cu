/**
 * @file fine_patch.cu
 * @brief Host-side utilities for FinePatch (compile-test for header).
 *
 * Phase 1 step 1: just verify the header compiles cleanly under nvcc.
 * Real implementations of fine-grid kernels live in:
 *   src/physics/amr/fine_patch_collision.cu   (Phase 1.2)
 *   src/physics/amr/fine_patch_streaming.cu   (Phase 1.3)
 */

#include "physics/amr/fine_patch.h"

namespace lbm {
namespace physics {
namespace amr {

// Compile-test: instantiate the type so any dependent compile error surfaces.
__host__ static void compile_test_only() {
    FinePatch patch;
    PatchExtentCoarse ext{0, 0, 0, 0, 0, 0};
    // Don't actually call allocate() (no CUDA context guarantee here).
    (void)patch.nx();
    (void)patch.dx();
    (void)patch.omega_nu_fine();
    (void)ext;
}

} // namespace amr
} // namespace physics
} // namespace lbm
