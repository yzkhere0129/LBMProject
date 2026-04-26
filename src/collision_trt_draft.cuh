/**
 * @file  collision_trt_draft.cuh
 * @brief REDUNDANT — TRT-EDM is already in production.
 *
 * Status: 2026-04-26 (Dawn-2 → discovered redundant during B integration).
 *
 * The Two-Relaxation-Time + EDM operator advertised in the original draft
 * already exists in the production codebase:
 *
 *   - Kernel       : `fluidTRTCollisionEDMKernel`
 *                    in src/physics/fluid/fluid_lbm.cu (line ~1533)
 *   - Host setter  : `FluidLBM::setTRT(float magic_parameter)`
 *                    in src/physics/fluid/fluid_lbm.cu (line ~2978)
 *   - Public API   : include/physics/fluid_lbm.h (line ~163)
 *   - Dispatch     : `FluidLBM::collisionBGKwithEDM` selects TRT branch
 *                    when omega_minus_ > 0 (fluid_lbm.cu:403-407)
 *   - Engaged by   : `MultiphysicsSolver` calls `setTRT(3/16)` at solver
 *                    construction (multiphysics_solver.cu:976) — every
 *                    production LPBF run is already TRT-EDM.
 *
 * What the production kernel includes that the draft missed:
 *   - Magic-parameter Λ recomputation when ω+ is adjusted by Smagorinsky LES
 *   - U_max clamp on u_shifted (Sprint-1 fix 2026-04-25)
 *   - LES-adjusted ω-effective with preserved Λ
 *
 * What was on the draft TODO that production STILL lacks:
 *   - An anchor test that TRT(ω-=ω+) ≡ BGK byte-equivalent (added separately
 *     at tests/validation/test_trt_degenerate_to_bgk.cu, 2026-04-26)
 *
 * Conclusion: this header is now documentation-only and never included.
 * Safe to delete in a future cleanup commit.
 */
#pragma once

#warning "src/collision_trt_draft.cuh is dead code — see header comment for production TRT location"
