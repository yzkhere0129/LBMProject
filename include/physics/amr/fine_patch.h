/**
 * @file fine_patch.h
 * @brief Static patch-based AMR: fine refinement region around an obstacle.
 *
 * Phase 1 (this header): defines the FinePatch data structure and basic
 * host-side configuration. Holds device PDF buffers (ping-pong), mask,
 * and sparse qfrac for the fine grid. Kernels live in
 *   src/physics/amr/fine_patch_collision.cu
 *   src/physics/amr/fine_patch_streaming.cu
 *
 * Phase 1 keeps the fine patch as an ISOLATED ISLAND (no coarse-fine
 * coupling). Coupling is added in Phase 2 (src/physics/amr/interface_coupling.cu).
 *
 * Refinement scheme: convective scaling (acoustic), 2× spatial refinement.
 *   dx_f = dx_c / 2
 *   dt_f = dt_c / 2
 *   nu_f = 2 * nu_c
 *   omega_f = 2 * omega_c / (4 - omega_c)      (Lagrava 2012 eq. 24)
 *
 * Reference: AMR_DESIGN_PROPOSAL.md §3-§4 in this worktree.
 */

#ifndef LBM_PHYSICS_AMR_FINE_PATCH_H
#define LBM_PHYSICS_AMR_FINE_PATCH_H

#include <cstddef>
#include <vector>
#include <stdexcept>

#include "utils/cuda_memory.h"

namespace lbm {
namespace physics {
namespace amr {

/**
 * @brief Axis-aligned bounding box on the COARSE grid (cell indices).
 *
 * Defines the region [i_lo, i_hi) × [j_lo, j_hi) × [k_lo, k_hi) in coarse
 * cell coordinates. The fine patch fills this region with 2× refinement.
 */
struct PatchExtentCoarse {
    int i_lo;
    int i_hi;
    int j_lo;
    int j_hi;
    int k_lo;
    int k_hi;

    int nx_coarse() const { return i_hi - i_lo; }
    int ny_coarse() const { return j_hi - j_lo; }
    int nz_coarse() const { return k_hi - k_lo; }
};

/**
 * @brief 2-level fine patch holding device-side LBM state.
 *
 * Memory layout: SoA D3Q27, same as the coarse driver
 *   (q-th population for cell id at offset id + q * n_cells_fine).
 *
 * Ownership: this object owns its CudaBuffer<float> members; they free
 * on destruction (move-only semantics through CudaBuffer's deleted copy).
 *
 * Phase 1 contract:
 *   - Patch runs in isolation (no inter-level transfer).
 *   - Initial state = freestream equilibrium at u_inf.
 *   - Optional NACA mask + qfrac stamped via stampNacaAirfoil4Digit at fine dx.
 *
 * Phase 2 additions (NOT in this header yet): coarse-snapshot buffer
 * for time interpolation; ghost-layer buffer for prolongation; flag
 * marking interface cells for BGK collision fallback.
 */
class FinePatch {
public:
    /**
     * @brief Construct an empty FinePatch ready for allocate().
     */
    FinePatch() = default;

    /**
     * @brief Allocate device memory for the fine patch.
     *
     * @param ext_coarse   AABB in coarse cell coordinates.
     * @param refine_factor  Must be 2 for Phase 1 (only 2× refinement supported).
     * @param dx_coarse    Coarse cell width [m] (for sanity prints).
     * @param omega_nu_coarse  Coarse shear relaxation rate (ω_ν on coarse).
     */
    void allocate(const PatchExtentCoarse& ext_coarse,
                  int refine_factor,
                  float dx_coarse,
                  float omega_nu_coarse) {
        if (refine_factor != 2) {
            throw std::runtime_error(
                "FinePatch: only refine_factor=2 supported in Phase 1");
        }
        ext_ = ext_coarse;
        refine_ = refine_factor;

        nx_f_ = ext_.nx_coarse() * refine_;
        ny_f_ = ext_.ny_coarse() * refine_;
        nz_f_ = ext_.nz_coarse() * refine_;
        n_cells_f_ = static_cast<std::size_t>(nx_f_) * ny_f_ * nz_f_;

        dx_f_ = dx_coarse / refine_;
        omega_nu_coarse_ = omega_nu_coarse;
        // Lagrava 2012 eq. 24: omega_f = 2 omega_c / (4 - omega_c)
        omega_nu_f_ = 2.0f * omega_nu_coarse / (4.0f - omega_nu_coarse);

        // PDF ping-pong (27 components SoA per cell)
        f_src_.reset(static_cast<int>(n_cells_f_ * 27));
        f_dst_.reset(static_cast<int>(n_cells_f_ * 27));

        // Solid mask (NACA airfoil stamped here at fine resolution)
        solid_mask_.reset(static_cast<int>(n_cells_f_));

        // Sparse qfrac (offsets + per-link q + per-link value)
        // qf_link_q_ / qf_link_val_ sized later via allocate_sparse_qfrac_links().
        // Phase 1 isolation test may skip qfrac entirely (set qf_link_q size 0).
        qf_offset_.reset(static_cast<int>(n_cells_f_ + 1));
    }

    // Accessors — for kernels and host-side build steps.
    int   nx() const { return nx_f_; }
    int   ny() const { return ny_f_; }
    int   nz() const { return nz_f_; }
    float dx() const { return dx_f_; }
    float omega_nu_coarse() const { return omega_nu_coarse_; }
    float omega_nu_fine()   const { return omega_nu_f_; }
    int   refine_factor()   const { return refine_; }
    const PatchExtentCoarse& extent_coarse() const { return ext_; }
    std::size_t n_cells() const { return n_cells_f_; }

    // Device pointers (raw, for kernel launches).
    float*         d_f_src()   { return f_src_.get(); }
    float*         d_f_dst()   { return f_dst_.get(); }
    unsigned char* d_solid()   { return solid_mask_.get(); }
    int*           d_qf_offset()  { return qf_offset_.get(); }
    unsigned char* d_qf_link_q()  { return qf_link_q_.get(); }
    float*         d_qf_link_val(){ return qf_link_val_.get(); }

    /**
     * @brief Swap f_src and f_dst (ping-pong after streaming).
     */
    void swap_pdf() { std::swap(f_src_, f_dst_); }

    /**
     * @brief Allocate sparse qfrac arrays once size is known (after build).
     */
    void allocate_sparse_qfrac_links(std::size_t n_links) {
        qf_link_q_.reset(static_cast<int>(n_links));
        qf_link_val_.reset(static_cast<int>(n_links));
    }

private:
    PatchExtentCoarse ext_{};
    int   refine_ = 2;
    int   nx_f_ = 0;
    int   ny_f_ = 0;
    int   nz_f_ = 0;
    std::size_t n_cells_f_ = 0;
    float dx_f_ = 0.0f;
    float omega_nu_coarse_ = 0.0f;
    float omega_nu_f_ = 0.0f;

    // Device storage (RAII; freed on destruction)
    utils::CudaBuffer<float>          f_src_;
    utils::CudaBuffer<float>          f_dst_;
    utils::CudaBuffer<unsigned char>  solid_mask_;
    utils::CudaBuffer<int>            qf_offset_;
    utils::CudaBuffer<unsigned char>  qf_link_q_;
    utils::CudaBuffer<float>          qf_link_val_;
};

// =====================================================================
// Kernel declarations (implementations in src/physics/amr/*.cu)
// =====================================================================

/**
 * @brief Cumulant collision kernel on the fine patch (verbatim clone of
 *        physics::cumulant::fluidCumulantCollisionKernel).
 *
 * Launch grid: dim3((nx+3)/4, (ny+3)/4, (nz+3)/4), dim3(4,4,4).
 * Phase 1: f_src == f_dst (in-place collision).
 */
__global__ void fineCumulantCollisionKernel(
    const float* f_src, float* f_dst,
    float* rho_out, float* ux_out, float* uy_out, float* uz_out,
    int nx, int ny, int nz,
    float omega_nu, float omega_b,
    float omega_3, float omega_4, float omega_5, float omega_6);

/**
 * @brief PULL streaming + single-node QBB on NACA wall (fine-patch variant).
 *
 * Phase 1 BC at patch edges: Y free-slip mirror, X bounce-back, Z periodic
 * (same as coarse kernel — replaced by ghost-layer prolongation in Phase 2).
 *
 * Launch grid: dim3((nx+3)/4, (ny+3)/4, (nz+3)/4), dim3(4,4,4).
 */
__global__ void fineStreamD3Q27_naca_qbb_sparse(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    const unsigned char* __restrict__ solid_mask,
    const int*           __restrict__ qf_offset,
    const unsigned char* __restrict__ qf_link_q,
    const float*         __restrict__ qf_link_val,
    int nx, int ny, int nz,
    float omega);

/**
 * @brief MEM force probe at NACA wall on the fine patch (verbatim clone
 *        of memForceNaca_QBB_sparse from the coarse driver).
 *
 * Writes into device-side double-precision accumulators (atomicAdd).
 */
__global__ void fineMemForceNaca_QBB_sparse(
    const float* __restrict__ f,
    const unsigned char* __restrict__ solid_mask,
    const int*           __restrict__ qf_offset,
    const unsigned char* __restrict__ qf_link_q,
    const float*         __restrict__ qf_link_val,
    float omega,
    int nx, int ny, int nz,
    double* Fx_acc, double* Fy_acc, double* Fz_acc);

} // namespace amr
} // namespace physics
} // namespace lbm

#endif // LBM_PHYSICS_AMR_FINE_PATCH_H
