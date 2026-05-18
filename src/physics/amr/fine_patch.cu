/**
 * @file fine_patch.cu
 * @brief Host-side implementation for FinePatch geometry build.
 *
 * Geometry build re-uses the coarse driver's host functions
 *   physics::aero::stampNacaAirfoil4Digit
 *   physics::aero::makeNacaQFractionSparse
 * with fine dx and a translated LE position so the airfoil sits at its
 * world coordinates inside the patch's local coordinate frame.
 */

#include "physics/amr/fine_patch.h"
#include "physics/aero/obstacle_geometry.h"

#include <cstring>
#include <stdexcept>

namespace lbm {
namespace physics {
namespace amr {

void FinePatch::buildNacaGeometry(float xLE_world,
                                  float yLE_world,
                                  float chord,
                                  float thick_pct,
                                  float alpha_rad,
                                  float dx_coarse)
{
    if (n_cells_f_ == 0) {
        throw std::runtime_error(
            "FinePatch::buildNacaGeometry: patch not allocated (call allocate() first)");
    }

    // Patch origin in world coordinates (lower corner of fine domain)
    const float patch_origin_x = ext_.i_lo * dx_coarse;
    const float patch_origin_y = ext_.j_lo * dx_coarse;
    const float patch_origin_z = ext_.k_lo * dx_coarse;

    // LE in fine-patch-local coords (so cell-center test at (i+0.5)·dx_f
    // matches world position).
    const float cx_LE_local = xLE_world - patch_origin_x;
    const float cy_LE_local = yLE_world - patch_origin_y;
    (void)patch_origin_z;  // z-extruded body, not used

    // Allocate host mask. CELL_FLUID = 0, CELL_SOLID = 1.
    std::vector<unsigned char> h_mask(n_cells_f_, 0);

    // Stamp NACA0012 (uses cell-center inside-test, same convention as coarse).
    physics::aero::stampNacaAirfoil4Digit(
        h_mask, nx_f_, ny_f_, nz_f_, dx_f_,
        cx_LE_local, cy_LE_local,
        chord, thick_pct, alpha_rad);

    // Build sparse qfrac (CSR-like). Returns SparseQFraction { offset[N+1], link_q[K], link_qfrac[K] }.
    auto sq = physics::aero::makeNacaQFractionSparse(
        h_mask, nx_f_, ny_f_, nz_f_, dx_f_,
        cx_LE_local, cy_LE_local,
        chord, thick_pct, alpha_rad);

    // Copy mask to device.
    CUDA_CHECK(cudaMemcpy(solid_mask_.get(), h_mask.data(),
                          n_cells_f_ * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    // Copy sparse-CSR arrays. offset has size n_cells_f_ + 1; link_q / link_qfrac
    // size = sq.offset.back() (= number of fractional links).
    const std::size_t n_links = sq.link_q.size();
    allocate_sparse_qfrac_links(n_links);
    CUDA_CHECK(cudaMemcpy(qf_offset_.get(), sq.offset.data(),
                          (n_cells_f_ + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (n_links > 0) {
        CUDA_CHECK(cudaMemcpy(qf_link_q_.get(), sq.link_q.data(),
                              n_links * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(qf_link_val_.get(), sq.link_qfrac.data(),
                              n_links * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Count solid cells for diagnostic.
    std::size_t n_solid = 0;
    for (auto v : h_mask) if (v) ++n_solid;

    // Print summary (driver wraps this if it wants quiet build).
    std::fprintf(stdout,
        " FinePatch geometry: %zu solid cells (%.4f%%), %zu sparse qfrac links\n",
        n_solid, 100.0 * (double)n_solid / (double)n_cells_f_, n_links);
}

} // namespace amr
} // namespace physics
} // namespace lbm
