/**
 * @file ib_markers.h
 * @brief Lagrangian markers for Immersed Boundary LBM (Wu-Shu 2009 style).
 *
 * Surface-only marker point clouds for IB-LBM. Replaces the stair-step solid
 * mask used by aero_naca0012_cumulant.cu. Markers store outward normals and
 * arc-length elements so spreading kernels can scale forces by ds_k.
 *
 * Conventions:
 *   - All positions in physical units [m] (same units as cell centres).
 *   - Outward normal points from solid into fluid. For α=0 NACA upper surface
 *     the normal is +y (rotated by α at output).
 *   - 2D generators ignore z; the extrudeZ() helper duplicates per z-layer
 *     so each layer is processed independently downstream (no z-interpolation).
 *
 * Symmetry guarantees:
 *   - At α=0, every NACA upper marker at (x, +y, n_x, +n_y) has a partner
 *     at (x, -y, n_x, -n_y) generated from the SAME parametric s. The pair
 *     differences should be at machine epsilon.
 *
 * References:
 *   Wu J. & Shu C. (2009). "Implicit velocity correction-based immersed
 *     boundary-lattice Boltzmann method." J. Comput. Phys. 228, 1963-1979.
 *   Roma A.M., Peskin C.S., Berger M.J. (1999). "An adaptive version of the
 *     immersed boundary method." J. Comput. Phys. 153, 509-534. (delta_h)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace lbm {
namespace physics {
namespace aero {

/**
 * @brief A single Lagrangian surface marker.
 *
 * Layout chosen for host-side AoS readability. GPU side is repacked into
 * SoA arrays by the marker upload helper (see make_ib_marker_soa()).
 */
struct IBMarker {
    float x, y, z;        ///< Position [m]
    float ds;             ///< Arc-length element [m] (for spread/force-reduce)
    float nx, ny, nz;     ///< Outward unit normal (z=0 for z-extruded thin-slab)
};

namespace detail {

/// NACA 4-digit symmetric thickness y_t(s)/c, s ∈ [0, 1], thick fraction t (e.g. 0.12).
inline float naca_yt(float s, float t) {
    if (s <= 0.0f) return 0.0f;
    if (s >= 1.0f) s = 1.0f;
    const float r = std::sqrt(s);
    return 5.0f * t * (0.2969f * r
                       - 0.1260f * s
                       - 0.3516f * s * s
                       + 0.2843f * s * s * s
                       - 0.1015f * s * s * s * s);
}

/// d y_t / d s, divergent at s=0 (sqrt term).
inline float naca_yt_deriv(float s, float t) {
    if (s <= 0.0f) s = 1e-8f;  // avoid 1/sqrt(0)
    const float inv_2r = 0.5f / std::sqrt(s);
    return 5.0f * t * (0.2969f * inv_2r
                       - 0.1260f
                       - 0.3516f * 2.0f * s
                       + 0.2843f * 3.0f * s * s
                       - 0.1015f * 4.0f * s * s * s);
}

/// Apply chord rotation by alpha around (xLE, yLE), translating from
/// chord-aligned (xc, yc) frame to world frame.
inline void rotate_about_LE(float xc, float yc, float alpha, float xLE, float yLE,
                            float& xw, float& yw) {
    const float ca = std::cos(alpha), sa = std::sin(alpha);
    xw = xLE + ca * xc - sa * yc;
    yw = yLE + sa * xc + ca * yc;
}

/// Rotate a 2D vector (vx, vy) by alpha (no translation, for normals).
inline void rotate_vec(float vx, float vy, float alpha,
                       float& vxw, float& vyw) {
    const float ca = std::cos(alpha), sa = std::sin(alpha);
    vxw = ca * vx - sa * vy;
    vyw = sa * vx + ca * vy;
}

} // namespace detail

/**
 * @brief Build NACA 4-digit symmetric airfoil markers in 2D (z=0 plane).
 *
 * Walks the upper surface from LE to TE at uniform arc-length ds_target,
 * adds explicit LE marker at (0,0) and TE markers along the finite TE
 * thickness, then mirrors upper markers to the lower surface for symmetry.
 *
 * @param chord     Chord length c [m].
 * @param thick_pct Thickness percent (12 for NACA0012).
 * @param alpha_rad Angle of attack [rad]. Positive = TE up.
 * @param xLE, yLE  Leading-edge position in world frame [m].
 * @param ds_target Target arc-length spacing [m] (≈ Δx for IB-LBM).
 *
 * @return Vector of markers, ordered: LE, upper s↑, TE chain, lower s↓.
 *
 * Symmetry property: at alpha_rad=0, upper markers at index k_up correspond
 * 1:1 to lower markers at index k_low = (n_upper + n_te + 1 + (n_upper - 1 - k_up))
 * with x equal and y opposite. See test_ib_markers for the assertion.
 */
inline std::vector<IBMarker> buildNacaMarkers2D(
    float chord, float thick_pct, float alpha_rad,
    float xLE, float yLE, float ds_target)
{
    const float t = thick_pct / 100.0f;

    // Step 1: high-resolution arc-length table of upper surface s ∈ [0, 1].
    constexpr int N_FINE = 4000;
    std::vector<float> s_fine(N_FINE + 1);
    std::vector<float> arclen(N_FINE + 1);
    arclen[0] = 0.0f;
    s_fine[0] = 0.0f;
    float x_prev = 0.0f, y_prev = 0.0f;
    for (int i = 1; i <= N_FINE; ++i) {
        const float s = float(i) / float(N_FINE);
        s_fine[i] = s;
        const float xc = s * chord;
        const float yc = detail::naca_yt(s, t) * chord;
        const float dx = xc - x_prev;
        const float dy = yc - y_prev;
        arclen[i] = arclen[i - 1] + std::sqrt(dx * dx + dy * dy);
        x_prev = xc; y_prev = yc;
    }
    const float L_upper = arclen[N_FINE];

    // Step 2: choose number of markers along upper surface.
    // n_upper = number of intervals; markers go at L = 0, ds, 2ds, ..., L_upper
    int n_upper = std::max(2, int(std::round(L_upper / ds_target)));
    const float ds_actual = L_upper / float(n_upper);

    std::vector<IBMarker> markers;
    markers.reserve(2 * (n_upper + 1) + 8);

    // ---- Upper surface markers (k=0 is LE, k=n_upper is TE-upper) ----
    std::vector<float> upper_x, upper_y;
    upper_x.reserve(n_upper + 1);
    upper_y.reserve(n_upper + 1);
    for (int k = 0; k <= n_upper; ++k) {
        const float L_target = k * ds_actual;
        // Find the fine bracket containing L_target.
        auto it = std::lower_bound(arclen.begin(), arclen.end(), L_target);
        int j = int(it - arclen.begin());
        if (j == 0) j = 1;
        if (j > N_FINE) j = N_FINE;
        const float L_lo = arclen[j - 1], L_hi = arclen[j];
        const float frac = (L_hi > L_lo) ? (L_target - L_lo) / (L_hi - L_lo) : 0.0f;
        const float s = s_fine[j - 1] + frac * (s_fine[j] - s_fine[j - 1]);

        const float xc = s * chord;
        const float yc = detail::naca_yt(s, t) * chord;

        // Outward normal for upper surface: rotate tangent +90° CCW.
        // Tangent ≈ (1, dy/ds) in chord units (s ∈ [0,1] is the chord coordinate
        // here). For LE (s≈0) tangent diverges; treat LE as horizontal-out normal.
        float nx, ny;
        if (k == 0) {
            // LE marker — outward normal points -x (upstream).
            nx = -1.0f; ny = 0.0f;
        } else if (k == n_upper) {
            // TE-upper marker: tangent essentially horizontal; outward = +y dominant.
            const float dy_ds = detail::naca_yt_deriv(s, t);
            float tx = 1.0f, ty = dy_ds;
            const float tm = std::sqrt(tx * tx + ty * ty);
            tx /= tm; ty /= tm;
            nx = -ty; ny = tx;
        } else {
            const float dy_ds = detail::naca_yt_deriv(s, t);
            float tx = 1.0f, ty = dy_ds;
            const float tm = std::sqrt(tx * tx + ty * ty);
            tx /= tm; ty /= tm;
            nx = -ty; ny = tx;     // rotate (tx, ty) by +90° CCW → (-ty, tx)
        }

        upper_x.push_back(xc);
        upper_y.push_back(yc);

        IBMarker m{};
        detail::rotate_about_LE(xc, yc, alpha_rad, xLE, yLE, m.x, m.y);
        m.z = 0.0f;
        m.ds = ds_actual;
        detail::rotate_vec(nx, ny, alpha_rad, m.nx, m.ny);
        m.nz = 0.0f;
        markers.push_back(m);
    }

    // ---- TE closure: walk from (1, +y_t(1)) down to (1, -y_t(1)) ----
    // Outward normal here is +x (downstream). Skip endpoints (already in
    // upper/lower lists); add only interior points if the gap is large enough.
    const float te_thick = 2.0f * detail::naca_yt(1.0f, t) * chord;
    const int n_te_interior = std::max(0, int(std::round(te_thick / ds_target)) - 1);
    const float te_y_top = +detail::naca_yt(1.0f, t) * chord;
    if (n_te_interior > 0) {
        const float ds_te = te_thick / float(n_te_interior + 1);
        for (int k = 1; k <= n_te_interior; ++k) {
            const float yc = te_y_top - k * ds_te;
            IBMarker m{};
            detail::rotate_about_LE(chord, yc, alpha_rad, xLE, yLE, m.x, m.y);
            m.z = 0.0f;
            m.ds = ds_te;
            detail::rotate_vec(1.0f, 0.0f, alpha_rad, m.nx, m.ny);
            m.nz = 0.0f;
            markers.push_back(m);
        }
    }

    // ---- Lower surface markers: mirror of upper, walked TE → LE ----
    // Include k=n_upper (lower TE marker, mirror of upper TE) so the airfoil
    // is geometrically closed. Skip k=0 (LE) — that marker is shared between
    // upper and lower and is already in the list.
    for (int k = n_upper; k >= 1; --k) {
        const float xc = upper_x[k];
        const float yc = -upper_y[k];   // mirror across chord

        // Lower surface outward normal: mirror of upper across y axis component.
        // upper had (nx_chord, ny_chord) with ny_chord > 0 for outward;
        // lower has (nx_chord, -ny_chord) — same x-component, flipped y.
        // Recompute from scratch to keep the two sides decoupled until we MIRROR.
        const float s_k = xc / chord;
        const float dy_ds = detail::naca_yt_deriv(s_k, t);
        float tx = 1.0f, ty = -dy_ds;          // tangent on lower (going LE→TE)
        const float tm = std::sqrt(tx * tx + ty * ty);
        tx /= tm; ty /= tm;
        // Outward = rotate tangent -90° CW (because we want -y component for lower).
        // Rotate (tx, ty) by -90° CW → (ty, -tx). For NACA upper-side
        // tangent (1, dy/ds) with dy/ds positive near LE we got nx=-ty, ny=tx
        // (positive y outward). Mirror: tangent (1, -dy/ds), outward = +ty,-tx.
        const float nx = ty;
        const float ny = -tx;

        IBMarker m{};
        detail::rotate_about_LE(xc, yc, alpha_rad, xLE, yLE, m.x, m.y);
        m.z = 0.0f;
        m.ds = ds_actual;
        detail::rotate_vec(nx, ny, alpha_rad, m.nx, m.ny);
        m.nz = 0.0f;
        markers.push_back(m);
    }

    return markers;
}

/**
 * @brief Build markers around a circle of given radius (z=0 plane).
 *
 * @param cx, cy     Centre in world frame [m].
 * @param radius     Circle radius [m].
 * @param ds_target  Target arc-length spacing [m].
 *
 * @return Vector of N ≈ round(2π R / ds) markers, evenly spaced in θ.
 */
inline std::vector<IBMarker> buildCircleMarkers2D(
    float cx, float cy, float radius, float ds_target)
{
    const float circumference = 2.0f * float(M_PI) * radius;
    int n = std::max(8, int(std::round(circumference / ds_target)));
    const float ds_actual = circumference / float(n);

    std::vector<IBMarker> markers;
    markers.reserve(n);
    for (int k = 0; k < n; ++k) {
        const float theta = (2.0f * float(M_PI) * float(k)) / float(n);
        const float ct = std::cos(theta), st = std::sin(theta);
        IBMarker m{};
        m.x = cx + radius * ct;
        m.y = cy + radius * st;
        m.z = 0.0f;
        m.ds = ds_actual;
        m.nx = ct;     // outward normal = radial unit vector
        m.ny = st;
        m.nz = 0.0f;
        markers.push_back(m);
    }
    return markers;
}

/**
 * @brief Replicate a 2D marker list across nz lattice layers (z-extrude).
 *
 * Each layer's marker has identical (x, y, ds, nx, ny) and z = (k+0.5)·dx,
 * matching the cell-centre convention used in obstacle_geometry.h.
 *
 * @param m2d  2D marker list (z must be 0; ignored).
 * @param nz   Number of z layers.
 * @param dx   Lattice spacing [m].
 *
 * @return Flat list of nz × m2d.size() markers.
 */
inline std::vector<IBMarker> extrudeZ(const std::vector<IBMarker>& m2d,
                                      int nz, float dx)
{
    std::vector<IBMarker> out;
    out.reserve(static_cast<size_t>(m2d.size()) * static_cast<size_t>(nz));
    for (int k = 0; k < nz; ++k) {
        const float zc = (float(k) + 0.5f) * dx;
        for (const auto& m : m2d) {
            IBMarker mm = m;
            mm.z = zc;
            out.push_back(mm);
        }
    }
    return out;
}

/**
 * @brief Pack AoS marker list into SoA buffers ready for cudaMemcpy.
 *
 * Caller owns the returned vectors. Pass their .data() to cudaMemcpy.
 */
struct IBMarkerSoA {
    std::vector<float> x, y, z, ds, nx, ny, nz;
    size_t size() const { return x.size(); }
};

inline IBMarkerSoA pack_soa(const std::vector<IBMarker>& m) {
    IBMarkerSoA s;
    const size_t n = m.size();
    s.x.resize(n);  s.y.resize(n);   s.z.resize(n);   s.ds.resize(n);
    s.nx.resize(n); s.ny.resize(n);  s.nz.resize(n);
    for (size_t i = 0; i < n; ++i) {
        s.x[i]=m[i].x;  s.y[i]=m[i].y;  s.z[i]=m[i].z;  s.ds[i]=m[i].ds;
        s.nx[i]=m[i].nx; s.ny[i]=m[i].ny; s.nz[i]=m[i].nz;
    }
    return s;
}

} // namespace aero
} // namespace physics
} // namespace lbm
