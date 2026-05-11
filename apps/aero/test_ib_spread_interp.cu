/**
 * @file test_ib_spread_interp.cu
 * @brief Phase B unit tests: IB spreading & interpolation correctness.
 *
 * Tests:
 *   T_B1  Partition of unity (host double, 1e-12 bar)
 *   T_B2  Interpolate uniform field u_x=1     (host double, 1e-12 bar)
 *   T_B3  Interpolate linear field u_x=2x+1   (host double, 1e-10 bar)
 *   T_B4  Spread total force conservation     (host double, 1e-12 bar)
 *   T_B5  GPU smoke: float kernels vs host double (1e-5 rel tolerance)
 *
 * T_B1-B4 prove the formula is implemented correctly (analytical-grade).
 * T_B5 proves the GPU port matches the formula at float precision.
 */

#include "physics/aero/ib_spread_interp.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(e) << " at " << __FILE__ << ":" \
              << __LINE__ << std::endl; std::exit(1); } } while (0)

using lbm::physics::aero::phi_rp3;
using lbm::physics::aero::interpolateVelocityKernel;
using lbm::physics::aero::spreadForceKernel;

// ===========================================================================
// Host double reference implementations (mirror the GPU kernel logic exactly,
// except in double precision and without atomic).
// ===========================================================================

namespace ref {

// Partition of unity: Σ_{27 cells} φ(rx) φ(ry) φ(rz). Uses an "infinite"
// scan of 5×5×5 to be safe (well covers 3-cell support); zero-φ cells drop.
double pou_at(double xL, double yL, double zL, double dx) {
    const int i_c = int(std::floor(xL / dx));
    const int j_c = int(std::floor(yL / dx));
    const int k_c = int(std::floor(zL / dx));
    double sum = 0.0;
    for (int di = -2; di <= 2; ++di)
    for (int dj = -2; dj <= 2; ++dj)
    for (int dk = -2; dk <= 2; ++dk) {
        const double xc = (i_c + di + 0.5) * dx;
        const double yc = (j_c + dj + 0.5) * dx;
        const double zc = (k_c + dk + 0.5) * dx;
        sum += phi_rp3<double>((xc - xL) / dx)
             * phi_rp3<double>((yc - yL) / dx)
             * phi_rp3<double>((zc - zL) / dx);
    }
    return sum;
}

// Generic interpolation. Field is callback (xc, yc, zc) → u value.
template <typename F>
double interp(F u_field, double xL, double yL, double zL,
              int nx, int ny, int nz, double dx) {
    const int i_c = int(std::floor(xL / dx));
    const int j_c = int(std::floor(yL / dx));
    const int k_c = int(std::floor(zL / dx));
    double sum = 0.0;
    for (int di = -1; di <= 1; ++di) {
        const int ii = i_c + di;
        if (ii < 0 || ii >= nx) continue;
        const double xc = (ii + 0.5) * dx;
        const double wx = phi_rp3<double>((xc - xL) / dx);
        if (wx == 0.0) continue;
        for (int dj = -1; dj <= 1; ++dj) {
            const int jj = j_c + dj;
            if (jj < 0 || jj >= ny) continue;
            const double yc = (jj + 0.5) * dx;
            const double wy = phi_rp3<double>((yc - yL) / dx);
            if (wy == 0.0) continue;
            for (int dkk = -1; dkk <= 1; ++dkk) {
                const int kk = k_c + dkk;
                if (kk < 0 || kk >= nz) continue;
                const double zc = (kk + 0.5) * dx;
                const double wz = phi_rp3<double>((zc - zL) / dx);
                if (wz == 0.0) continue;
                sum += u_field(xc, yc, zc) * wx * wy * wz;
            }
        }
    }
    return sum;
}

struct Field3D {
    int nx, ny, nz;
    double dx;
    std::vector<double> f;
    Field3D(int x, int y, int z, double d) : nx(x), ny(y), nz(z), dx(d),
                                             f(size_t(x) * y * z, 0.0) {}
    double& at(int i, int j, int k) {
        return f[size_t(i) + size_t(j) * nx + size_t(k) * size_t(nx) * ny];
    }
    double sum_volume() const {
        double s = 0;
        for (auto v : f) s += v * dx * dx * dx;
        return s;
    }
};

void spread_x(Field3D& fx, double xL, double yL, double zL,
              double ds, double FL_x) {
    const int i_c = int(std::floor(xL / fx.dx));
    const int j_c = int(std::floor(yL / fx.dx));
    const int k_c = int(std::floor(zL / fx.dx));
    const double ds_over_dx3 = ds / (fx.dx * fx.dx * fx.dx);
    for (int di = -1; di <= 1; ++di) {
        const int ii = i_c + di;
        if (ii < 0 || ii >= fx.nx) continue;
        const double xc = (ii + 0.5) * fx.dx;
        const double wx = phi_rp3<double>((xc - xL) / fx.dx);
        if (wx == 0.0) continue;
        for (int dj = -1; dj <= 1; ++dj) {
            const int jj = j_c + dj;
            if (jj < 0 || jj >= fx.ny) continue;
            const double yc = (jj + 0.5) * fx.dx;
            const double wy = phi_rp3<double>((yc - yL) / fx.dx);
            if (wy == 0.0) continue;
            for (int dkk = -1; dkk <= 1; ++dkk) {
                const int kk = k_c + dkk;
                if (kk < 0 || kk >= fx.nz) continue;
                const double zc = (kk + 0.5) * fx.dx;
                const double wz = phi_rp3<double>((zc - zL) / fx.dx);
                if (wz == 0.0) continue;
                fx.at(ii, jj, kk) += FL_x * wx * wy * wz * ds_over_dx3;
            }
        }
    }
}

} // namespace ref

// ===========================================================================
// Tests
// ===========================================================================

int main(int /*argc*/, char** /*argv*/) {
    int passed = 0, failed = 0;
    const double dx = 1.0;
    const int N = 16;  // grid extents for tests

    // ---------------- T_B1 — Partition of unity (host double) ----------------
    {
        std::cout << "\n=== T_B1. Partition of unity (host double, target < 1e-12) ===\n";
        const std::vector<std::array<double, 3>> positions = {
            {5.0, 5.0, 5.0},
            {5.3, 5.7, 5.5},
            {5.5, 5.5, 5.5},
            {5.99, 5.01, 5.5},
            {5.0000001, 5.0, 5.0},
        };
        double max_dev = 0.0;
        size_t worst = 0;
        for (size_t i = 0; i < positions.size(); ++i) {
            const auto& p = positions[i];
            const double pou = ref::pou_at(p[0], p[1], p[2], dx);
            const double dev = std::abs(pou - 1.0);
            std::cout << "  X = (" << p[0] << ", " << p[1] << ", " << p[2]
                      << ")  Σ φ³ = " << pou
                      << "  |dev| = " << dev << "\n";
            if (dev > max_dev) { max_dev = dev; worst = i; }
        }
        std::cout << "  max |dev| = " << max_dev
                  << "  (at position #" << worst << ")\n";
        const bool ok = (max_dev < 1e-12);
        std::cout << "  → " << (ok ? "PASS" : "FAIL") << "\n";
        if (ok) ++passed; else ++failed;
    }

    // ---------------- T_B2 — Interp uniform (host double) -------------------
    {
        std::cout << "\n=== T_B2. Interpolate uniform u_x=1 (host double, target < 1e-12) ===\n";
        auto u_uniform = [](double, double, double){ return 1.0; };
        const std::vector<std::array<double, 3>> positions = {
            {5.5, 5.5, 5.5},
            {5.37, 5.0, 5.0},
            {7.83, 9.21, 4.42},
        };
        double max_dev = 0.0;
        for (const auto& p : positions) {
            const double uL = ref::interp(u_uniform, p[0], p[1], p[2], N, N, N, dx);
            const double dev = std::abs(uL - 1.0);
            std::cout << "  X = (" << p[0] << ", " << p[1] << ", " << p[2]
                      << ")  u_L = " << uL
                      << "  |dev| = " << dev << "\n";
            max_dev = std::max(max_dev, dev);
        }
        std::cout << "  max |dev| = " << max_dev << "\n";
        const bool ok = (max_dev < 1e-12);
        std::cout << "  → " << (ok ? "PASS" : "FAIL") << "\n";
        if (ok) ++passed; else ++failed;
    }

    // ---------------- T_B3 — Interp linear field (host double) --------------
    {
        std::cout << "\n=== T_B3. Interpolate linear u_x=2x+1 (host double, target < 1e-10) ===\n";
        auto u_linear = [](double xc, double, double){ return 2.0 * xc + 1.0; };
        const double xL = 5.37;
        const double yL = 5.0, zL = 5.0;
        const double uL = ref::interp(u_linear, xL, yL, zL, N, N, N, dx);
        const double expected = 2.0 * xL + 1.0;
        const double dev = std::abs(uL - expected);
        std::cout << "  X = (" << xL << ", " << yL << ", " << zL << ")\n";
        std::cout << "  u_L = " << uL << "  expected = " << expected
                  << "  |dev| = " << dev << "\n";
        const bool ok = (dev < 1e-10);
        std::cout << "  → " << (ok ? "PASS" : "FAIL") << "\n";
        if (ok) ++passed; else ++failed;
    }

    // ---------------- T_B4 — Spread total-force conservation ---------------
    {
        std::cout << "\n=== T_B4. Spread conservation Σ f_x·dx³ = F·ds (host double, target < 1e-12) ===\n";
        ref::Field3D fx(N, N, N, dx);
        const double xL = 5.37, yL = 5.0, zL = 5.0;
        const double ds = 1.0;
        const double FL_x = 1.0;
        ref::spread_x(fx, xL, yL, zL, ds, FL_x);
        const double total = fx.sum_volume();
        const double expected = FL_x * ds;
        const double dev = std::abs(total - expected);
        std::cout << "  X = (" << xL << ", " << yL << ", " << zL << ") ds = " << ds << "\n";
        std::cout << "  Σ f_x·dx³ = " << total << "  expected = " << expected
                  << "  |dev| = " << dev << "\n";
        const bool ok = (dev < 1e-12);
        std::cout << "  → " << (ok ? "PASS" : "FAIL") << "\n";
        if (ok) ++passed; else ++failed;
    }

    // ---------------- T_B5 — GPU smoke: float kernels vs host double -------
    {
        std::cout << "\n=== T_B5. GPU smoke: float kernels vs host double (target rel < 1e-5) ===\n";
        const float dxf = 1.0f;
        const int Nx = 16, Ny = 16, Nz = 16;
        const size_t n_cells = size_t(Nx) * Ny * Nz;

        // ---- 5a) Interp uniform field u_x=1 ----
        std::vector<float> h_ux(n_cells, 1.0f), h_uy(n_cells, 0.0f), h_uz(n_cells, 0.0f);
        // Markers
        const std::vector<std::array<float, 3>> mpos = {
            {5.5f, 5.5f, 5.5f},
            {5.37f, 5.0f, 5.0f},
            {7.83f, 9.21f, 4.42f},
        };
        const int n_markers = int(mpos.size());
        std::vector<float> h_xL(n_markers), h_yL(n_markers), h_zL(n_markers),
                           h_ds(n_markers, 1.0f),
                           h_FLx(n_markers, 0.0f), h_FLy(n_markers, 0.0f), h_FLz(n_markers, 0.0f);
        for (int k = 0; k < n_markers; ++k) {
            h_xL[k] = mpos[k][0]; h_yL[k] = mpos[k][1]; h_zL[k] = mpos[k][2];
        }

        float *d_ux, *d_uy, *d_uz;
        float *d_xL, *d_yL, *d_zL, *d_ds, *d_FLx, *d_FLy, *d_FLz;
        float *d_uLx, *d_uLy, *d_uLz;
        float *d_Fx_field, *d_Fy_field, *d_Fz_field;

        CUDA_CHECK(cudaMalloc(&d_ux, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_uy, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_uz, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_xL, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_yL, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_zL, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ds, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_FLx, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_FLy, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_FLz, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_uLx, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_uLy, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_uLz, n_markers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Fx_field, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Fy_field, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Fz_field, n_cells * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_ux, h_ux.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_uy, h_uy.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_uz, h_uz.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xL, h_xL.data(), n_markers * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_yL, h_yL.data(), n_markers * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zL, h_zL.data(), n_markers * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ds, h_ds.data(), n_markers * sizeof(float), cudaMemcpyHostToDevice));

        const int block = 128;
        const int grid = (n_markers + block - 1) / block;
        interpolateVelocityKernel<<<grid, block>>>(
            d_ux, d_uy, d_uz, Nx, Ny, Nz, dxf,
            /*periodic_z=*/0,
            d_xL, d_yL, d_zL, n_markers,
            d_uLx, d_uLy, d_uLz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_uLx(n_markers), h_uLy(n_markers), h_uLz(n_markers);
        CUDA_CHECK(cudaMemcpy(h_uLx.data(), d_uLx, n_markers * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_uLy.data(), d_uLy, n_markers * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_uLz.data(), d_uLz, n_markers * sizeof(float), cudaMemcpyDeviceToHost));

        // Compare to host double interp.
        auto u_uniform = [](double, double, double){ return 1.0; };
        double max_rel = 0.0;
        std::cout << "  Subtest 5a: interp uniform u_x=1 (3 markers)\n";
        for (int k = 0; k < n_markers; ++k) {
            const double ref_v = ref::interp(u_uniform, h_xL[k], h_yL[k], h_zL[k],
                                             Nx, Ny, Nz, double(dxf));
            const double rel = std::abs(double(h_uLx[k]) - ref_v) / std::max(1e-30, std::abs(ref_v));
            std::cout << "    k=" << k << "  GPU=" << h_uLx[k] << "  host_d=" << ref_v
                      << "  rel=" << rel << "\n";
            max_rel = std::max(max_rel, rel);
        }
        const bool ok_5a = (max_rel < 1e-5);
        std::cout << "  Subtest 5a max rel = " << max_rel << "  → " << (ok_5a ? "PASS" : "FAIL") << "\n";

        // ---- 5b) Spread F=(1,0,0) ds=1 from single marker, sum field ----
        std::vector<float> h_FLx_one(n_markers, 0.0f);
        h_FLx_one[0] = 1.0f;  // only marker #0 carries force
        CUDA_CHECK(cudaMemcpy(d_FLx, h_FLx_one.data(), n_markers * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_FLy, h_FLy.data(), n_markers * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_FLz, h_FLz.data(), n_markers * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_Fx_field, 0, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_Fy_field, 0, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_Fz_field, 0, n_cells * sizeof(float)));

        // ds for marker #1, #2 doesn't matter since F=0.
        spreadForceKernel<<<grid, block>>>(
            d_xL, d_yL, d_zL, d_ds,
            d_FLx, d_FLy, d_FLz,
            n_markers, Nx, Ny, Nz, dxf,
            /*periodic_z=*/0,
            d_Fx_field, d_Fy_field, d_Fz_field);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_Fx_field(n_cells);
        CUDA_CHECK(cudaMemcpy(h_Fx_field.data(), d_Fx_field, n_cells * sizeof(float), cudaMemcpyDeviceToHost));

        double total_x = 0.0;
        for (auto v : h_Fx_field) total_x += double(v) * dxf * dxf * dxf;
        const double expected_x = 1.0;  // F_L_x · ds = 1·1
        const double rel_5b = std::abs(total_x - expected_x);
        std::cout << "  Subtest 5b: spread total Σf_x·dx³ = " << total_x
                  << " (expected " << expected_x << ", abs dev " << rel_5b << ")\n";
        const bool ok_5b = (rel_5b < 1e-5);
        std::cout << "  Subtest 5b → " << (ok_5b ? "PASS" : "FAIL") << "\n";

        const bool ok = ok_5a && ok_5b;
        std::cout << "  → " << (ok ? "PASS" : "FAIL") << "\n";
        if (ok) ++passed; else ++failed;

        // Cleanup
        cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
        cudaFree(d_xL); cudaFree(d_yL); cudaFree(d_zL); cudaFree(d_ds);
        cudaFree(d_FLx); cudaFree(d_FLy); cudaFree(d_FLz);
        cudaFree(d_uLx); cudaFree(d_uLy); cudaFree(d_uLz);
        cudaFree(d_Fx_field); cudaFree(d_Fy_field); cudaFree(d_Fz_field);
    }

    std::cout << "\n=========================================\n"
              << "Phase B test summary: " << passed << " PASS, " << failed << " FAIL\n";
    return (failed == 0) ? 0 : 1;
}
