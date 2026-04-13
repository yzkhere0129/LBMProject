/**
 * Round 2 Task 2 v2: Fixed energy conservation test
 *
 * Bug in v1: Q_boundary was computed at the Dirichlet boundary cell (kk=0)
 * which is pinned to T_bc → Q=0 always. Fix: use kk=1 for gradient.
 *
 * Also: compute Q at ALL 6 faces to verify adiabatic faces have zero flux.
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include "physics/thermal_fdm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

int main() {
    const int NX=20, NY=20, NZ=20;
    const int N = NX*NY*NZ;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const float rho=7900, cp=700, k_c=20;
    const float alpha = k_c/(rho*cp);
    const float T_bc = 300.0f;
    const float dV = dx*dx*dx;
    const float Q_density = 1e15f;
    const int hs = 5;
    const float P_source = Q_density * powf(hs*dx, 3);

    printf("================================================================\n");
    printf("  Task 2 v2: Energy Conservation (Fixed Q_boundary)\n");
    printf("  P_source = %.4f W, domain %d³, dx=%.1fμm, dt=%.0fns\n",
           P_source, NX, dx*1e6, dt*1e9);
    printf("================================================================\n");

    MaterialProperties mat = {};
    mat.rho_solid=rho; mat.rho_liquid=rho;
    mat.cp_solid=cp; mat.cp_liquid=cp;
    mat.k_solid=k_c; mat.k_liquid=k_c;
    mat.T_solidus=1650; mat.T_liquidus=1700;
    mat.T_vaporization=3200;

    ThermalFDM thermal(NX, NY, NZ, mat, alpha, false, dt, dx);

    std::vector<float> h_T(N, T_bc);
    thermal.initialize(h_T.data());

    std::vector<float> h_Q(N, 0.0f);
    int cx=NX/2, cy=NY/2, cz=NZ/2;
    for (int kk=cz-hs/2; kk<=cz+hs/2; kk++)
        for (int jj=cy-hs/2; jj<=cy+hs/2; jj++)
            for (int ii=cx-hs/2; ii<=cx+hs/2; ii++)
                h_Q[ii+jj*NX+kk*NX*NY] = Q_density;

    float *d_Q;
    cudaMalloc(&d_Q, N*sizeof(float));
    cudaMemcpy(d_Q, h_Q.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    FILE* csv = fopen("round2_audit/data/energy_test_A_v2.csv", "w");
    fprintf(csv, "step,t_us,E,dEdt,P_src,Q_zmin,Q_zmax,Q_xfaces,Q_yfaces,Q_total,residual_pct\n");

    auto computeE = [&](const std::vector<float>& T) -> double {
        double E = 0;
        for (int i = 0; i < N; i++) E += (double)rho * cp * T[i] * dV;
        return E;
    };

    // Heat flux at a Dirichlet face: use the FIRST INTERIOR cell (next to boundary)
    // Q = k × (T_interior - T_bc) / dx × A_face
    // At adiabatic face: Q = 0 (boundary cell equals interior → zero gradient)
    auto computeQ_face = [&](const std::vector<float>& T, int face) -> double {
        double Q = 0;
        // face: 0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max
        for (int a=0; a<(face<2?NY:NX); a++)
            for (int b=0; b<(face<4?NZ:NY); b++) {
                int ii, jj, kk;
                float T_int;  // temperature of first interior cell
                if (face == 0) { ii=1; jj=a; kk=b; }       // x_min: interior at i=1
                else if (face == 1) { ii=NX-2; jj=a; kk=b; } // x_max
                else if (face == 2) { ii=a; jj=1; kk=b; }     // y_min
                else if (face == 3) { ii=a; jj=NY-2; kk=b; }  // y_max
                else if (face == 4) { ii=a; jj=b; kk=1; }     // z_min
                else               { ii=a; jj=b; kk=NZ-2; }  // z_max

                T_int = T[ii + jj*NX + kk*NX*NY];

                // For Dirichlet face: Q = k * (T_int - T_bc) / dx * dx²
                // For adiabatic face: boundary cell = interior cell → gradient ≈ 0
                // But we need to compute what T_boundary actually is
                int bi, bj, bk;
                if (face == 0) { bi=0; bj=a; bk=b; }
                else if (face == 1) { bi=NX-1; bj=a; bk=b; }
                else if (face == 2) { bi=a; bj=0; bk=b; }
                else if (face == 3) { bi=a; bj=NY-1; bk=b; }
                else if (face == 4) { bi=a; bj=b; bk=0; }
                else               { bi=a; bj=b; bk=NZ-1; }

                float T_bnd = T[bi + bj*NX + bk*NX*NY];

                // Heat flux: q = k * (T_int - T_bnd) / dx, positive = outgoing
                Q += (double)k_c * (T_int - T_bnd) / dx * dx * dx;
            }
        return Q;
    };

    double E_prev = computeE(h_T);
    int total_steps = 10000;

    printf("  %6s %7s %12s %9s %9s %9s %9s %9s %9s %9s %8s\n",
           "Step", "t[μs]", "E[J]", "dE/dt", "P_src", "Q_zmin", "Q_zmax",
           "Q_xfac", "Q_yfac", "Q_tot", "Res[%]");

    for (int step = 1; step <= total_steps; step++) {
        thermal.applyFaceThermalBC(4, 2, dt, dx, T_bc);  // z_min Dirichlet
        for (int f = 0; f < 4; f++) thermal.applyFaceThermalBC(f, 1, dt, dx, T_bc);
        thermal.applyFaceThermalBC(5, 1, dt, dx, T_bc);  // z_max adiabatic

        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
        thermal.computeTemperature();
        thermal.addHeatSource(d_Q, dt);

        if (step % 1000 == 0) {
            thermal.copyTemperatureToHost(h_T.data());
            double E_now = computeE(h_T);
            double dEdt = (E_now - E_prev) / (1000 * dt);

            double Q_zmin = computeQ_face(h_T, 4);
            double Q_zmax = computeQ_face(h_T, 5);
            double Q_x = computeQ_face(h_T, 0) + computeQ_face(h_T, 1);
            double Q_y = computeQ_face(h_T, 2) + computeQ_face(h_T, 3);
            double Q_total = Q_zmin + Q_zmax + Q_x + Q_y;

            double residual = dEdt - P_source + Q_total;
            double res_pct = fabs(residual) / P_source * 100;

            printf("  %6d %7.1f %12.4e %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %8.2f\n",
                   step, step*dt*1e6, E_now, dEdt, P_source,
                   Q_zmin, Q_zmax, Q_x, Q_y, Q_total, res_pct);
            fprintf(csv, "%d,%.4f,%.8e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
                    step, step*dt*1e6, E_now, dEdt, P_source,
                    Q_zmin, Q_zmax, Q_x, Q_y, Q_total, res_pct);

            E_prev = E_now;
        }
    }
    fclose(csv);
    cudaFree(d_Q);

    printf("\n  If Res < 5%% at all times: FDM energy conservation is correct.\n");
    printf("  If Q_adiabatic (x,y,zmax) > 0: adiabatic BC is leaking.\n");
    return 0;
}
