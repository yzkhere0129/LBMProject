#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Simple 1D advection test
__global__ void test_advection_kernel(
    const float* f_old,
    float* f_new,
    float u,
    float dt,
    float dx,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Periodic boundaries
    int i_left = (i > 0) ? i - 1 : n - 1;
    int i_right = (i < n - 1) ? i + 1 : 0;

    // Upwind fluxes
    float flux_right, flux_left;
    if (u >= 0.0f) {
        flux_right = u * f_old[i];        // Material flows right from this cell
        flux_left = u * f_old[i_left];    // Material flows right from left neighbor
    } else {
        flux_right = u * f_old[i_right];  // Material flows left from right neighbor
        flux_left = u * f_old[i];         // Material flows left from this cell
    }

    // Finite volume update
    f_new[i] = f_old[i] - (dt / dx) * (flux_right - flux_left);
}

int main() {
    const int n = 50;
    const float dx = 2e-6f;  // 2 microns
    const float dt = 1e-7f;  // 0.1 microseconds
    const float u = 0.1f;    // 0.1 m/s (rightward)

    // Initialize: liquid (f=1) for i < 25, gas (f=0) for i >= 25
    std::vector<float> h_f(n, 0.0f);
    for (int i = 0; i < 25; ++i) {
        h_f[i] = 1.0f;
    }

    float *d_f_old, *d_f_new;
    cudaMalloc(&d_f_old, n * sizeof(float));
    cudaMalloc(&d_f_new, n * sizeof(float));
    cudaMemcpy(d_f_old, h_f.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Run advection for 1000 steps
    for (int step = 0; step < 1000; ++step) {
        test_advection_kernel<<<1, 64>>>(d_f_old, d_f_new, u, dt, dx, n);
        cudaDeviceSynchronize();

        // Debug: check interface position every 200 steps
        if (step % 200 == 0 || step == 999) {
            cudaMemcpy(h_f.data(), d_f_new, n * sizeof(float), cudaMemcpyDeviceToHost);
            int iface = -1;
            for (int i = 0; i < n; ++i) {
                if (h_f[i] > 0.4f && h_f[i] < 0.6f) {
                    iface = i;
                    break;
                }
            }
            if (iface == -1) {
                for (int i = 0; i < n - 1; ++i) {
                    if (h_f[i] > 0.9f && h_f[i+1] < 0.1f) {
                        iface = i;
                        break;
                    }
                }
            }
            std::cout << "Step " << step << ": interface at x=" << iface
                      << " (expected: " << (24 + (step+1) * 0.005) << ")" << std::endl;
        }

        // Swap buffers
        float* tmp = d_f_old;
        d_f_old = d_f_new;
        d_f_new = tmp;
    }

    // Copy result back
    cudaMemcpy(h_f.data(), d_f_old, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Find interface position
    int interface_pos = -1;
    for (int i = 0; i < n; ++i) {
        if (h_f[i] > 0.4f && h_f[i] < 0.6f) {
            interface_pos = i;
            break;
        }
    }

    if (interface_pos == -1) {
        // Look for sharp interface
        for (int i = 0; i < n - 1; ++i) {
            if (h_f[i] > 0.9f && h_f[i+1] < 0.1f) {
                interface_pos = i;
                break;
            }
        }
    }

    std::cout << "Initial interface: x = 24 (or 24.5)" << std::endl;
    std::cout << "Final interface: x = " << interface_pos << std::endl;
    std::cout << "Expected: x = 29-30 (displacement of ~5 cells)" << std::endl;
    std::cout << "CFL = " << u * dt / dx << " = " << (u * dt / dx) << std::endl;
    std::cout << "Expected displacement = CFL * num_steps = " << (u * dt / dx * 1000) << " cells" << std::endl;

    // Print fill levels around interface
    std::cout << "\nFill levels:" << std::endl;
    for (int i = 0; i < n; ++i) {
        if (h_f[i] > 0.01f && h_f[i] < 0.99f) {
            std::cout << "  f[" << i << "] = " << h_f[i] << std::endl;
        }
    }

    cudaFree(d_f_old);
    cudaFree(d_f_new);

    return 0;
}
