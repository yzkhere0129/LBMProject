/**
 * @file test_marangoni_benchmark.cu
 * @brief Marangoni (Thermocapillary) Benchmark Test with VTK Visualization
 *
 * This test implements the classic thermocapillary microchannel benchmark
 * for direct comparison with waLBerla's thermocapillary showcase.
 *
 * Physics: Marangoni-driven convection in a liquid layer with horizontal
 * temperature gradient. Surface tension decreases with temperature,
 * causing flow from hot to cold regions at the free surface.
 *
 * Expected phenomenon:
 *   - Surface flow from hot to cold (negative dσ/dT)
 *   - Return flow in bulk
 *   - Formation of convection cells
 *
 * Reference: waLBerla thermocapillary showcase
 *   /home/yzk/walberla/apps/showcases/Thermocapillary/
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

//==============================================================================
// CUDA Kernels (with extern "C" for proper linking)
//==============================================================================

extern "C" {

// Initialize temperature field with linear gradient
__global__ void initTemperatureLinearGradientKernel(
    float* temperature, int nx, int ny, int nz,
    float T_hot, float T_cold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        float x_frac = static_cast<float>(i) / static_cast<float>(nx - 1);
        temperature[idx] = T_hot * (1.0f - x_frac) + T_cold * x_frac;
    }
}

// Initialize fill level with tanh interface profile
__global__ void initFillLevelInterfaceKernel(
    float* fill_level, int nx, int ny, int nz,
    float interface_y, float interface_width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        float y_rel = static_cast<float>(j) - interface_y;
        float f = 0.5f * (1.0f - tanhf(2.0f * y_rel / interface_width));
        fill_level[idx] = fminf(1.0f, fmaxf(0.0f, f));
    }
}

// Compute Marangoni force: F = (dσ/dT) × ∇_s T × |∇f|
__global__ void computeMarangoniForceKernel(
    const float* temperature,
    const float* fill_level,
    float* fx, float* fy, float* fz,
    int nx, int ny, int nz,
    float dx, float dsigma_dT)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k >= 0 && k < nz) {
        int idx = i + j * nx + k * nx * ny;

        float f = fill_level[idx];
        if (f < 0.1f || f > 0.9f) {
            fx[idx] = 0.0f;
            fy[idx] = 0.0f;
            fz[idx] = 0.0f;
            return;
        }

        int ip = idx + 1;
        int im = idx - 1;
        int jp = idx + nx;
        int jm = idx - nx;

        float dfdx = (fill_level[ip] - fill_level[im]) / (2.0f * dx);
        float dfdy = (fill_level[jp] - fill_level[jm]) / (2.0f * dx);
        float dfdz = 0.0f;

        float grad_f_mag = sqrtf(dfdx*dfdx + dfdy*dfdy + dfdz*dfdz);
        if (grad_f_mag < 1e-10f) {
            fx[idx] = 0.0f;
            fy[idx] = 0.0f;
            fz[idx] = 0.0f;
            return;
        }

        float nx_n = dfdx / grad_f_mag;
        float ny_n = dfdy / grad_f_mag;
        float nz_n = 0.0f;

        float dTdx = (temperature[ip] - temperature[im]) / (2.0f * dx);
        float dTdy = (temperature[jp] - temperature[jm]) / (2.0f * dx);
        float dTdz = 0.0f;

        float grad_T_dot_n = dTdx * nx_n + dTdy * ny_n + dTdz * nz_n;
        float dTdx_s = dTdx - grad_T_dot_n * nx_n;
        float dTdy_s = dTdy - grad_T_dot_n * ny_n;
        float dTdz_s = dTdz - grad_T_dot_n * nz_n;

        float force_scale = dsigma_dT * grad_f_mag;

        fx[idx] = force_scale * dTdx_s;
        fy[idx] = force_scale * dTdy_s;
        fz[idx] = force_scale * dTdz_s;
    }
}

// Apply temperature boundary conditions
__global__ void applyTemperatureBCKernel(
    float* temperature, int nx, int ny, int nz,
    float T_hot, float T_cold)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < ny && k < nz) {
        temperature[0 + j * nx + k * nx * ny] = T_hot;
        temperature[(nx-1) + j * nx + k * nx * ny] = T_cold;
    }
}

} // extern "C"

namespace {

//==============================================================================
// Benchmark Configuration
//==============================================================================

struct MarangoniBenchmarkConfig {
    // Domain (quasi-2D, matching waLBerla microchannel)
    int nx = 200;        // X-direction (temperature gradient direction)
    int ny = 100;        // Y-direction (height: wall to free surface)
    int nz = 2;          // Z-direction (periodic, quasi-2D)

    // Physical parameters
    float dx = 2.0e-6f;  // Grid spacing: 2 μm
    float dt = 1.0e-9f;  // Time step: 1 ns

    // Temperature boundary conditions
    float T_hot = 2000.0f;   // Left wall temperature [K]
    float T_cold = 1800.0f;  // Right wall temperature [K]
    float T_ref = 1900.0f;   // Reference temperature [K]

    // Material: Ti-6Al-4V
    float rho = 4420.0f;           // Density [kg/m³]
    float mu = 4.0e-3f;            // Dynamic viscosity [Pa·s]
    float k = 30.0f;               // Thermal conductivity [W/(m·K)]
    float cp = 670.0f;             // Specific heat [J/(kg·K)]
    float sigma_0 = 1.52f;         // Surface tension at T_ref [N/m]
    float dsigma_dT = -0.26e-3f;   // Surface tension coefficient [N/(m·K)]

    // Interface position (fraction of ny from bottom)
    float interface_position = 0.7f;  // 70% liquid, 30% gas
    float interface_width = 3.0f;     // Interface width in cells

    // Simulation parameters
    int num_timesteps = 20000;
    int vtk_output_interval = 1000;
    int console_output_interval = 2000;

    // Output directory
    std::string output_dir = "marangoni_benchmark_output";

    // Derived quantities
    float nu() const { return mu / rho; }  // Kinematic viscosity
    float alpha() const { return k / (rho * cp); }  // Thermal diffusivity
    float Pr() const { return nu() / alpha(); }  // Prandtl number
    float dT() const { return T_hot - T_cold; }

    // Characteristic Marangoni velocity: U_M = |dσ/dT| × ΔT / μ
    float U_marangoni() const {
        return std::abs(dsigma_dT) * dT() / mu;
    }

    // Marangoni number: Ma = |dσ/dT| × ΔT × L / (μ × α)
    float Ma() const {
        float L = interface_position * ny * dx;
        return std::abs(dsigma_dT) * dT() * L / (mu * alpha());
    }

    // Reynolds number based on Marangoni velocity
    float Re() const {
        float L = interface_position * ny * dx;
        return U_marangoni() * L / nu();
    }
};

//==============================================================================
// VTK Writer for Benchmark
//==============================================================================

class MarangoniBenchmarkVTKWriter {
public:
    static void writeFields(
        const std::string& filename,
        const std::vector<float>& temperature,
        const std::vector<float>& fill_level,
        const std::vector<float>& ux,
        const std::vector<float>& uy,
        const std::vector<float>& uz,
        const std::vector<float>& fx,
        const std::vector<float>& fy,
        const std::vector<float>& fz,
        int nx, int ny, int nz,
        float dx, float time, float Ma)
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << filename << std::endl;
            return;
        }

        // Header
        file << "# vtk DataFile Version 3.0\n";
        file << "Marangoni Benchmark t=" << std::scientific << std::setprecision(4)
             << time << "s Ma=" << std::fixed << std::setprecision(1) << Ma << "\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        file << "ORIGIN 0.0 0.0 0.0\n";
        file << std::scientific << std::setprecision(6);
        file << "SPACING " << dx << " " << dx << " " << dx << "\n";
        file << std::defaultfloat;

        int n_points = nx * ny * nz;
        file << "\nPOINT_DATA " << n_points << "\n";

        // Velocity (vector)
        file << "VECTORS Velocity float\n";
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + j * nx + k * nx * ny;
                    file << ux[idx] << " " << uy[idx] << " " << uz[idx] << "\n";
                }
            }
        }

        // Temperature (scalar)
        file << "\nSCALARS Temperature float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + j * nx + k * nx * ny;
                    file << temperature[idx] << "\n";
                }
            }
        }

        // Fill level / Phase field (scalar)
        file << "\nSCALARS FillLevel float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + j * nx + k * nx * ny;
                    file << fill_level[idx] << "\n";
                }
            }
        }

        // Velocity magnitude (scalar)
        file << "\nSCALARS VelocityMagnitude float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + j * nx + k * nx * ny;
                    float vmag = std::sqrt(ux[idx]*ux[idx] + uy[idx]*uy[idx] + uz[idx]*uz[idx]);
                    file << vmag << "\n";
                }
            }
        }

        // Interface indicator (scalar) - 1 at interface, 0 elsewhere
        file << "\nSCALARS InterfaceIndicator float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + j * nx + k * nx * ny;
                    float f = fill_level[idx];
                    float indicator = (f > 0.1f && f < 0.9f) ? 1.0f : 0.0f;
                    file << indicator << "\n";
                }
            }
        }

        // Marangoni force (vector)
        file << "\nVECTORS MarangoniForce float\n";
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + j * nx + k * nx * ny;
                    file << fx[idx] << " " << fy[idx] << " " << fz[idx] << "\n";
                }
            }
        }

        file.close();
        std::cout << "VTK written: " << filename << std::endl;
    }
};

// Kernels defined outside anonymous namespace above

//==============================================================================
// Test Fixture
//==============================================================================

class MarangoniBenchmarkTest : public ::testing::Test {
protected:
    MarangoniBenchmarkConfig config;

    float* d_temperature = nullptr;
    float* d_fill_level = nullptr;
    float* d_ux = nullptr;
    float* d_uy = nullptr;
    float* d_uz = nullptr;
    float* d_fx = nullptr;
    float* d_fy = nullptr;
    float* d_fz = nullptr;

    std::vector<float> h_temperature;
    std::vector<float> h_fill_level;
    std::vector<float> h_ux, h_uy, h_uz;
    std::vector<float> h_fx, h_fy, h_fz;

    size_t num_cells;

    void SetUp() override {
        num_cells = config.nx * config.ny * config.nz;
        size_t size = num_cells * sizeof(float);

        cudaMalloc(&d_temperature, size);
        cudaMalloc(&d_fill_level, size);
        cudaMalloc(&d_ux, size);
        cudaMalloc(&d_uy, size);
        cudaMalloc(&d_uz, size);
        cudaMalloc(&d_fx, size);
        cudaMalloc(&d_fy, size);
        cudaMalloc(&d_fz, size);

        cudaMemset(d_ux, 0, size);
        cudaMemset(d_uy, 0, size);
        cudaMemset(d_uz, 0, size);
        cudaMemset(d_fx, 0, size);
        cudaMemset(d_fy, 0, size);
        cudaMemset(d_fz, 0, size);

        h_temperature.resize(num_cells);
        h_fill_level.resize(num_cells);
        h_ux.resize(num_cells);
        h_uy.resize(num_cells);
        h_uz.resize(num_cells);
        h_fx.resize(num_cells);
        h_fy.resize(num_cells);
        h_fz.resize(num_cells);

        system(("mkdir -p " + config.output_dir).c_str());
    }

    void TearDown() override {
        cudaFree(d_temperature);
        cudaFree(d_fill_level);
        cudaFree(d_ux);
        cudaFree(d_uy);
        cudaFree(d_uz);
        cudaFree(d_fx);
        cudaFree(d_fy);
        cudaFree(d_fz);
    }

    void initializeFields() {
        // Initialize on CPU and copy to GPU (avoids CUDA linking issues with kernels in test files)
        std::cout << "Initializing fields on CPU...\n";

        // Initialize temperature with linear gradient
        for (int k = 0; k < config.nz; ++k) {
            for (int j = 0; j < config.ny; ++j) {
                for (int i = 0; i < config.nx; ++i) {
                    int idx = i + j * config.nx + k * config.nx * config.ny;
                    float x_frac = static_cast<float>(i) / static_cast<float>(config.nx - 1);
                    h_temperature[idx] = config.T_hot * (1.0f - x_frac) + config.T_cold * x_frac;
                }
            }
        }

        // Initialize fill level with tanh interface profile
        float interface_y = config.interface_position * config.ny;
        std::cout << "Interface Y: " << interface_y << "\n";
        for (int k = 0; k < config.nz; ++k) {
            for (int j = 0; j < config.ny; ++j) {
                for (int i = 0; i < config.nx; ++i) {
                    int idx = i + j * config.nx + k * config.nx * config.ny;
                    float y_rel = static_cast<float>(j) - interface_y;
                    float f = 0.5f * (1.0f - std::tanh(2.0f * y_rel / config.interface_width));
                    h_fill_level[idx] = std::min(1.0f, std::max(0.0f, f));
                }
            }
        }

        // Copy to device
        size_t size = num_cells * sizeof(float);
        cudaMemcpy(d_temperature, h_temperature.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fill_level, h_fill_level.data(), size, cudaMemcpyHostToDevice);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after initialization: " << cudaGetErrorString(err) << "\n";
        }
        cudaDeviceSynchronize();
        std::cout << "Fields initialized.\n";
    }

    void computeForces() {
        // Compute Marangoni forces on CPU
        int nx = config.nx, ny = config.ny, nz = config.nz;
        float dx = config.dx;
        float dsigma_dT = config.dsigma_dT;

        for (int k = 0; k < nz; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    int idx = i + j * nx + k * nx * ny;

                    float f = h_fill_level[idx];
                    if (f < 0.1f || f > 0.9f) {
                        h_fx[idx] = 0.0f;
                        h_fy[idx] = 0.0f;
                        h_fz[idx] = 0.0f;
                        continue;
                    }

                    int ip = idx + 1;
                    int im = idx - 1;
                    int jp = idx + nx;
                    int jm = idx - nx;

                    float dfdx = (h_fill_level[ip] - h_fill_level[im]) / (2.0f * dx);
                    float dfdy = (h_fill_level[jp] - h_fill_level[jm]) / (2.0f * dx);

                    float grad_f_mag = std::sqrt(dfdx*dfdx + dfdy*dfdy);
                    if (grad_f_mag < 1e-10f) {
                        h_fx[idx] = 0.0f;
                        h_fy[idx] = 0.0f;
                        h_fz[idx] = 0.0f;
                        continue;
                    }

                    float nx_n = dfdx / grad_f_mag;
                    float ny_n = dfdy / grad_f_mag;

                    float dTdx = (h_temperature[ip] - h_temperature[im]) / (2.0f * dx);
                    float dTdy = (h_temperature[jp] - h_temperature[jm]) / (2.0f * dx);

                    float grad_T_dot_n = dTdx * nx_n + dTdy * ny_n;
                    float dTdx_s = dTdx - grad_T_dot_n * nx_n;
                    float dTdy_s = dTdy - grad_T_dot_n * ny_n;

                    float force_scale = dsigma_dT * grad_f_mag;

                    h_fx[idx] = force_scale * dTdx_s;
                    h_fy[idx] = force_scale * dTdy_s;
                    h_fz[idx] = 0.0f;
                }
            }
        }

        // Copy to device
        size_t size = num_cells * sizeof(float);
        cudaMemcpy(d_fx, h_fx.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy, h_fy.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fz, h_fz.data(), size, cudaMemcpyHostToDevice);
    }

    void copyToHost() {
        size_t size = num_cells * sizeof(float);
        cudaMemcpy(h_temperature.data(), d_temperature, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fill_level.data(), d_fill_level, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ux.data(), d_ux, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uy.data(), d_uy, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uz.data(), d_uz, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fx.data(), d_fx, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fy.data(), d_fy, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fz.data(), d_fz, size, cudaMemcpyDeviceToHost);
    }

    void writeVTK(int step) {
        copyToHost();

        float time = step * config.dt;
        std::ostringstream oss;
        oss << config.output_dir << "/marangoni_"
            << std::setfill('0') << std::setw(6) << step << ".vtk";

        MarangoniBenchmarkVTKWriter::writeFields(
            oss.str(),
            h_temperature, h_fill_level,
            h_ux, h_uy, h_uz,
            h_fx, h_fy, h_fz,
            config.nx, config.ny, config.nz,
            config.dx, time, config.Ma()
        );
    }

    void printDiagnostics(int step) {
        copyToHost();

        float max_force = 0.0f;
        float max_force_x = 0.0f;
        int max_i = 0, max_j = 0;

        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + j * config.nx;
                float f_mag = std::sqrt(h_fx[idx]*h_fx[idx] +
                                       h_fy[idx]*h_fy[idx] +
                                       h_fz[idx]*h_fz[idx]);
                if (f_mag > max_force) {
                    max_force = f_mag;
                    max_force_x = h_fx[idx];
                    max_i = i;
                    max_j = j;
                }
            }
        }

        float max_vel = 0.0f;
        float max_ux = 0.0f;
        for (size_t idx = 0; idx < num_cells; ++idx) {
            float v_mag = std::sqrt(h_ux[idx]*h_ux[idx] +
                                   h_uy[idx]*h_uy[idx] +
                                   h_uz[idx]*h_uz[idx]);
            if (v_mag > max_vel) {
                max_vel = v_mag;
                max_ux = h_ux[idx];
            }
        }

        float time = step * config.dt;
        std::cout << "\n========================================\n";
        std::cout << "Step " << step << " (t = " << std::scientific
                  << time << " s)\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Max Marangoni Force: " << max_force << " N/m^3\n";
        std::cout << "  at (i=" << max_i << ", j=" << max_j << ")\n";
        std::cout << "  Fx = " << max_force_x << " N/m^3\n";
        std::cout << "Max Velocity: " << max_vel << " m/s\n";
        std::cout << "  Ux = " << max_ux << " m/s\n";
        std::cout << "Expected U_M: " << config.U_marangoni() << " m/s\n";
        std::cout << "Ma = " << config.Ma() << ", Re = " << config.Re() << "\n";
        std::cout << "========================================\n";
    }
};

//==============================================================================
// Tests
//==============================================================================

TEST_F(MarangoniBenchmarkTest, ForceComputation) {
    std::cout << "\n=== Marangoni Benchmark: Force Computation ===\n";
    std::cout << "Domain: " << config.nx << " x " << config.ny << " x " << config.nz << "\n";
    std::cout << "dx = " << config.dx << " m, dt = " << config.dt << " s\n";
    std::cout << "T_hot = " << config.T_hot << " K, T_cold = " << config.T_cold << " K\n";
    std::cout << "dsigma/dT = " << config.dsigma_dT << " N/(m*K)\n";
    std::cout << "Ma = " << config.Ma() << ", Re = " << config.Re() << "\n";
    std::cout << "Expected U_M = " << config.U_marangoni() << " m/s\n\n";

    initializeFields();
    computeForces();
    copyToHost();

    float max_force = 0.0f;
    int interface_cells = 0;

    for (size_t idx = 0; idx < num_cells; ++idx) {
        float f_mag = std::sqrt(h_fx[idx]*h_fx[idx] +
                               h_fy[idx]*h_fy[idx] +
                               h_fz[idx]*h_fz[idx]);
        if (f_mag > max_force) {
            max_force = f_mag;
        }
        if (h_fill_level[idx] > 0.1f && h_fill_level[idx] < 0.9f) {
            interface_cells++;
        }
    }

    std::cout << "Interface cells: " << interface_cells << "\n";
    std::cout << "Max Marangoni force: " << max_force << " N/m^3\n";

    float expected_grad_T = config.dT() / (config.nx * config.dx);
    float expected_grad_f = 1.0f / (config.interface_width * config.dx);
    float expected_force = std::abs(config.dsigma_dT) * expected_grad_T * expected_grad_f;

    std::cout << "Expected force (order of magnitude): " << expected_force << " N/m^3\n";

    EXPECT_GT(max_force, 0.0f) << "Marangoni force should be non-zero";
    EXPECT_GT(max_force, expected_force * 0.01f) << "Force too small";
    EXPECT_LT(max_force, expected_force * 100.0f) << "Force too large";

    writeVTK(0);
    std::cout << "\nVTK output written to: " << config.output_dir << "/marangoni_000000.vtk\n";
}

TEST_F(MarangoniBenchmarkTest, ThermocapillaryFlowVisualization) {
    std::cout << "\n=== Marangoni Benchmark: Full Flow Simulation ===\n";
    std::cout << "This test generates VTK files for ParaView visualization.\n";
    std::cout << "Compare with waLBerla thermocapillary showcase.\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Domain: " << config.nx << " x " << config.ny << " x " << config.nz << "\n";
    std::cout << "  Physical size: " << (config.nx * config.dx * 1e6) << " x "
              << (config.ny * config.dx * 1e6) << " um\n";
    std::cout << "  Temperature: " << config.T_cold << " K (cold) to "
              << config.T_hot << " K (hot)\n";
    std::cout << "  Material: Ti-6Al-4V\n";
    std::cout << "  dsigma/dT = " << config.dsigma_dT << " N/(m*K)\n";
    std::cout << "  Ma = " << config.Ma() << "\n";
    std::cout << "  Re = " << config.Re() << "\n";
    std::cout << "  Pr = " << config.Pr() << "\n";
    std::cout << "  Expected Marangoni velocity: " << config.U_marangoni() << " m/s\n";
    std::cout << "\nSimulation: " << config.num_timesteps << " steps\n";
    std::cout << "VTK output every " << config.vtk_output_interval << " steps\n";
    std::cout << "Output directory: " << config.output_dir << "/\n\n";

    initializeFields();
    writeVTK(0);
    printDiagnostics(0);

    // Physical velocity update with viscous dissipation
    // At steady state: F_Marangoni = μ * ∇²u ≈ μ * u / L²
    // So: du/dt = F/ρ - ν * u / L²
    // Characteristic length: interface width * dx
    float L_char = config.interface_width * config.dx;
    float viscous_damping_rate = config.nu() / (L_char * L_char);

    // Use larger time step for visualization (not physical accuracy)
    // This accelerates convergence to steady state
    float effective_dt = 1.0e-7f;  // 100x larger than physical dt
    float force_to_accel = effective_dt / config.rho;
    float damping_factor = std::exp(-viscous_damping_rate * effective_dt);

    std::cout << "Physical parameters for velocity evolution:\n";
    std::cout << "  L_char = " << L_char << " m\n";
    std::cout << "  nu = " << config.nu() << " m²/s\n";
    std::cout << "  Viscous damping rate = " << viscous_damping_rate << " /s\n";
    std::cout << "  Effective dt = " << effective_dt << " s\n";
    std::cout << "  Damping factor per step = " << damping_factor << "\n\n";

    for (int step = 1; step <= config.num_timesteps; ++step) {
        // Compute Marangoni forces (CPU)
        computeForces();

        // Apply temperature BC (CPU)
        for (int k = 0; k < config.nz; ++k) {
            for (int j = 0; j < config.ny; ++j) {
                h_temperature[0 + j * config.nx + k * config.nx * config.ny] = config.T_hot;
                h_temperature[(config.nx-1) + j * config.nx + k * config.nx * config.ny] = config.T_cold;
            }
        }

        // Velocity update with viscous damping (simplified momentum equation)
        // du/dt = F/ρ - ν*u/L² → u_new = u*exp(-ν*dt/L²) + (F/ρ)*(L²/ν)*(1-exp(-ν*dt/L²))
        // Simplified: u_new = damping_factor * u + force_to_accel * F
        for (size_t idx = 0; idx < num_cells; ++idx) {
            h_ux[idx] = damping_factor * h_ux[idx] + force_to_accel * h_fx[idx];
            h_uy[idx] = damping_factor * h_uy[idx] + force_to_accel * h_fy[idx];
        }

        if (step % config.vtk_output_interval == 0) {
            // Copy to device for VTK writing
            size_t size = num_cells * sizeof(float);
            cudaMemcpy(d_ux, h_ux.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_uy, h_uy.data(), size, cudaMemcpyHostToDevice);
            writeVTK(step);
        }

        if (step % config.console_output_interval == 0) {
            printDiagnostics(step);
        }
    }

    writeVTK(config.num_timesteps);
    printDiagnostics(config.num_timesteps);

    copyToHost();

    float max_vel = 0.0f;
    float max_ux_at_interface = 0.0f;
    int interface_y = static_cast<int>(config.interface_position * config.ny);

    for (int i = 0; i < config.nx; ++i) {
        int idx = i + interface_y * config.nx;
        if (std::abs(h_ux[idx]) > std::abs(max_ux_at_interface)) {
            max_ux_at_interface = h_ux[idx];
        }
    }

    for (size_t idx = 0; idx < num_cells; ++idx) {
        float v_mag = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx]);
        if (v_mag > max_vel) {
            max_vel = v_mag;
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "FINAL RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "Max velocity: " << max_vel << " m/s\n";
    std::cout << "Max Ux at interface: " << max_ux_at_interface << " m/s\n";
    std::cout << "Expected U_M: " << config.U_marangoni() << " m/s\n";
    std::cout << "Ratio: " << (max_vel / config.U_marangoni()) << "\n";
    std::cout << "\nVTK files for ParaView: " << config.output_dir << "/marangoni_*.vtk\n";
    std::cout << "========================================\n";

    EXPECT_GT(max_vel, 0.0f) << "Flow should develop";

    std::cout << "\nFlow direction check:\n";
    std::cout << "  dsigma/dT < 0 means surface tension increases with cooling\n";
    std::cout << "  Flow goes from hot (low sigma) to cold (high sigma)\n";
    std::cout << "  Expected: +x direction (hot to cold)\n";
    std::cout << "  Actual Ux at interface: " << max_ux_at_interface << "\n";

    if (config.dsigma_dT < 0) {
        EXPECT_GT(max_ux_at_interface, 0.0f)
            << "Surface flow should be from hot (x=0) to cold (x=nx-1)";
    }
}

TEST_F(MarangoniBenchmarkTest, SanityCheck) {
    std::cout << "\n=== Marangoni Benchmark: Sanity Check ===\n";

    initializeFields();
    copyToHost();

    float T_at_0 = h_temperature[0];
    float T_at_end = h_temperature[config.nx - 1];

    std::cout << "Temperature at x=0: " << T_at_0 << " K (expected: " << config.T_hot << ")\n";
    std::cout << "Temperature at x=end: " << T_at_end << " K (expected: " << config.T_cold << ")\n";

    EXPECT_NEAR(T_at_0, config.T_hot, 1.0f);
    EXPECT_NEAR(T_at_end, config.T_cold, 1.0f);

    int liquid_cells = 0, gas_cells = 0, interface_cells = 0;
    for (size_t idx = 0; idx < num_cells; ++idx) {
        if (h_fill_level[idx] > 0.9f) liquid_cells++;
        else if (h_fill_level[idx] < 0.1f) gas_cells++;
        else interface_cells++;
    }

    std::cout << "Liquid cells: " << liquid_cells << "\n";
    std::cout << "Gas cells: " << gas_cells << "\n";
    std::cout << "Interface cells: " << interface_cells << "\n";

    EXPECT_GT(liquid_cells, 0) << "Should have liquid region";
    EXPECT_GT(gas_cells, 0) << "Should have gas region";
    EXPECT_GT(interface_cells, 0) << "Should have interface region";

    float expected_liquid_fraction = config.interface_position;
    float actual_liquid_fraction = static_cast<float>(liquid_cells + interface_cells/2) / num_cells;

    std::cout << "Expected liquid fraction: " << expected_liquid_fraction << "\n";
    std::cout << "Actual liquid fraction: " << actual_liquid_fraction << "\n";

    EXPECT_NEAR(actual_liquid_fraction, expected_liquid_fraction, 0.1f);
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "     MARANGONI BENCHMARK TEST WITH VTK VISUALIZATION\n";
    std::cout << "\n";
    std::cout << "  Thermocapillary-driven flow in liquid layer\n";
    std::cout << "  For comparison with waLBerla thermocapillary showcase\n";
    std::cout << "============================================================\n";
    std::cout << "\n";

    return RUN_ALL_TESTS();
}
