# MultiphysicsSolver Design Specification

**Purpose**: Integrate thermal, fluid, VOF, surface tension, and Marangoni solvers for LPBF simulation
**Target**: Phase 6 validation against literature (Marangoni velocity 0.5-2 m/s)
**Design Philosophy**: Modular, testable, walberla-inspired operator splitting

---

## 1. Class Architecture

### 1.1 Component Ownership

```
MultiphysicsSolver
├─ ThermalLBM (thermal evolution + phase change)
├─ FluidLBM (momentum evolution + LBM collision-streaming)
├─ VOFSolver (free surface tracking + interface reconstruction)
├─ SurfaceTension (curvature-based Laplace pressure)
├─ MarangoniEffect (thermocapillary force)
└─ LaserSource (shared_ptr, external heat source)
```

**Design Pattern**: Composition over inheritance (like walberla's BlockDataHandling)

**Rationale**:
- Each solver owns its device memory
- Clear separation of physics
- Independent testing of components
- Easy to disable individual effects

---

### 1.2 Data Flow

```
Step N → Step N+1:

1. Thermal Evolution:
   temperature[N] + laser_heat → temperature[N+1]
   ├─ Phase change (enthalpy method)
   └─ Updates liquid_fraction field

2. Interface Reconstruction:
   fill_level[N] → normals[N+1], curvature[N+1]
   ├─ PLIC or height-function method
   └─ Needed for force computation

3. Force Computation:
   temperature[N+1], fill_level[N], normals[N+1] → forces[N+1]
   ├─ Buoyancy: F = -ρ g β (T - T_ref)
   ├─ Surface tension: F = σ κ ∇f
   └─ Marangoni: F = (dσ/dT) ∇_s T |∇f| / h

4. Fluid Evolution:
   velocity[N] + forces[N+1] → velocity[N+1]
   ├─ LBM collision with force term
   └─ Streaming step

5. VOF Advection:
   fill_level[N] + velocity[N+1] → fill_level[N+1]
   ├─ Geometric advection (subcycled)
   └─ Mass conservation enforced
```

**Critical**: Temperature updated before forces (Marangoni depends on ∇T)

---

## 2. Memory Management Strategy

### 2.1 Device Memory Layout

**Each solver owns its own arrays**:
```cpp
class MultiphysicsSolver {
private:
    // Component solvers (encapsulate device memory)
    std::unique_ptr<ThermalLBM> thermal_solver_;  // Owns d_temperature
    std::unique_ptr<FluidLBM> fluid_solver_;      // Owns d_velocity, d_f (LBM dist)
    std::unique_ptr<VOFSolver> vof_solver_;       // Owns d_fill_level, d_normals, d_curvature

    // Force fields (owned by MultiphysicsSolver)
    float *d_force_buoyancy_x_, *d_force_buoyancy_y_, *d_force_buoyancy_z_;
    float *d_force_surface_x_, *d_force_surface_y_, *d_force_surface_z_;
    float *d_force_marangoni_x_, *d_force_marangoni_y_, *d_force_marangoni_z_;
    float *d_force_total_x_, *d_force_total_y_, *d_force_total_z_;

    // Material properties (device constant memory)
    // Declared in material_properties.h as: __constant__ MaterialProperties d_material;
};
```

**Total memory estimate** (for 200×150×100 domain):
```
N = 200 × 150 × 100 = 3,000,000 cells

ThermalLBM: 7 × float × N = 84 MB (D3Q7 distributions)
FluidLBM: 19 × float × N = 228 MB (D3Q19 distributions)
VOFSolver: 5 × float × N + 3 × float3 × N = 96 MB (fill, normals, curvature, etc.)
Forces: 9 × float × N = 108 MB (3 forces × 3 components each)

Total: ~516 MB (manageable on modern GPUs)
```

---

### 2.2 Access Pattern

**Option A: Getters (current, simple)**
```cpp
void MultiphysicsSolver::computeMarangoniForce() {
    const float* d_temp = thermal_solver_->getTemperatureDevicePtr();
    const float* d_fill = vof_solver_->getFillLevel();
    const float3* d_normals = vof_solver_->getInterfaceNormals();

    marangoni_->computeMarangoniForce(
        d_temp, d_fill, d_normals,
        d_force_marangoni_x_, d_force_marangoni_y_, d_force_marangoni_z_);
}
```

**Option B: Shared device pointers (like walberla's FieldAccessor)**
```cpp
// NOT RECOMMENDED for Phase 6 (adds complexity)
class FieldRegistry {
    std::map<std::string, void*> fields_;
public:
    template<typename T>
    T* get(const std::string& name);
};
```

**RECOMMENDATION**: **Option A (simple getters)**

**Rationale**:
- Explicit data flow (easier to debug)
- No hidden dependencies
- Type-safe (compiler checks pointer types)
- Sufficient for Phase 6 scope

---

## 3. Timestep Strategy (Adaptive)

### 3.1 Stability Constraints

```cpp
float MultiphysicsSolver::computeTimeStep() const {
    // 1. Thermal diffusion limit (explicit scheme)
    float alpha_max = config_.material.getThermalDiffusivity(T_max_);
    float dt_thermal = 0.5f * dx_ * dx_ / alpha_max;

    // 2. CFL limit (advection)
    float v_max = findMaxVelocity();  // Check current velocity field
    float dt_cfl = (v_max > 1e-6f) ? (0.3f * dx_ / v_max) : 1e-6f;

    // 3. Capillary wave limit (modified)
    // Standard: dt < sqrt(ρ dx³ / (2π σ))
    // This is extremely restrictive for small dx!
    // Workaround: Use larger effective dx or semi-implicit surface tension
    float rho = config_.material.rho_liquid;
    float sigma = config_.material.surface_tension;
    float dt_capillary = sqrtf(rho * dx_*dx_*dx_ / (2.0f * M_PI * sigma));

    // Apply safety factor to capillary (often too strict)
    dt_capillary *= 10.0f;  // Relaxation (monitor stability!)

    // 4. Viscous limit (usually not restrictive for LPBF)
    // dt < ρ dx² / μ
    float mu = config_.material.mu_liquid;
    float dt_viscous = rho * dx_ * dx_ / mu;

    // Take minimum, but enforce bounds
    float dt = fminf({dt_thermal, dt_cfl, dt_capillary, dt_viscous});
    dt = fmaxf(dt, config_.dt_min);
    dt = fminf(dt, config_.dt_max);

    return dt;
}
```

### 3.2 Timestep Monitoring

**Track which constraint is limiting**:
```cpp
struct TimestepInfo {
    float dt_thermal;
    float dt_fluid;
    float dt_capillary;
    float dt_viscous;
    float dt_used;
    float v_max;
    int limiting_factor;  // 0=thermal, 1=fluid, 2=capillary, 3=viscous
};

MultiphysicsSolver::TimestepInfo MultiphysicsSolver::getTimestepInfo() const {
    TimestepInfo info;
    info.dt_thermal = 0.5f * dx_ * dx_ / alpha_max_;
    info.dt_fluid = 0.3f * dx_ / v_max_;
    // ... compute others

    // Identify limiting factor
    std::vector<float> dts = {info.dt_thermal, info.dt_fluid, info.dt_capillary, info.dt_viscous};
    info.limiting_factor = std::distance(dts.begin(), std::min_element(dts.begin(), dts.end()));
    info.dt_used = current_dt_;

    return info;
}
```

**Use in validation**:
```cpp
if (step_count_ % 100 == 0) {
    auto ts_info = getTimestepInfo();
    std::cout << "Step " << step_count_ << ": dt = " << ts_info.dt_used * 1e9 << " ns, ";
    std::cout << "v_max = " << ts_info.v_max << " m/s, ";
    std::cout << "limited by: " << limit_names[ts_info.limiting_factor] << "\n";
}
```

---

## 4. Force Accumulation Algorithm

### 4.1 Separate Computation (Recommended)

```cpp
void MultiphysicsSolver::computeForces() {
    // Get current state
    const float* d_temp = thermal_solver_->getTemperatureDevicePtr();
    const float* d_fill = vof_solver_->getFillLevel();
    const float3* d_normals = vof_solver_->getInterfaceNormals();
    const float* d_curvature = vof_solver_->getCurvature();

    // 1. Buoyancy force (volumetric, in liquid regions)
    if (config_.enable_buoyancy) {
        computeBuoyancyForce(d_temp, d_fill,
                             d_force_buoyancy_x_, d_force_buoyancy_y_, d_force_buoyancy_z_);
    } else {
        cudaMemset(d_force_buoyancy_x_, 0, num_cells_ * sizeof(float));
        cudaMemset(d_force_buoyancy_y_, 0, num_cells_ * sizeof(float));
        cudaMemset(d_force_buoyancy_z_, 0, num_cells_ * sizeof(float));
    }

    // 2. Surface tension force (at interface)
    if (config_.enable_surface_tension) {
        surface_tension_->computeForce(d_fill, d_curvature, d_normals,
                                       d_force_surface_x_, d_force_surface_y_, d_force_surface_z_);
    } else {
        cudaMemset(d_force_surface_x_, 0, num_cells_ * sizeof(float));
        cudaMemset(d_force_surface_y_, 0, num_cells_ * sizeof(float));
        cudaMemset(d_force_surface_z_, 0, num_cells_ * sizeof(float));
    }

    // 3. Marangoni force (at interface)
    if (config_.enable_marangoni) {
        marangoni_->computeMarangoniForce(d_temp, d_fill, d_normals,
                                          d_force_marangoni_x_, d_force_marangoni_y_, d_force_marangoni_z_);
    } else {
        cudaMemset(d_force_marangoni_x_, 0, num_cells_ * sizeof(float));
        cudaMemset(d_force_marangoni_y_, 0, num_cells_ * sizeof(float));
        cudaMemset(d_force_marangoni_z_, 0, num_cells_ * sizeof(float));
    }

    // 4. Sum all forces
    sumForcesKernel<<<grid, block>>>(
        d_force_buoyancy_x_, d_force_buoyancy_y_, d_force_buoyancy_z_,
        d_force_surface_x_, d_force_surface_y_, d_force_surface_z_,
        d_force_marangoni_x_, d_force_marangoni_y_, d_force_marangoni_z_,
        d_force_total_x_, d_force_total_y_, d_force_total_z_,
        num_cells_);
}

__global__ void sumForcesKernel(
    const float* f_bx, const float* f_by, const float* f_bz,
    const float* f_sx, const float* f_sy, const float* f_sz,
    const float* f_mx, const float* f_my, const float* f_mz,
    float* f_tx, float* f_ty, float* f_tz,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    f_tx[idx] = f_bx[idx] + f_sx[idx] + f_mx[idx];
    f_ty[idx] = f_by[idx] + f_sy[idx] + f_my[idx];
    f_tz[idx] = f_bz[idx] + f_sz[idx] + f_mz[idx];
}
```

**Advantages**:
- Can monitor individual force components (for validation)
- Easy to disable forces (set to zero)
- Debugging: visualize each force separately

---

### 4.2 Buoyancy Force Implementation

```cpp
void MultiphysicsSolver::computeBuoyancyForce(const float* d_temperature,
                                               const float* d_fill_level,
                                               float* d_fx, float* d_fy, float* d_fz)
{
    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
              (ny_ + block.y - 1) / block.y,
              (nz_ + block.z - 1) / block.z);

    float T_ref = config_.material.T_liquidus;  // Reference temperature
    float rho_ref = config_.material.rho_liquid;
    float beta = /* thermal expansion coefficient */;  // Estimate: 1e-4 K⁻¹ for metals
    float g = 9.81f;  // m/s²

    buoyancyForceKernel<<<grid, block>>>(
        d_temperature, d_fill_level,
        d_fx, d_fy, d_fz,
        T_ref, rho_ref, beta, g,
        nx_, ny_, nz_);

    cudaDeviceSynchronize();
}

__global__ void buoyancyForceKernel(
    const float* temperature,
    const float* fill_level,
    float* force_x, float* force_y, float* force_z,
    float T_ref, float rho_ref, float beta, float g,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    float f = fill_level[idx];

    // Only in liquid regions
    if (f < 0.5f) {
        force_x[idx] = 0.0f;
        force_y[idx] = 0.0f;
        force_z[idx] = 0.0f;
        return;
    }

    float T = temperature[idx];
    float dT = T - T_ref;

    // Boussinesq approximation: F = -ρ g β ΔT ẑ
    // (negative sign: hot fluid rises, cold sinks)
    force_x[idx] = 0.0f;
    force_y[idx] = 0.0f;
    force_z[idx] = -rho_ref * g * beta * dT;  // Vertical only (z-direction)
}
```

**Note**: Thermal expansion coefficient β typically not stored in MaterialProperties.
**Estimate for Ti6Al4V**: β ≈ 1×10⁻⁴ K⁻¹ (order of magnitude, not critical for validation)

---

## 5. Main Integration Loop

```cpp
void MultiphysicsSolver::step(float dt) {
    // Use adaptive timestep if dt <= 0
    if (dt <= 0.0f) {
        dt = computeTimeStep();
    }

    // 1. Thermal evolution (includes laser heating + phase change)
    thermal_solver_->evolve(dt);

    // 2. Reconstruct interface (updates normals and curvature)
    vof_solver_->reconstructInterface();

    // 3. Compute all driving forces
    computeForces();

    // 4. Evolve fluid velocity with total force
    fluid_solver_->setBodyForce(d_force_total_x_, d_force_total_y_, d_force_total_z_);
    fluid_solver_->evolve(dt);

    // 5. Advect VOF with updated velocity (subcycling for stability)
    const float* d_velocity = fluid_solver_->getVelocityDevicePtr();

    int n_subcycles = config_.vof_subcycles;
    float dt_vof = dt / static_cast<float>(n_subcycles);

    for (int sub = 0; sub < n_subcycles; ++sub) {
        vof_solver_->advect(d_velocity, dt_vof);
    }

    // 6. Update time
    current_time_ += dt;
    current_dt_ = dt;
    step_count_++;
}
```

**Subcycling rationale**:
- VOF advection often requires smaller timestep than thermal/fluid
- CFL for VOF: v dt / dx < 0.5 (stricter than momentum CFL ~ 0.3)
- Subcycling avoids limiting entire simulation to VOF constraint

---

## 6. Validation Metrics Extraction

### 6.1 Melt Pool Geometry

```cpp
void MultiphysicsSolver::extractMeltPoolGeometry(float& width, float& depth, float& length) const {
    // Copy temperature to host
    std::vector<float> h_temp = thermal_solver_->getTemperatureFieldHost();

    float T_melt = config_.material.T_liquidus;

    // Find all cells above melting point
    std::vector<int> melt_cells;
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                int idx = i + nx_ * (j + ny_ * k);
                if (h_temp[idx] > T_melt) {
                    melt_cells.push_back(idx);
                }
            }
        }
    }

    if (melt_cells.empty()) {
        width = depth = length = 0.0f;
        return;
    }

    // Compute bounding box
    int i_min = nx_, i_max = 0;
    int j_min = ny_, j_max = 0;
    int k_min = nz_, k_max = 0;

    for (int idx : melt_cells) {
        int i = idx % nx_;
        int j = (idx / nx_) % ny_;
        int k = idx / (nx_ * ny_);

        i_min = std::min(i_min, i);
        i_max = std::max(i_max, i);
        j_min = std::min(j_min, j);
        j_max = std::max(j_max, j);
        k_min = std::min(k_min, k);
        k_max = std::max(k_max, k);
    }

    // Dimensions
    width = (j_max - j_min + 1) * dx_;   // Transverse (y-direction)
    length = (i_max - i_min + 1) * dx_;  // Scan direction (x-direction)
    depth = (k_max - k_min + 1) * dx_;   // Depth (z-direction)
}
```

---

### 6.2 Surface Velocity

```cpp
float MultiphysicsSolver::extractMaxSurfaceVelocity() const {
    // Get fields
    std::vector<float> h_fill = vof_solver_->getFillLevelHost();
    std::vector<float> h_vx, h_vy, h_vz;
    fluid_solver_->getVelocityFieldHost(h_vx, h_vy, h_vz);

    float v_max_surface = 0.0f;

    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                int idx = i + nx_ * (j + ny_ * k);

                // Interface cells
                if (h_fill[idx] > 0.1f && h_fill[idx] < 0.9f) {
                    float v_mag = sqrtf(h_vx[idx]*h_vx[idx] +
                                       h_vy[idx]*h_vy[idx] +
                                       h_vz[idx]*h_vz[idx]);

                    v_max_surface = std::max(v_max_surface, v_mag);
                }
            }
        }
    }

    return v_max_surface;
}
```

---

### 6.3 Force Ratio (Marangoni / Buoyancy)

```cpp
float MultiphysicsSolver::computeMarangoniBuoyancyRatio() const {
    // Copy force fields to host
    std::vector<float> h_fm_x(num_cells_), h_fm_y(num_cells_), h_fm_z(num_cells_);
    std::vector<float> h_fb_x(num_cells_), h_fb_y(num_cells_), h_fb_z(num_cells_);

    cudaMemcpy(h_fm_x.data(), d_force_marangoni_x_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fm_y.data(), d_force_marangoni_y_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fm_z.data(), d_force_marangoni_z_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_fb_x.data(), d_force_buoyancy_x_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fb_y.data(), d_force_buoyancy_y_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fb_z.data(), d_force_buoyancy_z_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute average magnitudes (only where non-zero)
    double sum_marangoni = 0.0, sum_buoyancy = 0.0;
    int count_m = 0, count_b = 0;

    for (int i = 0; i < num_cells_; ++i) {
        float mag_m = sqrtf(h_fm_x[i]*h_fm_x[i] + h_fm_y[i]*h_fm_y[i] + h_fm_z[i]*h_fm_z[i]);
        float mag_b = sqrtf(h_fb_x[i]*h_fb_x[i] + h_fb_y[i]*h_fb_y[i] + h_fb_z[i]*h_fb_z[i]);

        if (mag_m > 1e-6f) {
            sum_marangoni += mag_m;
            count_m++;
        }

        if (mag_b > 1e-6f) {
            sum_buoyancy += mag_b;
            count_b++;
        }
    }

    float avg_marangoni = (count_m > 0) ? (sum_marangoni / count_m) : 0.0f;
    float avg_buoyancy = (count_b > 0) ? (sum_buoyancy / count_b) : 1e-10f;  // Avoid division by zero

    return avg_marangoni / avg_buoyancy;
}
```

---

## 7. Error Handling and Stability Monitoring

### 7.1 NaN Detection

```cpp
bool MultiphysicsSolver::checkStability() const {
    // Check for NaN or Inf in critical fields

    auto check_field = [](const float* d_field, int N, const char* name) -> bool {
        std::vector<float> h_field(N);
        cudaMemcpy(h_field.data(), d_field, N * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; ++i) {
            if (std::isnan(h_field[i]) || std::isinf(h_field[i])) {
                std::cerr << "ERROR: NaN/Inf detected in " << name << " at index " << i << "\n";
                return false;
            }
        }
        return true;
    };

    bool stable = true;
    stable &= check_field(thermal_solver_->getTemperatureDevicePtr(), num_cells_, "temperature");
    stable &= check_field(fluid_solver_->getVelocityXDevicePtr(), num_cells_, "velocity_x");
    stable &= check_field(vof_solver_->getFillLevel(), num_cells_, "fill_level");

    return stable;
}
```

**Use in main loop**:
```cpp
void MultiphysicsSolver::evolveUntil(float t_end) {
    while (current_time_ < t_end) {
        step();

        // Check stability every 100 steps
        if (step_count_ % 100 == 0) {
            if (!checkStability()) {
                throw std::runtime_error("Simulation became unstable at t = " +
                                         std::to_string(current_time_));
            }
        }
    }
}
```

---

### 7.2 Mass Conservation Monitoring

```cpp
float MultiphysicsSolver::checkMassConservation() const {
    float mass_current = vof_solver_->computeTotalMass();

    if (initial_mass_ < 0.0f) {
        // First call: store initial mass
        initial_mass_ = mass_current;
        return 0.0f;
    }

    float error = fabsf(mass_current - initial_mass_) / initial_mass_;
    return error;
}
```

**Acceptance criterion**: Mass error < 1% (you have 0.2%, excellent!)

---

## 8. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `/home/yzk/LBMProject/include/physics/multiphysics_solver.h`
- [ ] Create `/home/yzk/LBMProject/src/physics/multiphysics_solver.cu`
- [ ] Implement constructor (initialize component solvers)
- [ ] Implement destructor (cleanup device memory)
- [ ] Implement `computeTimeStep()` with all stability criteria
- [ ] Unit test: Verify timestep computation logic

### Phase 2: Force Handling
- [ ] Allocate force field device memory
- [ ] Implement `computeBuoyancyForce()`
- [ ] Implement `computeForces()` (calls all force modules)
- [ ] Implement `sumForcesKernel()`
- [ ] Unit test: Verify force magnitudes are reasonable

### Phase 3: Integration Loop
- [ ] Implement `step(float dt)`
- [ ] Implement `evolveUntil(float t_end)`
- [ ] Add timestep info logging
- [ ] Test: Run 10 steps with zero forces, verify stability

### Phase 4: Validation Metrics
- [ ] Implement `extractMeltPoolGeometry()`
- [ ] Implement `extractMaxSurfaceVelocity()`
- [ ] Implement `computeMarangoniBuoyancyRatio()`
- [ ] Implement `extractMetrics()` (aggregate all)
- [ ] Unit test: Check metrics extraction on known fields

### Phase 5: Stability and I/O
- [ ] Implement `checkStability()` (NaN detection)
- [ ] Implement `checkMassConservation()`
- [ ] Implement `writeVTK()` (visualization output)
- [ ] Add progress logging (every N steps)

### Phase 6: Test 2C Integration
- [ ] Create `/home/yzk/LBMProject/tests/validation/test_marangoni_velocity_benchmark.cu`
- [ ] Run Test 2C (simplified melt pool)
- [ ] Debug if v < 0.1 m/s
- [ ] Iterate until 0.5 < v < 2 m/s

### Phase 7: Full LPBF
- [ ] Create `/home/yzk/LBMProject/tests/validation/test_lpbf_benchmark.cu`
- [ ] Implement laser scanning (moving heat source)
- [ ] Run full LPBF (200W, 1 m/s)
- [ ] Compare all metrics with literature

---

## 9. Performance Estimates

### 9.1 Computational Cost per Step

**Domain**: 200×150×100 = 3M cells, dx = 2 μm

**Per-step operations**:
```
ThermalLBM: ~14 memory accesses per cell (D3Q7 collision + streaming)
FluidLBM: ~38 memory accesses per cell (D3Q19 collision + streaming)
VOF advection: ~20 memory accesses per cell (geometric reconstruction)
Forces: ~15 memory accesses per cell (gradients + force computation)

Total: ~87 memory accesses × 3M cells = 261M memory ops
With 4 bytes/float: ~1 GB memory traffic per step
```

**GPU bandwidth** (RTX 3090): ~900 GB/s

**Time per step**: 1 GB / 900 GB/s ≈ 1.1 ms

**With dt = 0.1 μs**: 300 μs simulation requires 3000 steps → 3.3 seconds

**Actual runtime** (with kernel launch overhead): ~5-10 seconds expected

**Full LPBF (300 μs simulation)**: 1-2 hours total (including I/O)

---

### 9.2 Memory Usage

**Peak memory**: ~550 MB (calculated earlier)

**GPU requirements**: 2+ GB VRAM (for safety margin)

**Supported GPUs**: GTX 1660 Ti and above

---

## 10. Future Enhancements (Post-Phase 6)

### 10.1 Semi-Implicit Surface Tension

**Motivation**: Relax capillary timestep constraint

**Method**: Treat curvature implicitly
```
dt_capillary (explicit): 0.2 ns (too small!)
dt_capillary (implicit): 10 ns (100× improvement)
```

**Reference**: Popinet (2009), Denner & van Wachem (2015)

---

### 10.2 Adaptive Mesh Refinement

**Motivation**: Resolve interface (1 μm) while keeping bulk coarse (10 μm)

**Strategy**: Octree-based refinement (like walberla's blockforest)

**Expected speedup**: 5-10× for same accuracy

---

### 10.3 Multi-GPU Scaling

**Method**: Domain decomposition with MPI

**Challenge**: Load balancing (melt pool moves!)

**Benefit**: Simulate larger domains (mm-scale, multi-track)

---

## 11. References

**walberla Design Patterns**:
- Operator splitting (separate collision/streaming/forces)
- Field accessors (device pointer getters)
- Block-structured grids (for future AMR)

**Literature**:
- Khairallah et al. (2016) - Marangoni velocity benchmarks
- Brackbill et al. (1992) - CSF method for surface tension
- Hirt & Nichols (1981) - VOF method
- Sussman & Puckett (2000) - Level set methods

**Codebase**:
- `/home/yzk/LBMProject/LPBF_Validation_Quick_Reference.md` - Validation targets
- `/home/yzk/LBMProject/src/physics/vof/marangoni.cu` - Force implementation
- `/home/yzk/walberla/src/lbm/` - walberla LBM reference

---

**END OF SPECIFICATION**
