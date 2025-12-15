# LBM-CUDA Metal Melting Simulation Framework Architecture

## Executive Summary

This document defines the architecture for a modular, GPU-accelerated Lattice Boltzmann Method (LBM) framework specifically designed for simulating laser heating and melting of solid metals. The framework targets metal additive manufacturing processes including LPBF (Laser Powder Bed Fusion) and DED (Directed Energy Deposition).

Based on recent research (2024-2025) and the proven walberla framework architecture, this design emphasizes:
- **Modular block-structured domain decomposition**
- **GPU-first parallel computing strategy**
- **Enthalpy-based phase change model for high accuracy**
- **Marangoni convection modeling (critical for metal AM)**
- **Extensible physics coupling framework**

---

## 1. Core Modules and Responsibilities

### 1.1 LBM Core Module (`lbm_core/`)
**Responsibility**: Fundamental LBM operations and collision operators

- **Collision Operators**:
  - BGK (Bhatnagar-Gross-Krook) for basic flow
  - MRT (Multiple Relaxation Time) for improved stability
  - Block Triple-Relaxation-Time (B-TriRT) for phase interfaces
  - Cumulant-based for high Reynolds numbers

- **Streaming Operations**:
  - Pull/Push streaming kernels
  - Directional interpolation for accuracy
  - Optimized memory access patterns

- **Lattice Structures**:
  - D2Q9 (2D, 9 velocities) for testing
  - D3Q19 (3D, 19 velocities) standard
  - D3Q27 (3D, 27 velocities) high accuracy

### 1.2 Thermal Module (`thermal/`)
**Responsibility**: Heat transfer and thermal dynamics

- **Heat Conduction Solver**:
  - Thermal LBM with adjustable Prandtl number
  - Enthalpy-based formulation for phase change
  - Temperature-dependent material properties

- **Laser Heat Source**:
  - Ray tracing for laser-material interaction
  - Gaussian/Top-hat beam profiles
  - Multiple reflection handling
  - Absorption coefficient models

- **Heat Transfer Mechanisms**:
  - Conduction in solid/liquid phases
  - Convective heat transfer
  - Radiation heat loss
  - Evaporative cooling

### 1.3 Phase Change Module (`phase_change/`)
**Responsibility**: Solid-liquid phase transitions

- **Phase Tracking Methods**:
  - Enthalpy-based LBM (ELBM) - primary method
  - Phase-field LBM (PFLBM) - for interface dynamics
  - Immersed boundary LBM (IBLBM) - for complex geometries

- **Physical Models**:
  - Latent heat handling
  - Mushy zone treatment
  - Volume change during phase transition
  - Nucleation and growth kinetics

- **Material Database**:
  - Temperature-dependent properties
  - Phase diagram data
  - Thermophysical properties (Ti6Al4V, 316L SS, Inconel, etc.)

### 1.4 Fluid Dynamics Module (`fluid/`)
**Responsibility**: Liquid metal flow and surface phenomena

- **Flow Solvers**:
  - Incompressible/weakly compressible LBM
  - Free surface tracking (VOF/Level-set)
  - Multiphase flow handling

- **Surface Effects**:
  - Surface tension modeling
  - Marangoni convection (critical for metal AM)
  - Wetting dynamics
  - Contact angle models

- **Boundary Conditions**:
  - No-slip/free-slip walls
  - Pressure/velocity boundaries
  - Moving boundary treatment
  - Heat flux boundaries

### 1.5 GPU Computing Module (`gpu/`)
**Responsibility**: CUDA acceleration and parallel execution

- **Memory Management**:
  - Unified memory for CPU-GPU transfer
  - Pinned memory for async operations
  - Structure-of-Arrays (SoA) layout
  - Memory pooling and reuse

- **Kernel Organization**:
  - Fused kernels for collision-streaming
  - Grid-stride loops for flexibility
  - Shared memory optimization
  - Warp-level primitives

- **Multi-GPU Support**:
  - Domain decomposition strategies
  - Ghost cell communication
  - CUDA-aware MPI
  - Load balancing

### 1.6 Domain Management Module (`domain/`)
**Responsibility**: Computational domain and block structure

- **Block Structure** (walberla-inspired):
  - Uniform cubic blocks (256³ cells typical)
  - Ghost layer management
  - Adaptive mesh refinement (AMR) support
  - Load balancing strategies

- **Geometry Handling**:
  - STL import for complex geometries
  - Voxelization routines
  - Sparse matrix representation
  - Boundary mapping

### 1.7 Time Integration Module (`time_integration/`)
**Responsibility**: Temporal evolution and multi-physics coupling

- **Time Stepping**:
  - Explicit time integration
  - Adaptive time stepping
  - Subcycling for different physics
  - Stability monitoring

- **Coupling Strategies**:
  - Operator splitting for multi-physics
  - Strong coupling for critical interactions
  - Interpolation between different time scales

### 1.8 I/O and Visualization Module (`io/`)
**Responsibility**: Data input/output and visualization

- **Input Formats**:
  - Configuration files (YAML/JSON)
  - Material property databases
  - Geometry files (STL, VTK)
  - Restart checkpoints

- **Output Formats**:
  - VTK/ParaView compatible
  - HDF5 for large datasets
  - Real-time monitoring data
  - Statistical analysis outputs

- **In-situ Visualization**:
  - Temperature field rendering
  - Velocity vectors
  - Phase distribution
  - Melt pool geometry

### 1.9 Validation Module (`validation/`)
**Responsibility**: Verification and validation utilities

- **Benchmark Cases**:
  - Lid-driven cavity
  - Stefan problem (melting)
  - Marangoni convection test
  - Laser melting experiments

- **Analysis Tools**:
  - Conservation checks
  - Error norms
  - Convergence analysis
  - Performance metrics

---

## 2. Code Architecture Design

### 2.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  (Simulation runners, parameter studies, optimization)   │
├─────────────────────────────────────────────────────────┤
│                    Physics Coupling Layer                │
│  (Multi-physics coordinator, time stepping control)      │
├─────────────────────────────────────────────────────────┤
│                    Physics Modules Layer                 │
│  (Thermal, Phase Change, Fluid, Laser modules)          │
├─────────────────────────────────────────────────────────┤
│                    LBM Core Layer                        │
│  (Collision, Streaming, Lattice structures)             │
├─────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                  │
│  (GPU kernels, Memory management, Domain decomposition)  │
├─────────────────────────────────────────────────────────┤
│                    Hardware Abstraction Layer            │
│  (CUDA/HIP, MPI, CPU fallback)                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Design

```
Input Parameters → Domain Setup → Initial Conditions
                                         ↓
                              ┌──────────────────────┐
                              │   Time Step Loop     │
                              │                      │
                              │  1. Laser Heating    │
                              │  2. Thermal Diffusion│
                              │  3. Phase Change     │
                              │  4. Fluid Flow       │
                              │  5. LBM Collision    │
                              │  6. LBM Streaming    │
                              │  7. Boundary Update  │
                              │                      │
                              └──────────────────────┘
                                         ↓
                            Output/Visualization → Analysis
```

### 2.3 GPU Parallelization Strategy

**Block-Level Parallelism**:
- Each CUDA block handles a subdomain (e.g., 32×32×32 cells)
- Ghost cells for inter-block communication
- Coalesced memory access patterns

**Kernel Organization**:
```cuda
// Fused collision-streaming kernel
__global__ void collisionStreamingKernel(
    float* f_src,      // Source distribution functions
    float* f_dst,      // Destination distribution functions
    float* temperature,// Temperature field
    float* phase,      // Phase field (0=solid, 1=liquid)
    LaserParams laser, // Laser parameters
    MaterialProps mat, // Material properties
    int nx, int ny, int nz
);

// Thermal evolution kernel
__global__ void thermalKernel(
    float* temperature,
    float* enthalpy,
    float* phase,
    float* heat_source,
    MaterialProps mat,
    float dt
);

// Marangoni convection kernel
__global__ void marangoniKernel(
    float* velocity,
    float* temperature,
    float* surface_tension_grad,
    int nx, int ny, int nz
);
```

### 2.4 Module Interaction Design

```
LaserModule ──────► ThermalModule ─────► PhaseChangeModule
                           │                     │
                           ▼                     ▼
                    FluidModule ◄──────── MarangoniModule
                           │
                           ▼
                      LBMCore ◄───────────► GPUModule
                           │
                           ▼
                    DomainModule ────────► IOModule
```

---

## 3. File/Directory Structure

```
/home/yzk/LBMProject/
│
├── README.md                    # Project overview
├── ARCHITECTURE.md              # This document
├── CLAUDE.md                    # Development guidelines
├── CMakeLists.txt              # Root CMake configuration
├── LICENSE                     # License information
│
├── src/                        # Source code
│   ├── core/                   # Core LBM implementations
│   │   ├── lattice/           # Lattice structures (D2Q9, D3Q19, D3Q27)
│   │   ├── collision/         # Collision operators
│   │   ├── streaming/         # Streaming operations
│   │   ├── boundary/          # Boundary conditions
│   │   └── CMakeLists.txt
│   │
│   ├── physics/               # Physics modules
│   │   ├── thermal/          # Heat transfer
│   │   │   ├── conduction.cu
│   │   │   ├── convection.cu
│   │   │   ├── radiation.cu
│   │   │   └── laser_source.cu
│   │   │
│   │   ├── phase_change/     # Phase transitions
│   │   │   ├── enthalpy_method.cu
│   │   │   ├── phase_field.cu
│   │   │   ├── mushy_zone.cu
│   │   │   └── nucleation.cu
│   │   │
│   │   ├── fluid/            # Fluid dynamics
│   │   │   ├── navier_stokes.cu
│   │   │   ├── free_surface.cu
│   │   │   ├── marangoni.cu
│   │   │   └── surface_tension.cu
│   │   │
│   │   └── materials/        # Material properties
│   │       ├── material_database.h
│   │       ├── ti6al4v.cpp
│   │       ├── ss316l.cpp
│   │       └── inconel718.cpp
│   │
│   ├── gpu/                   # GPU-specific code
│   │   ├── kernels/          # CUDA kernels
│   │   │   ├── lbm_kernels.cu
│   │   │   ├── thermal_kernels.cu
│   │   │   ├── phase_kernels.cu
│   │   │   └── utility_kernels.cu
│   │   │
│   │   ├── memory/           # Memory management
│   │   │   ├── gpu_allocator.cu
│   │   │   ├── memory_pool.cu
│   │   │   └── transfer.cu
│   │   │
│   │   └── multi_gpu/        # Multi-GPU support
│   │       ├── decomposition.cu
│   │       ├── communication.cu
│   │       └── load_balance.cu
│   │
│   ├── domain/                # Domain management
│   │   ├── block_lattice.h
│   │   ├── block_manager.cpp
│   │   ├── geometry.cpp
│   │   ├── mesh_refinement.cpp
│   │   └── ghost_layer.cpp
│   │
│   ├── io/                    # Input/Output
│   │   ├── config_parser.cpp
│   │   ├── vtk_writer.cpp
│   │   ├── hdf5_handler.cpp
│   │   ├── checkpoint.cpp
│   │   └── logger.cpp
│   │
│   ├── coupling/              # Multi-physics coupling
│   │   ├── coupler.h
│   │   ├── time_stepper.cpp
│   │   ├── interpolation.cpp
│   │   └── operator_splitting.cpp
│   │
│   └── utils/                 # Utilities
│       ├── math_functions.h
│       ├── constants.h
│       ├── timer.cpp
│       └── error_handler.cpp
│
├── include/                    # Public headers
│   ├── lbm_metal_melting.h   # Main API header
│   └── config.h.in            # Configuration template
│
├── apps/                      # Application examples
│   ├── laser_melting/        # Basic laser melting
│   ├── lpbf_simulation/      # LPBF process
│   ├── ded_simulation/       # DED process
│   └── benchmark/            # Benchmark cases
│
├── tests/                     # Unit and integration tests
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── validation/           # Validation cases
│   └── performance/          # Performance benchmarks
│
├── python/                    # Python bindings and tools
│   ├── pybind/              # Python bindings
│   ├── preprocessing/        # Preprocessing scripts
│   ├── postprocessing/       # Analysis tools
│   └── visualization/        # Visualization scripts
│
├── docs/                      # Documentation
│   ├── api/                  # API documentation
│   ├── theory/               # Theory and equations
│   ├── tutorials/            # User tutorials
│   └── developer/            # Developer guide
│
├── external/                  # External dependencies
│   ├── cub/                  # CUB library
│   ├── thrust/               # Thrust library
│   └── json/                 # JSON parser
│
├── benchmarks/                # Benchmark data
│   ├── stefan_problem/       # Analytical solutions
│   ├── marangoni_flow/       # Reference data
│   └── experimental/         # Experimental validation
│
├── scripts/                   # Build and utility scripts
│   ├── build.sh
│   ├── run_tests.sh
│   ├── profile_gpu.sh
│   └── docker/               # Docker configurations
│
└── config/                    # Configuration files
    ├── default.yaml          # Default parameters
    ├── materials.yaml        # Material database
    └── laser_profiles.yaml   # Laser configurations
```

---

## 4. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)
**Priority: CRITICAL**

1. **Basic LBM Core**
   - Implement D3Q19 lattice structure
   - BGK collision operator
   - Basic streaming (pull scheme)
   - Simple boundary conditions (bounce-back)
   - Single GPU implementation

2. **Domain Infrastructure**
   - Block-structured domain setup
   - Ghost layer management
   - Basic geometry import (rectangular domains)
   - Memory allocation framework

3. **Minimal I/O**
   - Configuration file parser (YAML)
   - VTK output for ParaView
   - Basic logging system

**Deliverable**: Working LBM solver for isothermal flow on GPU

### Phase 2: Thermal Physics (Weeks 4-6)
**Priority: HIGH**

1. **Thermal LBM Implementation**
   - Double distribution function approach
   - Temperature evolution solver
   - Heat conduction with variable conductivity
   - Thermal boundary conditions

2. **Laser Heat Source**
   - Gaussian beam implementation
   - Surface absorption model
   - Heat source distribution
   - Time-dependent power control

3. **Material Properties**
   - Temperature-dependent properties
   - Database for common metals (Ti6Al4V, 316L)
   - Property interpolation routines

**Deliverable**: Laser heating simulation without phase change

### Phase 3: Phase Change (Weeks 7-9)
**Priority: HIGH**

1. **Enthalpy Method**
   - Total enthalpy formulation
   - Latent heat handling
   - Mushy zone model
   - Solid-liquid interface tracking

2. **Phase-Dependent Properties**
   - Viscosity variation with phase
   - Thermal property switching
   - Momentum damping in solid

3. **Validation**
   - 1D Stefan problem
   - 2D melting cavity
   - Comparison with analytical solutions

**Deliverable**: Complete melting/solidification capability

### Phase 4: Advanced Fluid Dynamics (Weeks 10-12)
**Priority: MEDIUM-HIGH**

1. **Free Surface Tracking**
   - Volume of Fluid (VOF) method
   - Interface reconstruction
   - Surface normal calculation

2. **Marangoni Convection**
   - Temperature-dependent surface tension
   - Surface gradient calculation
   - Tangential stress implementation

3. **Surface Tension Effects**
   - Young-Laplace pressure
   - Contact angle model
   - Wetting dynamics

**Deliverable**: Realistic melt pool dynamics with Marangoni flow

### Phase 5: Optimization & Multi-GPU (Weeks 13-15)
**Priority: MEDIUM**

1. **GPU Optimization**
   - Kernel fusion (collision-streaming)
   - Shared memory optimization
   - Memory access pattern optimization
   - Async operations and streams

2. **Multi-GPU Support**
   - Domain decomposition
   - MPI communication layer
   - Load balancing
   - Scalability testing

3. **Performance Tuning**
   - Profiling with nvprof/Nsight
   - Bottleneck identification
   - Parameter tuning
   - Benchmark suite

**Deliverable**: Optimized, scalable multi-GPU implementation

### Phase 6: Applications & Validation (Weeks 16-18)
**Priority: MEDIUM**

1. **LPBF Simulation**
   - Powder bed representation
   - Multiple laser passes
   - Layer-by-layer building
   - Defect prediction

2. **DED Simulation**
   - Powder feed modeling
   - Material deposition
   - Track overlap
   - Substrate preheating

3. **Experimental Validation**
   - Melt pool dimensions
   - Temperature history
   - Solidification rate
   - Microstructure prediction

**Deliverable**: Validated AM process simulations

---

## 5. Technical Recommendations

### 5.1 Critical Physics Models

Based on 2024-2025 research, the following physics are essential:

1. **Marangoni Convection** (CRITICAL for metal AM)
   - Primary driving force in melt pool dynamics
   - Use temperature-dependent surface tension coefficient
   - Implement tangential stress boundary condition
   - Consider compositional Marangoni effects for alloys

2. **Enthalpy-Based Phase Change** (RECOMMENDED)
   - Superior numerical stability
   - Handles latent heat naturally
   - Reduced numerical diffusion at interface
   - Compatible with LBM framework

3. **Block Triple-Relaxation-Time (B-TriRT)** (ADVANCED)
   - Minimizes numerical diffusion at phase interface
   - Better accuracy for solidification fronts
   - Recent 2024 development showing promise

### 5.2 GPU Optimization Strategies

1. **Memory Layout**
   ```cuda
   // Structure of Arrays (SoA) - RECOMMENDED
   struct DistributionFunctions {
       float* f0;  // All f0 values contiguous
       float* f1;  // All f1 values contiguous
       // ... f2-f18
   };

   // NOT Array of Structures (AoS)
   struct Cell {
       float f[19];  // Poor coalescing
   };
   ```

2. **Kernel Fusion**
   - Combine collision-streaming into single kernel
   - Reduce memory bandwidth requirements
   - Minimize kernel launch overhead

3. **Warp Efficiency**
   - Ensure 32-thread warps are fully utilized
   - Minimize branch divergence
   - Use warp-level primitives (__shfl_sync)

### 5.3 Numerical Considerations

1. **Stability Criteria**
   - Maintain τ > 0.5 + δt/2 for BGK stability
   - Monitor Mach number (Ma < 0.1 for incompressible)
   - Check CFL condition for thermal diffusion
   - Adaptive time stepping for phase change

2. **Accuracy Requirements**
   - Double precision for phase interface tracking
   - Single precision acceptable for bulk flow
   - Mixed precision strategy for optimization

3. **Conservation Properties**
   - Ensure mass conservation to machine precision
   - Monitor energy conservation in phase change
   - Check momentum conservation in free surface

### 5.4 Material-Specific Considerations

**For Ti6Al4V (Common in aerospace AM):**
- High temperature gradient sensitivity
- Strong Marangoni effects
- Rapid solidification rates
- Alpha-beta phase transitions

**For 316L Stainless Steel:**
- Lower thermal conductivity
- Different surface tension behavior
- Austenitic solidification
- Hot cracking susceptibility

**For Inconel 718:**
- Complex solidification behavior
- Precipitation hardening considerations
- High temperature strength
- Laves phase formation

### 5.5 Validation Strategy

1. **Unit Testing Hierarchy**
   - Individual kernel correctness
   - Module integration tests
   - Multi-physics coupling tests
   - Full system validation

2. **Benchmark Cases** (in order of complexity)
   - Poiseuille flow (basic LBM validation)
   - Lid-driven cavity (fluid dynamics)
   - 1D Stefan problem (phase change)
   - Marangoni convection in cavity
   - Static laser spot melting
   - Moving laser track
   - Multi-track overlap
   - Full LPBF/DED layer

3. **Experimental Validation Data**
   - High-speed camera melt pool geometry
   - Thermocouple temperature history
   - Cross-section metallography
   - In-situ X-ray imaging data

---

## 6. Risk Mitigation

### Technical Risks

1. **Numerical Instability at Phase Interface**
   - Mitigation: Implement B-TriRT or adaptive relaxation
   - Fallback: Use implicit phase change treatment

2. **GPU Memory Limitations**
   - Mitigation: Implement domain decomposition early
   - Use unified memory for overflow
   - Consider out-of-core algorithms

3. **Marangoni Instability**
   - Mitigation: Careful treatment of surface gradients
   - Use flux limiters if needed
   - Validate against analytical solutions

### Performance Risks

1. **Memory Bandwidth Bottleneck**
   - Mitigation: Kernel fusion, SoA layout
   - Monitor bandwidth utilization
   - Consider compression techniques

2. **Multi-GPU Scaling**
   - Mitigation: Minimize communication
   - Overlap computation and communication
   - Use CUDA-aware MPI

---

## 7. Integration with walberla Patterns

Based on walberla's proven architecture:

### Adopt:
- Block-structured domain (256³ cells per block)
- Ghost layer communication pattern
- Unified CPU/GPU code base approach
- Python-based code generation for kernels
- HDF5 for checkpointing

### Adapt:
- Modify block size for metal AM (smaller blocks for AMR)
- Extend boundary handling for free surfaces
- Add thermal-specific ghost cell updates
- Customize for single-phase to two-phase transitions

### Avoid:
- Over-generalization for early versions
- Complex template metaprogramming initially
- Full multigrid (not needed for explicit LBM)

---

## 8. Success Metrics

### Performance Targets:
- Single GPU: >90% bandwidth utilization
- Multi-GPU: >80% weak scaling efficiency
- Time to solution: <1 hour for single laser track

### Accuracy Targets:
- Melt pool dimensions: ±10% of experimental
- Peak temperature: ±50K of measurements
- Solidification rate: ±20% of analytical

### Software Quality:
- Unit test coverage: >80%
- Documentation: All public APIs documented
- Code review: All merges reviewed
- Performance regression: <5% per release

---

## Conclusion

This architecture provides a solid foundation for developing a state-of-the-art LBM-CUDA framework for metal melting simulations. The modular design ensures extensibility, while the GPU-first approach guarantees performance. By incorporating recent research advances (2024-2025) and learning from walberla's proven patterns, this framework will enable accurate and efficient simulation of laser-based metal additive manufacturing processes.

The phased implementation approach allows for incremental development with regular validation milestones, reducing technical risk while maintaining scientific rigor. The emphasis on Marangoni convection and enthalpy-based phase change reflects the latest understanding of the critical physics in metal AM processes.