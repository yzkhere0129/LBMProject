# Validation Test Configuration Templates

**Location**: `/home/yzk/LBMProject/config/validation/`
**Purpose**: Standardized JSON configuration files for group meeting validation cases
**Created**: 2026-01-13
**Status**: Ready for implementation

---

## Overview

This directory contains four standardized JSON configuration templates that define the validation test cases discussed in the group meeting. These templates serve as:

1. **Test specifications** - Complete parameter definitions for reproducible testing
2. **Documentation** - Self-documenting configuration with inline comments and references
3. **Implementation guides** - Details on physics models, numerical methods, and validation criteria
4. **Comparison baselines** - Expected results and success criteria for each test

---

## Configuration Files

### 1. test_3d_heat_diffusion.json

**Category**: Pure Thermal Diffusion
**Difficulty**: Basic
**Runtime**: 2-5 minutes
**Physics**: Thermal conduction only

**Purpose**: Validates the thermal LBM solver against analytical Gaussian diffusion solution.

**Key Parameters**:
- Domain: 9.5×9.5×9.5 mm (51×51×51 grid)
- Initial: Gaussian temperature pulse (Q = 1000 J)
- BC: Dirichlet T_wall = 273.13 K
- Material: ρ=1000, c_p=4186, k=1 (water-like)
- Duration: 0.1 s
- Analytical solution: 3D Gaussian diffusion

**Success Criteria**:
- L2 error < 5% at all output times
- Peak temperature matches analytical within 5%

**Reference**: Senior's Thesis Chapter 4, Section 4.2.1

---

### 2. test_couette_poiseuille.json

**Category**: Fluid Mechanics
**Difficulty**: Intermediate
**Runtime**: 5-10 minutes
**Physics**: Fluid dynamics (LBM D3Q19)

**Purpose**: Validates fluid solver with combined shear-driven (Couette) and pressure-driven (Poiseuille) flow.

**Key Parameters**:
- Domain: 1×1 m channel (64×128 grid)
- BC: Top plate U_top = 1 m/s, bottom no-slip
- Re = 100
- Body force: f_x = -6/Re (Poiseuille component)
- Analytical solution: u(y) = 3y² - 2y

**Success Criteria**:
- L2 error < 5% against analytical profile
- Wall velocities match boundary conditions
- Steady state convergence

**Reference**: Bruus (2008) Theoretical Microfluidics

---

### 3. test_rayleigh_taylor.json

**Category**: VOF Interface Tracking
**Difficulty**: Advanced
**Runtime**: 30-60 minutes
**Physics**: Fluid + VOF + Surface Tension + Buoyancy

**Purpose**: Validates VOF interface tracking with gravity-driven instability of heavy fluid over light fluid.

**Key Parameters**:
- Domain: 128×512 mm (16×64 grid, 8mm mesh)
- Interface: y = -0.05·cos(2πx) perturbation
- Densities: ρ_heavy = 1.225, ρ_light = 0.169 kg/m³
- Atwood number: At = 0.758
- Surface tension: σ = 1×10⁻² N/m
- Duration: 3.0 s

**Success Criteria**:
- Spike penetration depth at t=1s matches Gerris/Thibault
- Bubble rise height at t=1s matches Gerris/Thibault
- Mass conservation < 1%
- Qualitative: spike formation, bubble rise, Kelvin-Helmholtz rollups

**Reference**: Gerris benchmark, Thibault et al. (2013), Popinet (2009)

---

### 4. test_laser_melting.json

**Category**: Full LPBF Multiphysics
**Difficulty**: Expert
**Runtime**: 2-5 hours
**Physics**: ALL - Thermal + Fluid + VOF + Surface Tension + Marangoni + Phase Change + Laser

**Purpose**: Complete LPBF simulation validation - the "final boss" test that demonstrates the entire framework.

**Key Parameters**:
- Domain: 150×150×300 μm (40×80×80 mesh)
- Material: Steel (Fe)
- Laser: 50W stationary, r=50μm
- Duration: 30 μs (600 steps @ dt=0.05μs)
- All physics enabled: thermal advection, phase change, Marangoni, recoil pressure, evaporation, radiation

**Success Criteria**:
- Tier 1 (must pass): Numerical stability, mass < 1%, energy < 5%
- Tier 2 (should pass): T_max in [2000, 4000] K, v_max in [0.01, 5] m/s
- Tier 3 (quantitative): Within 30% of OpenFOAM/literature results

**Validation Targets**:
- Peak temperature: 2400-3200 K
- Melt pool depth: 30-80 μm
- Melt pool width: 80-150 μm
- Surface velocity: 0.1-1.0 m/s (Marangoni-driven)
- Reynolds number: 100-2000

**Reference**: OpenFOAM LaserbeamFoam, Khairallah et al. (2016)

---

## JSON Structure

Each configuration file follows a consistent structure:

```json
{
  "test_name": "...",
  "description": "...",
  "reference": "...",
  "test_category": "...",
  "difficulty": "...",
  "runtime_estimate": "...",

  "domain": { /* Grid and dimensions */ },
  "initial_conditions": { /* Initial state */ },
  "boundary_conditions": { /* BCs for all boundaries */ },
  "material_properties": { /* Physical properties */ },
  "time_parameters": { /* Time stepping and output */ },
  "physics_flags": { /* Which modules to enable */ },

  "validation": {
    "analytical_solution": { /* If available */ },
    "success_criteria": { /* Pass/fail thresholds */ },
    "expected_behavior": { /* Qualitative expectations */ }
  },

  "output": { /* File formats and diagnostics */ },
  "notes": [ /* Important information */ ],
  "implementation_notes": { /* Technical details */ }
}
```

---

## Usage

### Option 1: Direct JSON Parsing (Recommended for New Code)

```python
import json

# Load configuration
with open('config/validation/test_3d_heat_diffusion.json', 'r') as f:
    config = json.load(f)

# Access parameters
nx = config['domain']['grid']['nx']
T_wall = config['boundary_conditions']['all_boundaries']['temperature']
final_time = config['time_parameters']['final_time']

# Run simulation
run_simulation(config)
```

### Option 2: Convert to Existing .conf Format

```python
# Convert JSON to .conf format
def json_to_conf(json_path, conf_path):
    with open(json_path, 'r') as f:
        config = json.load(f)

    with open(conf_path, 'w') as f:
        f.write(f"nx = {config['domain']['grid']['nx']}\n")
        f.write(f"ny = {config['domain']['grid']['ny']}\n")
        # ... etc
```

### Option 3: Use as Reference for Manual Configuration

Simply read the JSON file and manually create corresponding `.conf` files based on your existing format.

---

## Validation Workflow

### Step 1: Select Test Case

Choose based on what you're validating:
- **Thermal solver only** → test_3d_heat_diffusion.json
- **Fluid solver only** → test_couette_poiseuille.json
- **VOF + multiphase** → test_rayleigh_taylor.json
- **Full LPBF system** → test_laser_melting.json

### Step 2: Review Configuration

```bash
# Pretty-print JSON for review
cat test_3d_heat_diffusion.json | python -m json.tool | less
```

### Step 3: Implement Test

Create corresponding C++/CUDA test file:
```cpp
// tests/validation/test_3d_heat_diffusion.cu
#include <gtest/gtest.h>
#include "config_loader.h"

TEST(Validation, Case1_HeatDiffusion) {
    // Load configuration
    auto config = load_json_config("config/validation/test_3d_heat_diffusion.json");

    // Setup domain
    Domain domain(config["domain"]["grid"]["nx"],
                  config["domain"]["grid"]["ny"],
                  config["domain"]["grid"]["nz"]);

    // Initialize physics
    ThermalLBM thermal(config["material_properties"]["thermal"]);

    // Run simulation
    run_simulation(domain, thermal, config);

    // Validate results
    float L2_error = compute_L2_error(T_numerical, T_analytical);
    EXPECT_LT(L2_error, config["validation"]["success_criteria"]["L2_error"]["threshold"]);
}
```

### Step 4: Run and Compare

```bash
# Build test
make test_3d_heat_diffusion

# Run test
./tests/validation/test_3d_heat_diffusion

# Compare results
python scripts/validate_results.py \
    --config config/validation/test_3d_heat_diffusion.json \
    --output output/case1_3d_heat_diffusion/ \
    --generate-report
```

### Step 5: Generate Validation Report

```bash
# Automatic report generation
python scripts/generate_validation_report.py \
    --test-suite config/validation/ \
    --results output/validation/ \
    --output VALIDATION_REPORT_$(date +%Y-%m-%d).md
```

---

## Configuration Field Reference

### Mandatory Fields

These fields must be present in all configuration files:

- `test_name`: Short descriptive name
- `description`: One-line purpose
- `domain.grid.{nx,ny,nz}`: Grid resolution
- `domain.dimensions.{x,y,z}`: Physical size
- `time_parameters.final_time`: Simulation duration
- `physics_flags.*`: Which modules to enable
- `output.directory`: Where to save results

### Optional Fields

- `material_properties.*`: Can reference material database
- `validation.analytical_solution`: Only if analytical solution exists
- `extensions.*`: For parameter studies
- `debugging_guide.*`: Troubleshooting tips

---

## Extending the Templates

### Adding a New Validation Case

1. **Copy template**:
   ```bash
   cp test_laser_melting.json test_my_new_case.json
   ```

2. **Modify parameters**:
   - Update `test_name`, `description`, `reference`
   - Adjust domain, material, physics flags
   - Define validation criteria

3. **Document expectations**:
   - Add `validation.success_criteria`
   - Include `expected_behavior`
   - Provide `debugging_guide`

4. **Test thoroughly**:
   - Run multiple times to ensure reproducibility
   - Verify output files are created
   - Check diagnostics make sense

### Adding New Configuration Fields

If you need new fields not in the templates:

```json
{
  "advanced_numerics": {
    "description": "Optional advanced settings",
    "tvd_limiter": "minmod",
    "flux_reconstruction": "MUSCL",
    "time_integrator": "RK3"
  }
}
```

Always include:
- Descriptive field names
- Units in separate `_unit` fields
- Comments in `_comment` fields
- Default values or "null" if auto-computed

---

## Validation Matrix

Use this matrix to track which tests pass:

| Test Case | Case 1 | Case 2 | Case 3 | Case 4 |
|-----------|--------|--------|--------|--------|
| **Configuration exists** | ✓ | ✓ | ✓ | ✓ |
| **Test implemented** | ☐ | ☐ | ☐ | ☐ |
| **Test passes** | ☐ | ☐ | ☐ | ☐ |
| **Meets criteria** | ☐ | ☐ | ☐ | ☐ |
| **Report generated** | ☐ | ☐ | ☐ | ☐ |

Check off boxes as you complete each stage.

---

## Troubleshooting

### JSON Parse Errors

```bash
# Validate JSON syntax
python -m json.tool test_3d_heat_diffusion.json > /dev/null
# If no output, JSON is valid

# Check specific file
jq '.' test_couette_poiseuille.json
```

### Missing Fields

If your code expects a field that's not in the JSON:

1. **Check if it's nested**: Use `config["section"]["subsection"]["field"]`
2. **Check spelling**: JSON is case-sensitive
3. **Add default value**: Handle missing fields gracefully
   ```python
   dt = config["time_parameters"].get("dt", None)
   if dt is None:
       dt = compute_dt_from_cfl(config)
   ```

### Units Confusion

All units are SI unless explicitly stated:
- Length: meters (m)
- Time: seconds (s)
- Temperature: Kelvin (K)
- Pressure: Pascal (Pa)
- Energy: Joules (J)
- Power: Watts (W)

Check the `_unit` field for each parameter to confirm.

---

## References

### Related Documentation

- **Existing validation tests**: `/home/yzk/LBMProject/tests/validation/`
- **Existing configs**: `/home/yzk/LBMProject/configs/validation/` (.conf format)
- **Validation framework**: `/home/yzk/LBMProject/docs/validation/VALIDATION_FRAMEWORK_COMPREHENSIVE.md`
- **Material database**: `/home/yzk/LBMProject/MATERIAL_DATABASE.yaml`
- **Group meeting summary**: `/home/yzk/LBMProject/GROUP_MEETING_EXECUTIVE_BRIEF.md`

### Literature References

1. **Case 1**: Senior's Thesis Chapter 4.2.1
2. **Case 2**: Bruus (2008) Theoretical Microfluidics
3. **Case 3**: Popinet (2009) JCP, Gerris benchmark suite
4. **Case 4**: Khairallah et al. (2016) Acta Materialia 108:36-45

### Code Examples

See these files for implementation patterns:
- `/home/yzk/LBMProject/tests/validation/test_3d_heat_diffusion_senior.cu`
- `/home/yzk/LBMProject/tests/validation/test_marangoni_velocity.cu`

---

## Next Steps

### Immediate (Week 1)

1. Review all four JSON configurations
2. Verify parameters match group meeting specifications
3. Identify any missing information
4. Create JSON parser utility if needed

### Short-term (Week 2-3)

1. Implement test harness for each case
2. Run Case 1 (simplest) to validate workflow
3. Run Case 2 and Case 3 in parallel
4. Debug any failures

### Mid-term (Week 4-6)

1. Run Case 4 (full LPBF) - expect 2-5 hours
2. Compare results with OpenFOAM LaserbeamFoam
3. Generate comprehensive validation report
4. Iterate on parameter calibration if needed

### Long-term (Beyond Week 6)

1. Add more validation cases as needed
2. Create automated validation pipeline
3. Integrate into CI/CD (GitHub Actions)
4. Publish validation results with paper

---

## Contributing

When adding new validation cases:

1. Follow the existing JSON structure
2. Include all mandatory fields
3. Add comprehensive `notes` and `implementation_notes`
4. Provide `debugging_guide` for common issues
5. Document expected results in `validation` section
6. Update this README with new case summary

---

## Version History

- **v1.0** (2026-01-13): Initial creation
  - Four validation cases from group meeting
  - Complete JSON schemas with validation criteria
  - Documentation and usage guide

---

## Contact

For questions about these configurations:
- **Physics/Validation**: Check `/home/yzk/LBMProject/docs/validation/`
- **Implementation**: See `/home/yzk/LBMProject/tests/validation/README.md`
- **Parameters**: Refer to `/home/yzk/LBMProject/MATERIAL_DATABASE.yaml`

---

**Status**: Configuration templates complete and ready for implementation
**Next Action**: Implement test harness and run Case 1 validation
**Estimated Time to Full Validation**: 4-6 weeks (all cases)
