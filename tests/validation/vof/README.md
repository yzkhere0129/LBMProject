# VOF Validation Tests

## Overview
This directory contains physics validation tests for the VOF solver, comparing numerical results against analytical solutions from literature.

## Tests

### 1. Zalesak's Disk (test_vof_advection_rotation.cu)
**Reference:** Zalesak, S. T. (1979). "Fully multidimensional flux-corrected transport algorithms for fluids." Journal of Computational Physics, 31(3), 335-362.

**Description:** Classic VOF benchmark. A slotted disk undergoes solid body rotation and should return to its original shape after 360°.

**Metrics:**
- Mass conservation: < 5% error
- L1 shape error: < 0.15
- Interface preservation

**Run:**
```bash
./test_vof_advection_rotation
```

---

### 2. Spherical Curvature (test_vof_curvature_sphere.cu)
**Analytical Solution:** κ = 2/R (mean curvature of sphere)

**Description:** Tests curvature computation for spherical droplets at multiple resolutions.

**Test Cases:**
- Large sphere (R=20): error < 10%
- Medium sphere (R=12): error < 20%
- Small sphere (R=8): error < 30%
- Isotropy check: CV < 0.3
- Sign convention validation

**Run:**
```bash
./test_vof_curvature_sphere
```

---

### 3. Cylindrical Curvature (test_vof_curvature_cylinder.cu)
**Analytical Solution:** κ = 1/R (in cross-section perpendicular to axis)

**Description:** Tests 2D curvature computation for cylindrical geometry.

**Test Cases:**
- Large cylinder (R=16): error < 15%
- Medium cylinder (R=10): error < 25%
- Cylinder vs sphere ratio: ~0.5
- Axial uniformity: CV < 0.2

**Run:**
```bash
./test_vof_curvature_cylinder
```

---

## Expected Tolerances

| Geometry | Radius | Expected Error | Reason |
|----------|--------|----------------|--------|
| Sphere | Large (R>15) | < 10% | Well-resolved |
| Sphere | Medium (R~12) | < 20% | Typical resolution |
| Sphere | Small (R<10) | < 30% | Under-resolved |
| Cylinder | Large (R>15) | < 15% | 2D geometry simpler |
| Cylinder | Medium (R~10) | < 25% | Typical resolution |

---

## Physics Background

### Curvature Definition
For a smooth interface with unit normal **n**:
```
κ = ∇·n
```

For specific geometries:
- **Sphere:** κ = 2/R (both principal curvatures equal 1/R)
- **Cylinder:** κ = 1/R (one principal curvature 1/R, other is 0)
- **Plane:** κ = 0 (both principal curvatures zero)

### Interface Normal
```
n = -∇f / |∇f|
```
Points from liquid (f=1) to gas (f=0).

### Numerical Challenges
1. **Discretization error:** Cartesian grid vs smooth interface
2. **Staircase effect:** Grid-aligned vs diagonal features
3. **Resolution limit:** Need ~10 cells per radius for good accuracy

---

## Compilation

```bash
cd /home/yzk/LBMProject/build
cmake ..
cmake --build . --target test_vof_advection_rotation -j4
cmake --build . --target test_vof_curvature_sphere -j4
cmake --build . --target test_vof_curvature_cylinder -j4
```

## Running All Validation Tests

```bash
cd /home/yzk/LBMProject/build
ctest -R "vof.*rotation|vof.*sphere|vof.*cylinder" --output-on-failure
```

## Interpreting Results

### Good Results
```
  Mean curvature: κ = 0.098
  Analytical: κ = 0.100
  Relative error: 2.0%
  ✓ Test passed
```

### Acceptable Results (boundary)
```
  Mean curvature: κ = 0.112
  Analytical: κ = 0.100
  Relative error: 12.0%
  ✓ Test passed (curvature within 20%)
```

### Failed Results
```
  Mean curvature: κ = 0.145
  Analytical: κ = 0.100
  Relative error: 45.0%
  ✗ Test FAILED: error too large
```

Failures may indicate:
- Bug in curvature computation kernel
- Incorrect normal calculation
- Interface reconstruction issues
- Grid resolution too coarse

---

## References

1. **Zalesak's Disk:**
   - Zalesak, S. T. (1979). JCP, 31(3), 335-362.
   - LeVeque, R. J. (1996). "High-resolution conservative algorithms for advection in incompressible flow." SIAM J. Numer. Anal., 33(2), 627-665.

2. **VOF Method:**
   - Koerner et al. (2005). "Lattice Boltzmann model for free surface flow." J. Stat. Phys., 121(1), 179-196.
   - Thuerey, N. (2007). PhD thesis, University of Erlangen-Nuremberg.

3. **Curvature Computation:**
   - Popinet, S. (2009). "An accurate adaptive solver for surface-tension-driven interfacial flows." JCP, 228(16), 5838-5866.
   - Francois et al. (2006). "A balanced-force algorithm for continuous and sharp interfacial surface tension models." JCP, 213(1), 141-173.
