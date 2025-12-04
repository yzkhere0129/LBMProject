# Gas Phase Module - Implementation Summary

## Quick Reference for Implementation

### 1. File Locations

**New Header Files** (created):
- `/home/yzk/LBMProject/include/physics/gas_phase_module.h` - Main module
- `/home/yzk/LBMProject/include/physics/evaporation_model.h` - Hertz-Knudsen model
- `/home/yzk/LBMProject/include/physics/recoil_pressure.h` - Recoil pressure

**Source Files to Create**:
```
/home/yzk/LBMProject/src/physics/gas/
    gas_phase_module.cu
    evaporation_model.cu
    recoil_pressure.cu
```

**Files to Modify**:
- `include/physics/multiphysics_solver.h` - Add gas phase config and member
- `src/physics/multiphysics/multiphysics_solver.cu` - Integration code

### 2. MultiphysicsConfig Extensions

Add to `MultiphysicsConfig` struct:
```cpp
// Gas Phase Configuration
bool enable_gas_phase = false;
bool enable_evaporation = true;
bool enable_recoil_pressure = true;
bool enable_evaporative_cooling = true;
GasMode gas_mode = GasMode::IMPLICIT;
float sticking_coefficient = 0.82f;
float recoil_coefficient = 0.54f;
float recoil_smoothing_width = 2.0f;
```

### 3. Key Equations

**Clausius-Clapeyron (saturation pressure)**:
```
p_sat(T) = p_ref * exp(L_v * M / R * (1/T_ref - 1/T))
```

**Hertz-Knudsen (mass flux)**:
```
m_dot = (1 - beta_r) * p_sat * sqrt(M / (2*pi*R*T))
```

**Recoil Pressure**:
```
P_recoil = 0.54 * p_sat
```

**Volumetric Force**:
```
F_recoil = P_recoil * (-n) * |grad(f)| / h_interface
```

### 4. Integration in step()

```cpp
void MultiphysicsSolver::step(float dt) {
    // 1. Laser source
    if (config_.enable_laser) applyLaserSource(dt);

    // 2. Thermal solve
    if (config_.enable_thermal) thermalStep(dt);

    // 3. Gas phase [NEW]
    if (config_.enable_gas_phase && gas_phase_) {
        // Compute evaporation
        gas_phase_->computeEvaporationMassFlux(
            getTemperature(),
            vof_->getFillLevel(),
            vof_->getInterfaceNormals());

        // Apply evaporative cooling
        if (config_.enable_evaporative_cooling) {
            gas_phase_->computeEvaporativeCooling();
            thermal_->addHeatSource(gas_phase_->getHeatSink(), dt);
        }
    }

    // 4. VOF advection (with mass source if gas enabled)
    if (vof_ && config_.enable_vof_advection) {
        vofStep(dt);  // Modified to accept mass source
    }

    // 5. Fluid solve
    if (fluid_) fluidStep(dt);

    current_time_ += dt;
}
```

### 5. Integration in computeTotalForce()

```cpp
// After Marangoni, before Darcy damping:
if (config_.enable_recoil_pressure && gas_phase_) {
    gas_phase_->addRecoilForce(
        getTemperature(),
        vof_->getFillLevel(),
        vof_->getInterfaceNormals(),
        d_fx, d_fy, d_fz);
}
```

### 6. Typical Parameter Values (Ti6Al4V)

| Parameter | Value | Units |
|-----------|-------|-------|
| T_boil | 3560 | K |
| L_vaporization | 8.9e6 | J/kg |
| M_molar (Ti) | 0.04788 | kg/mol |
| beta_r (sticking) | 0.82 | - |
| C_r (recoil) | 0.54 | - |
| Keyhole threshold | ~200 | W (at 50um spot) |

### 7. Expected Behaviors

1. **T < 2800 K**: Negligible evaporation
2. **T ~ 3560 K**: p_sat ~ 1 atm, P_recoil ~ 54 kPa
3. **T > 4000 K**: Strong evaporation, keyhole formation
4. **Conduction mode**: Surface depression < 1 spot radius
5. **Keyhole mode**: Depression > 1 spot radius, deep penetration

### 8. Implementation Priority

**Phase 7a (Core)**: 2-3 weeks
- [ ] EvaporationModel class + unit tests
- [ ] RecoilPressure class + unit tests
- [ ] Integration in computeTotalForce()

**Phase 7b (Energy/Mass)**: 1-2 weeks
- [ ] Evaporative cooling in thermal solver
- [ ] Mass source in VOF advection
- [ ] Energy balance validation

**Phase 7c (Integration)**: 1-2 weeks
- [ ] GasPhaseModule wrapper class
- [ ] Config extension
- [ ] Keyhole formation test

### 9. Validation Checklist

- [ ] p_sat(T_boil) = p_ambient (1 atm)
- [ ] m_dot increases monotonically with T
- [ ] Recoil force points INTO liquid
- [ ] Energy balance: P_laser = P_rad + P_evap + dE/dt
- [ ] Mass conservation at interface
- [ ] Keyhole forms above threshold power
- [ ] No NaN/divergence at high T

---

**Full Documentation**: `/home/yzk/LBMProject/docs/GasPhase_Module_Architecture_Design.md`
