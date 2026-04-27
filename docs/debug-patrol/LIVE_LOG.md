# Debug Patrol Live Log

**Started**: 2026-04-27 21:50
**Branch**: debug/patrol-2026-04-27 (off benchmark/conduction-316L @ 36d62b0)

## Mission scope

User mandate: "现在的代码库我认为应该是漏洞百出的，再多的纠错也不为过" — assume
the codebase is full of latent bugs, do exhaustive debug patrol during the
5060-runtime wait. Continuous, parallel, multi-subagent.

## Already-applied audit fixes (already on main)

From overnight code-audit pass 1 + 2 (90 findings total):
- BUG-3: run_simulation.cu thermal dt/dx (5-line patch)
- F-02: thermal_lbm.cu computeTemperatureKernel const-cast write (UB removed)
- F-04: FluidLBM/VOFSolver copy ctor delete (rule of five)
- F-06: multiphysics_solver.cu evap mass_loss_scale=1.0 round-trip removed (~190 MB host transfer/step saved)
- F-08: getMaxMetalVelocity / getMaxTemperature CUDA_CHECK
- F-12: int*int*int overflow widened
- F-13: vof_solver compression non-periodic BC warning
- F-01: verified false alarm via `apps/diag_evap_material_readback`

Plus separately:
- BUG-1 (recoil_pressure.cu Ti6Al4V constants) verified DOES NOT bite production
  (multiphysics_solver.cu reads correct material; dead code only)

## Open audit findings (not yet patched)

### HIGH (from pass 1)
- F-05: cudaMalloc/Free in vofStep every timestep (perf, ~ms cost per step)
- F-07: 198MB diagnostic copy DeviceToHost every 100 steps (perf)
- F-10: zeroForceKernelLocal duplicate of force_accumulator.cu kernel
- F-11: D3Q7 lattice weights {0.25f, 0.125f, ...} duplicated in 8 kernels
- F-16: ray_tracing_laser reduceSum allocates per call

### HIGH (from pass 2)
- F-03: 19/28 multiphysics integration tests are EXPECT_TRUE(true) stubs
- F-04: test_vof_mass_correction built but no add_test (CTest skips it)
- F-05: RayleighTaylorGerrisExact has no TIMEOUT (CI hang risk)
- F-07: validate() thermal diffusivity check is vacuous (defaults always positive)

### Many MEDIUM/LOW items

90 total - 7 fixed - ~9 above HIGH = ~74 more items to triage.

## Patrol queue (priority order)

1. ⏳ Replace 19 stub tests with real assertions (F-03 from pass 2)
2. ⏳ Fix F-04 add_test missing for vof_mass_correction
3. ⏳ Apply F-05 cudaMalloc move out of hot path
4. ⏳ Fresh audit pass on areas not previously covered
5. ⏳ Numerical-accuracy spot checks on collision kernels
6. ⏳ MEDIUM cleanups (F-08+ from pass 2)

## Currently active subagents

(filled as launched)

EOF
