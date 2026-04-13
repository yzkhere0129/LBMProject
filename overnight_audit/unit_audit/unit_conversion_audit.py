#!/usr/bin/env python3
"""
Phase 1.1: Complete Unit Conversion Audit for LBM-LPBF Solver
Pure physics calculations — no dependency on project CUDA code.
316L stainless steel, dx=2.5μm, dt=10ns
"""
import math

print("=" * 80)
print("  PHASE 1.1: UNIT CONVERSION AUDIT")
print("  316L Stainless Steel LPBF — dx=2.5μm, dt=10ns")
print("=" * 80)

# ============================================================================
# PHYSICAL PARAMETERS (316L)
# ============================================================================
rho     = 7900.0       # kg/m³
cp      = 700.0        # J/(kg·K)
k_cond  = 20.0         # W/(m·K)
mu      = 0.005        # Pa·s
T_sol   = 1650.0       # K
T_liq   = 1700.0       # K
T_boil  = 3200.0       # K
L_fus   = 260e3        # J/kg
L_vap   = 7.45e6       # J/kg
M_mol   = 0.0558       # kg/mol
sigma   = 1.75         # N/m
dsigma_dT = -4.3e-4    # N/(m·K)
absorb  = 0.35
P_laser = 150.0        # W
r_spot  = 25e-6        # m
v_scan  = 1.0          # m/s (1000 mm/s)

# Derived physical
nu_phys = mu / rho                     # kinematic viscosity [m²/s]
alpha_phys = k_cond / (rho * cp)       # thermal diffusivity [m²/s]
Pr = nu_phys / alpha_phys              # Prandtl number

print(f"\n--- Physical Parameters ---")
print(f"ν = μ/ρ = {nu_phys:.6e} m²/s")
print(f"α = k/(ρ·cp) = {alpha_phys:.6e} m²/s")
print(f"Pr = ν/α = {Pr:.4f}")

# ============================================================================
# LATTICE PARAMETERS
# ============================================================================
dx = 2.5e-6   # m
dt = 10e-9     # s

C_L = dx                          # length conversion
C_T = dt                          # time conversion
C_M = rho * dx**3                 # mass conversion
C_V = C_L / C_T                   # velocity conversion
C_F = C_M * C_L / C_T**2         # force conversion
C_P = C_M / (C_L * C_T**2)       # pressure conversion
C_E = C_M * C_L**2 / C_T**2      # energy conversion
C_F_density = C_M / (C_L**2 * C_T**2)  # force density [N/m³]

print(f"\n--- Conversion Factors ---")
print(f"C_L (length)   = {C_L:.4e} m")
print(f"C_T (time)     = {C_T:.4e} s")
print(f"C_M (mass)     = {C_M:.4e} kg")
print(f"C_V (velocity) = {C_V:.4e} m/s  (= dx/dt)")
print(f"C_F (force)    = {C_F:.4e} N")
print(f"C_P (pressure) = {C_P:.4e} Pa")
print(f"C_E (energy)   = {C_E:.4e} J")
print(f"C_F_density    = {C_F_density:.4e} N/m³")

# Lattice sound speed
cs_lat = 1.0 / math.sqrt(3.0)
cs_phys = cs_lat * dx / dt

print(f"\n--- Lattice Sound Speed ---")
print(f"cs_lat  = 1/√3 = {cs_lat:.6f}")
print(f"cs_phys = cs_lat × dx/dt = {cs_phys:.2f} m/s")
print(f"  → Any flow with v > {0.3*cs_phys:.1f} m/s violates Ma < 0.3")
print(f"  → Any flow with v > {0.1*cs_phys:.1f} m/s violates Ma < 0.1 (recommended)")

# Lattice viscosity and tau
nu_lat = nu_phys * dt / dx**2
tau = 3.0 * nu_lat + 0.5

print(f"\n--- Relaxation Time ---")
print(f"ν_lat = ν × dt/dx² = {nu_lat:.6f}")
print(f"τ = 3ν_lat + 0.5 = {tau:.6f}")
print(f"  → Distance from instability limit (0.5): {tau - 0.5:.6f}")
print(f"  → Status: {'CRITICAL (< 0.01 from limit)' if tau - 0.5 < 0.01 else 'MARGINAL' if tau - 0.5 < 0.05 else 'OK'}")

# Thermal lattice parameters
alpha_lat = alpha_phys * dt / dx**2
tau_thermal_d3q7 = 4.0 * alpha_lat + 0.5   # D3Q7: cs² = 1/4
tau_thermal_d3q19 = 3.0 * alpha_lat + 0.5  # if D3Q19 thermal

print(f"\n--- Thermal Parameters ---")
print(f"α_lat = α × dt/dx² = {alpha_lat:.6f}")
print(f"τ_thermal (D3Q7, cs²=1/4) = {tau_thermal_d3q7:.6f}")
print(f"τ_thermal (D3Q19, cs²=1/3) = {tau_thermal_d3q19:.6f}")
print(f"Pr = ν/α = {Pr:.4f}")
print(f"  → Pr < 1 means thermal diffuses faster than momentum")
print(f"  → D3Q7 thermal was replaced by FDM due to Pr deadlock")

# FDM stability (explicit Euler, 3D central diff)
Fo = alpha_phys * dt / dx**2
print(f"\nFDM Fourier number: Fo = α×dt/dx² = {Fo:.6f}")
print(f"  → 3D stability limit: Fo < 1/6 = {1/6:.6f}")
print(f"  → Status: {'STABLE' if Fo < 1/6 else 'UNSTABLE!'}")

# ============================================================================
# FORCE MAGNITUDE ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print(f"  FORCE MAGNITUDE ANALYSIS (all in lattice units)")
print(f"{'='*80}")

R_gas = 8.314  # J/(mol·K)
P_atm = 101325.0  # Pa
C_r = 0.54  # recoil coefficient

print(f"\n--- 1. Recoil Pressure ---")
print(f"{'T [K]':>8s} {'P_sat [kPa]':>12s} {'P_recoil [kPa]':>14s} {'P_recoil_lat':>12s} {'Δu_lat/step':>12s} {'Ma_single':>10s} {'VERDICT':>10s}")
for T in [3200, 3300, 3400, 3500, 3600, 3800, 4000]:
    exponent = (L_vap * M_mol / R_gas) * (1.0/T_boil - 1.0/T)
    exponent = min(exponent, 80)  # prevent overflow
    P_sat = P_atm * math.exp(exponent)
    P_recoil = C_r * P_sat
    P_recoil_lat = P_recoil / C_P
    # Velocity increment per step from recoil: Δu = (P/ρ) × (dt/dx) in lattice
    # More precisely: F_vol = P_recoil × |∇f| ≈ P_recoil / dx (for 1-cell interface)
    # a = F_vol / ρ = P_recoil / (ρ × dx)
    # Δu_lat = a × dt = P_recoil × dt / (ρ × dx) = P_recoil_lat / ρ_lat
    # But in lattice ρ_lat = 1 (if using standard LBM convention with ρ=1)
    # Actually: Δu_lat = P_recoil_lat × (dt²/(dx × ρ_lat))
    # Let me compute directly: Δu_phys = P_recoil/(ρ × dx) × dt (for interface cell)
    delta_u_phys = P_recoil / (rho * dx) * dt  # velocity per step [m/s] (acting on 1 interface cell of thickness dx)
    delta_u_lat = delta_u_phys / C_V
    Ma_single = delta_u_lat / cs_lat
    verdict = "RED" if delta_u_lat > 0.1 else "YELLOW" if delta_u_lat > 0.01 else "OK"
    print(f"{T:8.0f} {P_sat/1e3:12.1f} {P_recoil/1e3:14.1f} {P_recoil_lat:12.6f} {delta_u_lat:12.6f} {Ma_single:10.4f} {verdict:>10s}")

# Find critical temperature where Ma_single_step = 0.01 and 0.1
print(f"\n  Critical temperatures:")
for ma_target, label in [(0.01, "Ma_step=0.01 (safe)"), (0.05, "Ma_step=0.05 (marginal)"), (0.1, "Ma_step=0.1 (RED)")]:
    # Δu_lat = P_recoil × dt / (ρ × dx × C_V) = C_r × P_atm × exp(E×(1/Tb-1/T)) × dt / (ρ × dx²)
    # ma_target × cs_lat = above
    target_du = ma_target * cs_lat * C_V  # target Δu_phys
    target_P_recoil = target_du * rho * dx / dt
    target_P_sat = target_P_recoil / C_r
    if target_P_sat > P_atm:
        target_exp = math.log(target_P_sat / P_atm)
        E_factor = L_vap * M_mol / R_gas
        # target_exp = E_factor * (1/T_boil - 1/T_crit)
        # 1/T_crit = 1/T_boil - target_exp/E_factor
        inv_T = 1.0/T_boil - target_exp/E_factor
        if inv_T > 0:
            T_crit = 1.0 / inv_T
            print(f"  T for {label}: {T_crit:.0f} K (P_recoil={target_P_recoil/1e3:.1f} kPa)")
        else:
            print(f"  T for {label}: > 10000 K (always safe at current dx/dt)")
    else:
        print(f"  T for {label}: ≤ T_boil (already safe)")

print(f"\n--- 2. Marangoni Surface Stress ---")
# Typical temperature gradients in LPBF melt pool
for dTdy_label, dTdy in [("3.23e7 K/m (unit test)", 3.23e7), ("1e8 K/m (steep)", 1e8), ("5e7 K/m (moderate)", 5e7)]:
    tau_s = abs(dsigma_dT) * dTdy  # surface stress [N/m²]
    # CSF volumetric force: F_vol = tau_s × |∇f| ≈ tau_s / dx (for 1-cell interface)
    F_vol = tau_s / dx  # N/m³
    # Velocity increment per step: Δu = F_vol × dt / ρ
    delta_u_phys = F_vol * dt / rho
    delta_u_lat = delta_u_phys / C_V
    # Steady-state surface velocity (thin film): u_s ≈ tau_s × h / (2μ), h=melt depth
    for h_um in [10, 20, 30]:
        h = h_um * 1e-6
        u_steady = tau_s * h / (2 * mu)
        Ma_steady = u_steady / cs_phys
        print(f"  ∇T={dTdy_label}, h={h_um}μm: τ_s={tau_s:.0f} N/m², Δu_lat/step={delta_u_lat:.6f}, u_steady={u_steady:.1f} m/s, Ma_steady={Ma_steady:.3f}")

print(f"\n--- 3. Surface Tension (Laplace) ---")
for R_um in [10, 25, 50]:
    R = R_um * 1e-6
    kappa = 1.0 / R  # for sphere: 2/R, for cylinder: 1/R
    dP = sigma * kappa  # Laplace pressure (1D curvature)
    dP_2D = 2 * sigma / R  # sphere
    delta_u_phys = dP / rho * dt / dx
    delta_u_lat = delta_u_phys / C_V
    print(f"  R={R_um}μm: κ=1/R={kappa:.1e}/m, ΔP(cyl)={dP/1e3:.1f}kPa, ΔP(sphere)={dP_2D/1e3:.1f}kPa, Δu_lat/step={delta_u_lat:.6f}")

print(f"\n--- 4. Darcy Damping ---")
K_darcy = 1e6  # as configured
for fl in [0.01, 0.1, 0.5, 0.9, 1.0]:
    eps = 1e-3
    K_eff = K_darcy * (1 - fl)**2 / (fl**3 + eps) * dt  # lattice K = K_phys × dt
    # Semi-implicit: u = m / (ρ + K/2)
    # If K >> ρ, velocity is suppressed to near zero
    # Suppression factor: u/u_free = ρ/(ρ + K/2)
    K_lat = K_eff  # actually need to check units in code
    # In code: K_LU = D × dt where D is in physical [1/s]
    # D = darcy_coefficient × (1-fl)² / (fl³ + eps) [1/s]
    D = K_darcy * (1 - fl)**2 / (fl**3 + eps)
    K_LU = D * dt
    suppression = 1.0 / (1.0 + K_LU / 2.0)
    print(f"  fl={fl:.2f}: D={D:.2e} 1/s, K_LU={K_LU:.4f}, suppression={suppression:.6f} ({'locked' if suppression < 0.01 else 'damped' if suppression < 0.5 else 'free'})")

print(f"\n--- 5. Gravity/Buoyancy ---")
g = 9.81
beta = 1e-4  # thermal expansion coefficient (estimated)
delta_T = 2000.0
F_buoy = rho * g * beta * delta_T  # N/m³
delta_u_phys = F_buoy * dt / rho
delta_u_lat = delta_u_phys / C_V
print(f"  ΔT={delta_T:.0f}K, β={beta:.0e}/K: F_buoy={F_buoy:.2f} N/m³, Δu_lat/step={delta_u_lat:.2e}")
print(f"  → Buoyancy is negligible compared to other forces")

# ============================================================================
# dx/dt FEASIBILITY ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print(f"  dx/dt FEASIBILITY ANALYSIS")
print(f"{'='*80}")

# Current Ma at reported v_max
for v_max_label, v_max in [("v_max reported", 600), ("Marangoni steady (h=20μm, ∇T=5e7)", 107.5), ("Recoil at T=3500K", 0)]:
    if v_max > 0:
        Ma = v_max / cs_phys
        print(f"  {v_max_label}: v={v_max:.0f} m/s, Ma={Ma:.3f} {'→ UNSTABLE' if Ma > 0.3 else '→ OK'}")

# What dt is needed for Ma_recoil < 0.2 at T=3500K?
T_test = 3500
exp_val = (L_vap * M_mol / R_gas) * (1.0/T_boil - 1.0/T_test)
P_sat_test = P_atm * math.exp(exp_val)
P_recoil_test = C_r * P_sat_test

print(f"\n  At T={T_test}K: P_recoil = {P_recoil_test/1e3:.1f} kPa")

# The cumulative Ma depends on how many steps the recoil acts and viscous damping
# For a crude estimate: steady-state velocity from pressure balance
# P_recoil ≈ ρ × u² (dynamic pressure), u ≈ sqrt(P/ρ)
u_recoil_est = math.sqrt(P_recoil_test / rho)
print(f"  Estimated recoil-driven velocity: u ≈ √(P/ρ) = {u_recoil_est:.1f} m/s")
Ma_recoil = u_recoil_est / cs_phys
print(f"  Ma_recoil = {Ma_recoil:.3f}")

print(f"\n--- Alternative (dx, dt) combinations ---")
print(f"{'dx[μm]':>8s} {'dt[ns]':>8s} {'cs[m/s]':>8s} {'ν_lat':>8s} {'τ':>8s} {'Ma_recoil':>10s} {'Ma_Maran':>10s} {'NX':>5s} {'NY':>5s} {'NZ':>5s} {'N_total':>10s} {'status':>10s}")

# Marangoni steady velocity estimate: u_s = |dσ/dT| × ∇T × h / (2μ)
# With ∇T ≈ 5e7 K/m, h ≈ 20μm
u_marangoni_est = abs(dsigma_dT) * 5e7 * 20e-6 / (2 * mu)

for dx_um, dt_ns in [(2.5, 10), (2.5, 5), (2.5, 2), (5.0, 10), (5.0, 20), (5.0, 40), (10.0, 40), (10.0, 80), (10.0, 160)]:
    dx_t = dx_um * 1e-6
    dt_t = dt_ns * 1e-9
    cs_t = dx_t / (dt_t * math.sqrt(3))
    nu_lat_t = nu_phys * dt_t / dx_t**2
    tau_t = 3 * nu_lat_t + 0.5
    Ma_r = u_recoil_est / cs_t
    Ma_m = u_marangoni_est / cs_t
    # Domain: keep physical size 200×400×200 μm
    NX_t = int(200e-6 / dx_t)
    NY_t = int(400e-6 / dx_t)
    NZ_t = int(200e-6 / dx_t)
    N_t = NX_t * NY_t * NZ_t
    ok = tau_t > 0.51 and Ma_r < 0.3 and Ma_m < 0.3
    status = "FEASIBLE" if ok else "FAIL"
    if tau_t <= 0.51:
        status += "(τ!)"
    if Ma_r >= 0.3:
        status += "(Ma_r!)"
    print(f"{dx_um:8.1f} {dt_ns:8.0f} {cs_t:8.1f} {nu_lat_t:8.5f} {tau_t:8.5f} {Ma_r:10.3f} {Ma_m:10.3f} {NX_t:5d} {NY_t:5d} {NZ_t:5d} {N_t:10d} {status:>10s}")

print(f"\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
print(f"""
CURRENT STATE (dx=2.5μm, dt=10ns):
  cs_phys     = {cs_phys:.1f} m/s
  τ_fluid     = {tau:.6f} (margin from 0.5: {tau-0.5:.6f})
  Pr          = {Pr:.4f}

MACH NUMBER CRISIS:
  Recoil at T_boil (3200K):   u_est={math.sqrt(C_r * P_atm / rho):.1f} m/s, Ma={math.sqrt(C_r * P_atm / rho)/cs_phys:.3f}
  Recoil at T=3500K:          u_est={u_recoil_est:.1f} m/s, Ma={Ma_recoil:.3f}
  Marangoni steady (h=20μm):  u_est={u_marangoni_est:.1f} m/s, Ma={u_marangoni_est/cs_phys:.3f}
  Reported v_max=600 m/s:     Ma={600/cs_phys:.3f}

TAU CRISIS:
  τ = {tau:.6f}, margin = {tau-0.5:.6f}
  This is 0.3% above the instability limit.

FORCE SCALE ANALYSIS:
  Recoil at 3500K:      Δu_lat/step = {P_recoil_test * dt / (rho * dx**2):.6f}
  Marangoni (∇T=5e7):   Δu_lat/step = {abs(dsigma_dT)*5e7/dx*dt/rho/C_V:.6f}
  Surface tension (R=25μm): Δu_lat/step = {sigma/(25e-6)/rho*dt/dx/C_V:.6f}
""")
