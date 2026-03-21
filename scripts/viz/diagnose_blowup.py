#!/usr/bin/env python3
"""
4-Evidence Diagnostic for v_max=0 blow-up in LPBF ray-tracing simulation.

Evidence 1: NaN/Inf audit on velocity and temperature
Evidence 2: Energy deposition location vs fill_level
Evidence 3: Evaporation flux vs laser input at hotspot
Evidence 4: VOF phase field state in melt pool region
"""

import os, sys, struct
import numpy as np

NX, NY, NZ = 500, 150, 90
DX = 2e-6
DT = 8e-8
N = NX * NY * NZ
LU_TO_MS = DX / DT

VTK_DIR = "output_powder_bed_sim"
if not os.path.isdir(VTK_DIR):
    VTK_DIR = os.path.join(os.path.dirname(__file__), "../../build/output_powder_bed_sim")


def parse_vtk(filepath):
    """Parse all fields from ASCII VTK."""
    print(f"Parsing {os.path.basename(filepath)}...", flush=True)
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    fields = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('VECTORS'):
            fields[s.split()[1].lower()] = ('vector', i + 1)
        elif s.startswith('SCALARS'):
            fields[s.split()[1].lower()] = ('scalar', i + 2)
        i += 1

    result = {}
    for name, (kind, start) in fields.items():
        if kind == 'scalar':
            data = np.empty(N, dtype=np.float64)  # double for NaN detection
            idx = 0
            for line in lines[start:]:
                if idx >= N: break
                for v in line.split():
                    if idx >= N: break
                    data[idx] = float(v); idx += 1
            result[name] = data.reshape(NZ, NY, NX)
        else:
            data = np.empty((N, 3), dtype=np.float64)
            idx = 0
            for line in lines[start:]:
                if idx >= N: break
                parts = line.split()
                if len(parts) >= 3:
                    data[idx] = [float(p) for p in parts[:3]]
                    idx += 1
            result[name] = data.reshape(NZ, NY, NX, 3)
    return result


def evidence_1_nan_audit(d):
    """Check for NaN/Inf in velocity and temperature."""
    print("\n" + "="*60)
    print("  EVIDENCE 1: NaN / Inf Audit")
    print("="*60)

    vel = d['velocity']  # (NZ, NY, NX, 3)
    T = d['temperature']
    f = d['fill_level']

    for name, arr in [('vx', vel[:,:,:,0]), ('vy', vel[:,:,:,1]),
                      ('vz', vel[:,:,:,2]), ('temperature', T),
                      ('fill_level', f)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        n_total = arr.size
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        print(f"  {name:15s}: NaN={n_nan:6d}, Inf={n_inf:6d}, "
              f"min={vmin:12.4g}, max={vmax:12.4g}")

    # Velocity magnitude
    vmag = np.sqrt(vel[:,:,:,0]**2 + vel[:,:,:,1]**2 + vel[:,:,:,2]**2)
    vmag_phys = vmag * LU_TO_MS
    print(f"\n  v_mag (LU):   max={np.nanmax(vmag):.6f}")
    print(f"  v_mag (m/s):  max={np.nanmax(vmag_phys):.4f}")
    print(f"  v_mag == 0 everywhere? {np.nanmax(vmag) < 1e-15}")

    # Check if all velocity is exactly zero
    n_nonzero_v = (vmag > 1e-20).sum()
    print(f"  Cells with |v| > 0: {n_nonzero_v} / {N} ({100*n_nonzero_v/N:.2f}%)")

    # Check velocity in liquid region
    liquid = (T > 1648) & (f > 0.3)
    n_liquid = liquid.sum()
    if n_liquid > 0:
        vmag_liquid = vmag[liquid]
        print(f"\n  Liquid cells (T>1648, f>0.3): {n_liquid}")
        print(f"  v_mag in liquid: max={np.nanmax(vmag_liquid):.6f} LU, "
              f"mean={np.nanmean(vmag_liquid):.6f} LU")
    else:
        print(f"\n  WARNING: No liquid cells found (T>1648 & f>0.3)!")


def evidence_2_deposition_check(d):
    """Check where ray tracing deposits energy relative to fill_level."""
    print("\n" + "="*60)
    print("  EVIDENCE 2: Energy Deposition vs Fill Level")
    print("="*60)

    T = d['temperature']
    f = d['fill_level']

    # Hot cells = cells receiving significant laser energy
    # At t=200μs, laser is at x≈260μm = cell 130. Check region around it.
    T_threshold = 2000.0  # Cells significantly heated

    hot = T > T_threshold
    n_hot = hot.sum()

    if n_hot == 0:
        print("  No cells above 2000K!")
        return

    # fill_level distribution of hot cells
    f_hot = f[hot]
    print(f"  Cells with T > {T_threshold}K: {n_hot}")
    print(f"  fill_level distribution of hot cells:")
    print(f"    f = 0     (pure gas):      {(f_hot < 0.01).sum():6d} ({100*(f_hot < 0.01).sum()/n_hot:.1f}%)")
    print(f"    0 < f < 1 (interface):     {((f_hot >= 0.01) & (f_hot <= 0.99)).sum():6d} ({100*((f_hot >= 0.01) & (f_hot <= 0.99)).sum()/n_hot:.1f}%)")
    print(f"    f = 1     (pure metal):    {(f_hot > 0.99).sum():6d} ({100*(f_hot > 0.99).sum()/n_hot:.1f}%)")

    # Where is the HOTTEST cell?
    idx_max = np.unravel_index(np.argmax(T), T.shape)
    k, j, i = idx_max
    print(f"\n  T_max cell: ({i}, {j}, {k}) = ({i*DX*1e6:.0f}, {j*DX*1e6:.0f}, {k*DX*1e6:.0f}) μm")
    print(f"    T = {T[k,j,i]:.0f} K")
    print(f"    f = {f[k,j,i]:.4f}")
    print(f"    Is this a gas cell? {'YES — PROBLEM!' if f[k,j,i] < 0.01 else 'No (metal)'}")
    print(f"    Is this pure metal interior? {'YES' if f[k,j,i] > 0.99 else 'No'}")

    # Check T in pure gas cells near the hotspot
    gas_near = (f < 0.01) & (T > 1000)
    n_hot_gas = gas_near.sum()
    if n_hot_gas > 0:
        T_gas = T[gas_near]
        print(f"\n  ⚠ HOT GAS CELLS (f<0.01, T>1000K): {n_hot_gas}")
        print(f"    T range: {T_gas.min():.0f} – {T_gas.max():.0f} K")
        print(f"    This means energy is leaking into gas phase!")
    else:
        print(f"\n  ✓ No hot gas cells (good — energy stays in metal)")


def evidence_3_evaporation_check(d):
    """Check evaporation heat flux at hotspot."""
    print("\n" + "="*60)
    print("  EVIDENCE 3: Evaporation Cooling Capacity")
    print("="*60)

    T = d['temperature']
    f = d['fill_level']

    # Clausius-Clapeyron saturation pressure for 316L
    # P_sat(T) = P_atm * exp(L_vap * M / R * (1/T_boil - 1/T))
    T_boil = 3090.0  # K
    L_vap = 6.09e6   # J/kg
    M_molar = 55.845e-3  # kg/mol (Fe)
    R_gas = 8.314     # J/(mol·K)
    P_atm = 101325.0  # Pa
    alpha_evap = 0.18  # Hertz-Knudsen coefficient

    # Find hotspot
    idx_max = np.unravel_index(np.argmax(T), T.shape)
    k, j, i = idx_max
    T_hot = T[k,j,i]

    if T_hot > T_boil:
        # Clausius-Clapeyron
        P_sat = P_atm * np.exp(L_vap * M_molar / R_gas * (1.0/T_boil - 1.0/T_hot))
        # Hertz-Knudsen mass flux
        J_evap = alpha_evap * P_sat * np.sqrt(M_molar / (2 * np.pi * R_gas * T_hot))
        # Evaporation cooling power density [W/m³]
        q_evap = J_evap * L_vap / DX  # Per unit depth

        print(f"  Hotspot: T = {T_hot:.0f} K (at cell {i},{j},{k})")
        print(f"  P_sat(T) = {P_sat:.2e} Pa")
        print(f"  J_evap = {J_evap:.2e} kg/(m²·s)")
        print(f"  q_evap = {q_evap:.2e} W/m³ (cooling)")

        # Compare with laser input
        # Ray tracing deposits ~130W into spot area π(25μm)²
        P_laser = 200.0
        alpha_eff = 0.65
        spot_r = 25e-6
        spot_area = np.pi * spot_r**2
        q_laser_surface = P_laser * alpha_eff / spot_area  # W/m²
        q_laser_vol = q_laser_surface / DX  # W/m³ (one cell deep)

        print(f"\n  Laser input estimate:")
        print(f"    q_laser ≈ {q_laser_vol:.2e} W/m³ (in surface cell)")
        print(f"    q_evap / q_laser = {q_evap/q_laser_vol:.2f}")

        if q_evap < q_laser_vol:
            print(f"\n  ⚠ EVAPORATION CANNOT KEEP UP!")
            print(f"    At T={T_hot:.0f}K, evaporation removes only {100*q_evap/q_laser_vol:.0f}%")
            print(f"    Thermal runaway is expected!")
        else:
            print(f"\n  ✓ Evaporation should balance laser at this T")

        # Check at what T evaporation balances laser
        T_test = np.linspace(T_boil, 15000, 1000)
        P_test = P_atm * np.exp(L_vap * M_molar / R_gas * (1.0/T_boil - 1.0/T_test))
        J_test = alpha_evap * P_test * np.sqrt(M_molar / (2 * np.pi * R_gas * T_test))
        q_test = J_test * L_vap / DX
        balance_idx = np.argmin(np.abs(q_test - q_laser_vol))
        print(f"\n  Equilibrium T (where q_evap = q_laser): {T_test[balance_idx]:.0f} K")
    else:
        print(f"  T_hot = {T_hot:.0f} K (below boiling, no evaporation)")


def evidence_4_phase_field(d):
    """Check VOF fill_level in melt pool region."""
    print("\n" + "="*60)
    print("  EVIDENCE 4: VOF Phase Field in Melt Pool")
    print("="*60)

    T = d['temperature']
    f = d['fill_level']
    vel = d['velocity']

    # Find melt pool region (T > T_solidus)
    melt = T > 1648
    n_melt = melt.sum()
    print(f"  Cells with T > T_solidus: {n_melt}")

    if n_melt == 0:
        print("  No melt pool!")
        return

    f_melt = f[melt]
    print(f"  Fill level in melt pool (T > 1648K):")
    print(f"    f = 0     (gas):       {(f_melt < 0.01).sum():6d} ({100*(f_melt < 0.01).sum()/n_melt:.1f}%)")
    print(f"    0 < f < 0.5:           {((f_melt >= 0.01) & (f_melt < 0.5)).sum():6d} ({100*((f_melt >= 0.01) & (f_melt < 0.5)).sum()/n_melt:.1f}%)")
    print(f"    0.5 ≤ f < 1:           {((f_melt >= 0.5) & (f_melt <= 0.99)).sum():6d} ({100*((f_melt >= 0.5) & (f_melt <= 0.99)).sum()/n_melt:.1f}%)")
    print(f"    f = 1     (full metal): {(f_melt > 0.99).sum():6d} ({100*(f_melt > 0.99).sum()/n_melt:.1f}%)")

    # Specifically check: is the hottest region gas?
    T_thresh_list = [3000, 4000, 5000, 6000]
    print(f"\n  Fill level at extreme temperatures:")
    for T_thr in T_thresh_list:
        mask = T > T_thr
        n = mask.sum()
        if n > 0:
            f_sub = f[mask]
            gas_frac = (f_sub < 0.01).sum() / n * 100
            metal_frac = (f_sub > 0.5).sum() / n * 100
            print(f"    T > {T_thr}K: {n:5d} cells, gas={gas_frac:.0f}%, metal={metal_frac:.0f}%")
        else:
            print(f"    T > {T_thr}K: 0 cells")

    # Check if recoil blew away the surface
    # Look at z-profile at hotspot x
    idx_max = np.unravel_index(np.argmax(T), T.shape)
    k_hot, j_hot, i_hot = idx_max
    print(f"\n  Z-profile at hotspot column (i={i_hot}, j={j_hot}):")
    print(f"    {'z[μm]':>6s}  {'f':>6s}  {'T[K]':>8s}  {'|v|[LU]':>10s}")
    for k in range(max(0, k_hot-10), min(NZ, k_hot+10)):
        fv = f[k, j_hot, i_hot]
        tv = T[k, j_hot, i_hot]
        vv = np.sqrt(vel[k, j_hot, i_hot, 0]**2 +
                     vel[k, j_hot, i_hot, 1]**2 +
                     vel[k, j_hot, i_hot, 2]**2)
        marker = " ← HOT" if k == k_hot else ""
        print(f"    {k*DX*1e6:6.0f}  {fv:6.3f}  {tv:8.0f}  {vv:10.6f}{marker}")


def main():
    # Use step ~1000 (t=80μs) where v_max first drops to 0
    # and step ~2500 (t=200μs) for developed state
    for fname, label in [("powder_sim_000624.vtk", "t=50μs (early)"),
                         ("powder_sim_001248.vtk", "t=100μs (v=0 onset)"),
                         ("powder_sim_002496.vtk", "t=200μs (developed)")]:
        path = os.path.join(VTK_DIR, fname)
        if not os.path.exists(path):
            print(f"Skip: {fname} not found")
            continue

        print(f"\n{'#'*60}")
        print(f"  DIAGNOSTIC: {label} — {fname}")
        print(f"{'#'*60}")

        d = parse_vtk(path)
        evidence_1_nan_audit(d)
        evidence_2_deposition_check(d)
        evidence_3_evaporation_check(d)
        evidence_4_phase_field(d)


if __name__ == '__main__':
    main()
