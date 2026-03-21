#!/usr/bin/env python3
"""
Forensic Audit: Why powder particles don't collapse despite T>2000K flow.

Audit 1: Darcy damping vs liquid fraction — is mushy zone locking liquid?
Audit 2: VOF advection velocity — does VOF actually see the LBM velocity?
"""

import os, sys
import numpy as np

NX, NY, NZ = 500, 150, 65
DX = 2e-6
DT = 8e-8
N = NX * NY * NZ
IY_MID = NY // 2
LU_TO_MS = DX / DT

VTK_DIR = "output_powder_bed_sim"
T_SOLIDUS = 1658.0
T_LIQUIDUS = 1700.0


def parse_vtk(filepath):
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
            data = np.empty(N, dtype=np.float64)
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
                    data[idx] = [float(p) for p in parts[:3]]; idx += 1
            result[name] = data.reshape(NZ, NY, NX, 3)
    return result


def audit_1_darcy(d):
    """Check liquid_fraction and Darcy damping at hot cells."""
    print("\n" + "="*65)
    print("  AUDIT 1: Darcy Damping & Liquid Fraction in Melt Pool")
    print("="*65)

    T = d['temperature']
    f = d['fill_level']
    vel = d['velocity']
    fl = d.get('liquid_fraction', None)

    # Find hot metal cells
    hot_metal = (T > 2000) & (f > 0.3)
    n_hot = hot_metal.sum()
    print(f"  Cells with T>2000K & f>0.3: {n_hot}")

    if n_hot == 0:
        print("  No hot metal cells found!")
        return

    if fl is not None:
        fl_hot = fl[hot_metal]
        print(f"\n  Liquid fraction in hot metal (T>2000K, f>0.3):")
        print(f"    fl = 0   (pure solid):  {(fl_hot < 0.01).sum():5d} ({100*(fl_hot < 0.01).sum()/n_hot:.1f}%)")
        print(f"    0 < fl < 1 (mushy):     {((fl_hot >= 0.01) & (fl_hot <= 0.99)).sum():5d} ({100*((fl_hot >= 0.01) & (fl_hot <= 0.99)).sum()/n_hot:.1f}%)")
        print(f"    fl = 1   (pure liquid): {(fl_hot > 0.99).sum():5d} ({100*(fl_hot > 0.99).sum()/n_hot:.1f}%)")
        print(f"    fl mean: {fl_hot.mean():.4f}")
        print(f"    fl min:  {fl_hot.min():.4f}")
        print(f"    fl max:  {fl_hot.max():.4f}")

        # Darcy coefficient: K = C * (1 - fl) * rho * dt
        # With C=5e4, rho=6900, dt=8e-8:
        C_darcy = 5e4
        rho = 6900.0
        K_LU = C_darcy * (1.0 - fl_hot) * rho * DT
        print(f"\n  Darcy coefficient K_LU (= C*(1-fl)*ρ*dt):")
        print(f"    C = {C_darcy:.0e}, ρ = {rho:.0f}, dt = {DT:.0e}")
        print(f"    K_LU mean: {K_LU.mean():.6f}")
        print(f"    K_LU max:  {K_LU.max():.6f}")
        print(f"    K_LU at fl=0: {C_darcy * rho * DT:.6f}")
        print(f"    K_LU at fl=1: 0.000000")

        # With semi-implicit: u = m / (rho_LU + 0.5*K_LU)
        # If K_LU >> rho_LU (~1), velocity is killed
        rho_LU = 1.0  # LBM density
        suppression = rho_LU / (rho_LU + 0.5 * K_LU)
        print(f"\n  Velocity suppression factor u/u_free = ρ/(ρ + 0.5K):")
        print(f"    At fl_mean={fl_hot.mean():.3f}: factor = {suppression.mean():.4f} ({suppression.mean()*100:.1f}% of free velocity)")
        print(f"    At fl_min={fl_hot.min():.3f}:  factor = {(rho_LU / (rho_LU + 0.5*C_darcy*(1-fl_hot.min())*rho*DT)):.4f}")
        if suppression.mean() < 0.5:
            print(f"\n    ⚠ DARCY IS KILLING >50% OF VELOCITY IN THE MELT POOL!")
            print(f"    Even at T>2000K, liquid fraction is not reaching 1.0")
    else:
        print("  ⚠ liquid_fraction field NOT found in VTK!")
        print("  This is a critical data gap — cannot assess Darcy damping")

    # Temperature breakdown at specific thresholds
    print(f"\n  Temperature vs liquid fraction breakdown:")
    for T_thr in [1658, 1700, 2000, 2500, 3000]:
        mask = (T > T_thr) & (f > 0.3)
        n = mask.sum()
        if n > 0 and fl is not None:
            fl_sub = fl[mask]
            print(f"    T>{T_thr}K: {n:5d} cells, fl_mean={fl_sub.mean():.4f}, fl<0.5: {(fl_sub<0.5).sum()}")

    # Velocity in hot metal
    vmag = np.sqrt(vel[:,:,:,0]**2 + vel[:,:,:,1]**2 + vel[:,:,:,2]**2)
    vmag_phys = vmag * LU_TO_MS
    v_hot = vmag_phys[hot_metal]
    print(f"\n  Velocity in hot metal (T>2000K, f>0.3):")
    print(f"    v_max: {v_hot.max():.4f} m/s")
    print(f"    v_mean: {v_hot.mean():.4f} m/s")
    print(f"    v=0 cells: {(v_hot < 1e-6).sum()} / {n_hot} ({100*(v_hot < 1e-6).sum()/n_hot:.1f}%)")

    # Print detailed profile at hotspot
    idx_max = np.unravel_index(np.argmax(T), T.shape)
    k_hot, j_hot, i_hot = idx_max
    print(f"\n  Z-profile at hotspot (i={i_hot}, j={j_hot}, T_max={T[k_hot,j_hot,i_hot]:.0f}K):")
    print(f"    {'z[μm]':>6}  {'f':>6}  {'fl':>6}  {'T[K]':>7}  {'|v|[m/s]':>9}  {'K_LU':>8}  {'suppress':>8}")
    for k in range(max(0, k_hot-8), min(NZ, k_hot+8)):
        fv = f[k, j_hot, i_hot]
        tv = T[k, j_hot, i_hot]
        vv = vmag_phys[k, j_hot, i_hot]
        flv = fl[k, j_hot, i_hot] if fl is not None else -1
        klu = C_darcy * max(0, 1.0 - flv) * rho * DT if fl is not None else 0
        sup = rho_LU / (rho_LU + 0.5 * klu)
        marker = " ← HOT" if k == k_hot else ""
        print(f"    {k*DX*1e6:6.0f}  {fv:6.3f}  {flv:6.3f}  {tv:7.0f}  {vv:9.4f}  {klu:8.4f}  {sup:8.4f}{marker}")


def audit_2_vof_velocity(d):
    """Check if VOF advection receives actual LBM velocity."""
    print("\n" + "="*65)
    print("  AUDIT 2: VOF Advection Velocity Check")
    print("="*65)

    T = d['temperature']
    f = d['fill_level']
    vel = d['velocity']

    vmag = np.sqrt(vel[:,:,:,0]**2 + vel[:,:,:,1]**2 + vel[:,:,:,2]**2)
    vmag_phys = vmag * LU_TO_MS

    # Check velocity at VOF interface cells
    interface = (f > 0.01) & (f < 0.99)
    n_interface = interface.sum()
    print(f"  Interface cells (0.01 < f < 0.99): {n_interface}")

    if n_interface == 0:
        print("  No interface cells!")
        return

    v_interface = vmag_phys[interface]
    T_interface = T[interface]

    print(f"  Velocity at interface:")
    print(f"    v_max:  {v_interface.max():.4f} m/s")
    print(f"    v_mean: {v_interface.mean():.4f} m/s")
    print(f"    v=0:    {(v_interface < 1e-6).sum()} / {n_interface} ({100*(v_interface < 1e-6).sum()/n_interface:.1f}%)")

    # HOT interface cells specifically (where melting should cause flow)
    hot_interface = (f > 0.01) & (f < 0.99) & (T > T_LIQUIDUS)
    n_hot_int = hot_interface.sum()
    print(f"\n  HOT interface cells (f∈(0.01,0.99), T>{T_LIQUIDUS}K): {n_hot_int}")

    if n_hot_int > 0:
        v_hot_int = vmag_phys[hot_interface]
        print(f"    v_max:  {v_hot_int.max():.4f} m/s")
        print(f"    v_mean: {v_hot_int.mean():.4f} m/s")
        print(f"    v=0:    {(v_hot_int < 1e-6).sum()} / {n_hot_int} ({100*(v_hot_int < 1e-6).sum()/n_hot_int:.1f}%)")

        if (v_hot_int < 1e-6).sum() == n_hot_int:
            print(f"\n    ⚠⚠⚠ ALL HOT INTERFACE CELLS HAVE ZERO VELOCITY! ⚠⚠⚠")
            print(f"    VOF advection has NOTHING to work with!")
            print(f"    Possible causes:")
            print(f"    1. freezeSolidVelocityKernel is zeroing liquid cells")
            print(f"    2. Darcy damping is too strong (fl not reaching 1.0)")
            print(f"    3. Velocity output is taken AFTER freeze kernel")

    # Check velocity in bulk liquid (T > T_liquidus, f > 0.99)
    bulk_liquid = (T > T_LIQUIDUS) & (f > 0.99)
    n_bulk = bulk_liquid.sum()
    print(f"\n  Bulk liquid cells (T>{T_LIQUIDUS}K, f>0.99): {n_bulk}")
    if n_bulk > 0:
        v_bulk = vmag_phys[bulk_liquid]
        print(f"    v_max:  {v_bulk.max():.4f} m/s")
        print(f"    v_mean: {v_bulk.mean():.4f} m/s")
        print(f"    v=0:    {(v_bulk < 1e-6).sum()} / {n_bulk}")

    # Spatial check: XZ midplane velocity in powder zone
    print(f"\n  XZ Midplane velocity in powder zone (z=60-90μm):")
    z_powder_lo = 30  # cell index for z=60μm
    z_powder_hi = 45  # cell index for z=90μm
    for k in range(z_powder_lo, min(z_powder_hi, NZ)):
        v_slice = vmag_phys[k, IY_MID, :]
        f_slice = f[k, IY_MID, :]
        T_slice = T[k, IY_MID, :]
        metal = f_slice > 0.3
        hot_metal = metal & (T_slice > T_LIQUIDUS)
        n_m = metal.sum()
        n_hm = hot_metal.sum()
        v_max_k = v_slice[metal].max() if n_m > 0 else 0
        v_hot_k = v_slice[hot_metal].max() if n_hm > 0 else 0
        print(f"    z={k*DX*1e6:3.0f}μm: metal={n_m:3d}, hot_liquid={n_hm:3d}, "
              f"v_max_metal={v_max_k:.3f}, v_max_hot={v_hot_k:.3f} m/s")


def main():
    # Use t=200μs snapshot
    vtk_path = os.path.join(VTK_DIR, "powder_sim_002500.vtk")
    if not os.path.exists(vtk_path):
        # Try other names
        import glob
        candidates = sorted(glob.glob(os.path.join(VTK_DIR, "powder_sim_*.vtk")))
        if len(candidates) > 2:
            vtk_path = candidates[2]  # 3rd file ≈ t=100-200μs
        else:
            print("No suitable VTK found!"); sys.exit(1)

    print(f"Using: {vtk_path}")
    d = parse_vtk(vtk_path)

    print(f"\nAvailable fields: {list(d.keys())}")

    audit_1_darcy(d)
    audit_2_vof_velocity(d)


if __name__ == '__main__':
    main()
