#!/usr/bin/env python3
"""
Verify heat source implementation matches theory
Compare CUDA LBM implementation with expected Beer-Lambert absorption
"""
import numpy as np
import matplotlib.pyplot as plt

def beer_lambert_profile(z, P, eta, r0, delta):
    """
    Compute Beer-Lambert absorption profile

    Args:
        z: depth array [m]
        P: laser power [W]
        eta: absorptivity
        r0: beam radius [m]
        delta: penetration depth [m]

    Returns:
        Q: volumetric heat source [W/m³]
    """
    # Surface intensity at r=0
    I0 = (2 * P * eta) / (np.pi * r0**2)

    # Volumetric heat source (Beer-Lambert)
    beta = 1.0 / delta
    Q = I0 * beta * np.exp(-beta * z)

    return Q

def main():
    print("="*70)
    print("HEAT SOURCE VERIFICATION SCRIPT")
    print("Comparing CUDA LBM vs waLBerla Beer-Lambert absorption")
    print("="*70)

    # Test parameters (from test_laser_melting_senior.cu)
    P = 200.0  # W
    eta = 0.35
    r0 = 30e-6  # m (spot radius)

    # Compare penetration depths
    delta_cuda = 10e-6  # m (CUDA configured)
    delta_walberla_assumed = 50e-6  # m (waLBerla: assumed = beam radius)

    print(f"\nLaser Parameters:")
    print(f"  Power: {P} W")
    print(f"  Absorptivity: {eta}")
    print(f"  Spot radius: {r0*1e6:.1f} μm")
    print(f"  Expected absorbed power: {P*eta:.1f} W")

    print(f"\nPenetration Depths:")
    print(f"  CUDA: {delta_cuda*1e6:.1f} μm")
    print(f"  waLBerla (assumed): {delta_walberla_assumed*1e6:.1f} μm")
    print(f"  Ratio: {delta_walberla_assumed/delta_cuda:.1f}×")

    # Create depth array
    z = np.linspace(0, 150e-6, 1000)  # 0-150 μm (matches test domain)

    # Compute heat source profiles
    Q_cuda = beer_lambert_profile(z, P, eta, r0, delta_cuda)
    Q_walberla = beer_lambert_profile(z, P, eta, r0, delta_walberla_assumed)

    # Verify energy conservation
    # Note: Surface intensity already includes 2*P/(π*r0²), which when integrated
    # over Gaussian profile gives total power P. The volumetric source Q includes
    # the beta factor which integrates to 1. So integrating Q over z should give
    # surface intensity I, not total power.
    dz = z[1] - z[0]
    # Integration check: ∫ β·exp(-βz) dz from 0 to ∞ should equal 1
    integral_check_cuda = np.trapz(Q_cuda / (Q_cuda[0] / (1.0/delta_cuda)), dx=dz)
    integral_check_walberla = np.trapz(Q_walberla / (Q_walberla[0] / (1.0/delta_walberla_assumed)), dx=dz)

    # Peak intensity (at surface, at beam center)
    I0 = (2 * P * eta) / (np.pi * r0**2)

    print(f"\nEnergy Conservation Check:")
    print(f"  Beer-Lambert integral ∫ β·exp(-βz) dz should equal 1.0:")
    print(f"    CUDA: {integral_check_cuda:.4f} (error: {abs(integral_check_cuda-1.0)*100:.2f}%)")
    print(f"    waLBerla: {integral_check_walberla:.4f} (error: {abs(integral_check_walberla-1.0)*100:.2f}%)")
    print(f"  Peak surface intensity I₀ = {I0/1e6:.2f} MW/m²")
    print(f"  Total absorbed power (requires 2D radial integration):")
    print(f"    Expected: P × η = {P*eta:.1f} W")

    # Peak heat source comparison
    print(f"\nPeak Volumetric Heat Source (at surface):")
    print(f"  CUDA: {Q_cuda[0]/1e12:.3f} TW/m³")
    print(f"  waLBerla: {Q_walberla[0]/1e12:.3f} TW/m³")
    print(f"  Ratio (CUDA/waLBerla): {Q_cuda[0]/Q_walberla[0]:.2f}×")

    # Penetration depth metrics
    # Find depth where Q drops to 10% of peak
    idx_cuda_10 = np.argmax(Q_cuda < 0.1 * Q_cuda[0])
    idx_walberla_10 = np.argmax(Q_walberla < 0.1 * Q_walberla[0])

    print(f"\nEffective Absorption Depth (90% energy absorbed):")
    print(f"  CUDA: {z[idx_cuda_10]*1e6:.1f} μm (theory: {2.3*delta_cuda*1e6:.1f} μm)")
    print(f"  waLBerla: {z[idx_walberla_10]*1e6:.1f} μm (theory: {2.3*delta_walberla_assumed*1e6:.1f} μm)")

    # Expected temperature impact
    print(f"\nExpected Temperature Impact:")
    print(f"  Peak temperature ratio (steady-state):")
    print(f"    T_CUDA / T_waLBerla ≈ (Q_CUDA / Q_waLBerla)^0.5")
    print(f"    ≈ {np.sqrt(Q_cuda[0]/Q_walberla[0]):.2f}×")
    print(f"  Example: If T_waLBerla = 2000 K, then T_CUDA ≈ {2000 * np.sqrt(Q_cuda[0]/Q_walberla[0]):.0f} K")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    ax1.plot(z*1e6, Q_cuda/1e12, label=f'CUDA (δ={delta_cuda*1e6:.0f} μm)', linewidth=2.5, color='#e74c3c')
    ax1.plot(z*1e6, Q_walberla/1e12, label=f'waLBerla (δ={delta_walberla_assumed*1e6:.0f} μm)', linewidth=2.5, color='#3498db')
    ax1.axhline(Q_cuda[0]/1e12 * 0.1, color='gray', linestyle='--', alpha=0.5, label='10% level')
    ax1.set_xlabel('Depth below surface [μm]', fontsize=12)
    ax1.set_ylabel('Volumetric heat source [TW/m³]', fontsize=12)
    ax1.set_title('Beer-Lambert Absorption Profiles (Linear)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)

    # Log scale
    ax2.semilogy(z*1e6, Q_cuda/1e12, label=f'CUDA (δ={delta_cuda*1e6:.0f} μm)', linewidth=2.5, color='#e74c3c')
    ax2.semilogy(z*1e6, Q_walberla/1e12, label=f'waLBerla (δ={delta_walberla_assumed*1e6:.0f} μm)', linewidth=2.5, color='#3498db')
    ax2.axhline(Q_cuda[0]/1e12 * 0.1, color='gray', linestyle='--', alpha=0.5, label='10% level')
    ax2.axhline(Q_cuda[0]/1e12 * 0.01, color='gray', linestyle=':', alpha=0.5, label='1% level')
    ax2.set_xlabel('Depth below surface [μm]', fontsize=12)
    ax2.set_ylabel('Volumetric heat source [TW/m³] (log scale)', fontsize=12)
    ax2.set_title('Beer-Lambert Absorption Profiles (Log)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 150)
    ax2.set_ylim(1e-5, 10)

    plt.tight_layout()
    output_file = 'heat_source_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Generate data file for further analysis
    data_file = 'heat_source_profiles.csv'
    np.savetxt(data_file,
               np.column_stack([z*1e6, Q_cuda/1e12, Q_walberla/1e12]),
               header='z_um,Q_CUDA_TW_per_m3,Q_waLBerla_TW_per_m3',
               delimiter=',',
               comments='')
    print(f"Data saved to: {data_file}")

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    if abs(Q_cuda[0]/Q_walberla[0] - 1.0) > 0.5:
        print("⚠️  WARNING: Large heat source discrepancy detected!")
        print(f"   Peak heat source differs by {abs(Q_cuda[0]/Q_walberla[0] - 1.0)*100:.0f}%")
        print("   This will cause significant temperature differences.")
        print("\n   RECOMMENDATION:")
        print("   1. Verify waLBerla penetration depth parameter")
        print("   2. Match penetration depths between implementations")
        print("   3. Re-run comparison tests")
    else:
        print("✅ Heat source implementations are consistent")

    print("="*70)

if __name__ == '__main__':
    main()
