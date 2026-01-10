#!/usr/bin/env python3
"""
Marangoni Velocity Profile Validation Against Analytical Solutions

This script analyzes VTK output from test_marangoni_velocity.cu and compares
the computed velocity profiles against analytical solutions.

Analytical Solution (Young et al. 1959):
For thermocapillary flow in a thin layer with linear temperature gradient:
  v_s = (dσ/dT) × (∂T/∂x) × h / (2μ)

where:
  h = layer thickness
  μ = dynamic viscosity
  dσ/dT = surface tension temperature coefficient
  ∂T/∂x = temperature gradient (tangential to interface)

The velocity profile should vary linearly from the surface to the bottom wall.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# === PARAMETERS ===
VTK_DIR = "phase6_test2c_visualization"
OUTPUT_DIR = "marangoni_validation_results"

# Material properties (Ti6Al4V liquid)
RHO_LIQUID = 4110.0        # kg/m³
MU_LIQUID = 5.0e-3         # Pa·s
DSIGMA_DT = -2.6e-4        # N/(m·K)

# Expected from test configuration
DX = 2.0e-6                # m (2 μm resolution)
NX = NY = 64
NZ = 32

# Temperature field parameters (from test)
T_HOT = 2500.0             # K
T_COLD = 2000.0            # K
R_HOT = 30e-6              # 30 μm hot zone radius
R_DECAY = 50e-6            # 50 μm decay length

def load_latest_vtk(vtk_dir):
    """Load the latest VTK file from the output directory."""
    vtk_files = sorted(Path(vtk_dir).glob("marangoni_flow_*.vtk"))
    if not vtk_files:
        raise FileNotFoundError(f"No VTK files found in {vtk_dir}")

    latest_file = vtk_files[-1]
    print(f"Loading VTK file: {latest_file}")

    mesh = pv.read(str(latest_file))
    return mesh

def extract_temperature_gradient(mesh, center_idx):
    """
    Extract temperature gradient at the interface.

    Uses radial gradient from the hot center.
    """
    temperature = mesh["Temperature"]

    # Get grid dimensions
    dims = mesh.dimensions
    nx, ny, nz = dims[0], dims[1], dims[2]

    # Reshape to 3D
    T_3d = temperature.reshape((nz, ny, nx))

    # Extract mid-plane (z at interface)
    z_interface = nz // 10  # Interface at 10% height
    T_plane = T_3d[z_interface, :, :]

    # Compute gradient magnitude in x-y plane
    grad_y, grad_x = np.gradient(T_plane, DX, DX)
    grad_T_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Average gradient at interface (radial)
    center_i, center_j = nx // 2, ny // 2

    # Sample gradient in annular region around hot zone
    grad_samples = []
    for di in range(-5, 6):
        for dj in range(-5, 6):
            i = center_i + di
            j = center_j + dj
            r = np.sqrt(di**2 + dj**2) * DX

            # Sample at transition region (R_HOT to R_DECAY)
            if R_HOT <= r <= R_DECAY and 0 <= i < nx and 0 <= j < ny:
                grad_samples.append(grad_T_mag[j, i])

    avg_grad_T = np.mean(grad_samples) if grad_samples else 0.0
    max_grad_T = np.max(grad_samples) if grad_samples else 0.0

    return avg_grad_T, max_grad_T

def compute_analytical_velocity(grad_T, h_layer):
    """
    Compute analytical surface velocity from Young et al. (1959).

    v_s = |dσ/dT| × |∇T| × h / (2μ)

    Parameters:
    - grad_T: Temperature gradient magnitude [K/m]
    - h_layer: Layer thickness [m]

    Returns:
    - v_s: Surface velocity [m/s]
    """
    v_s = np.abs(DSIGMA_DT) * grad_T * h_layer / (2.0 * MU_LIQUID)
    return v_s

def extract_velocity_profile(mesh, extract_line='center'):
    """
    Extract velocity profile along a vertical line through the domain.

    Parameters:
    - mesh: PyVista mesh
    - extract_line: 'center' for central vertical line, or (x, y) coordinates

    Returns:
    - z_coords: Array of z-coordinates
    - v_mag: Array of velocity magnitudes
    - v_x, v_y, v_z: Velocity components
    """
    velocity = mesh["Velocity"]
    fill_level = mesh["FillLevel"]

    # Get grid dimensions
    dims = mesh.dimensions
    nx, ny, nz = dims[0], dims[1], dims[2]

    # Reshape fields
    v_x = velocity[:, 0].reshape((nz, ny, nx))
    v_y = velocity[:, 1].reshape((nz, ny, nx))
    v_z = velocity[:, 2].reshape((nz, ny, nx))
    fill = fill_level.reshape((nz, ny, nx))

    # Extract along vertical centerline
    i_center = nx // 2
    j_center = ny // 2

    z_coords = np.arange(nz) * DX
    v_mag_profile = np.sqrt(v_x[:, j_center, i_center]**2 +
                            v_y[:, j_center, i_center]**2 +
                            v_z[:, j_center, i_center]**2)

    vx_profile = v_x[:, j_center, i_center]
    vy_profile = v_y[:, j_center, i_center]
    vz_profile = v_z[:, j_center, i_center]
    fill_profile = fill[:, j_center, i_center]

    return z_coords, v_mag_profile, vx_profile, vy_profile, vz_profile, fill_profile

def extract_interface_velocity_statistics(mesh):
    """
    Extract velocity statistics at the interface.

    Returns:
    - max_v: Maximum velocity at interface
    - mean_v: Mean velocity at interface
    - std_v: Standard deviation
    - num_cells: Number of interface cells
    """
    velocity = mesh["Velocity"]
    fill_level = mesh["FillLevel"]

    # Compute velocity magnitude
    v_mag = np.linalg.norm(velocity, axis=1)

    # Interface cells: 0.1 < f < 0.9
    interface_mask = (fill_level > 0.1) & (fill_level < 0.9)

    if not np.any(interface_mask):
        return 0.0, 0.0, 0.0, 0

    v_interface = v_mag[interface_mask]

    max_v = np.max(v_interface)
    mean_v = np.mean(v_interface)
    std_v = np.std(v_interface)
    num_cells = np.sum(interface_mask)

    return max_v, mean_v, std_v, num_cells

def compute_analytical_profile(z_coords, h_layer, v_surface):
    """
    Compute analytical velocity profile for linear shear flow.

    For thermocapillary flow with surface stress and no-slip bottom:
      v(z) = v_s × (z / h)

    where v_s is the surface velocity.

    Parameters:
    - z_coords: Array of z-coordinates [m]
    - h_layer: Layer thickness [m]
    - v_surface: Surface velocity [m/s]

    Returns:
    - v_analytical: Array of analytical velocities [m/s]
    """
    # Find liquid region (z < z_interface)
    z_interface_idx = int(0.1 * len(z_coords))
    z_interface = z_coords[z_interface_idx]

    # Linear profile in liquid region
    v_analytical = np.zeros_like(z_coords)
    liquid_mask = z_coords <= z_interface
    v_analytical[liquid_mask] = v_surface * (z_coords[liquid_mask] / z_interface)

    return v_analytical

def plot_velocity_profile_comparison(z_coords, v_mag_profile, v_analytical,
                                     fill_profile, output_path):
    """
    Plot computed velocity profile vs analytical solution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity profile comparison
    ax1.plot(v_mag_profile * 1e3, z_coords * 1e6, 'o-',
             label='Simulated', markersize=4, linewidth=1.5)
    ax1.plot(v_analytical * 1e3, z_coords * 1e6, 'r--',
             label='Analytical (Linear)', linewidth=2)
    ax1.set_xlabel('Velocity Magnitude [mm/s]', fontsize=12)
    ax1.set_ylabel('Height z [μm]', fontsize=12)
    ax1.set_title('Marangoni Velocity Profile', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Fill level profile
    ax2.plot(fill_profile, z_coords * 1e6, 'b-', linewidth=2)
    ax2.axhline(z_coords[int(0.1*len(z_coords))] * 1e6,
                color='r', linestyle='--', label='Interface (10%)')
    ax2.set_xlabel('Fill Level [-]', fontsize=12)
    ax2.set_ylabel('Height z [μm]', fontsize=12)
    ax2.set_title('Fill Level Profile', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved velocity profile comparison: {output_path}")
    plt.close()

def plot_error_analysis(z_coords, v_mag_profile, v_analytical, fill_profile, output_path):
    """
    Plot error metrics and residuals.
    """
    # Compute error only in liquid region
    z_interface_idx = int(0.1 * len(z_coords))
    liquid_mask = np.arange(len(z_coords)) <= z_interface_idx

    # Relative error
    error = v_mag_profile - v_analytical
    rel_error = np.zeros_like(error)
    valid_mask = (v_analytical > 1e-6) & liquid_mask
    rel_error[valid_mask] = 100.0 * error[valid_mask] / v_analytical[valid_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute error
    ax1.plot(error[liquid_mask] * 1e3, z_coords[liquid_mask] * 1e6,
             'o-', markersize=4, linewidth=1.5)
    ax1.axvline(0, color='r', linestyle='--', linewidth=1)
    ax1.set_xlabel('Error (Simulated - Analytical) [mm/s]', fontsize=12)
    ax1.set_ylabel('Height z [μm]', fontsize=12)
    ax1.set_title('Absolute Error', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Relative error
    ax2.plot(rel_error[liquid_mask], z_coords[liquid_mask] * 1e6,
             'o-', markersize=4, linewidth=1.5)
    ax2.axvline(0, color='r', linestyle='--', linewidth=1)
    ax2.axvline(-20, color='orange', linestyle=':', linewidth=1, label='±20% tolerance')
    ax2.axvline(20, color='orange', linestyle=':', linewidth=1)
    ax2.set_xlabel('Relative Error [%]', fontsize=12)
    ax2.set_ylabel('Height z [μm]', fontsize=12)
    ax2.set_title('Relative Error', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved error analysis: {output_path}")
    plt.close()

def compute_error_metrics(v_mag_profile, v_analytical, z_coords):
    """
    Compute quantitative error metrics.

    Returns dictionary with:
    - L2 norm
    - L_inf norm
    - Mean absolute error (MAE)
    - Mean relative error (MRE)
    - Root mean square error (RMSE)
    """
    # Focus on liquid region only
    z_interface_idx = int(0.1 * len(z_coords))
    liquid_mask = np.arange(len(z_coords)) <= z_interface_idx

    v_sim = v_mag_profile[liquid_mask]
    v_ana = v_analytical[liquid_mask]

    # Remove zero-velocity regions for relative error
    nonzero_mask = v_ana > 1e-6
    v_sim_nz = v_sim[nonzero_mask]
    v_ana_nz = v_ana[nonzero_mask]

    # Error metrics
    error = v_sim - v_ana
    error_nz = v_sim_nz - v_ana_nz

    L2_norm = np.sqrt(np.sum(error**2) / len(error))
    L_inf_norm = np.max(np.abs(error))
    MAE = np.mean(np.abs(error))
    RMSE = np.sqrt(np.mean(error**2))

    # Relative error (only non-zero analytical values)
    if len(v_ana_nz) > 0:
        rel_error = error_nz / v_ana_nz
        MRE = np.mean(np.abs(rel_error)) * 100.0  # %
        max_rel_error = np.max(np.abs(rel_error)) * 100.0
    else:
        MRE = 0.0
        max_rel_error = 0.0

    metrics = {
        'L2_norm': L2_norm,
        'L_inf_norm': L_inf_norm,
        'MAE': MAE,
        'RMSE': RMSE,
        'MRE_percent': MRE,
        'max_rel_error_percent': max_rel_error
    }

    return metrics

def generate_validation_report(mesh, output_dir):
    """
    Generate complete validation report with plots and metrics.
    """
    Path(output_dir).mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("MARANGONI VELOCITY VALIDATION REPORT")
    print("="*60 + "\n")

    # === 1. Extract interface velocity statistics ===
    max_v, mean_v, std_v, num_cells = extract_interface_velocity_statistics(mesh)

    print("Interface Velocity Statistics:")
    print(f"  Number of interface cells: {num_cells}")
    print(f"  Maximum velocity: {max_v:.4f} m/s")
    print(f"  Mean velocity: {mean_v:.4f} m/s")
    print(f"  Standard deviation: {std_v:.4f} m/s")
    print()

    # === 2. Extract temperature gradient ===
    center_idx = (NX // 2, NY // 2)
    avg_grad_T, max_grad_T = extract_temperature_gradient(mesh, center_idx)

    print("Temperature Gradient:")
    print(f"  Average |∇T| at interface: {avg_grad_T:.3e} K/m")
    print(f"  Maximum |∇T| at interface: {max_grad_T:.3e} K/m")
    print(f"  Radial decay length: {R_DECAY*1e6:.1f} μm")
    print()

    # === 3. Compute analytical solution ===
    h_layer = NZ * DX * 0.1  # Liquid layer thickness (10% of domain)
    v_analytical_surface = compute_analytical_velocity(avg_grad_T, h_layer)

    print("Analytical Solution (Young et al. 1959):")
    print(f"  Layer thickness h: {h_layer*1e6:.2f} μm")
    print(f"  Temperature gradient: {avg_grad_T:.3e} K/m")
    print(f"  Surface tension coeff: {DSIGMA_DT*1e4:.2f} × 10⁻⁴ N/(m·K)")
    print(f"  Dynamic viscosity: {MU_LIQUID*1e3:.2f} mPa·s")
    print(f"  Expected surface velocity: {v_analytical_surface:.4f} m/s")
    print()

    # === 4. Extract velocity profile ===
    z_coords, v_mag_profile, vx, vy, vz, fill_profile = extract_velocity_profile(mesh)

    # Compute analytical profile
    v_analytical_profile = compute_analytical_profile(z_coords, h_layer, v_analytical_surface)

    # === 5. Compute error metrics ===
    metrics = compute_error_metrics(v_mag_profile, v_analytical_profile, z_coords)

    print("Error Metrics (Liquid Region Only):")
    print(f"  L2 norm: {metrics['L2_norm']:.6f} m/s")
    print(f"  L∞ norm: {metrics['L_inf_norm']:.6f} m/s")
    print(f"  MAE: {metrics['MAE']:.6f} m/s")
    print(f"  RMSE: {metrics['RMSE']:.6f} m/s")
    print(f"  Mean Relative Error: {metrics['MRE_percent']:.2f}%")
    print(f"  Max Relative Error: {metrics['max_rel_error_percent']:.2f}%")
    print()

    # === 6. Validation criteria ===
    print("Validation Criteria:")

    # Surface velocity comparison
    v_sim_surface = v_mag_profile[int(0.1*len(z_coords))]  # At interface
    rel_error_surface = 100.0 * abs(v_sim_surface - v_analytical_surface) / v_analytical_surface

    print(f"  Simulated surface velocity: {v_sim_surface:.4f} m/s")
    print(f"  Analytical surface velocity: {v_analytical_surface:.4f} m/s")
    print(f"  Relative error: {rel_error_surface:.2f}%")

    if rel_error_surface <= 20.0:
        print("  ✓ PASS: Surface velocity within 20% of analytical")
    else:
        print(f"  ✗ FAIL: Surface velocity error {rel_error_surface:.1f}% > 20%")

    # Mean relative error
    if metrics['MRE_percent'] <= 30.0:
        print(f"  ✓ PASS: Mean relative error {metrics['MRE_percent']:.1f}% < 30%")
    else:
        print(f"  ✗ FAIL: Mean relative error {metrics['MRE_percent']:.1f}% > 30%")

    # Flow direction check
    # Velocity should be primarily horizontal (x-y plane)
    v_horizontal = np.sqrt(vx**2 + vy**2)
    z_interface_idx = int(0.1 * len(z_coords))
    v_horiz_surface = v_horizontal[z_interface_idx]
    v_vert_surface = abs(vz[z_interface_idx])

    print(f"\nFlow Direction Check:")
    print(f"  Horizontal velocity at surface: {v_horiz_surface:.4f} m/s")
    print(f"  Vertical velocity at surface: {v_vert_surface:.4f} m/s")
    print(f"  Ratio (horizontal/vertical): {v_horiz_surface/max(v_vert_surface, 1e-10):.2f}")

    if v_horiz_surface > 5.0 * v_vert_surface:
        print("  ✓ PASS: Flow primarily tangential to interface")
    else:
        print("  ✗ WARNING: Significant vertical velocity component")

    print()

    # === 7. Generate plots ===
    plot_velocity_profile_comparison(
        z_coords, v_mag_profile, v_analytical_profile, fill_profile,
        Path(output_dir) / "velocity_profile_comparison.png"
    )

    plot_error_analysis(
        z_coords, v_mag_profile, v_analytical_profile, fill_profile,
        Path(output_dir) / "error_analysis.png"
    )

    # === 8. Save numerical results ===
    results_file = Path(output_dir) / "validation_metrics.txt"
    with open(results_file, 'w') as f:
        f.write("MARANGONI VELOCITY VALIDATION METRICS\n")
        f.write("="*60 + "\n\n")

        f.write("Analytical Solution:\n")
        f.write(f"  Surface velocity: {v_analytical_surface:.6e} m/s\n")
        f.write(f"  Temperature gradient: {avg_grad_T:.6e} K/m\n")
        f.write(f"  Layer thickness: {h_layer:.6e} m\n\n")

        f.write("Simulated Results:\n")
        f.write(f"  Surface velocity: {v_sim_surface:.6e} m/s\n")
        f.write(f"  Maximum interface velocity: {max_v:.6e} m/s\n")
        f.write(f"  Mean interface velocity: {mean_v:.6e} m/s\n\n")

        f.write("Error Metrics:\n")
        for key, val in metrics.items():
            f.write(f"  {key}: {val:.6e}\n")
        f.write(f"\nSurface velocity relative error: {rel_error_surface:.2f}%\n")

        f.write("\nValidation Status:\n")
        if rel_error_surface <= 20.0 and metrics['MRE_percent'] <= 30.0:
            f.write("  ✓ PASS\n")
        else:
            f.write("  ✗ FAIL\n")

    print(f"Validation metrics saved to: {results_file}")
    print("="*60 + "\n")

    return metrics, rel_error_surface

def main():
    """Main analysis workflow."""
    try:
        # Load VTK data
        mesh = load_latest_vtk(VTK_DIR)

        # Generate validation report
        metrics, rel_error = generate_validation_report(mesh, OUTPUT_DIR)

        # Return exit code based on validation
        if rel_error <= 20.0 and metrics['MRE_percent'] <= 30.0:
            print("Validation PASSED")
            return 0
        else:
            print("Validation FAILED")
            return 1

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())
