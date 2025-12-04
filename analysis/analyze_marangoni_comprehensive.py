#!/usr/bin/env python3
"""
Comprehensive Marangoni Flow Analysis for Multiphysics LBM Simulation

Analyzes:
- Marangoni-driven velocity field (surface tension gradient effects)
- Temperature field and gradients at liquid-gas interface
- Liquid fraction distribution
- Coupling between thermal and fluid fields
- Time evolution of Marangoni convection
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# === PARAMETERS (modify these) ===
VTK_DIR = "/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization"
VTK_PATTERN = "marangoni_flow_*.vtk"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/marangoni_results"

# Physical parameters (from simulation)
GRID_SPACING = 2e-6  # meters (2 microns)
SOLIDUS_TEMP = 1650.0  # K (typical for metals)
LIQUIDUS_TEMP = 1700.0  # K
BOILING_TEMP = 3100.0  # K (typical for metals)

# Analysis parameters
LIQUID_FRACTION_THRESHOLD = 0.5  # Define liquid region
SURFACE_FILL_LEVEL = (0.4, 0.6)  # Range to identify free surface
VERTICAL_SLICE_Y = 0.5  # Normalized Y position for cross-section

# === HELPER FUNCTIONS ===

def get_physical_coords(mesh):
    """Extract physical coordinates in millimeters"""
    points = mesh.points
    x_mm = points[:, 0] * 1e3  # Convert m to mm
    y_mm = points[:, 1] * 1e3
    z_mm = points[:, 2] * 1e3
    return x_mm, y_mm, z_mm

def compute_velocity_magnitude(velocity):
    """Compute velocity magnitude from vector field"""
    return np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2 + velocity[:, 2]**2)

def extract_surface_layer(mesh, fill_level_range=(0.4, 0.6)):
    """Extract cells near the free surface using fill level"""
    if 'FillLevel' not in mesh.array_names:
        # If FillLevel not available, use liquid fraction as proxy
        if 'LiquidFraction' in mesh.array_names:
            liq_frac = mesh['LiquidFraction']
            surface_mask = (liq_frac >= 0.3) & (liq_frac <= 0.7)
            return mesh.extract_points(surface_mask)
        else:
            # Return empty mesh
            return mesh.extract_points(np.zeros(mesh.n_points, dtype=bool))

    fill_level = mesh['FillLevel']
    surface_mask = (fill_level >= fill_level_range[0]) & (fill_level <= fill_level_range[1])
    return mesh.extract_points(surface_mask)

def compute_temperature_gradient(mesh):
    """Compute temperature gradient magnitude"""
    temp_grad = mesh.compute_derivative('Temperature', gradient='gradient')
    if 'gradient' in temp_grad.array_names:
        grad_vectors = temp_grad['gradient']
        grad_mag = np.sqrt(grad_vectors[:, 0]**2 + grad_vectors[:, 1]**2 + grad_vectors[:, 2]**2)
        return grad_mag
    return None

# === MAIN ANALYSIS ===

def analyze_single_timestep(vtk_file):
    """Analyze a single VTK file"""
    try:
        mesh = pv.read(vtk_file)
    except Exception as e:
        print(f"  WARNING: Failed to read {Path(vtk_file).name}: {e}")
        return None, None

    # Check that all required fields exist
    required_fields = ['Velocity', 'Temperature', 'LiquidFraction']
    for field in required_fields:
        if field not in mesh.array_names:
            print(f"  WARNING: Missing field '{field}' in {Path(vtk_file).name}")
            return None, None

    # Extract data
    velocity = mesh['Velocity']
    temperature = mesh['Temperature']
    liquid_frac = mesh['LiquidFraction']

    # FillLevel is optional
    fill_level = mesh['FillLevel'] if 'FillLevel' in mesh.array_names else np.ones(mesh.n_points)

    # Compute velocity magnitude
    vel_mag = compute_velocity_magnitude(velocity)
    mesh['VelocityMagnitude'] = vel_mag

    # Identify liquid region
    liquid_mask = liquid_frac > LIQUID_FRACTION_THRESHOLD

    # Extract surface layer
    surface = extract_surface_layer(mesh, SURFACE_FILL_LEVEL)

    results = {
        'filename': Path(vtk_file).name,
        'timestep': int(Path(vtk_file).stem.split('_')[-1]),

        # Velocity statistics (full domain)
        'vel_max_global': np.max(vel_mag),
        'vel_mean_global': np.mean(vel_mag),
        'vel_std_global': np.std(vel_mag),

        # Velocity in liquid region only
        'vel_max_liquid': np.max(vel_mag[liquid_mask]) if np.any(liquid_mask) else 0,
        'vel_mean_liquid': np.mean(vel_mag[liquid_mask]) if np.any(liquid_mask) else 0,

        # Temperature statistics
        'temp_max': np.max(temperature),
        'temp_min': np.min(temperature),
        'temp_mean': np.mean(temperature),
        'temp_mean_liquid': np.mean(temperature[liquid_mask]) if np.any(liquid_mask) else 0,

        # Surface statistics
        'surface_points': surface.n_points if surface.n_points > 0 else 0,
        'surface_temp_max': np.max(surface['Temperature']) if surface.n_points > 0 else 0,
        'surface_temp_min': np.min(surface['Temperature']) if surface.n_points > 0 else 0,
        'surface_vel_max': np.max(surface['VelocityMagnitude']) if surface.n_points > 0 else 0,

        # Liquid fraction statistics
        'liquid_volume_frac': np.mean(liquid_mask),
        'liquid_frac_max': np.max(liquid_frac),
    }

    # Compute temperature gradient at surface
    if surface.n_points > 0:
        grad_mag = compute_temperature_gradient(surface)
        if grad_mag is not None:
            results['surface_temp_gradient_max'] = np.max(grad_mag)
            results['surface_temp_gradient_mean'] = np.mean(grad_mag)
        else:
            results['surface_temp_gradient_max'] = 0
            results['surface_temp_gradient_mean'] = 0
    else:
        results['surface_temp_gradient_max'] = 0
        results['surface_temp_gradient_mean'] = 0

    return results, mesh

def analyze_all_timesteps():
    """Analyze all VTK files in directory"""
    vtk_files = sorted(glob.glob(f"{VTK_DIR}/{VTK_PATTERN}"))

    if len(vtk_files) == 0:
        print(f"ERROR: No VTK files found matching {VTK_DIR}/{VTK_PATTERN}")
        return None

    print(f"Found {len(vtk_files)} VTK files")
    print(f"Analyzing from {Path(vtk_files[0]).name} to {Path(vtk_files[-1]).name}")

    all_results = []

    for i, vtk_file in enumerate(vtk_files):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(vtk_files)}: {Path(vtk_file).name}")

        results, _ = analyze_single_timestep(vtk_file)
        if results is not None:
            all_results.append(results)

    return all_results

def create_summary_plots(all_results):
    """Create time evolution plots"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    timesteps = [r['timestep'] for r in all_results]

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Marangoni Flow Time Evolution', fontsize=16, fontweight='bold')

    # 1. Velocity magnitude evolution
    ax = axes[0, 0]
    ax.plot(timesteps, [r['vel_max_global'] for r in all_results], 'b-', linewidth=2, label='Global max')
    ax.plot(timesteps, [r['vel_max_liquid'] for r in all_results], 'r-', linewidth=2, label='Liquid max')
    ax.plot(timesteps, [r['vel_mean_liquid'] for r in all_results], 'g--', linewidth=1.5, label='Liquid mean')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Expected ~0.5 m/s')
    ax.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Expected max ~2.0 m/s')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Magnitude Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Surface velocity (Marangoni effect)
    ax = axes[0, 1]
    ax.plot(timesteps, [r['surface_vel_max'] for r in all_results], 'purple', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Surface Velocity (m/s)')
    ax.set_title('Surface Velocity (Marangoni-Driven)')
    ax.grid(True, alpha=0.3)

    # 3. Temperature evolution
    ax = axes[1, 0]
    ax.plot(timesteps, [r['temp_max'] for r in all_results], 'r-', linewidth=2, label='Max')
    ax.plot(timesteps, [r['temp_mean_liquid'] for r in all_results], 'orange', linewidth=2, label='Liquid mean')
    ax.axhline(SOLIDUS_TEMP, color='blue', linestyle='--', alpha=0.5, label=f'Solidus ({SOLIDUS_TEMP} K)')
    ax.axhline(LIQUIDUS_TEMP, color='cyan', linestyle='--', alpha=0.5, label=f'Liquidus ({LIQUIDUS_TEMP} K)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Surface temperature
    ax = axes[1, 1]
    ax.plot(timesteps, [r['surface_temp_max'] for r in all_results], 'r-', linewidth=2, label='Max')
    ax.plot(timesteps, [r['surface_temp_min'] for r in all_results], 'b-', linewidth=2, label='Min')
    ax.fill_between(timesteps,
                     [r['surface_temp_min'] for r in all_results],
                     [r['surface_temp_max'] for r in all_results],
                     alpha=0.3, color='orange')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Surface Temperature (K)')
    ax.set_title('Surface Temperature Range')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Temperature gradient at surface
    ax = axes[2, 0]
    ax.plot(timesteps, [r['surface_temp_gradient_max'] for r in all_results], 'darkred', linewidth=2, label='Max')
    ax.plot(timesteps, [r['surface_temp_gradient_mean'] for r in all_results], 'orange', linewidth=2, label='Mean')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Temperature Gradient (K/m)')
    ax.set_title('Surface Temperature Gradient (Drives Marangoni)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Liquid volume fraction
    ax = axes[2, 1]
    ax.plot(timesteps, [r['liquid_volume_frac'] * 100 for r in all_results], 'blue', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Liquid Volume (%)')
    ax.set_title('Liquid Volume Fraction')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = f"{OUTPUT_DIR}/marangoni_time_evolution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved time evolution plot: {output_file}")
    plt.close()

def analyze_final_state(vtk_file):
    """Detailed analysis of final timestep"""
    print(f"\n{'='*60}")
    print(f"Detailed Analysis of Final State: {Path(vtk_file).name}")
    print(f"{'='*60}")

    mesh = pv.read(vtk_file)

    # Get data
    velocity = mesh['Velocity']
    temperature = mesh['Temperature']
    liquid_frac = mesh['LiquidFraction']
    fill_level = mesh['FillLevel']

    vel_mag = compute_velocity_magnitude(velocity)
    mesh['VelocityMagnitude'] = vel_mag

    # Grid information
    dims = mesh.dimensions
    nx, ny, nz = dims
    print(f"\nGrid: {nx} x {ny} x {nz} = {mesh.n_points} points")
    print(f"Domain size: {nx*GRID_SPACING*1e3:.3f} x {ny*GRID_SPACING*1e3:.3f} x {nz*GRID_SPACING*1e3:.3f} mm")

    # Velocity analysis
    print(f"\n--- Velocity Field ---")
    print(f"Max velocity (global): {np.max(vel_mag):.4f} m/s")
    print(f"Mean velocity (global): {np.mean(vel_mag):.4f} m/s")
    print(f"Std velocity (global): {np.std(vel_mag):.4f} m/s")

    # Liquid region analysis
    liquid_mask = liquid_frac > LIQUID_FRACTION_THRESHOLD
    n_liquid = np.sum(liquid_mask)
    print(f"\n--- Liquid Region ({n_liquid} points, {100*n_liquid/mesh.n_points:.2f}%) ---")
    if n_liquid > 0:
        print(f"Max velocity (liquid): {np.max(vel_mag[liquid_mask]):.4f} m/s")
        print(f"Mean velocity (liquid): {np.mean(vel_mag[liquid_mask]):.4f} m/s")
        print(f"Max temperature (liquid): {np.max(temperature[liquid_mask]):.1f} K")
        print(f"Mean temperature (liquid): {np.mean(temperature[liquid_mask]):.1f} K")

    # Surface analysis
    surface = extract_surface_layer(mesh, SURFACE_FILL_LEVEL)
    print(f"\n--- Free Surface ({surface.n_points} points) ---")
    if surface.n_points > 0:
        surf_vel = surface['VelocityMagnitude']
        surf_temp = surface['Temperature']
        print(f"Max surface velocity: {np.max(surf_vel):.4f} m/s (Marangoni-driven)")
        print(f"Mean surface velocity: {np.mean(surf_vel):.4f} m/s")
        print(f"Surface temperature range: {np.min(surf_temp):.1f} - {np.max(surf_temp):.1f} K")
        print(f"Surface temperature delta: {np.max(surf_temp) - np.min(surf_temp):.1f} K")

        # Temperature gradient (drives Marangoni)
        grad_mag = compute_temperature_gradient(surface)
        if grad_mag is not None:
            print(f"Max temperature gradient: {np.max(grad_mag):.2e} K/m")
            print(f"Mean temperature gradient: {np.mean(grad_mag):.2e} K/m")

    # Create visualization plots
    create_final_state_plots(mesh, surface)

    return mesh, surface

def create_final_state_plots(mesh, surface):
    """Create detailed plots of final state"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Get mesh dimensions for slicing
    dims = mesh.dimensions
    nx, ny, nz = dims

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Final State: Marangoni Flow Analysis', fontsize=16, fontweight='bold')

    # Extract vertical center slice (XZ plane at Y=center)
    y_center = int(ny / 2)
    slice_xz = mesh.slice(normal='y', origin=mesh.center)

    # 1. Temperature field with velocity vectors
    ax = axes[0, 0]
    points = slice_xz.points
    temp = slice_xz['Temperature']
    vel = slice_xz['Velocity']

    # Create 2D grid for contour plot
    x_coords = points[:, 0] * 1e3  # mm
    z_coords = points[:, 2] * 1e3  # mm

    scatter = ax.scatter(x_coords, z_coords, c=temp, s=5, cmap='hot', vmin=300, vmax=np.max(temp))

    # Add velocity vectors (subsample for clarity)
    stride = max(1, len(x_coords) // 100)
    ax.quiver(x_coords[::stride], z_coords[::stride],
              vel[::stride, 0], vel[::stride, 2],
              scale=5, width=0.003, color='cyan', alpha=0.6)

    plt.colorbar(scatter, ax=ax, label='Temperature (K)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Temperature + Velocity Field (Center Slice)')
    ax.set_aspect('equal')

    # 2. Velocity magnitude distribution
    ax = axes[0, 1]
    vel_mag = mesh['VelocityMagnitude']
    liquid_mask = mesh['LiquidFraction'] > LIQUID_FRACTION_THRESHOLD

    ax.hist(vel_mag[vel_mag > 0], bins=50, alpha=0.7, color='blue', label='All points', edgecolor='black')
    if np.any(liquid_mask):
        ax.hist(vel_mag[liquid_mask], bins=50, alpha=0.7, color='red', label='Liquid only', edgecolor='black')

    ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Expected ~0.5 m/s')
    ax.axvline(2.0, color='darkred', linestyle='--', linewidth=2, label='Expected max ~2.0 m/s')
    ax.set_xlabel('Velocity Magnitude (m/s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Velocity Distribution')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Liquid fraction field
    ax = axes[1, 0]
    liq_frac = slice_xz['LiquidFraction']
    scatter = ax.scatter(x_coords, z_coords, c=liq_frac, s=5, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label='Liquid Fraction')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Liquid Fraction (Center Slice)')
    ax.set_aspect('equal')

    # 4. Surface temperature and velocity correlation
    if surface.n_points > 0:
        ax = axes[1, 1]
        surf_temp = surface['Temperature']
        surf_vel = surface['VelocityMagnitude']

        scatter = ax.scatter(surf_temp, surf_vel, c=surf_vel, s=20, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Velocity (m/s)')
        ax.set_xlabel('Surface Temperature (K)')
        ax.set_ylabel('Surface Velocity (m/s)')
        ax.set_title('Temperature-Velocity Coupling at Surface')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = f"{OUTPUT_DIR}/marangoni_final_state.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved final state plot: {output_file}")
    plt.close()

def print_summary_report(all_results):
    """Print comprehensive summary report"""
    print(f"\n{'='*70}")
    print(f"MARANGONI FLOW COMPREHENSIVE ANALYSIS REPORT")
    print(f"{'='*70}")

    print(f"\nAnalyzed {len(all_results)} timesteps")
    print(f"From timestep {all_results[0]['timestep']} to {all_results[-1]['timestep']}")

    # Find peak values across all timesteps
    max_vel_global = max(r['vel_max_global'] for r in all_results)
    max_vel_liquid = max(r['vel_max_liquid'] for r in all_results)
    max_vel_surface = max(r['surface_vel_max'] for r in all_results)
    max_temp = max(r['temp_max'] for r in all_results)
    max_grad = max(r['surface_temp_gradient_max'] for r in all_results)

    # Find timesteps where peaks occur
    ts_max_vel = [r['timestep'] for r in all_results if r['vel_max_global'] == max_vel_global][0]
    ts_max_surf_vel = [r['timestep'] for r in all_results if r['surface_vel_max'] == max_vel_surface][0]

    print(f"\n--- PEAK VALUES ACROSS SIMULATION ---")
    print(f"Maximum velocity (global): {max_vel_global:.4f} m/s at timestep {ts_max_vel}")
    print(f"Maximum velocity (liquid): {max_vel_liquid:.4f} m/s")
    print(f"Maximum surface velocity: {max_vel_surface:.4f} m/s at timestep {ts_max_surf_vel}")
    print(f"Maximum temperature: {max_temp:.1f} K")
    print(f"Maximum surface temp gradient: {max_grad:.2e} K/m")

    print(f"\n--- EXPECTED VS OBSERVED (Marangoni Flow) ---")
    print(f"Expected velocity range: 0.5 - 2.0 m/s")
    print(f"Observed max velocity: {max_vel_liquid:.4f} m/s")

    if max_vel_liquid >= 0.5 and max_vel_liquid <= 2.5:
        print(f"STATUS: WITHIN EXPECTED RANGE")
    elif max_vel_liquid < 0.5:
        print(f"STATUS: BELOW EXPECTED (possible weak Marangoni effect)")
    else:
        print(f"STATUS: ABOVE EXPECTED (strong Marangoni convection)")

    # Final state
    final = all_results[-1]
    print(f"\n--- FINAL STATE (timestep {final['timestep']}) ---")
    print(f"Velocity (liquid max): {final['vel_max_liquid']:.4f} m/s")
    print(f"Velocity (liquid mean): {final['vel_mean_liquid']:.4f} m/s")
    print(f"Surface velocity: {final['surface_vel_max']:.4f} m/s")
    print(f"Temperature (max): {final['temp_max']:.1f} K")
    print(f"Temperature (liquid mean): {final['temp_mean_liquid']:.1f} K")
    print(f"Surface temp gradient: {final['surface_temp_gradient_mean']:.2e} K/m (mean)")
    print(f"Liquid volume fraction: {final['liquid_volume_frac']*100:.2f}%")

    print(f"\n{'='*70}")

# === MAIN EXECUTION ===

if __name__ == "__main__":
    print("="*70)
    print("MARANGONI FLOW COMPREHENSIVE ANALYSIS")
    print("="*70)
    print(f"VTK Directory: {VTK_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    # Analyze all timesteps
    all_results = analyze_all_timesteps()

    if all_results is None or len(all_results) == 0:
        print("\nERROR: No valid VTK files could be analyzed")
        exit(1)

    print(f"\nSuccessfully analyzed {len(all_results)} valid VTK files")

    # Create time evolution plots
    create_summary_plots(all_results)

    # Detailed analysis of final state
    vtk_files = sorted(glob.glob(f"{VTK_DIR}/{VTK_PATTERN}"))
    final_mesh, final_surface = analyze_final_state(vtk_files[-1])

    # Print comprehensive report
    print_summary_report(all_results)

    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")
