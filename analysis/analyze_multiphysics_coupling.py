#!/usr/bin/env python3
"""
Multiphysics Coupling Analysis: Thermal-Fluid Interaction

Analyzes the coupling between thermal and fluid fields:
- Temperature-driven buoyancy effects
- Marangoni (surface tension gradient) driven flow
- Thermal convection patterns
- Phase change effects on flow
- Energy transport by convection vs conduction
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy.ndimage import gaussian_filter

# === PARAMETERS ===
VTK_DIR = "/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization"
VTK_PATTERN = "marangoni_flow_*.vtk"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/multiphysics_results"

# Select specific timesteps to analyze
TIMESTEPS_TO_ANALYZE = [1000, 3000, 5000, 7000, 10000]

# Physical parameters
GRID_SPACING = 2e-6  # meters
SOLIDUS_TEMP = 1650.0  # K
LIQUIDUS_TEMP = 1700.0  # K

# Analysis regions
LIQUID_FRACTION_THRESHOLD = 0.5
SURFACE_FILL_RANGE = (0.4, 0.6)

# === ANALYSIS FUNCTIONS ===

def compute_vorticity_magnitude(mesh):
    """Compute vorticity (curl of velocity) magnitude"""
    # Compute curl of velocity
    mesh_curl = mesh.compute_derivative('Velocity', gradient='curl', vorticity=True)

    if 'vorticity' in mesh_curl.array_names:
        vorticity = mesh_curl['vorticity']
        vort_mag = np.sqrt(vorticity[:, 0]**2 + vorticity[:, 1]**2 + vorticity[:, 2]**2)
        return vort_mag
    return None

def compute_reynolds_number(velocity_mag, length_scale, kinematic_viscosity=1e-6):
    """
    Estimate local Reynolds number
    Re = U * L / nu
    """
    Re = velocity_mag * length_scale / kinematic_viscosity
    return Re

def compute_peclet_number(velocity_mag, length_scale, thermal_diffusivity=2e-5):
    """
    Estimate Peclet number (ratio of advection to diffusion)
    Pe = U * L / alpha
    """
    Pe = velocity_mag * length_scale / thermal_diffusivity
    return Pe

def identify_convection_cells(mesh):
    """Identify convection patterns using vorticity"""
    vort_mag = compute_vorticity_magnitude(mesh)

    if vort_mag is None:
        print("Warning: Could not compute vorticity")
        return None, None

    mesh['Vorticity'] = vort_mag

    # Identify regions of significant vorticity (convection cells)
    vort_threshold = np.percentile(vort_mag[vort_mag > 0], 90)
    convection_mask = vort_mag > vort_threshold

    return vort_mag, convection_mask

def analyze_thermal_fluid_coupling(vtk_file):
    """Comprehensive analysis of thermal-fluid coupling"""
    mesh = pv.read(vtk_file)

    # Extract fields
    velocity = mesh['Velocity']
    temperature = mesh['Temperature']
    liquid_frac = mesh['LiquidFraction']

    vel_mag = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2 + velocity[:, 2]**2)
    mesh['VelocityMagnitude'] = vel_mag

    # Compute vorticity
    vort_mag, convection_mask = identify_convection_cells(mesh)

    # Identify liquid region
    liquid_mask = liquid_frac > LIQUID_FRACTION_THRESHOLD

    # Compute dimensionless numbers
    L_char = 50 * GRID_SPACING  # Characteristic length ~100 microns
    Re = compute_reynolds_number(vel_mag, L_char)
    Pe = compute_peclet_number(vel_mag, L_char)
    mesh['Reynolds'] = Re
    mesh['Peclet'] = Pe

    results = {
        'timestep': int(Path(vtk_file).stem.split('_')[-1]),

        # Velocity
        'vel_max': np.max(vel_mag),
        'vel_max_liquid': np.max(vel_mag[liquid_mask]) if np.any(liquid_mask) else 0,

        # Temperature
        'temp_max': np.max(temperature),
        'temp_min': np.min(temperature),
        'temp_range': np.max(temperature) - np.min(temperature),

        # Vorticity (convection strength)
        'vorticity_max': np.max(vort_mag) if vort_mag is not None else 0,
        'vorticity_mean': np.mean(vort_mag[liquid_mask]) if vort_mag is not None and np.any(liquid_mask) else 0,
        'convection_volume': np.sum(convection_mask) / len(convection_mask) if convection_mask is not None else 0,

        # Dimensionless numbers
        'Re_max': np.max(Re[liquid_mask]) if np.any(liquid_mask) else 0,
        'Re_mean': np.mean(Re[liquid_mask]) if np.any(liquid_mask) else 0,
        'Pe_max': np.max(Pe[liquid_mask]) if np.any(liquid_mask) else 0,
        'Pe_mean': np.mean(Pe[liquid_mask]) if np.any(liquid_mask) else 0,
    }

    return results, mesh

def create_coupling_visualization(mesh, timestep):
    """Create detailed visualization of thermal-fluid coupling"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Multiphysics Coupling Analysis - Timestep {timestep}', fontsize=16, fontweight='bold')

    # Get center slice (XZ plane)
    slice_xz = mesh.slice(normal='y', origin=mesh.center)
    points = slice_xz.points
    x_mm = points[:, 0] * 1e3
    z_mm = points[:, 2] * 1e3

    # 1. Temperature field
    ax = axes[0, 0]
    temp = slice_xz['Temperature']
    scatter = ax.scatter(x_mm, z_mm, c=temp, s=8, cmap='hot', vmin=300)
    plt.colorbar(scatter, ax=ax, label='Temperature (K)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Temperature Field')
    ax.set_aspect('equal')

    # 2. Velocity magnitude
    ax = axes[0, 1]
    vel_mag = slice_xz['VelocityMagnitude']
    scatter = ax.scatter(x_mm, z_mm, c=vel_mag, s=8, cmap='viridis', vmin=0)
    plt.colorbar(scatter, ax=ax, label='Velocity (m/s)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Velocity Magnitude')
    ax.set_aspect('equal')

    # 3. Vorticity (convection cells)
    ax = axes[0, 2]
    if 'Vorticity' in slice_xz.array_names:
        vort = slice_xz['Vorticity']
        vort_plot = np.log10(vort + 1e-10)  # Log scale for visualization
        scatter = ax.scatter(x_mm, z_mm, c=vort_plot, s=8, cmap='plasma')
        plt.colorbar(scatter, ax=ax, label='log10(Vorticity)')
        ax.set_title('Vorticity (Convection Cells)')
    else:
        ax.text(0.5, 0.5, 'Vorticity not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Vorticity')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_aspect('equal')

    # 4. Reynolds number
    ax = axes[1, 0]
    if 'Reynolds' in slice_xz.array_names:
        Re = slice_xz['Reynolds']
        Re_plot = np.clip(Re, 0, np.percentile(Re, 99))  # Clip outliers
        scatter = ax.scatter(x_mm, z_mm, c=Re_plot, s=8, cmap='coolwarm')
        plt.colorbar(scatter, ax=ax, label='Re')
        ax.set_title('Reynolds Number (Inertia/Viscous)')
    else:
        ax.text(0.5, 0.5, 'Reynolds not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Reynolds Number')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_aspect('equal')

    # 5. Peclet number
    ax = axes[1, 1]
    if 'Peclet' in slice_xz.array_names:
        Pe = slice_xz['Peclet']
        Pe_plot = np.clip(Pe, 0, np.percentile(Pe, 99))
        scatter = ax.scatter(x_mm, z_mm, c=Pe_plot, s=8, cmap='YlOrRd')
        plt.colorbar(scatter, ax=ax, label='Pe')
        ax.set_title('Peclet Number (Advection/Diffusion)')
    else:
        ax.text(0.5, 0.5, 'Peclet not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Peclet Number')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_aspect('equal')

    # 6. Liquid fraction
    ax = axes[1, 2]
    liq_frac = slice_xz['LiquidFraction']
    scatter = ax.scatter(x_mm, z_mm, c=liq_frac, s=8, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label='Liquid Fraction')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Liquid Fraction (Phase Field)')
    ax.set_aspect('equal')

    plt.tight_layout()

    output_file = f"{OUTPUT_DIR}/coupling_analysis_t{timestep:06d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved coupling visualization: {output_file}")
    plt.close()

def create_streamline_visualization(mesh, timestep):
    """Create streamline visualization showing flow patterns"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Create streamlines using PyVista
    # Extract a slice first for 2D-like visualization
    slice_xz = mesh.slice(normal='y', origin=mesh.center)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    points = slice_xz.points
    x_mm = points[:, 0] * 1e3
    z_mm = points[:, 2] * 1e3
    temp = slice_xz['Temperature']
    velocity = slice_xz['Velocity']

    # Background: temperature
    scatter = ax.scatter(x_mm, z_mm, c=temp, s=5, cmap='hot', vmin=300, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='Temperature (K)')

    # Overlay: velocity vectors (subsampled)
    stride = max(1, len(x_mm) // 150)
    vel_x = velocity[::stride, 0]
    vel_z = velocity[::stride, 2]
    vel_mag = np.sqrt(vel_x**2 + vel_z**2)

    # Color vectors by magnitude
    quiver = ax.quiver(x_mm[::stride], z_mm[::stride], vel_x, vel_z,
                       vel_mag, cmap='cool', scale=3, width=0.004, alpha=0.8)
    plt.colorbar(quiver, ax=ax, label='Velocity (m/s)')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(f'Flow Field with Temperature - Timestep {timestep}')
    ax.set_aspect('equal')

    output_file = f"{OUTPUT_DIR}/streamlines_t{timestep:06d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved streamline plot: {output_file}")
    plt.close()

def analyze_vertical_profiles(mesh, timestep):
    """Extract vertical profiles of temperature and velocity"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    dims = mesh.dimensions
    nx, ny, nz = dims

    # Extract vertical line at domain center
    x_center = nx // 2
    y_center = ny // 2

    # Get points along vertical (Z) direction
    vertical_points = []
    vertical_temp = []
    vertical_vel = []

    points = mesh.points
    temp = mesh['Temperature']
    vel_mag = mesh['VelocityMagnitude']

    # Extract data along vertical centerline
    for iz in range(nz):
        idx = x_center + y_center * nx + iz * nx * ny
        if idx < len(points):
            vertical_points.append(points[idx, 2] * 1e3)  # mm
            vertical_temp.append(temp[idx])
            vertical_vel.append(vel_mag[idx])

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Temperature profile
    ax = axes[0]
    ax.plot(vertical_temp, vertical_points, 'r-', linewidth=2)
    ax.axvline(SOLIDUS_TEMP, color='blue', linestyle='--', label=f'Solidus ({SOLIDUS_TEMP} K)')
    ax.axvline(LIQUIDUS_TEMP, color='cyan', linestyle='--', label=f'Liquidus ({LIQUIDUS_TEMP} K)')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(f'Vertical Temperature Profile - t={timestep}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity profile
    ax = axes[1]
    ax.plot(vertical_vel, vertical_points, 'b-', linewidth=2)
    ax.set_xlabel('Velocity Magnitude (m/s)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(f'Vertical Velocity Profile - t={timestep}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = f"{OUTPUT_DIR}/vertical_profiles_t{timestep:06d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved vertical profiles: {output_file}")
    plt.close()

# === MAIN EXECUTION ===

if __name__ == "__main__":
    print("="*70)
    print("MULTIPHYSICS COUPLING ANALYSIS")
    print("="*70)
    print(f"VTK Directory: {VTK_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    # Get all VTK files
    vtk_files = sorted(glob.glob(f"{VTK_DIR}/{VTK_PATTERN}"))

    if len(vtk_files) == 0:
        print(f"ERROR: No VTK files found")
        exit(1)

    print(f"Found {len(vtk_files)} VTK files")

    # Determine which timesteps to analyze
    available_timesteps = [int(Path(f).stem.split('_')[-1]) for f in vtk_files]
    timesteps_to_use = []

    for target_ts in TIMESTEPS_TO_ANALYZE:
        # Find closest available timestep
        closest = min(available_timesteps, key=lambda x: abs(x - target_ts))
        if closest not in timesteps_to_use:
            timesteps_to_use.append(closest)

    # Add final timestep
    if available_timesteps[-1] not in timesteps_to_use:
        timesteps_to_use.append(available_timesteps[-1])

    timesteps_to_use = sorted(timesteps_to_use)

    print(f"Analyzing timesteps: {timesteps_to_use}")

    # Analyze selected timesteps
    all_results = []

    for ts in timesteps_to_use:
        # Find VTK file for this timestep
        vtk_file = [f for f in vtk_files if int(Path(f).stem.split('_')[-1]) == ts][0]
        print(f"\nAnalyzing timestep {ts}: {Path(vtk_file).name}")

        results, mesh = analyze_thermal_fluid_coupling(vtk_file)
        all_results.append(results)

        # Create visualizations
        create_coupling_visualization(mesh, ts)
        create_streamline_visualization(mesh, ts)
        analyze_vertical_profiles(mesh, ts)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY: Thermal-Fluid Coupling Analysis")
    print(f"{'='*70}")

    for res in all_results:
        print(f"\nTimestep {res['timestep']}:")
        print(f"  Max velocity (liquid): {res['vel_max_liquid']:.4f} m/s")
        print(f"  Temperature range: {res['temp_range']:.1f} K")
        print(f"  Max vorticity: {res['vorticity_max']:.2e} 1/s")
        print(f"  Convection volume: {res['convection_volume']*100:.2f}%")
        print(f"  Reynolds (mean): {res['Re_mean']:.2f}, (max): {res['Re_max']:.2f}")
        print(f"  Peclet (mean): {res['Pe_mean']:.2f}, (max): {res['Pe_max']:.2f}")

    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")
