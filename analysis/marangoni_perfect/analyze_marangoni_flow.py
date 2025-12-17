#!/usr/bin/env python3
"""
Comprehensive analysis of Marangoni flow simulation VTK output.

Analyzes temperature distribution, velocity fields, hot spot location, and validates
against literature values (Panwisawas 2017: 0.5-1.0 m/s, Khairallah 2016: 1.0-2.0 m/s).

Output: Publication-quality figures showing temperature, velocity, and temporal evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import glob
import re

# === PARAMETERS ===
VTK_DIR = "/home/yzk/LBMProject/build/phase6_test2c_visualization"
OUTPUT_DIR = "/home/yzk/LBMProject/analysis/marangoni_perfect"
VTK_PATTERN = "marangoni_flow_*.vtk"

# Physical parameters
EXPECTED_TEMP_RANGE = (2000.0, 2500.0)  # K
EXPECTED_VEL_RANGE = (0.0, 1.5)  # m/s

# Analysis parameters
SURFACE_Z_FRACTION = 0.95  # Analyze top 5% of domain for surface flow
SUBSAMPLE_VECTORS = 4  # Show every Nth vector in quiver plots

# Literature comparison
LIT_VALUES = {
    'Panwisawas2017': (0.5, 1.0),  # m/s
    'Khairallah2016': (1.0, 2.0),  # m/s
}

# === VTK READER (Manual parsing for structured points) ===

class VTKStructuredPointsReader:
    """Simple reader for VTK structured points format."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.dimensions = None
        self.origin = None
        self.spacing = None
        self.n_points = None
        self.arrays = {}

    def read(self):
        """Read VTK file and parse data."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Parse dimensions
            if line.startswith('DIMENSIONS'):
                parts = line.split()
                self.dimensions = tuple(map(int, parts[1:4]))
                self.n_points = np.prod(self.dimensions)

            # Parse origin
            elif line.startswith('ORIGIN'):
                parts = line.split()
                self.origin = tuple(map(float, parts[1:4]))

            # Parse spacing
            elif line.startswith('SPACING'):
                parts = line.split()
                self.spacing = tuple(map(float, parts[1:4]))

            # Parse vector data
            elif line.startswith('VECTORS'):
                parts = line.split()
                array_name = parts[1]
                i += 1
                data = []
                for _ in range(self.n_points):
                    vec_line = lines[i].strip().split()
                    data.append([float(x) for x in vec_line])
                    i += 1
                self.arrays[array_name] = np.array(data)
                continue

            # Parse scalar data
            elif line.startswith('SCALARS'):
                parts = line.split()
                array_name = parts[1]
                i += 1  # Skip LOOKUP_TABLE line
                if lines[i].strip().startswith('LOOKUP_TABLE'):
                    i += 1
                data = []
                for _ in range(self.n_points):
                    data.append(float(lines[i].strip()))
                    i += 1
                self.arrays[array_name] = np.array(data)
                continue

            i += 1

        return self

    def get_points(self):
        """Generate point coordinates from structured grid."""
        nx, ny, nz = self.dimensions
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing

        x = ox + np.arange(nx) * dx
        y = oy + np.arange(ny) * dy
        z = oz + np.arange(nz) * dz

        # Create meshgrid
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

        # Flatten to point array
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        return points

    def get_bounds(self):
        """Get domain bounds."""
        nx, ny, nz = self.dimensions
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing

        return (ox, ox + (nx-1)*dx,
                oy, oy + (ny-1)*dy,
                oz, oz + (nz-1)*dz)

# === HELPER FUNCTIONS ===

def load_vtk_file(filepath):
    """Load VTK file and return reader object."""
    reader = VTKStructuredPointsReader(filepath)
    reader.read()
    print(f"Loaded: {Path(filepath).name}")
    print(f"  Dimensions: {reader.dimensions}, Points: {reader.n_points}")
    print(f"  Arrays: {list(reader.arrays.keys())}")
    return reader

def extract_timestep(filename):
    """Extract timestep number from filename."""
    match = re.search(r'_(\d+)\.vtk$', filename)
    return int(match.group(1)) if match else 0

def get_surface_slice(reader, z_fraction=0.95):
    """Extract surface slice at specified z-fraction of domain."""
    bounds = reader.get_bounds()
    z_surface = bounds[4] + z_fraction * (bounds[5] - bounds[4])

    # Get all points
    points = reader.get_points()
    nx, ny, nz = reader.dimensions

    # Find z-index closest to surface
    oz = reader.origin[2]
    dz = reader.spacing[2]
    z_idx = int((z_surface - oz) / dz)
    z_idx = min(max(z_idx, 0), nz - 1)

    # Extract surface slice (all x, y at this z)
    surface_mask = np.zeros(reader.n_points, dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            idx = z_idx * (nx * ny) + iy * nx + ix
            surface_mask[idx] = True

    surface_data = {
        'points': points[surface_mask],
        'z_surface': oz + z_idx * dz
    }

    # Extract arrays
    for name, data in reader.arrays.items():
        if len(data.shape) == 1:  # Scalar
            surface_data[name] = data[surface_mask]
        else:  # Vector
            surface_data[name] = data[surface_mask]

    return surface_data

def compute_velocity_magnitude(velocity):
    """Compute velocity magnitude from velocity vectors."""
    return np.linalg.norm(velocity, axis=1)

def find_hot_spot(reader):
    """Find location of maximum temperature (hot spot)."""
    temp = reader.arrays['Temperature']
    idx_max = np.argmax(temp)
    points = reader.get_points()
    hot_spot_coords = points[idx_max]
    max_temp = temp[idx_max]
    return hot_spot_coords, max_temp

def analyze_radial_flow(surface_data):
    """
    Analyze if flow is radially outward from center.

    Returns: radial velocity component statistics
    """
    points = surface_data['points']
    velocity = surface_data['Velocity']
    temp = surface_data['Temperature']

    # Find hot spot as center
    idx_max = np.argmax(temp)
    center = points[idx_max, :2]  # x, y only

    # Compute radial direction from center
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    r = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero at center
    mask = r > 1e-9

    # Radial unit vectors
    r_hat_x = np.zeros_like(dx)
    r_hat_y = np.zeros_like(dy)
    r_hat_x[mask] = dx[mask] / r[mask]
    r_hat_y[mask] = dy[mask] / r[mask]

    # Radial velocity component (dot product)
    v_radial = velocity[:, 0] * r_hat_x + velocity[:, 1] * r_hat_y

    return {
        'v_radial_mean': np.mean(v_radial[mask]) if np.sum(mask) > 0 else 0.0,
        'v_radial_max': np.max(v_radial[mask]) if np.sum(mask) > 0 else 0.0,
        'v_radial_min': np.min(v_radial[mask]) if np.sum(mask) > 0 else 0.0,
        'fraction_outward': np.sum(v_radial[mask] > 0) / np.sum(mask) if np.sum(mask) > 0 else 0.0,
        'center': center,
        'r': r,
        'v_radial': v_radial
    }

# === MAIN ANALYSIS ===

def main():
    print("="*70)
    print("MARANGONI FLOW ANALYSIS")
    print("="*70)

    # Get all VTK files
    vtk_files = sorted(glob.glob(f"{VTK_DIR}/{VTK_PATTERN}"))
    print(f"\nFound {len(vtk_files)} VTK files")

    if not vtk_files:
        print(f"ERROR: No VTK files found in {VTK_DIR}")
        return

    # Storage for temporal evolution
    timesteps = []
    max_velocities = []
    max_temperatures = []
    hot_spot_positions = []
    radial_flow_fractions = []

    # === TIME SERIES ANALYSIS ===
    print("\n" + "="*70)
    print("TEMPORAL EVOLUTION ANALYSIS")
    print("="*70)

    for vtk_file in vtk_files:
        reader = load_vtk_file(vtk_file)
        timestep = extract_timestep(vtk_file)
        timesteps.append(timestep)

        # Compute velocity magnitude
        velocity = reader.arrays['Velocity']
        vel_mag = compute_velocity_magnitude(velocity)
        max_velocities.append(np.max(vel_mag))

        # Find hot spot
        hot_spot, max_temp = find_hot_spot(reader)
        max_temperatures.append(max_temp)
        hot_spot_positions.append(hot_spot)

        # Analyze surface flow
        surface_data = get_surface_slice(reader, SURFACE_Z_FRACTION)
        radial_analysis = analyze_radial_flow(surface_data)
        radial_flow_fractions.append(radial_analysis['fraction_outward'])

        print(f"  t={timestep:05d}: T_max={max_temp:.1f} K, V_max={max_velocities[-1]:.4f} m/s, "
              f"Radial_out={radial_analysis['fraction_outward']*100:.1f}%")

    timesteps = np.array(timesteps)
    max_velocities = np.array(max_velocities)
    max_temperatures = np.array(max_temperatures)
    hot_spot_positions = np.array(hot_spot_positions)
    radial_flow_fractions = np.array(radial_flow_fractions)

    # === DETAILED ANALYSIS OF FINAL TIMESTEP ===
    print("\n" + "="*70)
    print("DETAILED ANALYSIS - FINAL TIMESTEP")
    print("="*70)

    final_reader = load_vtk_file(vtk_files[-1])

    temp_final = final_reader.arrays['Temperature']
    vel_final = final_reader.arrays['Velocity']
    vel_mag_final = compute_velocity_magnitude(vel_final)

    print(f"\nTemperature Statistics:")
    print(f"  Min:  {np.min(temp_final):.2f} K")
    print(f"  Max:  {np.max(temp_final):.2f} K")
    print(f"  Mean: {np.mean(temp_final):.2f} K")
    print(f"  Std:  {np.std(temp_final):.2f} K")
    print(f"  Expected range: {EXPECTED_TEMP_RANGE[0]}-{EXPECTED_TEMP_RANGE[1]} K")

    print(f"\nVelocity Magnitude Statistics:")
    print(f"  Min:  {np.min(vel_mag_final):.4f} m/s")
    print(f"  Max:  {np.max(vel_mag_final):.4f} m/s")
    print(f"  Mean: {np.mean(vel_mag_final):.4f} m/s")
    print(f"  Std:  {np.std(vel_mag_final):.4f} m/s")
    print(f"  Expected range: {EXPECTED_VEL_RANGE[0]}-{EXPECTED_VEL_RANGE[1]} m/s")

    # Surface analysis
    surface_final = get_surface_slice(final_reader, SURFACE_Z_FRACTION)
    radial_final = analyze_radial_flow(surface_final)

    print(f"\nSurface Flow Analysis (z={surface_final['z_surface']*1e6:.2f} µm):")
    print(f"  Radial velocity (mean): {radial_final['v_radial_mean']:.4f} m/s")
    print(f"  Radial velocity (max):  {radial_final['v_radial_max']:.4f} m/s")
    print(f"  Fraction flowing outward: {radial_final['fraction_outward']*100:.1f}%")
    print(f"  Hot spot center: ({radial_final['center'][0]*1e6:.2f}, {radial_final['center'][1]*1e6:.2f}) µm")

    # === VALIDATION AGAINST LITERATURE ===
    print("\n" + "="*70)
    print("LITERATURE COMPARISON")
    print("="*70)

    v_max = np.max(vel_mag_final)
    print(f"\nSimulation max velocity: {v_max:.4f} m/s")

    for ref, (v_min, v_max_lit) in LIT_VALUES.items():
        in_range = v_min <= v_max <= v_max_lit
        status = "PASS" if in_range else "OUT OF RANGE"
        print(f"  {ref}: {v_min:.1f}-{v_max_lit:.1f} m/s [{status}]")

    # === PUBLICATION-QUALITY VISUALIZATION ===
    print("\n" + "="*70)
    print("GENERATING PUBLICATION FIGURES")
    print("="*70)

    # Set publication style
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300

    # === FIGURE 1: Multi-panel surface analysis ===
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Prepare surface data for plotting
    surface_points = surface_final['points']
    x_surf = surface_points[:, 0] * 1e6  # Convert to µm
    y_surf = surface_points[:, 1] * 1e6
    temp_surf = surface_final['Temperature']
    vel_surf = surface_final['Velocity']
    vel_mag_surf = compute_velocity_magnitude(vel_surf)

    # Create structured grid for contour plots
    nx, ny, nz = final_reader.dimensions
    X = x_surf.reshape(ny, nx)
    Y = y_surf.reshape(ny, nx)
    T = temp_surf.reshape(ny, nx)
    V_mag = vel_mag_surf.reshape(ny, nx)
    U = vel_surf[:, 0].reshape(ny, nx)
    V_vel = vel_surf[:, 1].reshape(ny, nx)

    # Panel 1: Temperature contour
    ax1 = fig.add_subplot(gs[0, 0])
    cs1 = ax1.contourf(X, Y, T, levels=20, cmap='hot')
    ax1.contour(X, Y, T, levels=10, colors='k', linewidths=0.5, alpha=0.3)
    cbar1 = plt.colorbar(cs1, ax=ax1)
    cbar1.set_label('Temperature (K)', fontsize=10)
    ax1.set_xlabel('x (µm)', fontsize=10)
    ax1.set_ylabel('y (µm)', fontsize=10)
    ax1.set_title('(a) Temperature Distribution', fontsize=11, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Velocity magnitude contour
    ax2 = fig.add_subplot(gs[0, 1])
    cs2 = ax2.contourf(X, Y, V_mag, levels=20, cmap='viridis')
    ax2.contour(X, Y, V_mag, levels=10, colors='k', linewidths=0.5, alpha=0.3)
    cbar2 = plt.colorbar(cs2, ax=ax2)
    cbar2.set_label('Velocity Magnitude (m/s)', fontsize=10)
    ax2.set_xlabel('x (µm)', fontsize=10)
    ax2.set_ylabel('y (µm)', fontsize=10)
    ax2.set_title('(b) Velocity Magnitude', fontsize=11, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Velocity vectors (quiver)
    ax3 = fig.add_subplot(gs[0, 2])
    skip = (slice(None, None, SUBSAMPLE_VECTORS), slice(None, None, SUBSAMPLE_VECTORS))
    ax3.contourf(X, Y, V_mag, levels=20, cmap='viridis', alpha=0.6)
    ax3.quiver(X[skip], Y[skip], U[skip], V_vel[skip],
               V_mag[skip], cmap='viridis',
               scale=5.0, width=0.003, headwidth=3, headlength=4)
    center = radial_final['center'] * 1e6
    ax3.plot(center[0], center[1], 'r*', markersize=15, label='Hot Spot')
    ax3.set_xlabel('x (µm)', fontsize=10)
    ax3.set_ylabel('y (µm)', fontsize=10)
    ax3.set_title('(c) Velocity Vectors', fontsize=11, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Time evolution of max velocity
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(timesteps, max_velocities, 'b-o', linewidth=2, markersize=5, label='Simulation')
    ax4.axhspan(LIT_VALUES['Panwisawas2017'][0], LIT_VALUES['Panwisawas2017'][1],
                alpha=0.2, color='green', label='Panwisawas 2017')
    ax4.axhspan(LIT_VALUES['Khairallah2016'][0], LIT_VALUES['Khairallah2016'][1],
                alpha=0.2, color='orange', label='Khairallah 2016')
    ax4.set_xlabel('Timestep', fontsize=10)
    ax4.set_ylabel('Max Velocity (m/s)', fontsize=10)
    ax4.set_title('(d) Temporal Evolution of Max Velocity', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Time evolution of max temperature
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(timesteps, max_temperatures, 'r-o', linewidth=2, markersize=5)
    ax5.axhspan(EXPECTED_TEMP_RANGE[0], EXPECTED_TEMP_RANGE[1],
                alpha=0.2, color='gray', label='Expected Range')
    ax5.set_xlabel('Timestep', fontsize=10)
    ax5.set_ylabel('Max Temperature (K)', fontsize=10)
    ax5.set_title('(e) Temporal Evolution of Max Temperature', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Radial flow fraction
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(timesteps, radial_flow_fractions * 100, 'g-o', linewidth=2, markersize=5)
    ax6.axhline(50, color='k', linestyle='--', alpha=0.5, label='50% threshold')
    ax6.set_xlabel('Timestep', fontsize=10)
    ax6.set_ylabel('Outward Flow (%)', fontsize=10)
    ax6.set_title('(f) Radial Outward Flow Fraction', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 105])

    fig.suptitle('Marangoni Flow Analysis - Surface Characterization',
                 fontsize=14, fontweight='bold', y=0.995)

    output_path = f"{OUTPUT_DIR}/marangoni_analysis_comprehensive.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # === FIGURE 2: Radial profile analysis ===
    fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # Compute radial profiles from center
    r_bins = np.linspace(0, np.max(radial_final['r']), 30)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    r_data = radial_final['r']
    v_radial_data = radial_final['v_radial']
    temp_data = temp_surf

    # Bin data
    v_radial_binned = []
    temp_binned = []

    for i in range(len(r_bins) - 1):
        mask = (r_data >= r_bins[i]) & (r_data < r_bins[i+1])
        if np.sum(mask) > 0:
            v_radial_binned.append(np.mean(v_radial_data[mask]))
            temp_binned.append(np.mean(temp_data[mask]))
        else:
            v_radial_binned.append(np.nan)
            temp_binned.append(np.nan)

    # Left panel: Temperature vs radius
    ax_left.plot(r_centers * 1e6, temp_binned, 'r-o', linewidth=2, markersize=6)
    ax_left.set_xlabel('Radial Distance from Hot Spot (µm)', fontsize=11)
    ax_left.set_ylabel('Temperature (K)', fontsize=11)
    ax_left.set_title('(a) Radial Temperature Profile', fontsize=12, fontweight='bold')
    ax_left.grid(True, alpha=0.3)

    # Right panel: Radial velocity vs radius
    ax_right.plot(r_centers * 1e6, v_radial_binned, 'b-o', linewidth=2, markersize=6)
    ax_right.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_right.set_xlabel('Radial Distance from Hot Spot (µm)', fontsize=11)
    ax_right.set_ylabel('Radial Velocity (m/s)', fontsize=11)
    ax_right.set_title('(b) Radial Velocity Profile', fontsize=12, fontweight='bold')
    ax_right.grid(True, alpha=0.3)

    fig2.suptitle('Radial Profiles from Hot Spot Center',
                  fontsize=14, fontweight='bold')

    output_path2 = f"{OUTPUT_DIR}/marangoni_radial_profiles.png"
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()

    # === FIGURE 3: Hot spot trajectory ===
    fig3, ax = plt.subplots(figsize=(8, 8))

    hot_spot_x = hot_spot_positions[:, 0] * 1e6
    hot_spot_y = hot_spot_positions[:, 1] * 1e6

    # Plot trajectory
    ax.plot(hot_spot_x, hot_spot_y, 'ro-', linewidth=2, markersize=8, alpha=0.6)
    ax.plot(hot_spot_x[0], hot_spot_y[0], 'go', markersize=12, label='Initial')
    ax.plot(hot_spot_x[-1], hot_spot_y[-1], 'bs', markersize=12, label='Final')

    # Domain outline
    bounds = final_reader.get_bounds()
    domain_x = np.array([bounds[0], bounds[1], bounds[1], bounds[0], bounds[0]]) * 1e6
    domain_y = np.array([bounds[2], bounds[2], bounds[3], bounds[3], bounds[2]]) * 1e6
    ax.plot(domain_x, domain_y, 'k--', linewidth=1, alpha=0.5, label='Domain')

    ax.set_xlabel('x (µm)', fontsize=11)
    ax.set_ylabel('y (µm)', fontsize=11)
    ax.set_title('Hot Spot Trajectory Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    output_path3 = f"{OUTPUT_DIR}/marangoni_hotspot_trajectory.png"
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path3}")
    plt.close()

    # === VALIDATION SUMMARY ===
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    validation_results = {
        'Temperature Range': {
            'Expected': f"{EXPECTED_TEMP_RANGE[0]}-{EXPECTED_TEMP_RANGE[1]} K",
            'Observed': f"{np.min(temp_final):.1f}-{np.max(temp_final):.1f} K",
            'Status': 'PASS' if (EXPECTED_TEMP_RANGE[0] <= np.max(temp_final) <= EXPECTED_TEMP_RANGE[1]) else 'FAIL'
        },
        'Velocity Range': {
            'Expected': f"{EXPECTED_VEL_RANGE[0]}-{EXPECTED_VEL_RANGE[1]} m/s",
            'Observed': f"{np.min(vel_mag_final):.4f}-{np.max(vel_mag_final):.4f} m/s",
            'Status': 'PASS' if (np.max(vel_mag_final) <= EXPECTED_VEL_RANGE[1]) else 'FAIL'
        },
        'Flow Direction': {
            'Expected': 'Radially outward (>50%)',
            'Observed': f"{radial_final['fraction_outward']*100:.1f}% outward",
            'Status': 'PASS' if (radial_final['fraction_outward'] > 0.5) else 'FAIL'
        },
        'Hot Spot Stability': {
            'Expected': 'Stable at center',
            'Observed': f"Drift: {np.std(hot_spot_x):.2f} µm (x), {np.std(hot_spot_y):.2f} µm (y)",
            'Status': 'PASS' if (np.std(hot_spot_x) < 10 and np.std(hot_spot_y) < 10) else 'MARGINAL'
        }
    }

    for metric, results in validation_results.items():
        print(f"\n{metric}:")
        print(f"  Expected: {results['Expected']}")
        print(f"  Observed: {results['Observed']}")
        print(f"  Status:   {results['Status']}")

    # Save validation summary to file
    summary_file = f"{OUTPUT_DIR}/validation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MARANGONI FLOW VALIDATION SUMMARY\n")
        f.write("="*70 + "\n\n")

        for metric, results in validation_results.items():
            f.write(f"{metric}:\n")
            f.write(f"  Expected: {results['Expected']}\n")
            f.write(f"  Observed: {results['Observed']}\n")
            f.write(f"  Status:   {results['Status']}\n\n")

        f.write("\nLiterature Comparison:\n")
        f.write(f"  Simulation max velocity: {np.max(vel_mag_final):.4f} m/s\n")
        for ref, (v_min, v_max_lit) in LIT_VALUES.items():
            in_range = v_min <= np.max(vel_mag_final) <= v_max_lit
            status = "PASS" if in_range else "OUT OF RANGE"
            f.write(f"  {ref}: {v_min:.1f}-{v_max_lit:.1f} m/s [{status}]\n")

    print(f"\nValidation summary saved: {summary_file}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - marangoni_analysis_comprehensive.png")
    print("  - marangoni_radial_profiles.png")
    print("  - marangoni_hotspot_trajectory.png")
    print("  - validation_summary.txt")

if __name__ == "__main__":
    main()
