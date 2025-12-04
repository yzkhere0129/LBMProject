#!/usr/bin/env python3
"""
Marangoni Benchmark Visualization Script for ParaView

This script creates standardized visualizations of the Marangoni benchmark
results for comparison with waLBerla thermocapillary showcase.

Usage:
    # In ParaView Python shell:
    pvpython visualize_marangoni_benchmark.py marangoni_000000.vtk
    
    # Or load as a ParaView macro
"""

import sys
import os

def create_visualization_paraview(vtk_file, output_prefix="marangoni"):
    """
    Create Marangoni benchmark visualization using ParaView.
    Run this with: pvpython visualize_marangoni_benchmark.py <vtk_file>
    """
    from paraview.simple import *
    
    # Load data
    reader = LegacyVTKReader(FileNames=[vtk_file])
    reader.UpdatePipeline()
    
    # Get bounds for camera setup
    bounds = reader.GetDataInformation().GetBounds()
    center_x = (bounds[0] + bounds[1]) / 2
    center_y = (bounds[2] + bounds[3]) / 2
    
    # Create render view
    view = CreateView('RenderView')
    view.ViewSize = [1920, 1080]
    view.Background = [1, 1, 1]  # White background
    
    # Camera setup for 2D view (looking down Z axis)
    camera_distance = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 1.5
    view.CameraPosition = [center_x, center_y, camera_distance]
    view.CameraFocalPoint = [center_x, center_y, 0]
    view.CameraViewUp = [0, 1, 0]
    view.CameraParallelProjection = 1  # Orthographic projection
    
    # ========================================
    # 1. Temperature Field Visualization
    # ========================================
    temp_display = Show(reader, view)
    temp_display.Representation = 'Surface'
    ColorBy(temp_display, ('POINTS', 'Temperature'))
    
    # Set color map
    temp_lut = GetColorTransferFunction('Temperature')
    temp_lut.ApplyPreset('Cool to Warm', True)
    temp_lut.RescaleTransferFunction(1800.0, 2000.0)
    
    # Add color bar
    temp_bar = GetScalarBar(temp_lut, view)
    temp_bar.Title = 'Temperature [K]'
    temp_bar.ComponentTitle = ''
    temp_bar.TitleFontSize = 16
    temp_bar.LabelFontSize = 14
    temp_bar.Position = [0.85, 0.25]
    temp_bar.Visibility = 1
    
    # Render and save temperature visualization
    Render()
    SaveScreenshot(f'{output_prefix}_temperature.png', view, ImageResolution=[1920, 1080])
    print(f"Saved: {output_prefix}_temperature.png")
    
    # ========================================
    # 2. Velocity Field with Arrows (Glyph)
    # ========================================
    # Hide temperature display
    Hide(reader, view)
    
    # Create glyph for velocity vectors
    glyph = Glyph(Input=reader, GlyphType='Arrow')
    glyph.OrientationArray = ['POINTS', 'Velocity']
    glyph.ScaleArray = ['POINTS', 'Velocity']
    glyph.ScaleFactor = 10000.0  # Adjust based on velocity magnitude
    glyph.GlyphMode = 'Every Nth Point'
    glyph.Stride = 4  # Show every 4th point for clarity
    
    glyph_display = Show(glyph, view)
    ColorBy(glyph_display, ('POINTS', 'VelocityMagnitude'))
    
    # Velocity color map
    vel_lut = GetColorTransferFunction('VelocityMagnitude')
    vel_lut.ApplyPreset('Rainbow Uniform', True)
    vel_lut.RescaleTransferFunction(0.0, 0.01)
    
    # Show the base field too
    base_display = Show(reader, view)
    base_display.Representation = 'Surface'
    ColorBy(base_display, ('POINTS', 'FillLevel'))
    base_display.Opacity = 0.3
    
    # Add velocity color bar
    vel_bar = GetScalarBar(vel_lut, view)
    vel_bar.Title = 'Velocity [m/s]'
    vel_bar.ComponentTitle = ''
    vel_bar.TitleFontSize = 16
    vel_bar.LabelFontSize = 14
    vel_bar.Position = [0.85, 0.25]
    vel_bar.Visibility = 1
    
    # Hide temperature color bar
    temp_bar.Visibility = 0
    
    Render()
    SaveScreenshot(f'{output_prefix}_velocity_arrows.png', view, ImageResolution=[1920, 1080])
    print(f"Saved: {output_prefix}_velocity_arrows.png")
    
    # ========================================
    # 3. Streamlines
    # ========================================
    Hide(glyph, view)
    Hide(base_display, view)
    
    # Create stream tracer
    stream = StreamTracer(Input=reader, SeedType='Line')
    stream.Vectors = ['POINTS', 'Velocity']
    
    # Set seed line along interface (y = 70% of height)
    y_interface = bounds[2] + 0.7 * (bounds[3] - bounds[2])
    stream.SeedType.Point1 = [bounds[0], y_interface, 0]
    stream.SeedType.Point2 = [bounds[1], y_interface, 0]
    stream.SeedType.Resolution = 30
    stream.IntegrationDirection = 'BOTH'
    stream.MaximumStreamlineLength = bounds[1] - bounds[0]
    
    stream_display = Show(stream, view)
    stream_display.Representation = 'Surface'
    ColorBy(stream_display, ('POINTS', 'VelocityMagnitude'))
    
    # Show fill level as background
    fill_display = Show(reader, view)
    fill_display.Representation = 'Surface'
    ColorBy(fill_display, ('POINTS', 'FillLevel'))
    fill_lut = GetColorTransferFunction('FillLevel')
    fill_lut.RescaleTransferFunction(0.0, 1.0)
    fill_display.Opacity = 0.5
    
    Render()
    SaveScreenshot(f'{output_prefix}_streamlines.png', view, ImageResolution=[1920, 1080])
    print(f"Saved: {output_prefix}_streamlines.png")
    
    # ========================================
    # 4. Marangoni Force Visualization
    # ========================================
    Hide(stream, view)
    Hide(fill_display, view)
    
    # Show base field
    Show(reader, view)
    
    # Create threshold to show only interface region
    threshold = Threshold(Input=reader)
    threshold.Scalars = ['POINTS', 'InterfaceIndicator']
    threshold.ThresholdRange = [0.5, 1.0]
    
    # Create glyph for Marangoni force
    force_glyph = Glyph(Input=threshold, GlyphType='Arrow')
    force_glyph.OrientationArray = ['POINTS', 'MarangoniForce']
    force_glyph.ScaleArray = ['POINTS', 'MarangoniForce']
    force_glyph.ScaleFactor = 1e-8  # Adjust based on force magnitude
    force_glyph.GlyphMode = 'All Points'
    
    force_display = Show(force_glyph, view)
    force_display.DiffuseColor = [1.0, 0.0, 0.0]  # Red arrows
    
    Render()
    SaveScreenshot(f'{output_prefix}_marangoni_force.png', view, ImageResolution=[1920, 1080])
    print(f"Saved: {output_prefix}_marangoni_force.png")
    
    print(f"\nVisualization complete! Generated 4 images with prefix '{output_prefix}'")


def create_comparison_plot():
    """
    Create matplotlib comparison plots for Marangoni benchmark.
    Use this for quantitative comparison with waLBerla.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Example: Load data from CSV (exported from ParaView)
    # Modify paths as needed
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Temperature profile at x=center
    ax1 = axes[0, 0]
    ax1.set_xlabel('Y position [cells]')
    ax1.set_ylabel('Temperature [K]')
    ax1.set_title('Temperature Profile (x = center)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Velocity profile at interface
    ax2 = axes[0, 1]
    ax2.set_xlabel('X position [cells]')
    ax2.set_ylabel('Velocity Ux [m/s]')
    ax2.set_title('Interface Velocity Profile (y = interface)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Velocity magnitude contour
    ax3 = axes[1, 0]
    ax3.set_xlabel('X [cells]')
    ax3.set_ylabel('Y [cells]')
    ax3.set_title('Velocity Magnitude Contour')
    
    # Plot 4: Convergence history
    ax4 = axes[1, 1]
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Max Velocity [m/s]')
    ax4.set_title('Convergence History')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('marangoni_comparison.png', dpi=300)
    print("Saved: marangoni_comparison.png")


def print_usage():
    print("""
Marangoni Benchmark Visualization Script
========================================

Usage with ParaView:
    pvpython visualize_marangoni_benchmark.py <vtk_file>

Example:
    pvpython visualize_marangoni_benchmark.py marangoni_benchmark_output/marangoni_020000.vtk

This will generate:
    - marangoni_temperature.png     : Temperature field
    - marangoni_velocity_arrows.png : Velocity vectors
    - marangoni_streamlines.png     : Flow streamlines
    - marangoni_marangoni_force.png : Marangoni force at interface

For comparison with waLBerla:
    1. Run both simulations with same parameters
    2. Export centerline profiles from ParaView (Plot Over Line)
    3. Use matplotlib to overlay and compare
""")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)
    
    vtk_file = sys.argv[1]
    
    if not os.path.exists(vtk_file):
        print(f"Error: File not found: {vtk_file}")
        sys.exit(1)
    
    # Check if running in ParaView
    try:
        from paraview.simple import *
        create_visualization_paraview(vtk_file)
    except ImportError:
        print("ParaView not found. Run with: pvpython visualize_marangoni_benchmark.py <vtk_file>")
        print("\nAlternatively, load the VTK file directly in ParaView and follow these steps:")
        print("1. Color by 'Temperature' - use 'Cool to Warm' preset")
        print("2. Add Glyph filter for velocity arrows")
        print("3. Add StreamTracer for streamlines")
        print("4. Use Threshold on InterfaceIndicator to see Marangoni force")
