# ParaView Flow Visualization Tutorial
## Visualizing Melt Pool Convection from LBM Simulations

**Purpose:** Step-by-step guide to visualize velocity vectors and streamlines from Phase 5 thermal-fluid coupling simulations.

---

## Prerequisites

- Completed simulation run (VTK files generated)
- ParaView 5.10+ installed
- Basic familiarity with ParaView interface

## Installation

### Option 1: Download Pre-built Binary (Recommended)
```bash
wget https://www.paraview.org/paraview-downloads/download.php?submit=Download&version=v5.11&type=binary&os=Linux&downloadFile=ParaView-5.11.2-MPI-Linux-Python3.9-x86_64.tar.gz -O ParaView.tar.gz
tar -xzf ParaView.tar.gz
./ParaView-*/bin/paraview
```

### Option 2: System Package Manager
```bash
# Ubuntu/Debian
sudo apt install paraview

# Fedora/RHEL
sudo dnf install paraview
```

---

## Part 1: Loading and Basic Visualization

### Step 1: Load Time Series Data
1. Launch ParaView
2. **File → Open**
3. Navigate to `/home/yzk/LBMProject/build/visualization_output/`
4. Select **all** `laser_melting_flow_*.vtk` files
   - Use Shift+Click to select range
   - Or Ctrl+A to select all
5. Click **OK**
6. In Properties panel, click **Apply**

**What you see:** Grid loaded, default view shows domain outline

### Step 2: Visualize Temperature Field
1. In toolbar, change **Solid Color** dropdown to **Temperature**
2. Click **Play** button ▶ in animation controls
3. Observe temperature evolution over time

**Color Scale Adjustment:**
- Click on color bar legend
- **Edit Color Map**
- Choose preset: **Rainbow**, **Cool to Warm**, or **Black-Body Radiation**
- Adjust range: Min = 300 K, Max = 2500 K (or auto-range)

### Step 3: Visualize Phase Change
1. Change coloring to **LiquidFraction**
2. Adjust color range: 0.0 to 1.0
3. Blue = solid (fl = 0), Red = liquid (fl = 1), Green = mushy zone

**Interpretation:**
- 0.0: Fully solid
- 0.0-1.0: Mushy zone (partially melted)
- 1.0: Fully liquid

---

## Part 2: Velocity Vector Visualization

### Method 1: Arrow Glyphs (Recommended for Beginners)

1. **Select loaded data** in Pipeline Browser
2. **Filters → Common → Glyph**
3. In Glyph Properties:
   - **Scalars:** None (or Temperature for colored arrows)
   - **Vectors:** Velocity
   - **Glyph Type:** Arrow
   - **Scale Mode:** Vector
   - **Scale Factor:** Start with 1e-5, adjust based on arrow size
   - **Glyph Mode:** All Points (or Every Nth Point to reduce clutter)
4. Click **Apply**
5. **Color by:** VelocityMagnitude (shows flow speed)

**Fine-tuning:**
- **Too many arrows?** Set **Glyph Mode:** Every Nth Point (stride = 2 or 4)
- **Arrows too small?** Increase Scale Factor (try 5e-5)
- **Arrows too big?** Decrease Scale Factor (try 5e-6)

**Result:** Arrow field showing flow direction and magnitude

### Method 2: 3D Cone Glyphs (For Presentations)

Same as Method 1, but:
- **Glyph Type:** Cone (or 3D Arrow)
- **Resolution:** 12-24 for smooth cones
- More visually appealing but slower to render

---

## Part 3: Streamline Visualization

### Basic Streamlines

1. **Select original data** (not the Glyph)
2. **Filters → Common → Stream Tracer**
3. In Stream Tracer Properties:
   - **Vectors:** Velocity
   - **Seed Type:** Point Cloud
   - **Center:** [80e-6, 80e-6, 40e-6] (center of domain)
   - **Radius:** 20e-6 (seeding sphere radius)
   - **Number of Points:** 100 (number of streamlines)
   - **Maximum Streamline Length:** 200e-6 (twice domain size)
4. Click **Apply**
5. **Color by:** VelocityMagnitude

**What you see:** Curves tracing fluid flow paths

**Customization:**
- **More streamlines:** Increase Number of Points (200-500)
- **Longer streamlines:** Increase Maximum Length
- **Different seed region:** Adjust Center and Radius
- **Line width:** In Display properties → Line Width = 2-3

### Advanced: Seed from Melt Pool Only

1. First, create a **Threshold** filter:
   - **Filters → Common → Threshold**
   - **Scalars:** LiquidFraction
   - **Minimum:** 0.1 (only liquid/mushy regions)
   - **Apply**
2. Now apply Stream Tracer to **thresholded data**
3. **Seed Type:** Point Cloud
4. Seeds will only appear in melted regions

**Result:** Streamlines only where fluid is liquid

---

## Part 4: Combined Visualization

### Setup 1: Temperature + Velocity Arrows
1. Load data, color by Temperature
2. Add Glyph filter for velocity arrows
3. In Glyph properties, uncheck **Color by Temperature** (keep arrows white or by magnitude)
4. Adjust arrow size to not overwhelm temperature field

### Setup 2: Phase + Streamlines + Arrows
1. **Pipeline:**
   ```
   laser_melting_flow_*.vtk
   ├── Color by: PhaseState
   ├── Glyph (arrows, Every 4th Point)
   └── Stream Tracer (from liquid region)
   ```
2. Adjust opacity of base data: 50-70%
3. Streamlines and arrows remain fully opaque

### Setup 3: Slice View with Vectors
1. **Filters → Common → Slice**
2. **Plane:** XY Plane
3. **Origin:** [80e-6, 80e-6, 20e-6] (mid-height)
4. **Apply**
5. Add Glyph filter to slice
6. Color slice by Temperature or Velocity

**Advantage:** 2D slice reduces visual clutter, easier to see flow pattern

---

## Part 5: Animation and Video Export

### Create Animation
1. Load data as time series
2. Set up desired visualization (arrows, streamlines, etc.)
3. **View → Animation View** (bottom panel)
4. Click **Play** ▶ to preview
5. Adjust animation speed: **File → Animation → Time**
   - Duration: 10 seconds
   - Frame Rate: 30 FPS

### Export Video
1. **File → Save Animation**
2. Choose format:
   - **AVI** (Windows Media Video)
   - **OGGTM** (free codec)
   - **PNG Series** (for custom encoding later)
3. **Frame Rate:** 30 FPS
4. **Resolution:** 1920×1080 (or higher)
5. **Compression:** Best quality
6. Click **OK** and wait for rendering

**Tip:** For publication-quality videos, export PNG series and use FFmpeg:
```bash
ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
```

---

## Part 6: Advanced Techniques

### Vector Field Magnitude Contours
1. **Filters → Common → Contour**
2. **Contour By:** VelocityMagnitude
3. **Isosurfaces:** [0.001, 0.005, 0.01] m/s
4. **Apply**
5. Shows iso-surfaces of constant velocity magnitude

### Flow Vorticity
1. **Filters → Alphabetical → Compute Derivatives**
2. **Vectors:** Velocity
3. **Output Vector Type:** Vorticity
4. **Apply**
5. Color by: Vorticity (shows rotation strength)

### Particle Tracing (Time-dependent)
1. **Filters → Temporal → Particle Tracer**
2. **Seed Source:** Point Cloud
3. **Maximum Time:** Full simulation duration
4. Shows Lagrangian particle paths over time

---

## Troubleshooting

### Issue 1: Arrows Not Visible
**Symptoms:** Glyph filter applied, but no arrows appear

**Solutions:**
1. Check **Vectors** is set to "Velocity" (not "None")
2. Increase **Scale Factor** (try 1e-4, 1e-3)
3. Check velocity magnitude: may be too small
   - Color by VelocityMagnitude
   - Check color bar range (should be > 0)

### Issue 2: Streamlines Disappear Immediately
**Symptoms:** Stream Tracer shows only points, no lines

**Solutions:**
1. Increase **Maximum Streamline Length** (try 500e-6)
2. Check velocity field is not zero everywhere
3. Verify seed points are in fluid domain (not solid walls)
4. Change **Integrator Type:** from default to Runge-Kutta 4-5

### Issue 3: Too Slow / Laggy
**Symptoms:** ParaView freezes or is very slow

**Solutions:**
1. Reduce number of glyphs: **Every 4th Point** or **Every 8th Point**
2. Reduce number of streamlines: 50-100 instead of 500
3. Use **Wireframe** view instead of **Surface**
4. Close other applications to free RAM

### Issue 4: Velocity Seems Wrong
**Symptoms:** Arrows point in unexpected directions

**Diagnostics:**
1. **Info** tab → Check field ranges:
   - Velocity_X: should vary (not all zeros)
   - Velocity_Y: should vary
   - Velocity_Z: should vary
2. Create **Plot Over Line** filter:
   - **Filters → Data Analysis → Plot Over Line**
   - **X Axis:** arc_length
   - **Y Axis:** Velocity_Y
   - Should show variation

**Common causes:**
- Velocity is actually zero (physics issue, not visualization)
- Scale factor too large/small (arrows don't match actual magnitude)

---

## Example Workflow: Finding Convection Cells

**Goal:** Visualize buoyancy-driven convection rolls in melt pool

**Steps:**
1. Load data, advance to time when melting is significant
2. Create **Threshold**: LiquidFraction > 0.5 (only fully liquid)
3. Apply **Stream Tracer** to thresholded data
4. Adjust seeding to fill liquid region
5. Color streamlines by: TemperatureOrVelocityMagnitude
6. Add **Slice** filter (XY plane, mid-height)
7. On slice, add **Glyph** (arrows)
8. Observe: hot fluid rises (arrows point up), cool fluid sinks (arrows point down)

**Expected Pattern:** Closed circulation loops (convection cells)

---

## Physical Interpretation

### Velocity Magnitude Scale
- **0-0.001 m/s (0-1 mm/s):** Typical for high viscosity or weak buoyancy
- **0.001-0.01 m/s (1-10 mm/s):** Expected for metal melt pools
- **0.01-0.1 m/s (10-100 mm/s):** Very strong convection
- **> 0.1 m/s:** Unrealistic for thermal convection, check parameters

### Flow Patterns
- **Upward velocity in hot regions:** Buoyancy effect (less dense fluid rises)
- **Downward velocity in cool regions:** Gravity (denser fluid sinks)
- **Circular streamlines:** Convection rolls
- **No velocity in solid/mushy:** Darcy damping effect (flow suppressed)

### Phase Change Indicators
- **Expanding liquid region:** Melting progressing
- **Stationary liquid/solid interface:** Equilibrium reached
- **Mushy zone thickness:** Depends on temperature gradient

---

## References

- **ParaView Guide:** https://www.paraview.org/paraview-docs/
- **VTK File Format:** https://kitware.github.io/vtk-examples/site/VTKFileFormats/
- **Stream Tracer Tutorial:** https://www.paraview.org/Wiki/ParaView/Users_Guide/Filtering_Data#Stream_Tracer
- **Glyph Filter:** https://www.paraview.org/Wiki/ParaView/Users_Guide/Filtering_Data#Glyph

---

## Quick Reference Card

| Task | Filter | Key Setting |
|------|--------|-------------|
| Show velocity arrows | Glyph | Vectors: Velocity, Glyph Type: Arrow |
| Show flow paths | Stream Tracer | Vectors: Velocity, Seed: Point Cloud |
| Slice 2D view | Slice | Plane: XY, Origin: mid-height |
| Show only liquid | Threshold | Scalars: LiquidFraction, Min: 0.5 |
| Color by flow speed | (none) | Color by: VelocityMagnitude |
| Export video | File → Save Animation | Format: AVI or PNG series |

---

**Questions?** See `PHASE5_SUMMARY.md` for technical details or contact simulation team.

**Have fun visualizing! 🌊**
