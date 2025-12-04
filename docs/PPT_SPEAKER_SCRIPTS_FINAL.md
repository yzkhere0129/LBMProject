# Speaker Scripts for Group Meeting Presentation
## GPU-Accelerated LBM Platform for AM Simulation
### Nov 28th, 2025

---

## Slide 1: Title Slide
**Time: 30 seconds**

**Script:**

> "Good morning everyone. Today I will talk about my GPU platform for AM simulation.
>
> This platform uses LBM method. It runs on GPU. It is very fast.
>
> I will show you three things:
> - First, the platform design
> - Second, the validation results
> - Third, how I use AI to help development
>
> Let me start with why we choose LBM and GPU."

---

## Slide 2: Why LBM + GPU?
**Time: 45 seconds**

**Script:**

> "Why do we choose LBM with GPU? Two reasons.
>
> [Point to left diagram]
> First, LBM is naturally parallel. Look at this picture. In FVM, all cells must talk to each other. This is hard to parallelize. But in LBM, each cell computes independently. This is perfect for GPU.
>
> [Point to right chart]
> Second, look at the speed. Red bars show optimized LBM on GPU. Much faster than blue bars. Literature reports 136 times speedup.
>
> Our platform is 37 to 150 times faster than CPU FEM.
>
> Now let me show you the platform design."

---

## Slide 3: Platform Architecture
**Time: 50 seconds**

**Script:**

> "This is our platform structure. It has four layers.
>
> [Point to diagram from bottom to top]
>
> The bottom is LBM Core with CUDA. This is the GPU engine.
>
> The second layer is physics modules. Thermal, Fluid, VOF, Phase Change, Marangoni, Laser. These are building blocks.
>
> The third layer is Multiphysics Solver. It couples all physics together.
>
> The top layer is Applications. LPBF is working now. DED and EBM are planned.
>
> [Point to code screenshot]
> You can see our code structure here. It is modular. We can add new physics easily.
>
> Key point: this is a PLATFORM, not just one simulation."

---

## Slide 4: Physics Modules Status
**Time: 45 seconds**

**Script:**

> "Here is the status of our physics modules.
>
> [Point to table]
>
> We have seven modules.
>
> [Point to green checkmarks]
> Six modules are verified:
> - Thermal Conduction: D3Q7 MRT, validated against analytical solution
> - Phase Change: enthalpy method, validated against Stefan problem
> - Evaporation: Hertz-Knudsen model
> - Marangoni Effect: validated against Khairallah 2016
> - Surface Tension: CSF model, Laplace pressure test
> - Laser Heating: Gaussian plus Beer-Lambert
>
> [Point to yellow circle]
> Only Recoil Pressure needs more validation.
>
> Now let me show a real simulation case."

---

## Slide 5: LPBF Single Track Case - 120W
**Time: 40 seconds**

**Script:**

> "This is our LPBF simulation. Ti-6Al-4V material.
>
> [Point to parameters]
> Domain: 1400 by 300 by 200 microns
> Resolution: 2 microns
> Laser Power: 120 watts
> Scan Speed: 0.6 meters per second
> Duration: 1500 microseconds
>
> [Point to top image]
> Top image shows temperature field. Red is hot, blue is cold.
>
> [Point to bottom image]
> Bottom image shows the melt pool. You can see the liquid region.
>
> Compute time: about 60 minutes on RTX 3050 GPU.
>
> This is conduction mode - shallow and wide melt pool."

---

## Slide 6: LPBF Single Track Case - 300W
**Time: 40 seconds**

**Script:**

> "Now higher power: 300 watts. Same speed.
>
> [Point to top image]
> Look at the temperature. Much higher now. Peak temperature around 1900 Kelvin.
>
> [Point to bottom image]
> Look at the melt pool shape. It is deeper now. Starting to form keyhole.
>
> [Compare with previous slide]
> At 120 watts: shallow pool, conduction mode.
> At 300 watts: deeper pool, transition to keyhole mode.
>
> This matches physics. Higher power means deeper penetration.
>
> Now let me show the validation."

---

## Slide 7: Validation Results
**Time: 50 seconds**

**Script:**

> "This is the key slide - validation against experiment.
>
> [Point to paper reference]
> We compare with this paper from Optics and Laser Technology. They measured Ti-6Al-4V melt pool size.
>
> [Point to experimental image]
> This is their experiment. You can see the fusion zone and HAZ.
>
> [Point to simulation image]
> This is our simulation. Same laser parameters: 120 watts, 600 mm per second.
>
> [Point to numbers on right]
> Melt Pool Size comparison:
> - Our simulation: 44.0 microns width, 112.2 microns depth
> - Experiment: 42.7 microns width, 126.1 microns depth
>
> Marangoni Velocity:
> - Our result: 1.2 meters per second
> - Literature range: 0.5 to 2.0 meters per second
>
> Physics is validated. Surface tension drives melt pool flow correctly."

---

## Slide 8: Validation Results Summary
**Time: 45 seconds**

**Script:**

> "Here is the full summary table.
>
> [Point to table]
>
> Process parameters match exactly: 120 watts, 600 mm/s, Ti6Al4V.
>
> [Point to geometry rows]
> Melt Pool Width: 44.0 vs 42.7 microns. Error: 2.9 percent. Pass.
> Melt Pool Depth: 119.2 vs 126.1 microns. Error: minus 5.4 percent. Pass.
>
> [Point to velocity row]
> Marangoni Velocity: 1.2 m/s. Literature says 0.3 to 1.5 m/s. We are in range. Pass.
>
> [Point to mode row]
> Melting mode is conduction. This matches experiment.
>
> [Point to left summary]
> Energy conservation error is less than 5 percent.
> Simulation runs 1500 microseconds with no divergence.
> Speed: 37 to 150 times faster than CPU FEM."

---

## Slide 9: Current Limitations
**Time: 40 seconds**

**Script:**

> "Now let me be honest about limitations.
>
> [Point to first section]
> First, Incomplete Physics.
> - No gas phase. No vapor, no bubble, no spatter.
> - Simplified mushy zone. No dendrites.
> - This limits defect prediction.
>
> [Point to second section]
> Second, Simplified Laser Model.
> - We use Gaussian heat source only.
> - No reflection. No multiple scattering in keyhole.
> - For deep keyhole, this matters.
>
> [Point to third section]
> Third, No Rigorous Benchmark.
> - Similar projects like waLBerla have no native multiphase thermal.
> - Our performance numbers are estimates.
>
> Platform works, but still early stage.
>
> Now let me talk about AI workflow."

---

## Slide 10: AI-Assisted Development Workflow - Problem
**Time: 50 seconds**

**Script:**

> "Shengfeng showed this workflow in October. Let me explain my improvement.
>
> [Point to top workflow diagram]
> This is the workflow. Start with requirements. AI helps polish. Human reviews. Code agent writes code. Run simulation. Human reviews results.
>
> [Point to Human Review box - red outline]
> The problem is Human Review. When I get simulation results, how do I describe problems to AI?
>
> [Point to problem list]
> Four challenges:
> - Description is lossy. I might miss things.
> - Strange bugs have no words. I don't know how to describe.
> - VTK files are huge. Gigabytes. AI cannot read directly.
> - 3D phenomena are hard to verbalize.
>
> [Point to VTK Analysis Agent]
> My solution: add VTK Analysis Agent to the loop."

---

## Slide 11: AI-Assisted Development Workflow - Solution
**Time: 50 seconds**

**Script:**

> "Here is how VTK Analysis Agent works.
>
> [Point to flow diagram]
> Run simulation. Get VTK files. VTK Analysis Agent writes Python script. Script extracts data. Data goes to Code Agent. Code Agent fixes bugs.
>
> [Point to key principle]
> Very important: AI writes scripts, not reads data.
>
> Why? AI makes mistakes with units and numbers. Python does not.
>
> [Point to my role]
> What I do:
> - Review the visualization in ParaView
> - Tell AI what to analyze: check temperature, check velocity, check interface
> - AI writes the script
> - Script gives me numbers
>
> This way, AI helps but does not make calculation errors."

---

## Slide 12: AI-Assisted Development Workflow - Code Examples
**Time: 40 seconds**

**Script:**

> "Here are real examples.
>
> [Point to left code - agent definition]
> This is the VTK Analysis Agent I defined. It knows how to write Python scripts for VTK data.
>
> [Point to middle code - ParaView script]
> This is a script AI wrote. It reads VTK files, sets up visualization, creates animation.
>
> [Point to right code - analysis script]
> This is another script. It extracts temperature and liquid fraction at different depths. Compares with physical expectations.
>
> All scripts are saved. I can run them again. I can modify them.
>
> This is my improvement to Shengfeng's workflow."

---

## Slide 13: "AI Write Codes, I Make Decision"
**Time: 50 seconds**

**Script:**

> "Let me summarize my role. AI writes codes. I make decisions.
>
> [Point to four boxes]
>
> [Point to Architect]
> First, Architect. I designed LBM plus GPU approach. I designed modular framework. AI cannot make these big decisions.
>
> [Point to Physics Judge]
> Second, Physics Judge. I identified unphysical results. Temperature too high? Flow direction wrong? I diagnosed root causes.
>
> [Point to Debug Director]
> Third, Debug Director. I spotted anomalies in visualization. I narrowed the scope for AI. I told AI where to look.
>
> [Point to Quality Gate]
> Fourth, Quality Gate. I reviewed code. I rejected over-engineering. I made sure physics is correct.
>
> [Point to bottom quote]
> This project is large and complex. It validates the workflow works for real development."

---

## Slide 14: Roadmap & Summary
**Time: 45 seconds**

**Script:**

> "[Point to timeline]
> Here is our roadmap.
>
> Now: LPBF basic simulation validated.
>
> Next: Complete physical field. Add recoil pressure, add evaporation. Better benchmark with more cases.
>
> Then: Other AM processes. DED, EBM, LIFT.
>
> Future: Multi-GPU for large domains. Eventually productization.
>
> [Point to Key Achievements box]
> What we achieved:
> - Modular GPU-LBM platform
> - 11 physics modules
> - 37 to 150 times speedup
> - AI-assisted workflow that works
>
> [Point to quote]
> One sentence summary: A fast, modular platform for AM process exploration.
>
> Thank you."

---

## Slide 15: Thanks
**Time: 15 seconds**

**Script:**

> "Thank you for listening.
>
> I am happy to answer questions."

---

# Q&A Preparation

## Q1: "Why LBM instead of FVM?"

> "Two reasons. First, LBM has no pressure Poisson solver. In FVM, you must solve a global equation every time step. This is hard to parallelize. LBM does not need this.
>
> Second, LBM is explicit and local. Each cell computes independently. Perfect for GPU.
>
> Trade-off: LBM interface is more diffuse. But for fast exploration, this is acceptable."

---

## Q2: "How do you handle gas phase?"

> "We use void boundary. When liquid fraction is below 0.1, we set velocity to zero. This is simple but works for metals. Why? Density ratio is 3400 to 1. Gas effect is very small."

---

## Q3: "The code is written by AI. What did YOU do?"

> "Good question. AI writes code, but AI cannot make decisions.
>
> I decided the architecture: LBM plus GPU, modular design.
> I judged physics: is this result correct? What is wrong?
> I directed debugging: where to look, what to fix.
> I controlled quality: reject bad code, ensure physics is right.
>
> AI is a powerful tool. But human expertise is still essential."

---

## Q4: "How accurate is 'good enough'?"

> "For engineering purpose, less than 10 percent error is acceptable. We achieved less than 6 percent for melt pool geometry. This is good for process parameter exploration."

---

## Q5: "Why not let AI read VTK directly?"

> "Two reasons. First, VTK files are huge. Gigabytes per frame. Token cost would be astronomical.
>
> Second, AI makes calculation mistakes. Wrong units, wrong magnitudes. But Python scripts don't make these mistakes. So AI writes the script, script does the calculation. This is safe."

---

# Time Summary

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title | 30 sec |
| 2 | Why LBM+GPU | 45 sec |
| 3 | Architecture | 50 sec |
| 4 | Physics Modules | 45 sec |
| 5 | Case 120W | 40 sec |
| 6 | Case 300W | 40 sec |
| 7 | Validation | 50 sec |
| 8 | Validation Summary | 45 sec |
| 9 | Limitations | 40 sec |
| 10 | AI Workflow Problem | 50 sec |
| 11 | AI Workflow Solution | 50 sec |
| 12 | Code Examples | 40 sec |
| 13 | My Role | 50 sec |
| 14 | Roadmap | 45 sec |
| 15 | Thanks | 15 sec |
| **Total** | | **~10 min** |

---

# Tips for Delivery

1. **Practice the numbers**: 44.0, 42.7, 2.9%, 119.2, 126.1, -5.4%, 1.2 m/s. Say them clearly.

2. **Point and pause**: When you say "look at this", point first, then speak.

3. **Slow down at key points**:
   - "AI writes scripts, not reads data"
   - "AI writes codes, I make decisions"
   - "2.9 percent error"

4. **Be confident about limitations**: Slide 9 shows you are honest. This is good.

5. **Eye contact**: Look at audience, especially when saying conclusions.

6. **Breathe**: Pause between slides. Take a breath.
