# PPT Design Specification
## A GPU-Accelerated LBM Platform for Additive Manufacturing Process Simulation

**Presentation Duration**: 10 minutes
**Audience**: AM simulation experts (familiar with AM, some LBM knowledge)
**Language**: English
**Style**: Technical but accessible, honest about limitations

---

## Overall Design Guidelines

### Color Scheme
```
Primary:    #2C3E50 (Dark blue - titles, text)
Accent:     #E67E22 (Orange - highlights, our work)
Success:    #27AE60 (Green - completed items)
Warning:    #F1C40F (Yellow - in progress)
Light:      #ECF0F1 (Light gray - backgrounds)
White:      #FFFFFF (Slide background)
```

### Typography
```
Title:      36-40pt, Bold
Subtitle:   24-28pt, Regular
Body:       18-20pt, minimum 16pt
```

### Layout Principles
- **60-70% figures**, 30-40% text per slide
- Maximum **30 words** per slide (excluding figure labels)
- Use figures to tell the story, text only for key points

---

## Slide-by-Slide Specification

**NOTE**: This presentation has TWO parts:
1. **Part 1 (Slides 1-3)**: AI Workflow - Methodology introduction (~2.5 min)
2. **Part 2 (Slides 4-10)**: Technical - The LBM Platform (~7.5 min)

**Logic**: First introduce AI workflow → Then show project results → Good results validate the workflow

---

### SLIDE 1: Title Slide

**Layout**: Centered, clean

**Content**:
```
A GPU-Accelerated LBM Platform
for Additive Manufacturing Process Simulation

Subtitle: First Implementation — Laser Powder Bed Fusion

[Name]
[Date] | Group Meeting
```

**Visual**: Optional small melt pool image in corner (from xz_slice_analysis.png, t=400μs frame)

**Speaker Notes**: Brief introduction, 15 seconds

---

### SLIDE 2: AI-Assisted Development Workflow

**Layout**: Full-page flow diagram

**Main Figure**: `ai_workflow_comparison.png`
- Path: `/home/yzk/LBMProject/docs/figures/ai_workflow_comparison.png`

**Content Structure**:

**Top Section - 学长's Workflow (brief mention)**:
```
AI-Assisted Code Development (presented before by [学长's name]):
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│Reqmt │ → │ AI   │ → │Human │ → │ Code │ → │Human │→ Loop
│ List │   │Polish│   │Review│   │Agent │   │Review│
└──────┘   └──────┘   └──────┘   └──────┘   └──────┘
```

**Middle Section - Challenges in Previous Workflow** (详细展开):
```
Challenges of Human Review in CFD/Simulation Development:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DESCRIBING RESULTS TO AI IS LOSSY
   • Human descriptions are biased/incomplete
   • Weird phenomena? Don't even know how to describe
   • Information loss between what you see and what AI understands

2. VTK FILES ARE IMPOSSIBLE FOR AI
   • Single frame: GB-scale for 3D results
   • AI cannot read VTK/VTU formats (Gemini, GPT, etc.)
   • Even if possible, astronomical token cost

3. SCREENSHOTS LOSE CRITICAL INFO
   • AI vision is weak for numerical analysis
   • Cannot extract actual values from images
   • Especially bad for numerical accuracy issues

4. COMPLEX 3D PHENOMENA
   • Melt pool shape, keyhole geometry
   • Internal flow patterns
   • Phase boundaries in 3D space
   → Human struggles to describe to AI
```

**My Improvement: VTK Analysis Agent**:
```
Solution Architecture:
┌──────────┐    ┌─────────────────┐    ┌──────────┐    ┌──────────┐
│   Run    │ →  │  VTK Analysis   │ →  │  Human   │ →  │  Code    │
│   Sim    │    │     Agent       │    │  Guide   │    │  Agent   │
└──────────┘    └─────────────────┘    └──────────┘    └──────────┘
                       │
                       ↓
            Writes Python scripts to extract:
            • T, v ranges & distributions
            • Interface shape & position
            • Phase fractions & locations
            • Whatever human asks for

Key: Human still reviews visualization
     But VTK Agent extracts structured metrics
     → Code Agent gets precise, actionable info
```

**Bottom Section - Critical Principle** (NEW):
```
⚠️ Critical: Never let AI calculate directly
   AI often gets units, magnitudes, variable names wrong

✓ AI writes scripts → Scripts execute calculations
```

**Speaker Notes** (1.5 min):
- Briefly mention: 学长 presented this workflow before, I'm using it
- My project validates this workflow — it's a large, complex CFD project
- My improvement: VTK Analysis Agent for simulation result feedback
- Key principle: AI plans and writes code, but never does calculations itself
  - AI makes mistakes with units, magnitudes, variable names
  - Scripts are reliable, AI planning is valuable

---

### SLIDE 3: My Role in AI Development

**Layout**: Full-page figure

**Main Figure**: `my_role_in_ai_dev.png`
- Path: `/home/yzk/LBMProject/docs/figures/my_role_in_ai_dev.png`

**Content**:
```
"AI writes code, I make decisions"

┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  ARCHITECT  │ │PHYSICS JUDGE│ │DEBUG DIRECTOR│ │QUALITY GATE │
├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤
│• LBM+GPU    │ │• Identify   │ │• Spot       │ │• Review code│
│  approach   │ │  wrong      │ │  anomalies  │ │• Reject over│
│• Modular    │ │  results    │ │• Narrow     │ │  engineering│
│  framework  │ │• Diagnose   │ │  scope      │ │• Ensure     │
│• Phase-by-  │ │  root cause │ │• Guide AI   │ │  correctness│
│  phase      │ │• Select     │ │  direction  │ │             │
│             │ │  benchmarks │ │             │ │             │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

**Bottom Box**:
```
This project: Large scale (11 modules, multi-physics coupling)
             Complex CFD problem (GPU + LBM + free surface + phase change)
             → Validates the AI workflow is effective for real development
```

**Speaker Notes** (1 min):
- Be direct: Yes, AI wrote the code
- But human makes all key decisions:
  - Architecture choices (why LBM, why GPU, which physics to include)
  - Physics judgment (is this result correct? what's wrong?)
  - Debug direction (where to look, what to fix)
  - Quality control (reject bad code, ensure physical correctness)
- This project is large and complex — it demonstrates the workflow works

---

### SLIDE 4: Why LBM + GPU?

**Layout**: Full-page figure with minimal text overlay

**Main Figure**: `tradeoff_triangle.png`
- Path: `/home/yzk/LBMProject/docs/figures/tradeoff_triangle.png`
- Shows Speed-Accuracy-Generality trade-off
- Our position (LBM+GPU) marked in orange

**Text Overlay** (bottom or side):
```
Why LBM + GPU?
• Explicit time-stepping → massive parallelism
• No Poisson solver → GPU-friendly
• Trade-off: Speed over precision
```

**Key Message**: "Balanced trade-off for rapid process exploration"

**Speaker Notes** (1 min):
- AM processes share common physics (melting, solidification, fluid flow)
- Need fast simulation for parameter exploration
- LBM is naturally parallel, GPU amplifies this
- Acknowledge trade-off: we sacrifice some precision for speed

---

### SLIDE 3: Platform Architecture

**Layout**: Full-page architecture diagram

**Main Figure**: `architecture_diagram.png`
- Path: `/home/yzk/LBMProject/docs/figures/architecture_diagram.png`
- Four-layer structure:
  - APPLICATION LAYER (LPBF ✓, DED, EBM planned)
  - MULTIPHYSICS SOLVER
  - PHYSICS MODULES (Thermal, Fluid, VOF, Phase Change, Marangoni, Laser)
  - LBM CORE (CUDA)

**Text** (bottom):
```
Modular design: Add new AM process without changing physics core
```

**Speaker Notes** (1.5 min):
- Emphasize modularity: physics modules are reusable across different AM processes
- LPBF is first application, DED/EBM can reuse same physics
- Config-driven: change parameters without recompilation
- This is a platform, not just a single application

---

### SLIDE 4: Physics Modules

**Layout**: Full-page table

**Main Figure**: `fig6_physics_modules.png`
- Path: `/home/yzk/LBMProject/docs/figures/fig6_physics_modules.png`
- Shows 11 physics modules with implementation method and validation status
- 10 verified (green), 1 partial (yellow)

**Text** (bottom):
```
11 modules implemented, 10 verified against analytical solutions or literature
```

**Table Content** (for reference):
| Module | Implementation | Status | Validation |
|--------|---------------|--------|------------|
| Thermal Conduction | D3Q7 MRT | ✓ Verified | Analytical solution |
| Phase Change | Enthalpy method | ✓ Verified | Stefan problem |
| Evaporation | Hertz-Knudsen | ✓ Verified | Mass flux match |
| Marangoni Effect | CSF + ∇T | ✓ Verified | Khairallah 2016 |
| Surface Tension | CSF model | ✓ Verified | Laplace pressure |
| Recoil Pressure | T-dependent | ○ Partial | Needs validation |
| Darcy Damping | Mushy zone | ✓ Verified | Solidification |
| Buoyancy | Boussinesq | ✓ Verified | Natural convection |
| Laser Heating | Gaussian+Beer | ✓ Verified | Energy integral |
| Radiation BC | Stefan-Boltzmann | ✓ Verified | T⁴ law |
| Substrate Cooling | Newton cooling | ✓ Verified | Energy balance |

**Speaker Notes** (1.5 min):
- Cover key modules: Thermal (D3Q7), Fluid (D3Q19), Phase Change (enthalpy)
- Highlight Marangoni: critical for melt pool dynamics
- Mention gas phase is simplified as void boundary (sufficient for metal AM due to ~3400:1 density ratio)

---

### SLIDE 5: Case Study — LPBF Simulation

**Layout**: Left panel (parameters) + Right panel (figure, 60% width)

**Main Figure**: `xz_slice_analysis.png`
- Path: `/home/yzk/LBMProject/docs/figures/xz_slice_analysis.png`
- Four-frame evolution: t = 0, 200, 300, 400 μs
- Shows melt pool formation and keyhole development

**Left Panel - Parameters Table**:
```
LPBF Single Track — Ti-6Al-4V
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Domain        400 × 300 × 200 μm³
Resolution    2 μm
Laser Power   350 W
Scan Speed    0.4 m/s
Spot Size     70 μm
Duration      500 μs (laser on)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compute Time  ~1 min (single GPU)
```

**Text** (bottom):
```
Keyhole formation captured — melt pool depth ~100 μm at t = 400 μs
```

**Speaker Notes** (2 min):
- Walk through the four frames: initial flat surface → heating → melting → keyhole
- Point out the keyhole formation at t=400μs
- Mention runtime: entire simulation ~1 minute on consumer GPU
- This demonstrates the platform works for realistic LPBF conditions

---

### SLIDE 6: Validation Results

**Layout**: Main figure with key metrics

**Main Figure**: `fig1_energy_balance.png`
- Path: `/home/yzk/LBMProject/presentation_figures/fig1_energy_balance.png`
- Top panel: Power balance (laser input, evaporation, radiation)
- Bottom panel: Energy error < 5%

**Key Points** (side or overlay):
```
Validation Summary
━━━━━━━━━━━━━━━━━━
✓ Energy conservation: < 5% error
✓ Evaporation activates at T > T_boil
✓ Numerical stability: 500+ μs runs

Performance:
• 37-150× faster than CPU FEM
• 49M cells·steps/s throughput
```

**Optional Secondary Figure**: `fig6_performance.png` (if space allows)
- Path: `/home/yzk/LBMProject/presentation_figures/fig6_performance.png`
- Shows speedup comparison with ANSYS Fluent estimate

**Speaker Notes** (1.5 min):
- Energy balance shows the simulation is physically consistent
- Evaporation kicks in when temperature exceeds boiling point — expected behavior
- Performance: significant speedup, but acknowledge it's not a direct comparison (different physics modules)
- Quantitative validation against experimental data is ongoing work

---

### SLIDE 7: Current Limitations

**Layout**: Full-page table

**Main Figure**: `limitations_table.png`
- Path: `/home/yzk/LBMProject/docs/figures/limitations_table.png`

**Table Content**:
| Limitation | Current Status | Note |
|------------|---------------|------|
| Interface diffusion | 3-5 cells (vs 1-2 FVM) | LBM inherent |
| Gas phase | Void boundary | Sufficient for metals |
| Single GPU | ~10M cells max | Multi-GPU planned |
| Temperature calibration | WIP | Absorption tuning |

**Footer Text**:
```
These are inherent trade-offs of LBM approach, not implementation bugs
```

**Speaker Notes** (1 min):
- Be honest: every method has trade-offs
- Interface diffusion: LBM-VOF is more diffusive than FVM-VOF, this is fundamental
- Gas phase: simplified as void, but okay for metals (density ratio ~3400:1)
- Single GPU: current limitation, multi-GPU is planned
- Temperature: calibration in progress, absorption coefficient needs tuning

---

### SLIDE 8: AI-Assisted Development Workflow

**Layout**: Full-page flow diagram

**Main Figure**: `ai_workflow_comparison.png`
- Path: `/home/yzk/LBMProject/docs/figures/ai_workflow_comparison.png`

**Flow Diagram Content**:
```
Previous Workflow (学长's approach - briefly mention):
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│Reqmt │ → │ AI   │ → │Human │ → │ Code │ → │Human │→ Loop
│ List │   │Polish│   │Review│   │Agent │   │Review│
└──────┘   └──────┘   └──────┘   └──────┘   └──────┘
                                              ↑
                              ⚠️ Bottleneck: Describing
                                 simulation results to AI

My Improvement:
┌──────┐   ┌────────────┐   ┌──────┐   ┌──────┐
│ Run  │ → │VTK Analysis│ → │Human │ → │ Code │
│ Sim  │   │  Agent     │   │Guide │   │Agent │
└──────┘   └────────────┘   └──────┘   └──────┘
                 │
                 ↓
      Writes Python scripts to extract:
      • T, v ranges & distributions
      • Interface geometry
      • Phase fractions
```

**Key Points** (text):
```
Challenge in CFD + AI:
• VTK files: GB-scale, impossible for AI to read
• Human description: information loss
• Screenshots: AI vision loses numerical precision

Solution: VTK Analysis Agent
• Writes scripts to extract key metrics
• Human guides analysis direction
• Structured output for Code Agent
```

**Speaker Notes** (1.5 min):
- Acknowledge: 学长's workflow was presented before, this builds on it
- Problem: In CFD, simulation results are hard to describe to AI
  - VTK files are huge (GB per frame)
  - Complex 3D phenomena hard to put into words
  - Screenshots lose numerical info
- Solution: VTK Analysis Agent writes Python scripts
  - Extracts T/v ranges, interface shapes, phase fractions
  - Human reviews visualization, guides what to analyze
  - Structured metrics → Code Agent can understand
- This is essential for large-scale 3D CFD projects

---

### SLIDE 9: My Role in AI-Assisted Development

**Layout**: Full-page figure

**Main Figure**: `my_role_in_ai_dev.png`
- Path: `/home/yzk/LBMProject/docs/figures/my_role_in_ai_dev.png`

**Content (shown in figure)**:
```
What I Did (AI writes code, I make decisions):

1. ARCHITECT
   • Chose LBM+GPU approach
   • Designed modular framework
   • Phase-by-phase development strategy

2. PHYSICS JUDGE
   • Identified physically wrong results
   • Diagnosed root cause (which module?)
   • Selected validation benchmarks

3. DEBUG DIRECTOR
   • Spotted anomalies in visualizations
   • Narrowed problem scope
   • Guided AI to correct direction

4. QUALITY GATE
   • Reviewed generated code
   • Rejected over-engineering
   • Ensured physical correctness
```

**Right Side Box**:
```
┌─────────────────────────────┐
│                             │
│  "AI writes code,           │
│   I make decisions."        │
│                             │
│  • Architecture             │
│  • Physics judgment         │
│  • Debugging direction      │
│  • Quality control          │
│                             │
└─────────────────────────────┘
```

**Speaker Notes** (1 min):
- Be direct: Yes, AI wrote most of the code
- But AI cannot:
  - Decide technical approach (LBM vs FVM, GPU vs CPU)
  - Judge if results are physically correct
  - Know which direction to debug
  - Decide when to accept trade-offs vs fix bugs
- Example: Marangoni direction bug — I saw flow going wrong way, told AI to check sign
- Example: Temperature too high — I knew T>6000K is unphysical, guided calibration
- Human expertise remains essential for decision-making

---

### SLIDE 10: Roadmap & Summary

**Layout**: Timeline (top half) + Summary points (bottom half)

**Top Half - Timeline**: Use `roadmap_timeline.png`
- Path: `/home/yzk/LBMProject/docs/figures/roadmap_timeline.png`

**Bottom Half - Summary**:
```
Key Achievements
━━━━━━━━━━━━━━━━
✓ Modular GPU-LBM platform for AM simulation
✓ 11 physics modules, 10 validated
✓ 37-150× speedup demonstrated
✓ LPBF case study working
✓ AI-assisted workflow improvement

"A fast, modular platform for AM process exploration"
```

**Speaker Notes** (1 min):
- Roadmap: LPBF done → Multi-track → Other AM → Multi-GPU
- Platform, not just one simulation
- AI-assisted development works for complex CFD
- Questions?

---

### (OPTIONAL) SLIDE 11: Backup - Scorecard

Only show if asked for more details

**Figure**: `fig7_scorecard.png`
- Path: `/home/yzk/LBMProject/presentation_figures/fig7_scorecard.png`

**Layout**: Left panel (bullet points) + Right panel (scorecard figure)

**Main Figure**: `fig7_scorecard.png`
- Path: `/home/yzk/LBMProject/presentation_figures/fig7_scorecard.png`
- Project status scorecard

**Left Panel**:
```
Key Achievements
━━━━━━━━━━━━━━━━
✓ Modular GPU-LBM platform built
✓ 11 physics modules, 10 validated
✓ 37-150× speedup demonstrated
✓ LPBF simulation working

One-liner:
"A fast, modular simulation platform
 for AM process exploration"
```

**Closing**:
```
Questions?
```

**Speaker Notes** (1 min):
- Summarize: built a platform, not just a one-off simulation
- Validated core physics, demonstrated on LPBF
- Speed advantage enables rapid parameter exploration
- Open for questions

---

## Figure Files Summary

All figures are ready to use:

| Slide | Figure File | Full Path |
|-------|-------------|-----------|
| 1 | (optional) xz_slice crop | `/home/yzk/LBMProject/docs/figures/xz_slice_analysis.png` |
| 2 | **ai_workflow_comparison.png** | `/home/yzk/LBMProject/docs/figures/ai_workflow_comparison.png` |
| 3 | **my_role_in_ai_dev.png** | `/home/yzk/LBMProject/docs/figures/my_role_in_ai_dev.png` |
| 4 | tradeoff_triangle.png | `/home/yzk/LBMProject/docs/figures/tradeoff_triangle.png` |
| 5 | architecture_diagram.png | `/home/yzk/LBMProject/docs/figures/architecture_diagram.png` |
| 6 | fig6_physics_modules.png | `/home/yzk/LBMProject/docs/figures/fig6_physics_modules.png` |
| 7 | xz_slice_analysis.png | `/home/yzk/LBMProject/docs/figures/xz_slice_analysis.png` |
| 8 | fig1_energy_balance.png | `/home/yzk/LBMProject/presentation_figures/fig1_energy_balance.png` |
| 8 | fig6_performance.png (opt) | `/home/yzk/LBMProject/presentation_figures/fig6_performance.png` |
| 9 | limitations_table.png + roadmap_timeline.png | See paths below |
| 10 | fig7_scorecard.png (optional) | `/home/yzk/LBMProject/presentation_figures/fig7_scorecard.png` |

**Additional paths**:
- limitations_table.png: `/home/yzk/LBMProject/docs/figures/limitations_table.png`
- roadmap_timeline.png: `/home/yzk/LBMProject/docs/figures/roadmap_timeline.png`

---

## Time Allocation

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title | 15 sec |
| 2 | **AI Workflow** (学长's + my improvement) | 1.5 min |
| 3 | **My Role** ("AI writes, I decide") | 1 min |
| 4 | Why LBM+GPU (Trade-off) | 1 min |
| 5 | Architecture | 1 min |
| 6 | Physics Modules | 1 min |
| 7 | Case Study (LPBF) | 1.5 min |
| 8 | Validation | 1 min |
| 9 | Limitations + Roadmap | 1 min |
| 10 | Summary | 30 sec |
| **Total** | | **~10 min** |

**Two Parts**:
- Part 1 (Slides 1-3): AI Workflow introduction ~2.5 min
- Part 2 (Slides 4-10): Technical content ~7.5 min

**Logic**: AI Workflow → Project Demo → Good results validate the workflow

---

## Key Messages to Convey

1. **This is a PLATFORM, not just one simulation**
   - Modular design allows extension to other AM processes
   - LPBF is the first validated application

2. **LBM+GPU is a trade-off choice**
   - Fast (37-150× vs CPU)
   - But less precise than FVM (interface diffusion, simplified gas)
   - Suitable for rapid exploration, not final engineering design

3. **Honest about limitations**
   - Interface diffuses over 3-5 cells
   - Gas phase is void boundary
   - Temperature calibration ongoing
   - These are method trade-offs, not bugs

4. **Work is validated but ongoing**
   - Energy conservation verified
   - Core physics working
   - More experimental validation planned

---

## Things NOT to Include

1. **Do NOT show `fig5_literature_comparison.png`**
   - Shows +103% temperature error, -3-36× velocity error
   - Too negative for presentation; mention verbally that "calibration is WIP"

2. **Do NOT claim "100× faster" without context**
   - Speedup depends on what you compare
   - Always say "compared to CPU FEM with similar physics"

3. **Do NOT overstate accuracy**
   - LBM-VOF is more diffusive than FVM-VOF
   - Be honest: this is a rapid exploration tool

---

## Q&A Preparation

**Likely Questions**:

1. **Why LBM instead of FVM?**
   - Natural parallelism, no pressure solver, GPU-friendly
   - Trade-off: less accurate interface

2. **How do you handle gas phase?**
   - Void boundary (velocity zeroed for fill < 0.1)
   - Justified by ~3400:1 density ratio for metals

3. **What about experimental validation?**
   - Energy conservation validated
   - Quantitative comparison with experiments is ongoing
   - Temperature calibration in progress

4. **Can this handle keyhole welding?**
   - Yes, demonstrated in case study
   - But interface precision may limit deep keyhole accuracy

5. **Performance compared to commercial tools?**
   - 37-150× faster than CPU FEM estimate
   - Not direct comparison (different physics completeness)
   - Our advantage: open, customizable, GPU-native

---

## Appendix: Technical Details (if asked)

**Grid**:
- Thermal: D3Q7 (7 velocities)
- Fluid: D3Q19 (19 velocities)
- Resolution: typically 1-2 μm

**Time stepping**:
- dt = 0.1 μs (explicit)
- CFL < 0.5

**Material** (Ti-6Al-4V):
- T_solidus = 1878 K
- T_liquidus = 1928 K
- T_boil = 3533 K
- σ = 1.65 N/m, dσ/dT = -2.6×10⁻⁴ N/(m·K)

**Performance**:
- ~49 M cells·steps/s on RTX 3050
- 8M cells domain runs in ~100 seconds

---

*Document generated: 2024-11-27*
*For: Group meeting presentation preparation*
