# Speaker Scripts for 10-Minute Group Meeting Presentation

## LBM-CUDA Platform for LPBF Simulation

---

## SLIDE 1: Title Slide

**Time: 15 seconds**

### Script:

> "Good morning everyone. Today I will talk about a GPU-LBM platform for additive manufacturing simulation. This is our first application -- Laser Powder Bed Fusion, or LPBF."

### Key Points:
- Speak clearly, make eye contact
- Pause briefly after title

---

## SLIDE 2: AI-Assisted Development Workflow

**Time: 1 minute 30 seconds**

### Script:

> "Before I show the project, let me talk about HOW we built it.
>
> [Point to top section]
> My senior colleague showed this workflow before. The idea is simple: AI writes code, human reviews code. This works well for most software.
>
> [Point to middle section - challenges]
> But for CFD simulation, we have a BIG problem. How do I tell AI what is wrong?
>
> Think about it:
> - VTK files are huge. One frame can be gigabytes. AI cannot read this.
> - Screenshots? AI vision is weak. It cannot read numbers from images.
> - I describe the problem in words? I might miss important things. Sometimes I don't even know how to describe a strange bug.
>
> [Point to bottom section - my solution]
> So here is my small improvement: VTK Analysis Agent.
>
> This agent does NOT read the VTK file directly. Instead, it WRITES Python scripts. The scripts extract key data: temperature range, velocity range, interface shape, phase fractions.
>
> [Emphasize this point]
> One critical rule: Never let AI calculate numbers directly. AI makes mistakes with units and magnitudes. AI writes the script, script does the calculation. This is safe."

### Key Points:
- Point to each section as you explain
- Slow down at "Never let AI calculate directly" - this is important
- Keep it conversational, like explaining to a friend

---

## SLIDE 3: My Role in AI Development

**Time: 1 minute**

### Script:

> "Now, you might ask: if AI writes code, what do YOU do?
>
> [Point to the four boxes]
> Let me explain my four roles:
>
> First, ARCHITECT. I decided to use LBM plus GPU. I designed the modular structure. AI cannot make these big decisions.
>
> Second, PHYSICS JUDGE. When results look wrong, I know it. For example, if temperature is 6000 Kelvin, that is too high. AI does not know physics like I do.
>
> Third, DEBUG DIRECTOR. I watch the simulation. I see strange flow. I tell AI: check the Marangoni sign. I narrow down the problem.
>
> Fourth, QUALITY GATE. I review the code. I reject over-complicated solutions. I make sure the physics is correct.
>
> [Pause]
> In short: AI writes code, I make decisions. This project is large and complex. It proves this workflow works."

### Key Points:
- Count on fingers: "First... Second... Third... Fourth..."
- The quote "AI writes code, I make decisions" - say it clearly
- Sound confident, not defensive

---

## SLIDE 4: Why LBM + GPU?

**Time: 1 minute**

### Script:

> "Now let me show the project itself.
>
> [Point to triangle diagram]
> In simulation, you cannot have everything. This triangle shows the trade-off: Speed, Accuracy, and Generality. Pick two, lose one.
>
> [Point to our position]
> We chose LBM plus GPU. Why?
>
> LBM has explicit time stepping. This means massive parallelism. Every cell can compute at the same time.
>
> Also, LBM has no Poisson solver. FVM needs to solve pressure equation -- this is hard to parallelize. LBM does not need this.
>
> [Be honest]
> The trade-off? We lose some precision. Interface is more diffuse than FVM. But for fast exploration of parameters, this is acceptable."

### Key Points:
- Point to triangle corners as you mention each trade-off
- "No Poisson solver" - this is the key GPU advantage
- Be honest about the trade-off

---

## SLIDE 5: Platform Architecture

**Time: 1 minute**

### Script:

> "Here is the platform structure.
>
> [Point from bottom to top]
> At the bottom: LBM Core with CUDA. This is the engine.
>
> Middle layer: Physics Modules. Thermal, Fluid, VOF, Phase Change, Marangoni, Laser heating. These are building blocks.
>
> Top layer: Applications. LPBF is done. DED and EBM are planned.
>
> [Emphasize]
> The key point: this is a PLATFORM, not just one simulation. Physics modules can be reused. When we add DED later, we don't rewrite everything. Just add new laser model."

### Key Points:
- Move hand from bottom to top showing the layers
- Emphasize "platform" - this is important for showing value
- "Reusable modules" is the selling point

---

## SLIDE 6: Physics Modules

**Time: 1 minute**

### Script:

> "We have 11 physics modules.
>
> [Point to table]
> Let me highlight the important ones:
>
> Thermal conduction uses D3Q7 lattice. Validated against analytical solution.
>
> Phase change uses enthalpy method. Validated against Stefan problem.
>
> Marangoni effect -- this is critical for melt pool flow. Validated against Khairallah 2016.
>
> [Point to status column]
> You can see: 10 modules are verified with green check. Only recoil pressure is partial -- we need more validation data.
>
> One note: gas phase is simplified as void boundary. Why? Metal to gas density ratio is 3400 to 1. Gas effect is very small. This simplification is acceptable."

### Key Points:
- Don't read every module - pick 3-4 important ones
- Green checks are visual proof of validation
- Explain the gas simplification honestly

---

## SLIDE 7: Case Study -- LPBF Simulation

**Time: 1 minute 30 seconds**

### Script:

> "Now let me show a real simulation.
>
> [Point to parameters]
> This is Ti-6Al-4V, a common titanium alloy. Laser power 350 watts, scan speed 0.4 meters per second. Domain is 400 by 300 by 200 microns. Resolution is 2 microns.
>
> [Point to the four frames]
> Look at the evolution:
>
> At t equals zero, flat surface.
>
> At 200 microseconds, laser heats the surface, melting starts.
>
> At 300 microseconds, melt pool grows deeper.
>
> At 400 microseconds, you can see keyhole forming. Melt pool depth is about 100 microns.
>
> [Mention performance]
> Total compute time: about 1 minute on a single GPU. This is fast. Traditional FEM would take hours or days."

### Key Points:
- Walk through the four frames slowly
- Point to the keyhole at t=400
- "1 minute" - emphasize the speed

---

## SLIDE 8: Validation -- Energy Conservation

**Time: 1 minute**

### Script:

> "First validation: energy balance.
>
> [Point to top panel]
> Top chart shows power over time. Blue is laser input. Orange is evaporation loss. Green is radiation loss.
>
> You can see: evaporation turns on when temperature exceeds boiling point. This is correct physics.
>
> [Point to bottom panel]
> Bottom chart shows energy error. It stays below 5 percent. This means our simulation conserves energy well.
>
> Energy conservation is fundamental. If energy is wrong, nothing else can be trusted."

### Key Points:
- "Below 5 percent" - say this number clearly
- Connect physics to the chart (evaporation at boiling point)

---

## SLIDE 9: Validation -- Melt Pool Depth

**Time: 1 minute**

### Script:

> "Second validation: melt pool depth.
>
> [Point to comparison]
> We compare with Ye et al. They did CFD-VOF simulation of Ti-6Al-4V.
>
> Their melt pool depth: 45 microns.
> Our melt pool depth: 44 microns.
>
> Difference: only minus 2.2 percent.
>
> [Emphasize]
> This is excellent agreement. Melt pool depth is the most important metric. It controls heat penetration and weld quality.
>
> Note: their power is 100 watts, ours is 300 watts. So width is different. But depth matches well because scan speed is the same."

### Key Points:
- "44 versus 45 microns" - say these numbers clearly
- "-2.2 percent" - this is the key result
- Explain why depth matters more than width

---

## SLIDE 10: Validation -- Marangoni Flow

**Time: 45 seconds**

### Script:

> "Third validation: Marangoni flow velocity.
>
> Marangoni effect is surface tension driven flow. It is critical for melt pool mixing.
>
> [Point to comparison]
> Khairallah 2016 reports flow velocity between 0.5 and 2.0 meters per second.
>
> Our result: 1.2 meters per second. This is within the range.
>
> This shows our temperature field and flow field are coupled correctly."

### Key Points:
- "0.5 to 2.0" and "1.2" - say numbers clearly
- "Within the range" = PASS

---

## SLIDE 11: Validation Summary Table

**Time: 30 seconds**

### Script:

> "[Point to table]
> Here is the summary.
>
> Melt pool depth: minus 2.2 percent error. PASS.
>
> Marangoni velocity: within literature range. PASS.
>
> Energy conservation: below 5 percent error. PASS.
>
> Three key metrics, all validated."

### Key Points:
- Quick summary - don't repeat details
- "Three metrics, all pass" - confident conclusion

---

## SLIDE 12: Current Limitations

**Time: 1 minute**

### Script:

> "Now let me be honest about limitations.
>
> [Point to each row]
> First, interface diffusion. Our interface spreads over 3 to 5 cells. FVM-VOF is sharper, only 1 to 2 cells. This is a fundamental LBM limitation, not a bug.
>
> Second, gas phase. We use void boundary. This is okay for metals because density ratio is 3400 to 1. But for keyhole vapor dynamics, we need better model.
>
> Third, single GPU. Current code runs on one GPU only. Maximum about 10 million cells. Multi-GPU is planned.
>
> Fourth, temperature calibration. We are still tuning absorption coefficient. This is work in progress.
>
> [Pause]
> These are trade-offs of our approach. We chose speed over precision. For parameter exploration, this is the right choice."

### Key Points:
- Be honest and calm - limitations are normal
- "Trade-offs, not bugs" - frame it correctly
- End with positive framing (right choice for exploration)

---

## SLIDE 13: Roadmap and Summary

**Time: 45 seconds**

### Script:

> "[Point to timeline]
> Roadmap: LPBF single track is done. Next is multi-track simulation. Then other AM processes like DED. Finally, multi-GPU support.
>
> [Point to summary]
> Key achievements:
> - Modular GPU-LBM platform built
> - 11 physics modules, 10 validated
> - 37 to 150 times faster than CPU
> - LPBF simulation working
>
> [Final message]
> This is a fast, modular platform for AM process exploration.
>
> And it proves: AI-assisted development works for complex CFD projects.
>
> Thank you. Any questions?"

### Key Points:
- Quick summary - hit the highlights
- End with two key messages: platform value + AI workflow validation
- "Questions?" - pause and make eye contact

---

# Q&A Preparation

## Q1: "Why LBM instead of FVM?"

> "Two reasons. First, LBM has no pressure solver. In FVM, you must solve Poisson equation every time step. This is hard to parallelize. LBM does not need this. Second, LBM is explicit. Every cell computes independently. Perfect for GPU.
>
> Trade-off: interface is more diffuse. But for fast exploration, this is acceptable."

---

## Q2: "How do you handle gas phase?"

> "We use void boundary. When liquid fraction is below 0.1, we set velocity to zero. This is simple but effective for metals. Why? Density ratio is 3400 to 1. Gas momentum is negligible."

---

## Q3: "What about experimental validation?"

> "We validated against literature CFD results. Melt pool depth matches within 2 percent. Marangoni velocity is in reported range. Experimental validation is next step -- we are planning high-speed camera measurements."

---

## Q4: "The code is written by AI. What did YOU do?"

> "Good question. AI writes code, but AI cannot make decisions.
>
> I decided the architecture: LBM plus GPU, modular design.
>
> I judged physics correctness: is this result reasonable? What is wrong?
>
> I directed debugging: where to look, what to fix.
>
> I controlled quality: reject bad code, ensure physics is right.
>
> AI is a tool. A powerful tool. But human expertise is still essential for making the right decisions."

---

## Q5: "Performance compared to FLOW-3D or commercial tools?"

> "We estimate 37 to 150 times faster than CPU FEM. But this is not a direct comparison -- physics modules are different. Our advantage: open source, customizable, GPU-native from the start."

---

# Time Summary

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title | 15 sec |
| 2 | AI Workflow | 1 min 30 sec |
| 3 | My Role | 1 min |
| 4 | Why LBM+GPU | 1 min |
| 5 | Architecture | 1 min |
| 6 | Physics Modules | 1 min |
| 7 | Case Study | 1 min 30 sec |
| 8 | Energy Validation | 1 min |
| 9 | Depth Validation | 1 min |
| 10 | Marangoni Validation | 45 sec |
| 11 | Validation Summary | 30 sec |
| 12 | Limitations | 1 min |
| 13 | Roadmap & Summary | 45 sec |
| **Total** | | **~12 min** |

**Note**: Script is slightly over 10 minutes. When practicing, you can speed up Slides 8-11 (validation section) or combine Slides 10-11 into one.

---

# Tips for Delivery

1. **Practice the numbers**: 44 vs 45 microns, -2.2%, 1.2 m/s, 0.5-2.0 m/s. Say them clearly.

2. **Point and pause**: When showing figures, point first, then speak. Give audience time to look.

3. **Slow down at key points**:
   - "Never let AI calculate directly"
   - "AI writes code, I make decisions"
   - "-2.2 percent error"

4. **Be confident about limitations**: Say them calmly. This shows maturity and honesty.

5. **Eye contact**: Look at your advisor when saying key conclusions.

6. **Breathe**: Pause between slides. Take a breath. This is natural.
