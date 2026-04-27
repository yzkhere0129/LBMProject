# Numerical-Accuracy Audit of Collision Kernels

**Branch**: `debug/patrol-2026-04-27`
**File audited**: `src/physics/fluid/fluid_lbm.cu` (3111 lines)
**Auxiliary files**: `src/physics/fluid/smagorinsky_les.cuh`, `src/core/lattice/d3q19.cu`,
`src/physics/force_accumulator.cu`
**Commit**: `7fbdfd7` (HEAD of `debug/patrol-2026-04-27`)
**Audit date**: 2026-04-27
**Auditor profile**: CFD numerics specialist; no execution / runtime check, pure
line-by-line vs textbook derivation.

## Mission scope

Find subtle correctness defects in the collision kernels that would not be
caught by existing tests. The user explicitly asked for adversarial review
under the assumption "the codebase is full of latent bugs."

7 audit areas (per INBOX request):

1. EDM forcing implementation (`fluidBGKCollisionEDMKernel` family)
2. Semi-implicit Darcy denominator
3. TRT (Λ=3/16) decomposition
4. Regularized (Latt-Chopard 2006) projection
5. Smagorinsky LES branch
6. Equilibrium f_eq formula
7. CFL force limiter math (`applyCFLLimitingKernel`)

---

## Methodology

For each kernel:
1. Restate the textbook formula (with citation).
2. Verify lattice constants (D3Q19 ex/ey/ez/w/opposite) by orthogonality and
   isotropy. (Done with python sanity check — opposite table, weights sum,
   second-moment isotropy `Σ w c_α c_β = cs² δ_αβ` all pass.)
3. Trace what the code computes line by line.
4. Compare. Flag any drift.

I take "potential bug" = "code differs from textbook in a way that could
distort physics under realistic operating conditions"; "confirmed bug" =
"code is provably wrong by derivation or by self-inconsistency between two
related routines"; "style" = "math is right but easy to misread."

---

## Area 1 — EDM forcing implementation

### Textbook (Kupershtokh, Comput. Math. Appl. 58:862-872, 2009)

For body force `F` (lattice units, dt=1):

```
Δu  =  F · dt / ρ                                       (Eq. 9)
f_i^new  =  f_i  -  ω (f_i - f_i^eq(ρ, u))
                 +  [f_i^eq(ρ, u + Δu) - f_i^eq(ρ, u)]   (Eq. 11)
```

The "EDM shift" `f^eq(ρ,u+Δu) − f^eq(ρ,u)` lives entirely in the equilibrium
subspace, so no anisotropic distribution residue accumulates over time —
this is EDM's only theoretical advantage over Guo. The CE expansion gives
`∂_t(ρu) + ∇·(...) = F` to second order, identical to Guo.

`u` here is the conserved velocity `u = m/ρ`, where `m = Σ c_q f_q` is the
pre-collision momentum.

### Code variants under audit

* `fluidBGKCollisionEDMKernel` (lines 1388-1511)
* `fluidTRTCollisionEDMKernel` (lines 1536-1667)
* `fluidRegularizedCollisionEDMKernel` (lines 1684-1835)
* `fluidRegularizedCollisionGuoKernel` (lines 1855-2000) — Guo variant, not EDM

All EDM kernels share this skeleton:

```
m_rho   = Σ f_q
m_α     = Σ c_qα f_q
denom   = max(m_rho + 0.5·K, ρ_min)
u_bare  = m / denom            // Σ c f / (ρ + K/2)
Δu      = F / denom            // F / (ρ + K/2)        ← Sprint-1 fix
u_phys  = u_bare + 0.5·Δu
u_shifted = u_bare + Δu
... clamp u_bare and u_shifted to U_MAX = 0.25
ux/uy/uz[id] = u_phys
... (collision)
feq_bare    = feq(ρ, u_bare)
feq_shifted = feq(ρ, u_shifted)
f_dst = f - ω·(f - feq_bare) + (feq_shifted - feq_bare)
```

### Findings

#### Finding 1.1 — `Δu = F/(ρ + 0.5K)` is a Sprint-1 hybrid, not textbook EDM (POTENTIAL BUG)

The textbook EDM formula is `Δu = F/ρ`. The code uses `Δu = F/(ρ+0.5K)`,
documented in lines 1443-1448 as a Sprint-1 fix to prevent "Marangoni leak"
through the mushy zone. The change is intentional but is a *hybrid* of
EDM and Crank-Nicolson Darcy treatment.

**Limit checks**:
* K → 0 (pure liquid): `Δu = F/ρ`, reduces to standard EDM. ✓
* K → ∞ (full solid): `Δu → 0`, force is screened out. ✓ (physically
  reasonable but is NOT what standard EDM does.)

**Concern**: in the standard EDM, the post-collision momentum is
`m_post = m + ρ·Δu = m + F`. This recovers the correct macroscopic
momentum balance `dm/dt = F`. With the hybrid `Δu = F/(ρ+0.5K)`,
`m_post = m + ρ·F/(ρ+0.5K)`. So the actual force injected into the
momentum equation is `F·ρ/(ρ+0.5K)`, NOT `F`. In the K→∞ limit, no
force is injected even if F≠0 — this is in fact the desired behaviour
for solid cells, but it overlaps with what semi-implicit Darcy is
supposed to do *in u_post*. The two effects compound:

```
m_post   = m + ρ·F/(ρ+0.5K)
u_post   = m_post / (ρ+0.5K)        // next step macroscopic
        = u_bare·ρ/(ρ+0.5K)  +  ρ·F / (ρ+0.5K)²
```

Compare to a clean Crank-Nicolson semi-implicit Darcy:
```
(ρ+0.5K)·u_new = (ρ-0.5K)·u + F
u_new = u·(ρ-0.5K)/(ρ+0.5K) + F/(ρ+0.5K)
```

These are different. In the mushy zone, the EDM-hybrid path **damps
momentum more strongly** than clean Crank-Nicolson Darcy. Not a sign
error, not unstable, but quantitatively distinct from any single
textbook scheme. This may explain (or contribute to) the "raised track
fails" observation in the Sprint-1 overnight memo (groove instead of
ridge).

**Severity**: POTENTIAL BUG — math is internally consistent but does
not correspond to a published, CE-verified scheme. The `K → ∞` limit
behaviour is overcompensated.

**Location**: lines 1449-1451 (BGK), 1592-1594 (TRT), 1735-1737 (Reg),
1908-1910 (Reg-Guo).

#### Finding 1.2 — collision kernel and post-streaming macroscopic kernel disagree on `u_phys` formula (CONFIRMED BUG)

Inside `fluidBGKCollisionEDMKernel` (line 1455):
```c
u_phys = u_bare + 0.5·Δu
       = m/(ρ+0.5K) + 0.5·F/(ρ+0.5K)
       = (m + 0.5·F)/(ρ+0.5K)        // Darcy-aware in BOTH numerator and force
```

Inside `computeMacroscopicSemiImplicitDarcyEDMKernel` (lines 2049-2062),
which the multiphysics solver calls *after* streaming on every step:
```c
u_bare = m/(ρ+0.5K)            // Darcy-aware
u_phys = u_bare + 0.5·F·(1/m_rho)
       = m/(ρ+0.5K) + 0.5·F/ρ        // Darcy-aware in m only, NOT in F
```

These two formulas give different physical-velocity outputs for the same
state `(m, ρ, K, F)`:
```
Δu_phys = u_phys_collision - u_phys_macro
       = 0.5·F · [1/(ρ+0.5K) - 1/ρ]
       = -0.5·F · 0.5K / (ρ·(ρ+0.5K))
```

In the mushy zone with `ρ_LU ≈ 1, K_LU ~ 10, F_LU ~ 1e-3`:
```
Δu_phys ≈ -0.5 · 1e-3 · 5 / (1 · 6) ≈ -4·10⁻⁴ LU
```
which is comparable to the velocities being measured (~1e-3 LU). So this
is a **40% relative discrepancy** in the mushy zone.

**Effect**: between collision and post-stream macroscopic, `ux/uy/uz`
gets overwritten with a different formula. The next step then uses the
post-stream value to build forces (Marangoni, Darcy, recoil). The
*collision-time* velocity write becomes effectively dead, but the
collision-time `u_phys` was the one consistent with what the EDM shift
encodes. So downstream physics sees the "wrong" `u_phys`.

This is the kind of bug that:
* makes `v_max` diagnostics *just slightly off* (no NaN, no obvious
  failure),
* compounds over thousands of steps in the mushy zone,
* would never be caught by a Stefan-1D or natural-convection benchmark
  because both are pure-liquid (K=0 ⇒ formulas agree).

**Severity**: CONFIRMED BUG — the two formulas disagree by an algebraic
quantity, not a numerical error. Either both should use `(ρ+0.5K)` for
the F denominator (post-Sprint-1 convention), or both should use raw `ρ`
(pre-Sprint-1 convention). Cannot be one-of-each.

**Suggested fix**: line 2060-2062 of the macroscopic kernel should read

```c
u_x = u_bare_x + 0.5f * force_x[id] * inv_denom;   // matches collision
u_y = u_bare_y + 0.5f * force_y[id] * inv_denom;
u_z = u_bare_z + 0.5f * force_z[id] * inv_denom;
```

#### Finding 1.3 — EDM kernel reads `f_local` from global memory twice (STYLE)

In `fluidBGKCollisionEDMKernel` lines 1497 and 1505:
```c
for (int q = 0; q < 19; q++) f_local[q] = f_src[id + q * n_cells];
...
for (int q = 0; q < D3Q19::Q; ++q) {
    float f = f_local[q];                  // OK
    ...
}
```
But the moment computation at lines 1423-1429 already reads `f_src` once
into local accumulators. So `f_src` is read 19 + 19 = 38 times. A clean
implementation would store all 19 into `f_local[]` once at the top.
Performance only, not correctness.

**Severity**: STYLE.

---

## Area 2 — Semi-implicit Darcy denominator

### Textbook

Crank-Nicolson treatment of Darcy drag `F_darcy = -K·u`:
```
ρ · (u_new - u) / dt = F_other - K · (u + u_new)/2
(ρ + 0.5·K·dt) · u_new = (ρ - 0.5·K·dt) · u + F_other · dt
```

In LU (dt=1): denominator is `ρ + 0.5·K_LU`. Numerator's `(ρ-0.5K)·u + F`
is the algebraically correct form. Note the **two** Darcy contributions:
one in the denominator, one in the numerator (multiplying old u).

### Code path

`computeDarcyCoefficientKernel` (force_accumulator.cu line 649-680):
```c
fl   = clamp(liquid_fraction, 0, 1)
fs   = 1 - fl
K_factor = darcy_coeff · fs² / (fl³ + 1e-3)            [units: 1/s, physical]
darcy_K[idx] = K_factor · dt                           [unitless, LU]
```

Carman-Kozeny formula with ε = 1e-3 (mush regularization). `darcy_coeff`
is configured as a physical [1/s] scaling. Multiplied by `dt` to get LU.

**Comment at line 678** says "rho_phys is NOT included — K_LU adds to ρ_LU
(≈1), not ρ_phys". This is correct: the LBM denominator is `ρ_LU + 0.5·K_LU`
where ρ_LU = m/ρ_ref ≈ 1, not the SI density.

### Findings

#### Finding 2.1 — denominator factor 0.5 is correct Crank-Nicolson (NOT A BUG)

The 0.5 in `denom = m_rho + 0.5·K` is the canonical Crank-Nicolson
half-step. Sometimes implementations use `denom = m_rho + K` (full
implicit) which is more dissipative but more stable; some use
`denom = m_rho + 0.25·K` for a different time-discretization. The 0.5
value is mathematically standard.

**Severity**: NOT A BUG.

#### Finding 2.2 — numerator-side Darcy term is missing (POTENTIAL BUG)

The clean Crank-Nicolson scheme has:
```
u_new = [(ρ - 0.5·K)·u + F·dt] / (ρ + 0.5·K)
```

The code computes `u_bare = m / (ρ + 0.5·K)` which corresponds to:
```
u_new = m / (ρ + 0.5·K)                  (treating m as carrying old momentum ρu)
```

This is NOT the same as Crank-Nicolson. Specifically, the `m` here is the
**pre-collision** momentum, which after streaming includes the previous
step's relaxation. So algebraically, `m ≈ ρ·u + correction_from_relaxation`.
The kernel implicitly trusts that `m/(ρ+0.5K)` already encodes
the (ρ−0.5K)·u part, which is **not** generally true.

In the K→∞ limit:
* Clean CN: `u_new = (ρ−0.5K)·u/(ρ+0.5K) + F/(ρ+0.5K) → -u + F·0` (sign-flip
  of u, force vanishes). Sign-flip is the well-known instability of
  CN at very high K, why people sometimes go fully implicit.
* Code: `u_bare = m/(ρ+0.5K) → 0` (since m bounded but denominator → ∞).
  No sign flip.

So the code's behaviour is **better** than naive CN at K→∞. But it is
**not** the textbook scheme either — it's closer to fully-implicit Darcy
on the LBM moment.

CE analysis (sketch): consider K constant in space, no other forces. Then
`m_post = (1-ω)m + ω·ρ·u_eq + 0` = `(1-ω)m + ω·ρ·u_bare`
       = `(1-ω)m + ω·m/(1+0.5K)` (with ρ=1)
       = `m·[(1-ω) + ω/(1+0.5K)]`
       = `m·[1 - ω·0.5K/(1+0.5K)]`

So `m_new/m = 1 - ω·0.5K/(1+0.5K)`. Per step, the moment decays by factor
`1 - ω·0.5K/(1+0.5K)`. Compare to clean CN where momentum decay per step
is `(ρ-0.5K)/(ρ+0.5K) = (1-0.5K)/(1+0.5K)` (with ρ=1, dt=1).
At ω = 1 (BGK τ=1), the LBM-Darcy decay rate is `0.5K/(1+0.5K)`, vs
CN decay rate `2·0.5K/(1+0.5K) = K/(1+0.5K)`. The LBM-Darcy is ω times
the CN rate. This matches the standard CE result that body forces in
LBM with Guo enter as `(1-ω/2)`-modulated source terms, consistent with
ω-dependent damping. No bug in this aspect — just a subtlety to remember
when interpreting `K` calibration.

**Severity**: NOT A BUG, but easy to misinterpret. Style note: comment
out the implicit assumption that m carries the (ρ−0.5K)u information.

#### Finding 2.3 — `RHO_MIN = 1e-6f` floor in denom is asymmetric (POTENTIAL BUG)

Line 1436:
```c
float denom = fmaxf(m_rho + 0.5f * K, RHO_MIN);
```

`RHO_MIN = 1e-6f`. This protects against `m_rho = 0` (gas/empty cell) but
allows `K` to be unbounded, so denom can be enormous. That's fine for the
mushy zone. But: the floor only fires if `m_rho + 0.5·K < 1e-6` which
**only occurs when both m_rho ≈ 0 AND K ≈ 0**, i.e. the empty cell case.
In the empty cell case, the velocity isn't physical anyway. Functionally
correct.

However, in `computeMacroscopicSemiImplicitDarcyEDMKernel` (line 2048),
the gating is different:
```c
if (m_rho > 1e-10f && !isnan(m_rho)) {
    ...
}
```

Here the threshold is `1e-10` and there's no symmetric check inside the
collision kernel. **Inconsistent thresholds across two kernels operating
on the same cells** — could cause divergent behaviour at exactly the gas
boundary (where one kernel writes a finite u_bare and the other writes
zero).

**Severity**: STYLE / LOW. Unify the threshold (suggest `1e-10f` everywhere
and replace `fmaxf(..., RHO_MIN)` floor with the explicit `if`).

---

## Area 3 — TRT (Λ=3/16) decomposition

### Textbook (Ginzburg & Adler 1994; Ginzburg et al. 2008)

For each direction `q` (with opposite `q̄ = opposite[q]`):
```
f_q^+      = (f_q + f_{q̄})/2          (symmetric)
f_q^-      = (f_q - f_{q̄})/2          (anti-symmetric)
f_q^{eq+}  = (f_q^eq + f_{q̄}^eq)/2
f_q^{eq-}  = (f_q^eq - f_{q̄}^eq)/2
f_q^{post} = f_q  -  ω⁺·(f_q^+ - f_q^{eq+})  -  ω⁻·(f_q^- - f_q^{eq-})
```

Equivalently using neq splits:
```
neq_q     = f_q   - f_q^eq
neq_q̄    = f_{q̄} - f_{q̄}^eq
f_s_neq  = (neq_q + neq_q̄)/2    = (f_q - f_q^{eq})^+
f_a_neq  = (neq_q - neq_q̄)/2    = (f_q - f_q^{eq})^-
f_q^{post} = f_q  -  ω⁺·f_s_neq  -  ω⁻·f_a_neq
```

For q=0 (rest, self-opposite): `f_a_neq = 0` automatically; reduces to BGK
with `ω⁺`. ✓

Magic parameter: `Λ = (τ⁺ - 0.5)·(τ⁻ - 0.5)`; choosing Λ = 3/16 cancels
the leading-order error in straight-walled bounce-back, making TRT
"as good as" no-slip on coordinate-aligned walls.

### Code (`fluidTRTCollisionEDMKernel` lines 1647-1666)

```c
for (int q = 0; q < D3Q19::Q; ++q) {
    float f_q = f_local[q];
    int q_bar = opposite[q];
    float f_qbar = f_local[q_bar];
    float feq_bare_q    = feq(q,     ρ, u_bare);
    float feq_bare_qbar = feq(q_bar, ρ, u_bare);
    float neq_q    = f_q    - feq_bare_q;
    float neq_qbar = f_qbar - feq_bare_qbar;
    float f_s_neq  = 0.5f * (neq_q + neq_qbar);
    float f_a_neq  = 0.5f * (neq_q - neq_qbar);
    float feq_shifted = feq(q, ρ, u_shifted);
    f_dst[id + q * n_cells] = f_q
        - omega_eff       * f_s_neq
        - omega_minus_eff * f_a_neq
        + (feq_shifted - feq_bare_q);
}
```

### Findings

#### Finding 3.1 — TRT decomposition is correctly implemented (NOT A BUG)

The neq-split formulation matches the textbook. The opposite table is
verified correct (Σ c_q + c_{opp(q)} = 0 for all q; rest is self-opposite).
For q=0: `q_bar = 0`, so `f_qbar = f_q`, `neq_qbar = neq_q`, `f_s_neq = neq_q`,
`f_a_neq = 0`. So `f_dst = f_q - ω⁺·neq_q - 0 + EDM_shift = BGK + EDM`.
Correct degeneration.

**Severity**: NOT A BUG.

#### Finding 3.2 — TRT magic parameter under LES is recomputed using base ω (POTENTIAL BUG)

Lines 1640-1644:
```c
float tau_eff = 1.0f / omega_eff;
float Lambda = (1.0f/omega - 0.5f) * (1.0f/omega_minus - 0.5f);
float tau_minus_eff = 0.5f + Lambda / (tau_eff - 0.5f);
float omega_minus_eff = 1.0f / tau_minus_eff;
```

`omega` is the **base** ω⁺ (from physical viscosity), `omega_eff` is the
LES-augmented ω⁺. Λ is computed from base ω⁺ × base ω⁻. Then ω⁻_eff is
chosen so that Λ remains constant under the LES augmentation.

Mathematical analysis: under heavy LES, `tau_eff = 5.0` (the safety clamp
in `computeSmagorinskyOmega`). With `Λ = 3/16`:
```
tau_minus_eff = 0.5 + 0.1875 / 4.5 = 0.542
omega_minus_eff = 1.846
```

That is *just* under 2.0, the linear stability boundary. Under base
viscosity (`tau_0 = 0.51`), `tau_minus_0 = 0.5 + 0.1875/0.01 = 19.25`,
`omega_minus_0 = 0.052` — heavily underrelaxed. So LES drives ω⁻ from
~0.05 to ~1.85, a 35× change.

**Concern**: as Cs Q_mag/ρ varies cell-to-cell, `omega_minus_eff` varies
too, while ω⁺_eff is also adapting. The TRT magic-parameter property
(boundary anchoring) was derived for *constant* Λ across the domain,
which the code does preserve. But for *strongly variable* viscosity due
to LES, the TRT advantages may be partly lost. Not a bug — it's a
deliberate design choice (preserve Λ rather than preserve ω⁻).

**Severity**: STYLE / DESIGN NOTE. No fix needed unless boundary errors
under LES are observed.

#### Finding 3.3 — `feq_shifted` uses post-clamp `u_shifted` (NOT A BUG)

Lines 1620-1632 clamp `u_shifted` to U_MAX = 0.25 if magnitude exceeds.
The clamping happens before `feq_shifted` is computed in the inner loop.
Without this, `feq_shifted` could violate Ma < 0.43 (the truncated-Hermite
validity bound). Correct.

#### Finding 3.4 — opposite[opposite[q]] = q (assumption holds, verified)

The neq-split scheme implicitly assumes `opposite(opposite(q)) = q`. The
D3Q19 opposite table satisfies this (verified in python: each opposite
maps to a self-inverse pair). No bug.

---

## Area 4 — Regularized (Latt-Chopard 2006) projection

### Textbook (Latt & Chopard, Math. Comput. Simul. 72:165, 2006)

```
1. Compute ρ, u from f
2. f_q^eq = w_q·ρ·(1 + 3·c·u + 4.5·(c·u)² - 1.5·u²)
3. Π^{neq}_{αβ} = Σ_q (f_q - f_q^eq) c_qα c_qβ
4. f_q^{neq,reg} = w_q/(2·cs⁴) · Π^{neq}_{αβ} · (c_qα c_qβ - cs²·δ_αβ)
5. f_q^{post} = f_q^eq + (1-ω)·f_q^{neq,reg}
```

Step 4 is the projection onto the 2nd-order Hermite subspace: it discards
all moments of order > 2 (ghost moments in D3Q19) before relaxation.
Constants:  `cs² = 1/3, cs⁴ = 1/9, 1/(2·cs⁴) = 9/2 = 4.5`.

### Code (`fluidRegularizedCollisionEDMKernel`, lines 1762-1834)

```c
const float cs2 = 1.0f / 3.0f;
const float coeff = 4.5f;            // 9/2 = 1/(2 cs⁴)
...
for (int q = 0; q < 19; q++) {
    float Qxx = cx*cx - cs2;         // Hermite basis Q_q
    float Qyy = cy*cy - cs2;
    float Qzz = cz*cz - cs2;
    float Qxy = cx*cy;
    float Qxz = cx*cz;
    float Qyz = cy*cz;
    float PiQ = Pi_xx*Qxx + Pi_yy*Qyy + Pi_zz*Qzz
              + 2.0f*(Pi_xy*Qxy + Pi_xz*Qxz + Pi_yz*Qyz);
    float f_neq_reg = w[q] * coeff * PiQ;
    float feq_bare    = feq(q, ρ, u_bare);
    float feq_shifted = feq(q, ρ, u_shifted);
    f_dst[id + q*n_cells] = feq_bare + (1-ω_eff)·f_neq_reg + (feq_shifted - feq_bare);
}
```

### Verification

Python round-trip test (see audit notebook): set `Π^{neq} = e_x⊗e_x` and
verify that `Σ f_neq_reg c_qα c_qβ` recovers the input.

```
Round-trip Pi=e_x⊗e_x:
  [[ 1.000e+00  0.000e+00  0.000e+00]
   [ 0.000e+00  6.939e-17  0.000e+00]
   [ 0.000e+00  0.000e+00  6.939e-17]]
Round-trip Pi_xy=1 (Pi_yx=1):
  [[0.  1.  0.]
   [1.  0.  0.]
   [0.  0.  0.]]
```

Both round-trips are exact within FP roundoff. The Hermite projection is
correctly implemented.

### Findings

#### Finding 4.1 — projection formula and constants are correct (NOT A BUG)

`coeff = 4.5f` matches `9/2 = 1/(2·cs⁴)`. All 6 independent components of
the symmetric stress tensor accumulated. Off-diagonal contribution
correctly multiplied by 2 (since Π_xy = Π_yx in `Σ Π_αβ Q_αβ`).

**Severity**: NOT A BUG.

#### Finding 4.2 — `Π^{neq}` measured against `feq_bare`, regularized collision adds EDM shift (CONSISTENT)

The Π^{neq} measurement uses `feq(u_bare)` (line 1766). The reconstruction
adds the EDM shift `feq_shifted - feq_bare` separately. Since the EDM
shift lives entirely in the equilibrium subspace (verified above:
Σ c f_neq_reg = 0, so adding the shift doesn't perturb f_neq_reg), this
is consistent.

**Severity**: NOT A BUG.

#### Finding 4.3 — Bulk viscosity = shear viscosity (NOT A BUG, design property)

The projection does NOT subtract the trace `(Π^{neq}_xx + Π^{neq}_yy +
Π^{neq}_zz)/3 · δ_αβ` before relaxation. So both deviatoric (shear) and
volumetric (bulk) parts of Π are relaxed at ω⁺ — bulk viscosity equals
shear. This is a known property of single-relaxation-rate regularized BGK
and is fine for incompressible flow. For low-Mach AM problems (Ma ~
0.05-0.4), the volumetric part is small.

For a TRT-style or MRT regularized variant, one would relax bulk and
shear separately. Out of scope here.

**Severity**: NOT A BUG (design choice).

#### Finding 4.4 — Guo regularized variant uses `feq(u_eq)` for Π measurement (CONSISTENT WITH GUO CONVENTION)

In `fluidRegularizedCollisionGuoKernel` line 1941, `feq` for Π is computed
using `u_eq = u_bare + 0.5·F/(ρ+0.5K)` (the Guo half-shifted velocity), NOT
`u_bare`. This matches Guo's convention (Π should be measured against the
equilibrium of the observable velocity). And the post-collision formula
uses `feq(u_eq)` plus the explicit Guo source term S_q.

```
S_q = (1 - ω/2) · w_q · [ (c_q - u_eq)/cs²  +  (c_q · u_eq) · c_q / cs⁴ ] · F
```

Code at lines 1986-1992 expands this exactly. Constants
`inv_cs2 = 3, inv_cs4 = 9` match `1/cs² = 3, 1/cs⁴ = 9`. The
`guo_prefactor = 1 - 0.5·omega_eff` correctly cancels the ω-dependent
artifact at τ → 0.5.

**Severity**: NOT A BUG.

---

## Area 5 — Smagorinsky LES branch

### Textbook (Hou et al., J. Comput. Phys. 118:329, 1996)

```
S_ij     = -1/(2·ρ·cs²·τ) · Π^{neq}_ij           (CE relation)
|S|      = √(2 S_ij S_ij)
ν_sgs    = (Cs · Δ)² · |S|                       (Smagorinsky)
ν_eff    = ν_0 + ν_sgs
τ_eff    = ν_eff/cs² + 0.5
```

This is implicit because |S| depends on τ (through Π/τ in the CE relation).
The exact algebraic solution (Hou 1996 Appendix):
```
Q_mag    = √(2 Π^{neq}_ij Π^{neq}_ij)
τ_eff    = 0.5·(τ_0 + √(τ_0² + 18·Cs²·Q_mag/ρ))
```

### Code (`smagorinsky_les.cuh` lines 49-128)

Computes Π^{neq} from `f_local - feq(u)` (lines 81-92, equilibrium uses
the local ρ, u passed in — typically `u_bare`). Then:
```c
float Q2 = Qxx² + Qyy² + Qzz² + 2·(Qxy² + Qxz² + Qyz²);
float Q_mag = sqrtf(2.0f * Q2);
float tau_0 = 1.0f / omega_0;
float tau_eff = 0.5f * (tau_0 + sqrtf(tau_0² + 18.0f*Cs²·Q_mag/max(ρ,1e-6)));
tau_eff = max(0.505, min(tau_eff, 5.0));
return 1.0f / tau_eff;
```

### Verification (independent derivation)

```
nu_sgs = Cs² · |S|  (delta = 1 LU)
       = Cs²·|Π|/(2 ρ cs² τ)
nu_eff = cs²(τ-0.5) + Cs²·|Π|/(2 ρ cs² τ)
τ_eff = nu_eff/cs² + 0.5 = τ + Cs²·|Π|/(2 ρ cs⁴ τ)
τ_eff·τ = τ² + Cs²·|Π|/(2 ρ cs⁴)
```
Self-consistency `τ = τ_eff`:
```
τ_eff² - τ_0·τ_eff = Cs²·|Π|/(2 ρ cs⁴)
τ_eff = 0.5·(τ_0 + √(τ_0² + 4·Cs²·|Π|/(2 ρ cs⁴)))
      = 0.5·(τ_0 + √(τ_0² + 2·Cs²·|Π|/(ρ cs⁴)))
2/cs⁴ = 18 (cs⁴ = 1/9), so:
τ_eff = 0.5·(τ_0 + √(τ_0² + 18·Cs²·|Π|/ρ))   ✓
```

The code's `18.0f * Cs * Cs * Q_mag / ρ` matches exactly.

### Findings

#### Finding 5.1 — algebraic Smagorinsky formula correct (NOT A BUG)

The exact algebraic solution is implemented. The `0.505 ≤ τ_eff ≤ 5.0`
clamp prevents divergence at exactly τ_eff → 0.5. ν_t = ν_eff − ν_0 ≥ 0
always since `√(τ_0² + nonneg) ≥ τ_0` ⇒ τ_eff ≥ τ_0.

**Severity**: NOT A BUG.

#### Finding 5.2 — Smagorinsky uses `u_bare`, not `u_phys` or `u_shifted` (CONSISTENT WITH EDM CONVENTION)

`computeSmagorinskyOmega` is called with `u_bare` (BGK kernel line 1501,
TRT kernel line 1638, Reg kernel line 1781). This is the velocity that
`feq` in Π^{neq} should reference. Consistent.

The Reg-Guo kernel calls it with `u_eq` (line 1955), matching its
internal Guo convention. Also consistent.

**Severity**: NOT A BUG.

#### Finding 5.3 — Cs constant interpretation, `ν_sgs = Cs² · |S|` vs `(Cs Δ)² · |S|` (DOC CLARITY)

Docstring (line 14-17) says `ν_sgs = (Cs·Δ)²·|S|`, which is the standard
Smagorinsky form. With Δ = 1 (LU), the formula reduces to `Cs²·|S|`,
which the code uses. The auxiliary constant in the algebraic solution is
`18·Cs²` not `18·(Cs·Δ)²`. Since Δ = 1 in LU this is fine, but if anyone
ever calls this with `Δ ≠ 1` (e.g. for AMR coarsening), they'd need to
rewrite as `18·(Cs·Δ)²`.

**Severity**: STYLE / future-proofing.

---

## Area 6 — Equilibrium f_eq formula

### Textbook (truncated Hermite expansion, 2nd order in u)

```
f_q^eq = w_q · ρ · (1 + (c_q·u)/cs² + (c_q·u)²/(2 cs⁴) - u·u/(2 cs²))
       = w_q · ρ · (1 + 3·(c·u) + 4.5·(c·u)² - 1.5·u²)        for cs² = 1/3
```

### Code (`d3q19.cu` line 132-147)

```c
float eu = ex[q]*ux + ey[q]*uy + ez[q]*uz;
float u2 = ux*ux + uy*uy + uz*uz;
return w[q] * rho * (1.0f + 3.0f*eu + 4.5f*eu*eu - 1.5f*u2);
```

### Findings

#### Finding 6.1 — equilibrium formula correct, no FP-conversion bug (NOT A BUG)

All numerical coefficients (`1.0f, 3.0f, 4.5f, -1.5f`) are exact in IEEE
FP32. Lattice constants `ex/ey/ez` are int (exactly representable for
{-1, 0, 1}). `w[q]` weights are `1/3, 1/18, 1/36` which are NOT exact
(1/3 = 0.333333... rounds in FP32) but the truncation error is ~3e-8 per
weight, well below CE truncation order O(u³).

A double-precision variant `computeEquilibriumDouble` exists for
high-accuracy paths (line 150-158). Used in TRT walberla-aligned kernels.

**Severity**: NOT A BUG.

#### Finding 6.2 — D3Q19 lattice tables verified (NOT A BUG)

Sanity-check by python (see Methodology):
* Σ w_q = 1.0 (exactly)
* Σ w_q c_qα = 0 for all α (1st moment)
* Σ w_q c_qα c_qβ = (1/3) δ_αβ (2nd moment, isotropic with cs² = 1/3)
* opposite[opposite[q]] = q for all q
* c_q + c_{opposite[q]} = 0 for all q

All checks pass.

**Severity**: NOT A BUG.

#### Finding 6.3 — `computeVelocity` divide-by-zero guard (LOW)

Line 196-203 of `d3q19.cu`: `RHO_EPSILON = 1e-8f`. If ρ < 1e-8, velocity
is set to 0. This is a sane guard for free-surface gas cells. However,
this `RHO_EPSILON` is decoupled from the `RHO_MIN = 1e-6f` used in
collision kernel and `1e-10f` used in macroscopic kernel. **Three
different ρ thresholds across files**.

**Severity**: STYLE — unify thresholds.

---

## Area 7 — CFL force limiter

### Code (`force_accumulator.cu` lines 701-761)

For each cell:
```c
v_current = |u|
v_new     = |u + F|             (predicted velocity if F applied)
v_ramp    = ramp_factor · v_target

if (v_new > v_target) {
    excess = v_current - v_target · ramp_factor
    if (excess > 0)
        scale = exp(-2·excess / v_target)
    else
        scale = max(0, v_target - v_current) / f_mag         // (clamped to ≤ 1)
} else if (v_new > v_ramp) {
    excess_ratio = (v_new - v_ramp) / (v_target - v_ramp)
    scale_at_target = min(1, (v_target - v_current) / f_mag)
    scale = max(0.01, 1 - excess_ratio · (1 - scale_at_target))
}                                  // else scale = 1 unchanged
F *= scale
```

### Claim from comment (line 735)

> "Smooth exponential damping instead of hard cutoff
> Avoids discontinuous force jump at v_target boundary"

### Verification

Continuity at `v_new = v_target` (boundary between gradual branch and
exp damping branch):

* **Approaching from below** (`v_new = v_target⁻`):
  `excess_ratio → 1`, `scale → scale_at_target = min(1, (v_target - v_current)/f_mag)`.

* **Approaching from above** (`v_new = v_target⁺`):
  `excess = v_current - v_ramp`.
  - If `v_current ≤ v_ramp`: `excess ≤ 0`, falls to else branch:
    `scale = max(0, v_target - v_current)/f_mag`. Same as below — CONTINUOUS in this case.
  - If `v_current > v_ramp`: `excess > 0`, `scale = exp(-2·(v_current-v_ramp)/v_target)`.

Compare the two at `v_current > v_ramp`:
* From below: `scale = min(1, (v_target - v_current)/f_mag)`. Note `f_mag` enters.
* From above: `scale = exp(-2·(v_current-v_ramp)/v_target)`. NO `f_mag` dependence.

For a fixed `v_current > v_ramp`, the value depends on `f_mag`. But
`v_new = v_target` requires a specific relationship between `f_mag`,
`v_current`, and the geometry of the velocity vectors. Generically,
the two formulas give different scale at the boundary.

**Concrete example**: take `v_target = 1, ramp_factor = 0.8, v_ramp = 0.8,
v_current = 0.9, f_mag = 0.1`, force aligned with velocity, so `v_new = 1.0`.
* Gradual branch: `excess_ratio = (1 - 0.8)/(1 - 0.8) = 1.0`,
  `scale_at_target = min(1, (1 - 0.9)/0.1) = 1.0`, `scale = 1 - 1·(1-1) = 1.0`.
* Exp damping branch: `excess = 0.9 - 0.8 = 0.1`, `scale = exp(-0.2) = 0.819`.

So at the v_new = v_target boundary, scale jumps from 1.0 to 0.819. **This
contradicts the comment claim.** A 18% force jump at the boundary.

### Findings

#### Finding 7.1 — CFL limiter is discontinuous at v_new = v_target (POTENTIAL BUG)

The two branches of the CFL limiter produce different scale values at the
boundary `v_new = v_target` when `v_current > v_ramp`. The discontinuity
can be ~20% scale jump. This contradicts the docstring/comment claim of
"smooth exponential damping ... avoids discontinuous force jump".

In practice, the discontinuity appears only at the precise instant
`v_new = v_target`, which is a measure-zero event in continuous time.
But over GPU SIMD execution and FP32 noise, neighboring cells can
straddle the boundary and produce force-magnitude rifts (~20%) at the
boundary, which can manifest as **noisy v-fields with periodic spatial
ridging** in the v ~ v_target regime.

The bound `scale ∈ [0.01, 1]` does hold in both branches (so it's not
unbounded), but the function is NOT continuous in `(v_current, F)` jointly.

**Severity**: POTENTIAL BUG. Severity depends on operational regime:
if forces are usually well below v_target (no clipping), this never
fires. If forces are pushed near the target (recoil hot spots, melt
pool boil), it does fire and creates spatial banding.

**Suggested fix**: smoothly blend the two formulas. One option:
```c
// Use exp-damping uniformly when v_current > v_ramp
if (v_current > v_ramp) {
    float excess = v_current - v_ramp;
    scale = expf(-2.0f * excess / (v_target + 1e-12f));
} else if (v_new > v_ramp) {
    // gradual scaling for cells under v_ramp but pushing past it
    float excess_ratio = (v_new - v_ramp) / (v_target - v_ramp + 1e-12f);
    float scale_at_target = (v_target - v_current) / (f_mag + 1e-12f);
    scale_at_target = fminf(scale_at_target, 1.0f);
    scale = 1.0f - excess_ratio * (1.0f - scale_at_target);
    scale = fmaxf(scale, 0.01f);
}
```
This makes the regime decision based on `v_current` (cell's current state)
rather than the predicted `v_new`, eliminating the boundary.

#### Finding 7.2 — `delta_v_allowed/f_mag` can saturate clamp at exactly 1.0 (LOW)

Line 743: `scale = delta_v_allowed / (f_mag + 1e-12f); scale = fminf(scale, 1.0f);`.
For `delta_v_allowed = 0` (exactly at v_target), scale = 0 — force entirely
cancelled. The 0.01 floor is NOT applied in this branch (only in the
gradual branch). Inconsistency with line 754.

**Severity**: LOW.

---

## Summary table

| Area | Confirmed | Potential | Style/clarity |
|---|:---:|:---:|:---:|
| 1. EDM forcing | **1.2** (collision/macro u_phys mismatch) | 1.1 (Δu = F/(ρ+0.5K) hybrid) | 1.3 |
| 2. Semi-implicit Darcy | — | 2.2 (CN numerator missing, but better than naive) | 2.3 |
| 3. TRT decomposition | — | 3.2 (Λ-preserving LES boundary) | — |
| 4. Regularized projection | — | — | — |
| 5. Smagorinsky LES | — | — | 5.3 |
| 6. Equilibrium f_eq | — | — | 6.3 |
| 7. CFL force limiter | **7.1** (discontinuity at v_target boundary) | — | 7.2 |

**Total**: 2 confirmed, 3 potential, 5 style.

Confirmed bugs:
* **1.2**: Two kernels disagree on `u_phys` formula in mushy zone. ~40%
  relative discrepancy at typical K_LU. Corrupts post-streaming velocity
  feed to next-step force computation. SHOULD FIX.
* **7.1**: CFL limiter has ~20% scale discontinuity at v_new = v_target.
  Contradicts its own docstring. SHOULD FIX (cosmetic + numerical noise).

Potential bugs (intentional design that may or may not match paper):
* **1.1**: Sprint-1 hybrid `Δu = F/(ρ+0.5K)` deviates from textbook
  Kupershtokh EDM. Internally consistent, limits OK, but compounds with
  semi-implicit Darcy in a non-standard way. Document or revise to a
  published scheme.
* **2.2**: The `m/(ρ+0.5K)` form does NOT match clean Crank-Nicolson but
  matches an LBM-friendly scheme that decays momentum at ω·CN-rate. Not
  a bug, but easy to misinterpret.
* **3.2**: TRT magic parameter Λ preserved under LES, ω⁻ allowed to
  drift to ~1.85. No instability but anti-symmetric channel becomes
  ~over-relaxed under heavy LES. Watch.

---

## Recommendation queue (priority order)

### P0 — fix immediately (confirmed bugs)

1. **Finding 1.2** — Unify `u_phys` formula between
   `fluidBGKCollisionEDMKernel`/`fluidTRTCollisionEDMKernel`/
   `fluidRegularizedCollisionEDMKernel` and
   `computeMacroscopicSemiImplicitDarcyEDMKernel`. Either:
   (a) macroscopic kernel uses `(ρ+0.5K)` for the F denominator (matches
       collision, post-Sprint-1 convention), or
   (b) collision kernels use bare `ρ` for the F denominator (pre-Sprint-1).
   Decision must be CE-justified. (a) is internally consistent with
   "EDM hybrid" interpretation; (b) reverts Sprint-1's intent.

   Patch site (option a): `src/physics/fluid/fluid_lbm.cu:2059-2062`,
   replace `inv_rho` with `inv_denom` for the force half-shift.

   **Test**: write a unit test that runs one step in mushy zone (K = 1, F = 1e-3)
   and asserts that `ux/uy/uz` from collision kernel matches `ux/uy/uz`
   from macroscopic kernel within 1e-7 absolute.

2. **Finding 7.1** — Replace the v_new-based regime split with a
   v_current-based regime split in `applyCFLLimitingKernel` and
   `applyCFLLimitingAdaptiveKernel`. (See Suggested fix in §7.)

   Patch site: `src/physics/force_accumulator.cu:701-761` and 766-860.

### P1 — investigate/document (potential bugs)

3. **Finding 1.1** — Decide and document whether `Δu = F/(ρ+0.5K)` or
   `Δu = F/ρ` is the canonical EDM in this codebase. Sprint-1 made the
   change for "Marangoni leak", but the choice changes the effective
   momentum sink. Add a CE derivation comment near the kernels.

4. **Finding 2.3 + 6.3** — Unify ρ_min thresholds across collision,
   macroscopic, and `D3Q19::computeVelocity` (suggest 1e-10f everywhere).

### P2 — style/future-proofing

5. **Finding 1.3** — Hoist the second `f_local` read inside BGK EDM
   kernel up to the moment computation (one read instead of two).

6. **Finding 5.3** — Make `Δ` (lattice spacing) explicit in
   `computeSmagorinskyOmega` for AMR-readiness.

7. **Finding 3.2** — Add comment noting that under heavy LES, ω⁻ can
   approach 2.0 with Λ=3/16; consider Λ floor for production cases.

---

## What I did NOT find

* **No sign errors** in any kernel.
* **No FP-conversion bugs** — coefficients are all exact in FP32 except
  weights (which use `1.0f/3.0f` etc., consistent across files).
* **No opposite-table mistakes** — exhaustively verified.
* **No 2nd-moment isotropy violations** — verified by python.
* **No missing factor-of-2 in cross terms** of regularized projection.
* **No τ-clamp leak** — all kernels clamp ω inside their respective
  sub-kernels (collisionBGK_EDM uses ρ_min floor in denom; Smagorinsky
  clamps τ_eff ∈ [0.505, 5.0]; constructor warns at τ < 0.51).

The two confirmed bugs (1.2, 7.1) are both *coupling* errors — single
kernels are correct in isolation, but the *handoff* between them is
inconsistent. This pattern is hard to catch with unit tests because each
kernel passes its own unit test.

---

## Confidence

* **确定** (verified by independent derivation): findings 1.2, 4.1-4.4, 5.1-5.2,
  6.1-6.2, 7.1.
* **较确定** (textbook comparison, no runtime test): findings 1.1, 1.3, 2.2,
  3.1, 3.4.
* **需验证** (depends on operational regime): finding 3.2 (LES boundary
  drift), 7.2 (clamp asymmetry).
* **不确定** (would need CE expansion or extensive simulation to confirm):
  the *physics impact* of finding 1.1 — whether the EDM-hybrid Darcy
  treatment is observable in melt pool measurements vs published Crank-
  Nicolson. The finding itself (not in any published paper) is certain.

---

## File locations referenced (all under `/home/yzk/LBMProject_debug_patrol`)

* `src/physics/fluid/fluid_lbm.cu:1388-1511` (BGK+EDM)
* `src/physics/fluid/fluid_lbm.cu:1536-1667` (TRT+EDM)
* `src/physics/fluid/fluid_lbm.cu:1684-1835` (Regularized+EDM)
* `src/physics/fluid/fluid_lbm.cu:1855-2000` (Regularized+Guo)
* `src/physics/fluid/fluid_lbm.cu:2009-2078` (post-stream macroscopic, EDM)
* `src/physics/fluid/fluid_lbm.cu:2583-2650` (post-stream macroscopic, Guo)
* `src/physics/fluid/fluid_lbm.cu:2979-2984` (setTRT)
* `src/core/lattice/d3q19.cu:30-83` (D3Q19 tables)
* `src/core/lattice/d3q19.cu:132-147` (`computeEquilibrium` float)
* `src/core/lattice/d3q19.cu:150-158` (`computeEquilibriumDouble`)
* `src/physics/fluid/smagorinsky_les.cuh:49-128`
* `src/physics/force_accumulator.cu:649-680` (Carman-Kozeny K)
* `src/physics/force_accumulator.cu:701-761` (CFL limiter)
* `src/physics/force_accumulator.cu:766-860` (CFL adaptive)
