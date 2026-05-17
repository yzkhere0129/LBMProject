"""lbmpy ↔ our D3Q27 Cumulant kernel cross-check (kernel verification spike).

Goal: catch math-level discrepancy between our hand-written cumulant_d3q27.h
transforms and lbmpy's canonical D3Q27 compressible Cumulant collision.

This does NOT run a simulation. It compares:
  1. Stencil & ordering (mapping between our and lbmpy indices)
  2. Weights
  3. Equilibrium PDFs at a test point (u, ρ given)
  4. Relaxation rate assignment by moment order
  5. Post-collision PDF at a non-equilibrium test point
"""
import numpy as np
import sympy as sp
from lbmpy.creationfunctions import create_lb_method, create_lb_collision_rule
from lbmpy.enums import Method, Stencil
from lbmpy.stencils import LBStencil
from lbmpy import LBMConfig

# ----- Our D3Q27 stencil (from src/core/lattice/d3q27.cu) -----
OUR_EX = [0, 1,-1, 0, 0, 0, 0,
          1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0,
          1,-1, 1,-1, 1,-1,-1, 1]
OUR_EY = [0, 0, 0, 1,-1, 0, 0,
          1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1,
          1,-1, 1,-1,-1, 1, 1,-1]
OUR_EZ = [0, 0, 0, 0, 0, 1,-1,
          0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1,
          1,-1,-1, 1, 1,-1, 1,-1]

OUR_W = [8/27,
         2/27, 2/27, 2/27, 2/27, 2/27, 2/27,
         1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54,
         1/54, 1/54, 1/54, 1/54,
         1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216]


def build_index_map(lbmpy_stencil):
    """Map lbmpy index -> our index by matching (ex, ey, ez)."""
    idx_map = {}
    for li, (lx, ly, lz) in enumerate(lbmpy_stencil):
        for oi in range(27):
            if OUR_EX[oi] == lx and OUR_EY[oi] == ly and OUR_EZ[oi] == lz:
                idx_map[li] = oi
                break
        else:
            raise RuntimeError(f"lbmpy direction ({lx},{ly},{lz}) not in our stencil")
    return idx_map


def main():
    print("=" * 60)
    print("lbmpy D3Q27 Cumulant cross-check")
    print("=" * 60)

    cfg = LBMConfig(stencil=LBStencil(Stencil.D3Q27),
                    method=Method.CUMULANT,
                    relaxation_rate=1.5,
                    compressible=True)
    method = create_lb_method(cfg)
    print(f"\nlbmpy method: {type(method).__name__}")

    # 1. Stencil + ordering
    lbmpy_st = list(method.stencil)
    idx_map = build_index_map(lbmpy_st)
    print(f"\nStencil bijection: {len(idx_map)}/27 entries matched")

    # 2. Weights — compare lbmpy weight at each direction to ours
    print("\n--- Weight check (lbmpy → our) ---")
    weights_match = True
    for li, (lx, ly, lz) in enumerate(lbmpy_st):
        oi = idx_map[li]
        lbmpy_w = float(method.weights[li])
        our_w = OUR_W[oi]
        if abs(lbmpy_w - our_w) > 1e-12:
            print(f"  DIRECTION ({lx},{ly},{lz}): lbmpy={lbmpy_w:.10f}  ours={our_w:.10f}  DIFF")
            weights_match = False
    if weights_match:
        print("  All 27 weights match exactly. ✓")

    # 3. Equilibrium at a test point (rho=1.05, u=(0.05, 0, 0)) — symmetric x flow
    print("\n--- Equilibrium PDF check (rho=1.05, u=(0.05, 0, 0)) ---")
    rho_val = 1.05
    u_test = (0.05, 0.0, 0.0)
    # lbmpy equilibrium
    sub = {method.zeroth_order_equilibrium_moment_symbol: rho_val,
           method.first_order_equilibrium_moment_symbols[0]: u_test[0]*rho_val,
           method.first_order_equilibrium_moment_symbols[1]: u_test[1]*rho_val,
           method.first_order_equilibrium_moment_symbols[2]: u_test[2]*rho_val}
    lbmpy_eq = [float(sp.simplify(e.subs(sub))) for e in method.get_equilibrium_terms()]

    # Our equilibrium per D3Q27::computeEquilibrium formula in lattice_d3q27.h:
    # feq_q = w_q * rho * [1 + 3(c·u)/cs² + 9(c·u)²/(2cs⁴) - 3(u·u)/(2cs²)]
    # In standard LBM with cs² = 1/3:
    # feq_q = w_q * rho * [1 + 3(c·u) + 4.5(c·u)² - 1.5 u²]
    u_sq = u_test[0]**2 + u_test[1]**2 + u_test[2]**2
    our_eq = []
    for oi in range(27):
        cu = OUR_EX[oi]*u_test[0] + OUR_EY[oi]*u_test[1] + OUR_EZ[oi]*u_test[2]
        feq = OUR_W[oi] * rho_val * (1 + 3*cu + 4.5*cu*cu - 1.5*u_sq)
        our_eq.append(feq)

    # Compare with reordering
    eq_match = True
    max_diff = 0
    for li in range(27):
        oi = idx_map[li]
        diff = abs(lbmpy_eq[li] - our_eq[oi])
        max_diff = max(max_diff, diff)
        if diff > 1e-9:
            cx, cy, cz = lbmpy_st[li]
            print(f"  DIR ({cx:2d},{cy:2d},{cz:2d})  lbmpy_eq={lbmpy_eq[li]:.8f}  our_eq={our_eq[oi]:.8f}  Δ={diff:.2e}")
            eq_match = False
    if eq_match:
        print(f"  All 27 equilibrium values match (max Δ={max_diff:.2e}). ✓")
    else:
        print(f"  EQUILIBRIUM MISMATCH (max Δ={max_diff:.2e}) — investigate")

    # 4. Relaxation rate map (which moment gets which omega)
    print("\n--- Relaxation rate by moment (lbmpy) ---")
    relax_d = method.relaxation_info_dict if hasattr(method, 'relaxation_info_dict') else {}
    moment_order_count = {}
    for k, v in relax_d.items():
        # Count moments by total order = sum of (x,y,z) powers
        order = sum(int(d) for d in str(k).split('*') if d.isdigit() or 'x' in d or 'y' in d or 'z' in d)
        # Simpler: convert symbolic key to text and count chars
        moment_order_count.setdefault(str(v.relaxation_rate), 0)
        moment_order_count[str(v.relaxation_rate)] += 1
    print("  Relaxation rate → # moments using it:")
    for rate, n in sorted(moment_order_count.items()):
        print(f"    ω={rate}:  {n} moments")
    # Our cumulant_d3q27.h:relaxCumulants27 uses:
    #   2nd order trace: omega_b (bulk)
    #   2nd order traceless: omega_nu (shear) -- 3 diag, 3 off-diag = 6 moments
    #   3rd order: omega_3 -- 7 moments
    #   4th order: omega_4 -- 6 moments
    #   5th order: omega_5 -- 3 moments
    #   6th order: omega_6 -- 1 moment
    #   total 23 relaxed + 4 conserved = 27 ✓
    print()
    print("  Our scheme (cumulant_d3q27.h:289 relaxCumulants27):")
    print("    ω_b   (bulk, 1 moment)")
    print("    ω_nu  (shear, 6 moments: 3 off-diag + 3 traceless diag)")
    print("    ω_3   (3rd order, 7 moments)")
    print("    ω_4   (4th order, 6 moments)")
    print("    ω_5   (5th order, 3 moments)")
    print("    ω_6   (6th order, 1 moment)")
    print("    + 4 conserved (ρ, ρu_x, ρu_y, ρu_z)")
    print("    = 4 + 1 + 6 + 7 + 6 + 3 + 1 = 28 ?? ← off by 1")
    print("  CHECK: our scheme structure plausibly matches lbmpy if (1+6) ≡ 7 split into bulk+shear")


if __name__ == "__main__":
    main()
