"""Physical and numerical parameters for the feature-fill level-set model (via or trench).

Ported verbatim from research/via_fill_sim/params.py (read-only reference) + `geometry` switch.
Units everywhere: cm, s, mol, A, V, K (concentrations in mol/cm3, lengths in cm unless a name ends in _um).
Defaults are the literature set from R1_params.json (2026-10-26):
  Braun et al. JES 166:D3259 (2019) / JES 167:013510 (2020) Cl-/polyether S-NDR set (high k_plus),
  D_Cu from NIST/Braun, kappa estimated, kinetics referred to 22 C.
Replace any of them by constructing  Params(k_minus=3e5, ...)  or  dataclasses.replace(p, ...).
Every number here is a *placeholder for this bath* (suppressor identity, Ea, kappa are not measured).

Transport switch: `transport="enhanced"` (default) keeps the constant migration enhancement
D_c = E_mig*D_cu and kappa = kappa_22(T), bit-identical to the pre-Task-6 engine.  `transport="tertiary"`
sets D_c = D_cu and adds explicit Cu2+/H+/SO4^2- transport with Scharfetter-Gummel face fluxes and a
locally evaluated kappa = F^2/(RT) sum z_i^2 D_i c_i (feature/engine/fields.py, StepSolver tertiary
path).  See research/tertiary_transport_notes.md."""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from ...core.constants import F, M_CU, RHO_CU   # identical values to the research engine

# R_GAS kept at the research-engine value 8.314 (platesim.core.constants has 8.314462618; using it
# would shift Arrhenius i0 by ~0.005 % and break the bit-identical via regression in the tests)
R_GAS = 8.314          # J/(mol K)


# ---------------------------------------------------------------- transport helpers (copied from
# research/agent_reports_2026-10-17/fill_model_v2.py so this folder is self-contained)
def _eta_water(T):
    """Water viscosity proxy (Vogel form) used for Stokes-Einstein scaling, T in C."""
    return 2.414e-5 * 10 ** (247.8 / (T + 273.15 - 140))


def stokes_einstein_factor(T_C, T_ref_C=25.0):
    """D(T)/D(T_ref) by Stokes-Einstein: (T/Tref) * eta(Tref)/eta(T)."""
    return ((T_C + 273.15) / (T_ref_C + 273.15)) * (_eta_water(T_ref_C) / _eta_water(T_C))


def migration_enhancement(c_cu, others):
    """E = 1 + (2/c0) * int c_Cu dphi (steady 1-D Nernst-Planck, electroneutral, only Cu2+ reacts).
    others: list of (z, c_bulk) for the non-reacting ions (mol/L). Identical to fill_model_v2.enhancement."""
    def ccu(phi):
        return -sum(z * c * np.exp(-z * phi) for z, c in others) / 2.0
    assert abs(ccu(0) - c_cu) < 1e-6 * max(1, c_cu), 'bulk not electroneutral'
    phiL = brentq(ccu, -40, 0)
    integral, _ = quad(ccu, phiL, 0)
    return 1 + 2 * integral / c_cu


def default_enhancement():
    """E for the Nano3D bath: Cu 65 g/L as CuSO4 + H2SO4 8 g/L (fully dissociated). ~1.67."""
    cu = 65 / M_CU
    h2so4 = 8 / 98.079
    return migration_enhancement(cu, [(+1, 2 * h2so4), (-2, cu + h2so4)])


# ---------------------------------------------------------------- the parameter set
@dataclass
class Params:
    # ---- bath / electrolyte
    c_bulk: float = 65 / M_CU * 1e-3          # mol/cm3 Cu2+
    D_cu25: float = 2.65e-6                    # cm2/s at 25 C (NIST / Braun), SE scaling with T
    E_mig: float = field(default_factory=default_enhancement)   # migration enhancement, D_eff = E * D_Cu
    kappa_22: float = 0.065                    # S/cm at 22 C
    kappa_per_K: float = 0.018                 # fractional change per K
    # ---- tertiary transport (explicit Cu2+ / H+ / sulfate, local kappa; Params.transport switch)
    # 'enhanced' (default, legacy): D_c = E_mig*D_cu and kappa = kappa_22(T).  Bit-identical to the
    # pre-Task-6 engine.  'tertiary': D_c = D_cu (migration comes from the explicit potential), the
    # species Cu2+/H+/SO4^2- are transported with Scharfetter-Gummel face fluxes and
    # kappa = F^2/(RT) sum z^2 D c is evaluated locally.  See research/tertiary_transport_notes.md.
    transport: str = "enhanced"                # 'enhanced' | 'tertiary'
    D_H25: float = 9.31e-5                     # cm2/s, H+ at 25 C (limiting conductivity, aqion/CRC)
    D_SO425: float = 1.07e-5                   # cm2/s, SO4^2- at 25 C
    c_h_bulk: float = field(default_factory=lambda: 2 * (8 / 98.079) * 1e-3)            # mol/cm3 H+
    c_so4_bulk: float = field(default_factory=lambda: (65 / M_CU + 8 / 98.079) * 1e-3)   # mol/cm3 SO4
    # Ka2 of HSO4- -> H+ + SO4 (mol/L).  Recorded for the pool re-equilibration; the default bath
    # treatment is fully dissociated (HSO4 = 0), so the tertiary path does not use it unless a caller
    # sets it.  Notes: 1.0e-2 at 25 C, ~1.05e-2 at 22 C.
    Ka2: float | None = 1.05e-2
    tertiary_inner: int = 2                    # inner species -> kappa -> potential passes per sweep
    tertiary_chi_max: float = 30.0             # |chi| cap per face (Bernoulli argument, ~0.77 V)
    # ---- kinetics (22 C reference, Arrhenius on all i0)
    i0_u: float = 2.0e-3                       # A/cm2, unsuppressed, at c_bulk
    alpha_u: float = 0.5
    i0_s: float = 1.3e-5                       # A/cm2, suppressed
    alpha_s: float = 0.5
    i0_a: float = 2.0e-3                       # A/cm2, anodic (reverse) exchange
    alpha_a: float = 1.5
    Ea_i0: float = 35e3                        # J/mol
    T_ref_C: float = 22.0
    # ---- suppressor S-NDR (Wheeler/Moffat/Josell type; Braun high-k set)
    suppressor: bool = True                    # False -> theta = 0 everywhere (no suppressor)
    s_bulk: float = 8e-8                       # mol/cm3 (80 umol/L)
    D_s25: float = 1.0e-6                      # cm2/s at 25 C (SE scaling)
    Gamma: float = 9.2e-12                     # mol/cm2 (saturation coverage)
    k_plus: float = 2.5e9                      # cm3/(mol s)  (k_plus*s_bulk = 200 /s)
    k_minus: float = 1.0e6                     # 1/cm of growth: rate = k_minus * v[cm/s] in 1/s.  TUNED (V5): literature
                                               # value 1e5 (Braun 2020) never breaks down in a 30-100 um via; the
                                               # breakdown threshold lies between 6e5 and 1e6 at 2.5 mA/cm2, 40 C
    theta0: float = 0.0                        # initial coverage
    s_transient: bool = True                   # include dS/dt storage in the suppressor equation
    k_rd: float = 0.0                          # reverse-pulse suppressor desorption (calibration parameter)
    # ---- optional CEAC accelerator coverage (off by default)
    use_accel: bool = False
    c_acc: float = 5e-9                        # mol/cm3
    k_acc: float = 1.76e6                      # cm3/(mol s)
    b_acc: float = 4.5e-3 / 2.6e-5             # i0 multiplier slope: i0 *= (1 + b_acc*theta_a)
    # ---- convection / boundary layer
    delta_um: float = 20.0                     # diffusion boundary layer above the field
    D_t0: float = 1e-3                         # cm2/s eddy diffusivity at the via mouth
    k_mix: float = 1.0                         # well-mixed depth in via diameters (0-1.6)
    # ---- resolved cavity flow (trench_full only; the LBM solve is lazy-imported by sim.py)
    flow_every: int = 10                       # re-solve the cavity flow every N level-set steps
    flow_move_cells: float = 1.0               # ... or when the front has moved this many cells
    flow_shear_s: float | None = None          # wall shear rate 1/s: None -> from delta_um; 0 -> no flow
    # Discretisation of the advective transport of Cu2+ and suppressor by the resolved flow.
    # 'gradient' (default): upwind  sum_faces F_f (c_f - c_P), i.e. u.grad(c).  Every row of the operator
    #   sums to zero, so a uniform concentration is preserved even when the face fluxes of the LBM velocity
    #   field are NOT discretely divergence free (they are not: cell-mean interpolation + the truncation at
    #   the Dirichlet top plane leave the column sums of +div(Hs u c) non-zero, which used to inject a
    #   spurious source/sink -c div(Hs u) into every cell and rippled the trench_full field: 74-111 um
    #   bumps on the shoulders with the divergence form vs a flat 83.0-83.5 um with 'gradient').
    # 'divergence': legacy conservative form +div(Hs u c) (column sums zero).  Kept only for comparison;
    #   it is not discretely stable against the LBM flux imbalance and re-introduces the ripple.
    adv_form: str = "gradient"                 # 'gradient' | 'divergence'
    # ---- solver for the resolved cavity flow (trench_full only; see sim.ViaSim.cavity_flow)
    # 'lbm' (default, bit-identical legacy path): platesim.trenchflow.cavity_lbm, cell-centred velocities
    #   interpolated to the faces by StepGeom.face_fluxes and discretised according to adv_form above.
    # 'stokes': platesim.trenchflow.stokes.solve_stokes, whose MAC face velocities are discretely
    #   divergence free.  The advective operator of every advected species is then ALWAYS the conservative
    #   'divergence' form built directly from those face fluxes (StepGeom.mac_face_fluxes: no
    #   interpolation, no Hs weighting) plus the open-top boundary flux through the Dirichlet plane, so
    #   adv_form is IGNORED for this path (it only selects the LBM discretisation).  With that operator the
    #   Cu2+ balance closes (cu_balance_rel) because the advective net influx through the plane is counted.
    # The flow itself is only used for geometry='trench_full' (periodic x); otherwise both solvers are idle.
    flow_solver: str = "lbm"                   # 'lbm' | 'stokes'
    # Discretisation of the advective face flux of the Cu2+/suppressor equations.  Only used when
    # flow_solver='stokes' (the LBM path keeps adv_form/upwind and is bit-identical for either value).
    # 'upwind' (default): separate diffusive conductance g plus the first-order upwind advective flux F
    #   (i.e. today's Stokes result, bit for bit).  At dx 8 um the cell Peclet number F/g ~ 70, so the
    #   numerical diffusion u dx/2 >> D swamps the physical one and the fill profile is grid dependent.
    # 'sg': exponentially fitted (Scharfetter-Gummel) face flux  N = g [B(-P) c_p - B(P) c_q],
    #   P = F/g, B(x) = x/(exp(x)-1), replacing the separate g + F on every interior stencil face.
    #   P -> 0 recovers pure diffusion, P >> 1 recovers upwind, and it is exact for the steady 1-D
    #   advection-diffusion problem with constant coefficients, so the solution is grid independent in
    #   the boundary-layer sense.  The top Dirichlet plane keeps its diffusive coupling plus the upwind
    #   advective terms (it is a modelling boundary, not a resolved face).  Not available for
    #   transport='tertiary' (derive raises NotImplementedError).
    adv_scheme: str = "upwind"                 # 'upwind' | 'sg'
    # ---- growth
    efficiency: float = 1.0                    # current efficiency for growth velocity and Cu2+ sink/source
    # ---- model switches
    ohmic: bool = True                         # False -> phi = 0 (uniform eta)
    # ---- geometry
    geometry: str = "via"                      # 'via' = axisymmetric (r, z); 'trench' = planar 2-D (x, z)
                                               # half-domain x in [0, R], symmetry at x = 0 and x = R, all
                                               # r-weightings (face areas, volumes, galvanostat, set current)
                                               # replaced by weight 1 / per-unit-length quantities;
                                               # 'trench_full' = planar 2-D, FULL pitch x in [0, pitch] with
                                               # pitch = d / f_open, two walls (no mirror), periodic in x,
                                               # dx identical to the 'trench' half-domain
    H_um: float = 300.0                        # via depth / trench depth
    max_overburden_um: float | None = None     # None -> 0.4 H for d >= 500 um else 80 um
    dx_um: float | None = None                 # None -> clamp(a/15, 0.75, 8)
    corner_cells: float = 1.0                  # initial corner rounding radius in cells
    # ---- numerics
    eps_fac: float = 1.2                       # interface half-width eps = eps_fac * dx
    n_active_eps: float = 3.5                  # solve where psi > -n_active_eps * eps
    cfl: float = 0.5
    dt_max: float = 120.0                      # s
    dt_adapt: bool = False                     # adaptive ceiling: dt_max_eff in [dt_max, dt_max_adapt]
    dt_max_adapt: float = 900.0                # s, absolute ceiling of the adaptive dt_max controller
    dt_tol: float = 2e-3                       # target relative per-step change of the interface speed
    dt_min: float = 1e-3
    reinit_every: int = 4
    fix_tol_bal: float = 2e-3                  # galvanostatic balance tolerance inside the iteration
    fix_tol_c: float = 3e-3
    fix_tol_theta: float = 1e-2
    fix_maxit: int = 16
    aa_depth: int = 5                          # Anderson acceleration depth (0 = plain Gauss-Seidel)
    fill_stop: float = 0.995                   # stop when via is this solid
    verbose: bool = False
    debug_iter: bool = False                   # print the fixed-point history of every step

    def to_dict(self):
        return asdict(self)


@dataclass
class Phys:
    """Temperature-dependent derived quantities (cm, s, mol, A, V)."""
    T_C: float
    T_K: float
    f_RT: float          # F/(R T), 1/V
    D_cu: float          # cm2/s plain Cu2+ diffusivity
    D_c: float           # enhanced: E_mig*D_cu (migration lumped in); tertiary: D_cu (explicit potential)
    D_s: float           # suppressor diffusivity
    kappa: float         # S/cm
    i0_u: float
    i0_s: float
    i0_a: float
    Mfac: float          # growth velocity per unit current density: v[cm/s] = Mfac * i[A/cm2]
    # ---- tertiary transport (populated for both paths; only 'tertiary' uses them)
    transport: str = "enhanced"
    D_H: float = 0.0     # cm2/s H+
    D_so4: float = 0.0   # cm2/s SO4^2-
    c_h_bulk: float = 0.0
    c_so4_bulk: float = 0.0
    z_cu: float = 2.0
    z_h: float = 1.0
    z_so4: float = -2.0
    kappa_ne: float = 0.0   # bulk Nernst-Einstein conductivity F^2/(RT) sum z^2 D c (S/cm)


TRANSPORTS = ("enhanced", "tertiary")
ADV_FORMS = ("gradient", "divergence")
FLOW_SOLVERS = ("lbm", "stokes")
ADV_SCHEMES = ("upwind", "sg")


def derive(p: Params, T_C: float) -> Phys:
    T_K = T_C + 273.15
    fD = stokes_einstein_factor(T_C)
    D_cu = p.D_cu25 * fD
    arr = math.exp(-p.Ea_i0 / R_GAS * (1 / T_K - 1 / (p.T_ref_C + 273.15)))
    if p.transport not in TRANSPORTS:
        raise ValueError(f"transport must be one of {TRANSPORTS}, got {p.transport!r}")
    if p.adv_form not in ADV_FORMS:
        raise ValueError(f"adv_form must be one of {ADV_FORMS}, got {p.adv_form!r}")
    if p.flow_solver not in FLOW_SOLVERS:
        raise ValueError(f"flow_solver must be one of {FLOW_SOLVERS}, got {p.flow_solver!r}")
    if p.adv_scheme not in ADV_SCHEMES:
        raise ValueError(f"adv_scheme must be one of {ADV_SCHEMES}, got {p.adv_scheme!r}")
    if p.adv_scheme == "sg" and p.transport == "tertiary":
        raise NotImplementedError(
            "adv_scheme='sg' is not implemented for transport='tertiary': that path already carries the "
            "Scharfetter-Gummel migration flux in StepGeom.sg_operator, and folding the advective Peclet "
            "number into its Bernoulli argument is not wired up.  Use adv_scheme='upwind' (default).")
    tertiary = p.transport == "tertiary"
    D_H = p.D_H25 * fD
    D_so4 = p.D_SO425 * fD
    f_RT = F / (R_GAS * T_K)
    return Phys(
        T_C=T_C, T_K=T_K, f_RT=f_RT,
        D_cu=D_cu, D_c=D_cu if tertiary else p.E_mig * D_cu, D_s=p.D_s25 * fD,
        kappa=p.kappa_22 * (1 + p.kappa_per_K * (T_C - 22.0)),
        i0_u=p.i0_u * arr, i0_s=p.i0_s * arr, i0_a=p.i0_a * arr,
        Mfac=M_CU / (2 * F * RHO_CU) * p.efficiency,
        transport=p.transport, D_H=D_H, D_so4=D_so4,
        c_h_bulk=p.c_h_bulk, c_so4_bulk=p.c_so4_bulk,
        kappa_ne=F * f_RT * (4 * D_cu * p.c_bulk + D_H * p.c_h_bulk + 4 * D_so4 * p.c_so4_bulk),
    )


if __name__ == '__main__':
    p = Params()
    for T in (22, 25, 40):
        print(T, derive(p, T))
    print('E =', p.E_mig, ' c_bulk =', p.c_bulk)
