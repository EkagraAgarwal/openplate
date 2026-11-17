"""Cell-local additive coverage state of the 3-D tile engine (3-D additives port, Phases 1-2).

Scope: the two coverages the 2-D engine solves *without* any bulk additive transport -- the Cl- adlayer
``theta_cl`` and the Cl-gated suppressor ``theta_s`` (``platesim.feature.engine.additives``, wired in 2-D
at ``feature/engine/fields.py:1546-1553``) -- plus, in Phase 2, the **transported accelerator** of a
``Params.additives='acc'`` run.  All three are quasi-steady on a plating step (relaxation times <= 0.1 s
for the Cl-/suppressor pair, 5-30 s for the accelerator against ``dt`` >= 1 s), so
``FieldSolver.solve`` updates them inside its outer fixed-point loop from the current local growth
velocity and the current bulk accelerator field; no ODE state is carried across time steps.

Equations, per interface cell (cm, s, mol, A; the 2-D balances of ``additives.py:174-187,220-236``)

    v          = Mfac * i_local                       growth velocity, cm/s, cathodic positive
    a_cl       = k_cl c_cl_bulk / Gamma_cl            [1/s]        (Cl- adsorption, bulk Cl-)
    theta_cl*  = a_cl / (a_cl + k_des_cl + k_inc_cl v)             (Langmuir + burial)
    a_s        = k_plus s_bulk theta_cl*^n_cl         [1/s]        (Cl-gated polyether adsorption)
    theta_s*   = a_s / (a_s + k_minus v + k_des_s)                 (S-NDR: breakdown where v is high)
    Fcov       = 1 - theta_s_eff (1 - i0_s / i0_u)                 (site-fraction i0 interpolation)
    theta_s_eff = min(theta_s, 1 - theta_a)                        (CEAC displacement)

Phase 2 transported accelerator (``additives='acc'``; ``D_acc25``, ``Gamma_acc``, ``k_inc_a`` new in
``Params``, all [J]).  With the bulk concentration normalised by the bath value, ``alpha = c_a/c_acc``,
the accelerator obeys a **steady** bulk balance below the bulk plane (quasi-steady per field solve, as
``eta``/``phi``/``ct`` are; plan 2.5 with the storage term dropped for Phase 2) and a quasi-steady
surface balance at every interface cell:

    -div(D_acc Hs grad alpha) + w q_a alpha = w s_a          in the electrolyte,  alpha = 1 at the plane
    q_a = Gamma_acc k_acc (1 - theta_a)                      [cm/s]  adsorption (implicit)
    s_a = Gamma_acc k_des_a theta_a / c_acc                  [cm/s]  thermal desorption (explicit, 0 default)
    theta_a* = k_acc c_a / (k_acc c_a + k_inc_a v + k_des_a)          (Langmuir + incorporation)
             = k_acc c_acc alpha / (k_acc c_acc alpha + k_inc_a v + k_des_a)

The sink is the *net* adsorption minus desorption at the quasi-steady coverage, so at convergence
``Gamma_acc k_inc_a v theta_a`` (incorporation) is the only irreversible loss: the bulk-plane influx equals
the surface consumption (``info['a_balance_rel']``, the Phase-2 transport-closure test).  Because the
accelerator is consumed at the front, it is depleted wherever the diffusion path from the bulk plane is
long -- below a field surface and more so at the bottom of a recess, by an amount set by depth/supply and
the surface Damkohler number ``Gamma_acc k_acc (1-theta_a) delta / D_acc`` rather than by the feature
width once the width exceeds the depletion length (``tools/accelerator_probe.py`` measures this on a
narrow and a wide trench).  This is the *transport-limited differential* the Step-0 probe
(``tools/i0_factor_probe.py``) showed is needed to lower floor/field.  A flat surface gives the analytic
1-D steady profile ``alpha(z) = (D + q z)/(D + q delta)``, linear in ``z``.

Optional advection by the resolved flow uses the same MAC face velocities and Scharfetter-Gummel operator
as Cu2+ (``operators.assemble_advdiff``).

Not implemented here (phases 3-4 of the plan), left as explicit hooks:
  * Phase 3 -- surface (extension) transport and the CEAC curvature/area-change term ``-theta_a kappa v``
    of the accelerator balance: the coverage above is a local algebraic function of the front, so a
    shrinking feature cannot yet enrich its accelerator.  The leveller ``theta_l`` is still 0, hence the
    ``(1 - strength_l theta_l)`` factor of ``AdditiveSet.combined_i0_factor`` is exactly 1.
  * Phase 4 Level A -- pulse-reverse phase-resolved coverages -- is implemented
    (:func:`periodic_phase_means` plus the ``pulse_coverage='resolved'`` branch of
    ``FieldSolver.solve``).  Level B (cycle-resolved bulk additive surface concentration) is not:
    ``info['additives']['pulse_coverage']['supply_ratio_a']`` > 1 means the Level-A recovery is
    overstated and Level B is needed.
  * The transported **suppressor** field ``s`` of plan 2.5 (D_s25, Gamma_s, k_plus sink) is not wired:
    Phase 2 here transports the accelerator only (the scope's optional part of "accelerator (and
    suppressor)").  The suppressor keeps the bulk ``s_bulk`` of Phase 1, which is the right default for
    the floor/field question: a badly supplied suppressor would be *depleted* at the floor and therefore
    push floor/field UP (plan 0.2), i.e. the wrong sign.  It shares this module's solver hook, so wiring
    it is a local addition to ``FieldSolver.solve`` rather than new machinery.

Constants come from ``Params`` only.  Their defaults are the 2-D engine's values
(``AdditiveSet`` in ``feature/engine/additives.py`` and ``Params.s_bulk/Gamma/k_plus/k_minus/c_acc/k_acc``)
plus the four new accelerator-transport constants (``D_acc25``, ``Gamma_acc``, ``k_inc_a``, ``k_des_a``),
which have no 2-D counterpart; none is calibrated on measured data, so every number this module produces
is an *effective*, uncalibrated map (flagged ``uncalibrated`` in ``solve(...)['info']['additives']``).
"""
from __future__ import annotations

import numpy as np

from ..feature.engine.additives import (displace_suppressor, langmuir_steady_state,
                                        suppressor_cl_rate, suppressor_i0_factor)

__all__ = ["COV_KEYS", "ACC_KEY", "coverage_init", "steady_state", "accelerator_steady_state",
           "accelerator_sink", "accelerator_source", "coverage_factor", "front_cells",
           "periodic_phase_means"]

COV_KEYS = ("theta_cl", "theta_s", "theta_a")
ACC_KEY = "a"                     # normalised bulk accelerator field c_a / c_acc (mode 'acc' only)


def coverage_init(shape, theta0=0.0, with_acc=False):
    """Fresh coverage state on a full (nx, ny, nz) tile grid.

    ``theta0`` (``Params.theta0``) seeds the three coverages; the default 0 starts the outer loop on the
    additive-free branch (``Fcov = 1`` exactly).  Full-grid float64 arrays: only the cells in the front
    band are ever updated, the rest stay at ``theta0``.  ``with_acc`` adds the normalised bulk
    accelerator field ``a`` (``Params.additives='acc'``), initialised at the bath value 1.
    """
    nx, ny, nz = (int(v) for v in shape)
    if min(nx, ny, nz) < 1:
        raise ValueError(f"invalid coverage grid shape {shape!r}")
    th = np.clip(float(theta0), 0.0, 1.0)
    cov = {key: np.full((nx, ny, nz), th, dtype=np.float64) for key in COV_KEYS}
    if with_acc:
        cov[ACC_KEY] = np.ones((nx, ny, nz), dtype=np.float64)
    return cov


def steady_state(i_A_cm2, p, mfac):
    """Quasi-steady ``(theta_cl, theta_s)`` at the local current density ``i_A_cm2``.

    ``p`` is a :class:`platesim.feature.engine.params.Params`, ``mfac`` the derived growth velocity
    per current density (``Phys.Mfac``, cm3/C).  Arrays in/out; the result is clipped to [0, 1].
    ``p.suppressor=False`` pins ``theta_s = 0`` (no suppressor at all, as in the 2-D engine), which
    keeps the whole additive path exactly equal to the additive-free one.
    """
    i = np.asarray(i_A_cm2, dtype=np.float64)
    v = mfac * np.maximum(i, 0.0)                        # cm/s, growth only (burial)
    a_cl = p.k_cl * p.c_cl_bulk / max(p.Gamma_cl, 1e-300)
    theta_cl = langmuir_steady_state(a_cl, p.k_des_cl + p.k_inc_cl * v)
    if not p.suppressor:
        return theta_cl, np.zeros_like(theta_cl)
    a_s = suppressor_cl_rate(p.s_bulk, theta_cl, p.k_plus, n_cl=p.n_cl)
    theta_s = langmuir_steady_state(a_s, p.k_minus * v + p.k_des_s)
    return theta_cl, theta_s


def accelerator_steady_state(i_A_cm2, a_norm, p, mfac):
    """Quasi-steady accelerator coverage ``theta_a`` at the local bulk field and current.

        theta_a = k_acc c_acc alpha / (k_acc c_acc alpha + k_inc_a v + k_des_a)

    ``a_norm`` is the local normalised bulk concentration ``c_a / c_acc`` (array; 1 in an additive-free
    region), ``v = Mfac * max(i, 0)`` the local growth velocity [cm/s] (incorporation only while growing).
    Exactly the Langmuir form of ``AdditiveSet.accelerator_step`` at ``d theta/dt = 0`` without the CEAC
    curvature term (Phase 3).  Clipped to [0, 1]; exactly 0 when ``c_acc <= 0`` or ``alpha = 0``.
    """
    a_norm = np.clip(np.asarray(a_norm, dtype=np.float64), 0.0, None)
    v = mfac * np.maximum(np.asarray(i_A_cm2, dtype=np.float64), 0.0)
    ads = p.k_acc * p.c_acc * a_norm
    return langmuir_steady_state(ads, p.k_inc_a * v + p.k_des_a)


def accelerator_sink(theta_a, p):
    """Implicit adsorption coefficient of the bulk accelerator balance [cm/s].

        q_a = Gamma_acc k_acc (1 - theta_a)

    Multiplied by ``w = |grad Hs|`` [1/cm] it is the volumetric sink rate [1/s] of the normalised
    concentration ``alpha``; multiplied by ``c_acc alpha`` it is the surface flux [mol/(cm2 s)].  Zero
    when the surface is saturated (``theta_a = 1``), so the balance never over-consumes.
    """
    th = np.clip(np.asarray(theta_a, dtype=np.float64), 0.0, 1.0)
    return p.Gamma_acc * p.k_acc * (1.0 - th)


def accelerator_source(theta_a, p):
    """Explicit desorption source of the bulk balance, normalised by ``c_acc`` [cm/s].

        s_a = Gamma_acc k_des_a theta_a / c_acc

    Zero for the default ``k_des_a = 0`` (the 2-D accelerator has no thermal desorption).  Enters the
    right-hand side as ``w s_a``; it returns adsorbed accelerator to the electrolyte and is counted on
    the source side of the transport closure.
    """
    th = np.clip(np.asarray(theta_a, dtype=np.float64), 0.0, 1.0)
    if p.k_des_a == 0.0 or p.c_acc <= 0.0:
        return np.zeros_like(th)
    return p.Gamma_acc * p.k_des_a * th / p.c_acc


def coverage_factor(theta_s, p, theta_a=0.0):
    """Multiplier on ``i0_u`` from the coverages (Phases 1-2).

        Fcov = [1 - theta_s_eff (1 - i0_s/i0_u)] (1 + b_acc theta_a)

    with the CEAC displacement ``theta_s_eff = min(theta_s, 1 - theta_a)`` applied first.  With
    ``theta_a = 0`` this is the 2-D site-fraction interpolation ``suppressor_i0_factor`` and is exactly 1
    at ``theta_s = 0``.  Phase 3 adds the leveller blocking factor ``(1 - strength_l theta_l)`` of
    ``AdditiveSet.combined_i0_factor`` (Params has no ``strength_l`` yet, so the leveller term is a
    documented hook, not a silent parameter).
    """
    th_a = np.clip(np.asarray(theta_a, dtype=np.float64), 0.0, 1.0)
    th_s = displace_suppressor(theta_s, th_a)
    return suppressor_i0_factor(th_s, p.i0_u, p.i0_s) * (1.0 + p.b_acc * th_a)


def front_cells(psi):
    """(i, j, k) of the first electrolyte cell above the highest solid cell of every column.

    Same front-crossing search as :func:`platesim.tile3d.sim.top_height`, used for coverage
    diagnostics (floor/field means in the series) and cheap enough to call once per saved frame.
    """
    a = np.asarray(psi)
    solid = a <= 0
    k = a.shape[2] - 1 - np.argmax(solid[:, :, ::-1], axis=2)        # highest solid cell
    k = np.clip(k + 1, 0, a.shape[2] - 1)                             # first electrolyte cell
    ii, jj = np.meshgrid(np.arange(a.shape[0]), np.arange(a.shape[1]), indexing="ij")
    return ii, jj, k


def periodic_phase_means(a, L, taus):
    """Periodic steady state of ``dx/dt = a - R_k x`` over forward/reverse/off phases.

    ``a`` is the adsorption rate per cell ``(n,)`` [1/s], ``L`` the loss rates per phase
    ``(3, n)`` [1/s] (``R_k = a + L_k``), ``taus`` the phase durations ``(3,)`` [s].
    Returns ``(x_start, xbar)`` each ``(3, n)``: the coverage at the start of each phase and
    the time mean within each phase, in the periodic steady state (forward -> reverse -> off).

    Per phase ``x_end = w + (x_start - w) e^{-R tau}`` with ``w = a/R``; composing gives
    ``x0 = A x0 + B`` (``A = prod e^{-R_k tau_k}``), solved as ``x0 = B/(1-A)``.  Phase means
    ``xbar_k = w_k + (x_k,start - w_k)(1 - e^{-R_k tau_k})/(R_k tau_k)`` (-> ``x_k,start``
    when ``R_k tau_k -> 0``).  ``expm1`` is used for accuracy at small ``R tau``.  Phases
    with ``tau`` 0 are the identity map.  When every ``R_k tau_k`` is 0 (i.e. ``a = L = 0``)
    the ODE is ``dx/dt = 0`` and the periodic state is undetermined: this function returns 0
    and the caller must keep the previous coverage (``dx/dt = 0`` stays where it was).

    All inputs must be finite and nonnegative; outputs are clipped to [0, 1].
    """
    a_v = np.atleast_1d(np.asarray(a, dtype=np.float64))
    L_m = np.asarray(L, dtype=np.float64)
    t_v = np.asarray(taus, dtype=np.float64).ravel()
    if t_v.shape != (3,):
        raise ValueError(f"taus must have shape (3,), got {t_v.shape}")
    if a_v.ndim != 1:
        raise ValueError(f"a must have shape (n,), got {a_v.shape}")
    n = a_v.shape[0]
    if L_m.shape != (3, n):
        raise ValueError(f"L must have shape (3, {n}), got {L_m.shape}")
    if not (np.all(np.isfinite(a_v)) and np.all(np.isfinite(L_m)) and np.all(np.isfinite(t_v))):
        raise ValueError("periodic_phase_means inputs must be finite")
    if a_v.min() < 0.0 or L_m.min() < 0.0 or t_v.min() < 0.0:
        raise ValueError("periodic_phase_means inputs must be nonnegative")
    R = a_v[None, :] + L_m                                  # (3, n)
    rtau = R * t_v[:, None]                                 # (3, n)
    one_minus_E = -np.expm1(-rtau)                          # 1 - exp(-rtau), accurate at small rtau
    E = np.exp(-rtau)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(R > 0.0, a_v[None, :] / np.where(R > 0.0, R, 1.0), 0.0)
    wE = w * one_minus_E                                    # w (1 - E), 0 when tau 0 or R 0
    A = E[0] * E[1] * E[2]                                  # (n,)
    B = wE[0] * E[1] * E[2] + wE[1] * E[2] + wE[2]          # (n,)
    denom = -np.expm1(-(rtau[0] + rtau[1] + rtau[2]))      # 1 - A without cancellation at small R tau
    x0 = np.where(denom != 0.0, B / np.where(denom != 0.0, denom, 1.0), 0.0)
    x_start = np.empty((3, n), dtype=np.float64)
    x_start[0] = x0
    x_start[1] = E[0] * x0 + wE[0]
    x_start[2] = E[1] * x_start[1] + wE[1]
    xbar = np.empty((3, n), dtype=np.float64)
    for k in range(3):
        denom_k = rtau[k]
        factor = np.where(denom_k > 0.0,
                          one_minus_E[k] / np.where(denom_k > 0.0, denom_k, 1.0), 1.0)
        xbar[k] = w[k] + (x_start[k] - w[k]) * factor
        xbar[k] = np.where(t_v[k] == 0.0, x_start[k], xbar[k])
    x_start = np.clip(x_start, 0.0, 1.0)
    xbar = np.clip(xbar, 0.0, 1.0)
    return x_start, xbar
