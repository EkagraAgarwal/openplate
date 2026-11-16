"""Cell-local additive coverage state of the 3-D tile engine (additives_3d_plan.md, Phase 1).

Scope: the two coverages the 2-D engine already solves *without* any bulk additive transport, so
they port to the 3-D tile without a new field: the Cl- adlayer ``theta_cl`` and the Cl-gated
suppressor ``theta_s`` (``platesim.feature.engine.additives``, wired in 2-D at
``feature/engine/fields.py:1546-1553``).  Both are quasi-steady on a plating step (relaxation times
<= 0.1 s against ``dt`` >= 1 s), so ``FieldSolver.solve`` updates them inside its outer fixed-point
loop from the current local growth velocity and never carries an ODE state across time steps.

Equations, per interface cell (cm, s, mol, A; the 2-D balances of ``additives.py:174-187,220-236``)

    v          = Mfac * i_local                       growth velocity, cm/s, cathodic positive
    a_cl       = k_cl c_cl_bulk / Gamma_cl            [1/s]        (Cl- adsorption, bulk Cl-)
    theta_cl*  = a_cl / (a_cl + k_des_cl + k_inc_cl v)             (Langmuir + burial)
    a_s        = k_plus s_bulk theta_cl*^n_cl         [1/s]        (Cl-gated polyether adsorption)
    theta_s*   = a_s / (a_s + k_minus v + k_des_s)                 (S-NDR: breakdown where v is high)
    Fcov       = 1 - theta_s_eff (1 - i0_s / i0_u)                 (site-fraction i0 interpolation)
    theta_s_eff = min(theta_s, 1 - theta_a)                        (CEAC displacement)

``Fcov`` multiplies the unsuppressed exchange current density ``i0_u`` of every interface cell.  The
outer loop relaxes ``theta <- theta + beta (theta* - theta)`` with ``beta = Params.cov_relax``,
halved (floored at 0.2) when the change grows between two passes (the heuristic of the pulse branch,
``TP/fields.py:528-530``).  The map is monotone in theta_s and is started from ``theta = 0``, i.e.
from the additive-free branch, so it climbs monotonically to the suppressed fixed point.

Not implemented here (phases 2-4 of the plan), left as explicit hooks:
  * Phase 2 -- bulk transport of the suppressor / leveller: ``c_s`` is the bulk ``s_bulk``
    everywhere (no ``s`` field), so a floor-depleted suppressor cannot be represented yet.
  * Phase 3 -- CEAC accelerator and transported leveller coverages plus surface (extension)
    transport: ``theta_a = theta_l = 0``, hence the ``(1 + b_acc theta_a)`` and
    ``(1 - strength_l theta_l)`` factors of ``AdditiveSet.combined_i0_factor`` are exactly 1.
    :func:`coverage_factor` already accepts ``theta_a``/``theta_l`` so only the state has to be
    added.
  * Phase 4 -- pulse-reverse phase-resolved coverages: the quasi-steady values below use the
    cycle-mean local current; ``b_ex``/``k_rd`` stripping is not applied.

Constants come from ``Params`` only.  Their defaults are the 2-D engine's values
(``AdditiveSet`` in ``feature/engine/additives.py`` and ``Params.s_bulk/Gamma/k_plus/k_minus``);
none is calibrated on measured data, so every number this module produces is an *effective*,
uncalibrated suppression map (flagged ``uncalibrated`` in ``solve(...)['info']['additives']``).
"""
from __future__ import annotations

import numpy as np

from ..feature.engine.additives import (displace_suppressor, langmuir_steady_state,
                                        suppressor_cl_rate, suppressor_i0_factor)

__all__ = ["COV_KEYS", "coverage_init", "steady_state", "coverage_factor", "front_cells"]

COV_KEYS = ("theta_cl", "theta_s")


def coverage_init(shape, theta0=0.0):
    """Fresh coverage state on a full (nx, ny, nz) tile grid.

    ``theta0`` (``Params.theta0``) seeds both coverages; the default 0 starts the outer loop on the
    additive-free branch (``Fcov = 1`` exactly).  Full-grid float64 arrays: only the cells in the
    front band are ever updated by :func:`steady_state` callers, the rest stay at ``theta0``.
    """
    nx, ny, nz = (int(v) for v in shape)
    if min(nx, ny, nz) < 1:
        raise ValueError(f"invalid coverage grid shape {shape!r}")
    th = np.clip(float(theta0), 0.0, 1.0)
    return {key: np.full((nx, ny, nz), th, dtype=np.float64) for key in COV_KEYS}


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


def coverage_factor(theta_s, p, theta_a=0.0):
    """Multiplier on ``i0_u`` from the coverages (Phase 1: only the suppressor is nonzero).

        Fcov = [1 - theta_s_eff (1 - i0_s/i0_u)] (1 + b_acc theta_a)

    with the CEAC displacement ``theta_s_eff = min(theta_s, 1 - theta_a)`` applied first.  With
    ``theta_a = 0`` (Phase 1) this is the 2-D site-fraction interpolation ``suppressor_i0_factor`` and
    is exactly 1 at ``theta_s = 0``.  Phase 3 adds the leveller blocking factor
    ``(1 - strength_l theta_l)`` of ``AdditiveSet.combined_i0_factor`` (Params has no ``strength_l``
    yet, so the leveller term is a documented hook, not a silent parameter).
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
