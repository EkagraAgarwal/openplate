"""Additive surface-chemistry kernels for the feature-fill engine (plan v2 item D2).

Standalone, pure, numpy-vectorised kernels for the four additive families of an acid-copper
TSV bath.  Nothing here imports the engine or mutates engine state: the lead wires these
functions into ``fields.py`` / ``sim.py`` (see "Wiring" below).

Units follow the engine convention (``params.py``):

    concentration   mol/cm3
    length          cm          (arclength ``L``, curvature ``kappa`` [1/cm], velocity [cm/s])
    Gamma           mol/cm2     (saturation coverage, i.e. mol of sites per cm2)
    coverage theta  dimensionless site fraction, kept in [0, 1]
    rate constants  k_ads [cm3/(mol s)], k_des [1/s], k_inc/k_minus [1/cm] (per cm of growth)

The growth velocity is the same object the level set is advected with,
``v = Mfac * i`` [cm/s] (``sim.py:434``, ``vfield = ph.Mfac * i_s``), positive along the
outward normal ``n``; the curvature is ``kappa = div n`` [1/cm] (the engine's
``kap = -curvature_div_n(psi, g)`` in ``sim.py:548``, so ``+kappa`` there).  With that sign
convention the area element obeys
``dA/dt = kappa * v * A`` and a shrinking front (``v < 0``) enriches every coverage, while the
1-D update ``theta_new = theta_old L_old / L_new`` is the exact discrete counterpart.

Every coverage ODE in this module is linear in theta with frozen coefficients over one step
``d theta/dt = a - R theta``, and is integrated with the exact exponential solution

    theta(t+dt) = w + (theta_old - w) exp(-R dt),    w = a/R (the Langmuir steady state),

which saturates at ``a/R`` for ``R dt >> 1`` (any dt, so the answer is the backward-Euler /
quasi-static one) and never leaves [0, 1].  No explicit Euler anywhere; the result is
unconditionally bounded (and clipped to [0, 1] in the unphysical shrinking-surface branch).

Wiring (for the lead, no engine file touched here)
--------------------------------------------------
* ``fields.StepGeom.solve`` / ``sweep_enh`` / ``sweep_ter``: replace the ``theta_be`` call by
  ``AdditiveSet.suppressor_step(theta_s_old, dt, theta_cl[act], c_s=p.s_bulk*s_t, v=Mfac*ct*ci)``
  and use ``suppressor_sink(c_s, theta_s, theta_cl, Gamma_s, k_plus)`` as the molar sink in the
  transported-suppressor equation (``extra = w * sink``; the engine's ``w`` is an interface area
  in cm2).  ``suppressor_i0_factor`` reproduces today's ``(1-theta) i0_u + theta i0_s`` exactly,
  so ``AdditiveSet(i0_u=2e-3, i0_s=1.3e-5)`` is bit-compatible with the legacy kinetics.
* ``fields.Kinetics.cath``: multiply by ``combined_i0_factor`` (suppressor x accelerator x
  leveller) instead of the bare ``acc = 1 + b_acc theta_a``.
* ``sim.py`` (after ``psi_new = advect(...)``, currently lines ~547-557): replace the explicit
  ``ta + dt*(...)`` update with ``accelerator_step(theta_a, dt, c_a, vext, kappa, ...)``; on a
  1-D arclength front (front-point model) use ``accelerator_step_arclength`` with the old/new
  segment lengths, which conserves the surface amount exactly.
* Reverse pulse: call ``reverse_pulse_desorption`` on ``st['theta']`` / ``st['theta_l']`` with
  ``i_a`` the anodic current and ``k_rd`` from the set (the engine's ``b_ex`` is the same
  mechanism, ``k_rd * Ia / 1e-3`` per second).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AdditiveSet",
    "accelerator_source",
    "accelerator_step",
    "accelerator_step_arclength",
    "area_conserving_update",
    "chloride_sink",
    "chloride_step",
    "displace_suppressor",
    "langmuir_steady_state",
    "leveller_i0_factor",
    "leveller_sink",
    "leveller_step",
    "reverse_pulse_desorption",
    "reverse_pulse_dissolution_factor",
    "suppressor_cl_rate",
    "suppressor_i0_factor",
    "suppressor_sink",
    "suppressor_step",
]


# ======================================================================================= numerics
def _linear_step(x_old, dt, source, rate):
    """Exact step of ``dx/dt = source - rate x`` over ``dt`` with frozen coefficients.

    x(t+dt) = x_old exp(-rate dt) + source dt phi(-rate dt),  phi(z) = (1 - e^-z)/z.

    Stable as ``rate dt -> 0`` (phi -> 1), exact in the stiff limit (x -> source/rate), and
    finite for negative ``rate`` (a shrinking surface, ``dA/dt < 0``), where ``x`` grows; the
    caller clips to the physical interval.  ``rate dt`` is clipped to +-700, far beyond the
    point where exp under/overflows, so a huge dt never produces a NaN.
    """
    x0 = np.asarray(x_old, dtype=float)
    dt_a = np.asarray(dt, dtype=float)
    a = np.asarray(source, dtype=float)
    r = np.asarray(rate, dtype=float)
    z_full = r * dt_a
    z = np.clip(z_full, -300.0, 700.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(-z)
        # x(t+dt) = w + (x0 - w) e^-z with the steady state w = source/rate; this form is exact
        # for all z (no a*dt*phi + clipped-z distortion) and saturates at w as z -> +inf.
        w = a / np.where(r != 0.0, r, 1.0)
        out = w + (x0 - w) * e
        small = np.abs(z_full) < 1e-8
        out = np.where(small, x0 + a * dt_a, out)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def _langmuir_step(theta_old, dt, a_rate, k_loss):
    """Exact step of ``d theta/dt = a (1 - theta) - k_loss theta = a - (a + k_loss) theta``.

    Returns theta clipped to [0, 1]; unconditionally bounded for any ``dt >= 0`` because the
    exponential solution of a linear Langmuir balance can never leave the interval for
    ``theta_old in [0, 1]``, ``a >= 0`` and ``k_loss >= 0`` (and the clip only acts in the
    unphysical shrinking-surface branch ``k_loss < 0``).
    """
    a = np.asarray(a_rate, dtype=float)
    k = np.asarray(k_loss, dtype=float)
    return np.clip(_linear_step(theta_old, dt, a, a + k), 0.0, 1.0)


def langmuir_steady_state(ads_rate, des_rate):
    """Langmuir equilibrium coverage ``theta_ss = a / (a + k_des)`` (scalar or array)."""
    a = np.asarray(ads_rate, dtype=float)
    k = np.asarray(des_rate, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / (a + k)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


def displace_suppressor(theta_s, theta_a):
    """CEAC displacement: an adsorbed accelerator occupies sites the suppressor cannot keep.

    ``theta_s_new = min(theta_s, 1 - theta_a)``, clipped to [0, 1].  Applied after the
    accelerator update so that the two organic coverages never exceed a monolayer
    (Wheeler/Moffat 2003; [J] the displacing amount is taken as one-for-one).
    """
    th_s = np.clip(np.asarray(theta_s, dtype=float), 0.0, 1.0)
    th_a = np.clip(np.asarray(theta_a, dtype=float), 0.0, 1.0)
    return np.minimum(th_s, 1.0 - th_a)


def area_conserving_update(theta_old, L_old, L_new, cap=1.0):
    """CEAC geometric update on a 1-D arclength-parametrised front.

    The surface *amount* of adsorbate on a front segment is ``sigma = theta L`` (dimensionless
    mol of sites per unit depth).  When the front is advected, a segment of length ``L_old``
    becomes ``L_new`` while ``sigma`` is unchanged (no reaction over the geometric sub-step):

        theta_new = theta_old * L_old / L_new,     capped at ``cap`` (=1).

    ``L`` is the arclength in cm; pass an array with one entry per front point.  A shrinking
    front (``L_new < L_old``) enriches the coverage - the physical origin of CEAC superfill -
    and a growing front dilutes it.  Degenerate points with ``L_new <= 0`` end up at 0.
    The cap breaks conservation by design (a monolayer cannot be over-filled).
    """
    th = np.clip(np.asarray(theta_old, dtype=float), 0.0, cap)
    lo = np.asarray(L_old, dtype=float)
    ln = np.asarray(L_new, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ln > 0.0, lo / np.where(ln > 0.0, ln, 1.0), 0.0)
    return np.clip(th * ratio, 0.0, cap)


# ======================================================================================= 1. chloride
def chloride_sink(c_cl, theta_cl, k_cl):
    """Surface consumption flux of Cl- [mol/(cm2 s)]:  ``k_cl c_cl (1 - theta_cl)``.

    ``k_cl`` [cm/s] is the adsorption velocity (a surface-kinetic or mass-transfer coefficient);
    it is related to the site balance by ``k_cl = k_ads_cl Gamma_cl`` so that
    ``Gamma_cl d theta_cl/dt = -sink + Gamma_cl k_des theta_cl``.  The net molar loss from the
    electrolyte to the surface is ``sink - Gamma_cl k_des theta_cl`` (desorption returns Cl-).
    """
    return k_cl * np.asarray(c_cl, dtype=float) * (1.0 - np.clip(theta_cl, 0.0, 1.0))


def chloride_step(theta_old, dt, c_cl, v=0.0, k_cl=1.0e-3, Gamma_cl=1.0e-9,
                  k_des=10.0, k_inc=1.0e4):
    """Chloride coverage balance, one exact step.

        d theta_cl/dt = k_ads c_cl (1 - theta_cl) - k_des theta_cl - k_inc v theta_cl

    with ``k_ads = k_cl / Gamma_cl`` [cm3/(mol s)] (see :func:`chloride_sink`), ``k_des`` [1/s],
    ``k_inc`` [1/cm] and ``v`` the normal growth velocity [cm/s] (>0 growth).  The last term is
    burial/consumption of the adlayer by the advancing front.  Steady state
    ``theta_ss = a/(a + k_des + k_inc v)``, ``a = k_ads c_cl`` (Langmuir).
    Returns theta clipped to [0, 1] for any dt.
    """
    a = k_cl * np.asarray(c_cl, dtype=float) / Gamma_cl
    return _langmuir_step(theta_old, dt, a, k_des + k_inc * np.asarray(v, dtype=float))


# ======================================================================================= 2. suppressor
def suppressor_cl_rate(c_s, theta_cl, k_plus, n_cl=1.0):
    """Suppressor adsorption rate [1/s] on the Cl-decorated surface.

        a_s = k_plus c_s theta_cl**n_cl

    ``k_plus`` [cm3/(mol s)], ``c_s`` [mol/cm3].  The Cl-/polyether co-adsorption picture of
    Braun et al. 2019 (JES 166:D3259) and Braun 2020 (JES 167:013510) is that the polyether
    (PEG-type) suppressor only adsorbs where Cl- has first decorated the copper, so the local
    Cl coverage gates the suppressor rate.  ``n_cl`` is a site-sharing exponent; ``n_cl = 1``
    is the linear (one Cl per adsorbing segment) choice, tagged [J] because the papers give the
    mechanism, not an explicit rate law.  At ``theta_cl = 0`` the rate is exactly zero for
    ``n_cl > 0``; it is monotone non-decreasing in ``theta_cl``.
    """
    th_cl = np.clip(np.asarray(theta_cl, dtype=float), 0.0, 1.0)
    return k_plus * np.asarray(c_s, dtype=float) * th_cl ** n_cl


def suppressor_sink(c_s, theta_s, theta_cl, Gamma_s, k_plus, n_cl=1.0):
    """Molar sink of the transported suppressor at the interface [mol/(cm2 s)].

        ``Gamma_s k_plus c_s theta_cl**n_cl (1 - theta_s)``

    Mirrors the engine's ``extra = w * Gamma * k_plus * (1 - theta)`` in
    ``fields.StepGeom.solve`` (there the Cl factor is implicitly 1).  ``Gamma_s`` [mol/cm2].
    """
    a = suppressor_cl_rate(c_s, theta_cl, k_plus, n_cl=n_cl)
    return Gamma_s * a * (1.0 - np.clip(theta_s, 0.0, 1.0))


def suppressor_step(theta_old, dt, c_s, theta_cl, v=0.0, k_plus=2.5e9, k_minus=1.0e6,
                    k_des=0.0, Gamma_s=9.2e-12, n_cl=1.0):
    """Cl-mediated suppressor coverage (S-NDR) for one step.

        d theta_s/dt = k_plus c_s theta_cl**n_cl (1 - theta_s) - k_minus v theta_s
                       - k_des theta_s

    Adsorption is gated by the local Cl coverage (:func:`suppressor_cl_rate`); the
    ``k_minus v`` term is deactivation/consumption of the suppressor by growth (Braun 2020,
    [cm^-1]); ``k_des`` is an optional thermal desorption [1/s], zero by default as in the
    engine.  ``Gamma_s`` is unused by the balance itself and is accepted so a caller can pass
    the same keyword set as :func:`suppressor_sink`.  Exact exponential integration; theta in
    [0, 1] for any dt.  The S-NDR feedback (suppressor breaks down where the current, hence v,
    is high) enters through ``v`` and is closed by the caller's potential solve.
    """
    a = suppressor_cl_rate(c_s, theta_cl, k_plus, n_cl=n_cl)
    return _langmuir_step(theta_old, dt, a, k_minus * np.asarray(v, dtype=float) + k_des)


def suppressor_i0_factor(theta_s, i0_u=2.0e-3, i0_s=1.3e-5):
    """Suppression factor on the exchange current density, ``i0_eff / i0_u``.

        ``1 - theta_s (1 - i0_s / i0_u)``

    the site-fraction interpolation the engine already uses
    (``i = (1-theta) i0_u + theta i0_s`` in ``fields.Kinetics``/``sweep_*``), clipped to
    ``[i0_s/i0_u, 1]``.  With ``i0_u = 2.0e-3`` and ``i0_s = 1.3e-5`` A/cm2 (Braun Table I) this
    is bit-compatible with the legacy kinetics.
    """
    th = np.clip(np.asarray(theta_s, dtype=float), 0.0, 1.0)
    ratio = i0_s / i0_u
    return np.clip(1.0 - th * (1.0 - ratio), ratio, 1.0)


# ======================================================================================= 3. accelerator
def accelerator_source(c_a, k_a):
    """Accelerator adsorption rate [1/s]: ``k_a c_a`` (``k_a`` [cm3/(mol s)])."""
    return k_a * np.asarray(c_a, dtype=float)


def accelerator_step(theta_old, dt, c_a, v=0.0, kappa=0.0, k_a=1.76e6, k_inc=0.0):
    """CEAC accelerator balance on a moving level-set front (Wheeler/Moffat 2003).

        d theta_a/dt = -theta_a kappa v + k_a c_a (1 - theta_a) - k_inc v theta_a

    ``kappa = div n`` [1/cm] and ``v`` [cm/s] along the same normal, so ``-theta_a kappa v`` is
    the dilution/concentration of the fixed surface amount by area change
    (``dA/dt = kappa v A``; ``-kappa v = dL/dt / L`` in 1-D).  ``k_inc`` [1/cm] is direct
    incorporation of the accelerator into the growing deposit (often small for SPS; default 0
    keeps the engine's current behaviour).  Exact exponential integration with frozen
    coefficients, theta clipped to [0, 1].

    For an explicit 1-D arclength front prefer :func:`accelerator_step_arclength`, whose
    geometric part is exact at any dt (this continuum form is only second-order accurate in
    ``kappa v dt`` per step).
    """
    a = accelerator_source(c_a, k_a)
    rate = np.asarray(kappa, dtype=float) * np.asarray(v, dtype=float) + k_inc * np.asarray(v, dtype=float)
    return _langmuir_step(theta_old, dt, a, rate)


def accelerator_step_arclength(theta_old, dt, c_a, v, L_old, L_new, k_a=1.76e6, k_inc=0.0):
    """CEAC update on a 1-D arclength-parametrised front, exact in the geometry.

    Works with the surface amount ``sigma = theta L`` (sites per unit depth).  On a segment
    whose length goes from ``L_old`` to ``L_new`` over the step, at the mid-length
    ``L_mid = (L_old + L_new)/2``,

        d sigma/dt = L_mid [ k_a c_a (1 - theta) - k_inc v theta ],   theta = sigma / L_mid,

    which is ``d sigma/dt = A - R sigma`` with ``A = L_mid k_a c_a`` and
    ``R = k_a c_a + k_inc v``, integrated exactly.  Then ``theta_new = sigma_new / L_new``,
    capped at 1.  With ``k_a = k_inc = 0`` this is *exactly* the area-conserving map
    ``theta_new = theta_old L_old / L_new`` (the ``1/R`` enrichment of CEAC) to round-off,
    for any dt, which is what makes superfill of a shrinking via bottom reproducible.

    ``L_old``/``L_new`` in cm (array over front points); ``v`` in cm/s.
    """
    th = np.clip(np.asarray(theta_old, dtype=float), 0.0, 1.0)
    lo = np.asarray(L_old, dtype=float)
    ln = np.asarray(L_new, dtype=float)
    sigma = th * lo
    l_mid = 0.5 * (lo + ln)
    a_rate = accelerator_source(c_a, k_a)
    rate = a_rate + k_inc * np.asarray(v, dtype=float)
    sigma_new = _linear_step(sigma, dt, a_rate * l_mid, rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_new = np.where(ln > 0.0, sigma_new / np.where(ln > 0.0, ln, 1.0), 0.0)
    return np.clip(theta_new, 0.0, 1.0)


# ======================================================================================= 4. leveller
def leveller_sink(c_l, theta_l, k_l):
    """Molar sink of the transported leveller at the interface [mol/(cm2 s)]:
    ``k_l c_l (1 - theta_l)``, ``k_l`` [cm/s] the local mass-transfer (supply) velocity.

    Because the sink is proportional to the *local* front concentration, a protrusion that
    sees a larger diffusive/forced-convective supply (larger ``c_l`` at the front, or a larger
    ``k_l`` at an exposed corner) adsorbs proportionally more leveller.  This is the mechanism
    that flattens shoulder bumps (plan v2 D2; Moffat 2005/2023).
    """
    return k_l * np.asarray(c_l, dtype=float) * (1.0 - np.clip(theta_l, 0.0, 1.0))


def leveller_step(theta_old, dt, c_l, v=0.0, k_l=1.0e-3, Gamma_l=2.0e-10,
                  k_des=1.0, k_inc=1.0e5):
    """Transport-limited leveller coverage, one exact step.

        d theta_l/dt = (k_l c_l / Gamma_l)(1 - theta_l) - k_des theta_l - k_inc v theta_l

    ``k_l c_l / Gamma_l`` [1/s] is the supply-normalised adsorption rate (``Gamma_l``
    [mol/cm2] the leveller saturation coverage), ``k_des`` [1/s] deactivation/desorption,
    ``k_inc v`` [1/cm * cm/s] burial by growth.  Steady state is Langmuir,
    ``theta_ss = a/(a + k_des + k_inc v)`` with ``a = k_l c_l/Gamma_l``, so the coverage
    orders with the local supply ``k_l c_l`` - larger on protrusions.  Exact exponential
    integration; theta in [0, 1] for any dt.
    """
    a = k_l * np.asarray(c_l, dtype=float) / Gamma_l
    return _langmuir_step(theta_old, dt, a, k_des + k_inc * np.asarray(v, dtype=float))


def leveller_i0_factor(theta_l, strength=1.0, exponent=1.0):
    """Blocking factor on i0 from leveller coverage: ``1 - strength theta_l**exponent``,
    clipped to [0, 1].  ``exponent = 1`` is area blocking [J]; some CEAC/leveler models use a
    stronger (e.g. 3/2) dependence, hence the knob.
    """
    th = np.clip(np.asarray(theta_l, dtype=float), 0.0, 1.0)
    return np.clip(1.0 - strength * th ** exponent, 0.0, 1.0)


# ======================================================================================= 5. reverse pulse
def reverse_pulse_desorption(theta_old, dt, i_a, k_rd=20.0, i_ref=1.0e-3):
    """Additive desorption during the anodic (reverse) pulse, exact exponential decay.

        d theta/dt = -k_rd (i_a / i_ref) theta      =>      theta_new = theta_old exp(-k_rd t i_a/i_ref)

    ``i_a`` [A/cm2] the local anodic current density, ``i_ref = 1e-3`` A/cm2 (1 mA/cm2) so
    ``k_rd`` [1/s per mA/cm2] is the same calibration parameter the engine already exposes
    (``params.k_rd``, used as ``b_ex = k_rd duty_r Ia / 1e-3`` in ``fields.py``).  During the
    forward pulse use :func:`suppressor_step` / :func:`leveller_step` instead; this is the
    reverse-pulse loss only.  theta is clipped to [0, 1] (degenerate inputs are floored at 0).
    """
    th = np.clip(np.asarray(theta_old, dtype=float), 0.0, 1.0)
    decay = np.clip(k_rd * np.asarray(i_a, dtype=float) / i_ref * np.asarray(dt, dtype=float), 0.0, 700.0)
    return np.clip(th * np.exp(-decay), 0.0, 1.0)


def reverse_pulse_dissolution_factor(theta_s=0.0, theta_a=0.0, theta_l=0.0,
                                     block_s=1.0, block_a=0.5, block_l=1.0):
    """Coverage-dependent anodic dissolution factor on the reverse pulse: ``i0_a_eff/i0_a``.

    Additives passivate the surface against dissolution, so the anodic exchange is scaled by

        f = 1 - (block_s theta_s + block_a theta_a + block_l theta_l),   clipped to [0, 1].

    i.e. additive-free (bright) copper dissolves fastest and the covered flanks/shoulders are
    protected - the leveling action of a pulse-reverse waveform.  Blocking coefficients are
    [J] (no measured coverage-resolved anodic kinetics); ``block_a = 0.5`` reflects the weaker
    anodic passivation of the accelerator layer.  With no coverage ``f = 1`` exactly.
    """
    th_s = np.clip(np.asarray(theta_s, dtype=float), 0.0, 1.0)
    th_a = np.clip(np.asarray(theta_a, dtype=float), 0.0, 1.0)
    th_l = np.clip(np.asarray(theta_l, dtype=float), 0.0, 1.0)
    return np.clip(1.0 - (block_s * th_s + block_a * th_a + block_l * th_l), 0.0, 1.0)


# ======================================================================================= 6. parameters
@dataclass(frozen=True)
class AdditiveSet:
    """Literature-order default constants for the four additive families.

    Provenance: ``[B19]/[B20]`` Braun 2019 JES 166:D3259 / Braun 2020 JES 167:013510 (high-k
    S-NDR set, also the engine's ``Params``); ``[W03]`` Wheeler, Moffat & Josell 2013 JES
    160:D3260 and Moffat 2005 IBM JRD 49:19 (CEAC); ``[J]`` judgement / not measured.

    The suppressor block copies ``Params`` exactly so a default ``AdditiveSet`` reproduces the
    legacy engine kinetics through :meth:`suppressor_i0_factor`.  Every other block is flagged
    ``[J]`` because the team bath has no voltammetry yet (plan v2 D2: "until then literature
    sets, flagged").  All concentrations mol/cm3, lengths cm, energies eV-free (pure kinetics).

    Attributes
    ----------
    c_cl, Gamma_cl, k_cl, k_des_cl, k_inc_cl : chloride
        Bulk Cl- 1 mM [B19] (the Braun high-k set needs ~1 mM), monolayer 1e-9 mol/cm2 [J],
        surface-kinetic velocity 1e-3 cm/s [J] (=> k_ads = 1e6 cm3/mol/s), thermal desorption
        10 /s [J], burial 1e4 /cm [J].
    c_s, Gamma_s, k_plus, k_minus, k_des_s, n_cl : suppressor
        [B20] c_s 80 umol/L, Gamma_s 9.2e-12 mol/cm2, k_plus 2.5e9 cm3/(mol s), k_minus 1e6 /cm
        (already the engine-tuned value; the Braun 2020 1e5 /cm never breaks down in a
        30-100 um via), no thermal desorption [J], linear Cl gating n_cl = 1 [J].
    i0_u, i0_s : kinetics [B20] 2.0e-3 / 1.3e-5 A/cm2 (ratio 154).
    c_a, k_a, k_inc_a, b_acc : accelerator
        [J] 5e-9 mol/cm3, 1.76e6 cm3/(mol s) (engine value), burial 1e5 /cm [J],
        b_acc = 4.5e-3/2.6e-5 = 173 (engine value, i0 *= 1 + b_acc theta_a).
    c_l, Gamma_l, k_l, k_des_l, k_inc_l, strength_l : leveller
        [J] 1e-8 mol/cm3 (~10 uM; commercial levelers are 1-30 ppm), 2e-10 mol/cm2,
        1e-3 cm/s supply velocity, 1 /s deactivation, 1e5 /cm burial, unit blocking strength.
    k_rd, i_ref, block_s, block_a, block_l : reverse pulse
        [J] k_rd 20 /s per mA/cm2 (a few-ms reverse pulse) and unit blocking coefficients
        except 0.5 for the accelerator; ``i_ref`` 1e-3 A/cm2 matches ``params.k_rd``'s units.
    """

    # ---- chloride [B19] concentration, [J] kinetics
    c_cl: float = 1.0e-6          # mol/cm3 (1 mM Cl-)
    Gamma_cl: float = 1.0e-9      # mol/cm2, Cl adlayer
    k_cl: float = 1.0e-3          # cm/s  (=> k_ads_cl = 1e6 cm3/(mol s))
    k_des_cl: float = 10.0        # 1/s
    k_inc_cl: float = 1.0e4       # 1/cm
    # ---- suppressor [B19/B20]
    c_s: float = 8.0e-8           # mol/cm3
    Gamma_s: float = 9.2e-12      # mol/cm2
    k_plus: float = 2.5e9         # cm3/(mol s)
    k_minus: float = 1.0e6        # 1/cm
    k_des_s: float = 0.0          # 1/s [J]
    n_cl: float = 1.0             # [J] linear Cl gating
    i0_u: float = 2.0e-3          # A/cm2
    i0_s: float = 1.3e-5          # A/cm2
    # ---- accelerator [J], engine-compatible
    c_a: float = 5.0e-9           # mol/cm3
    k_a: float = 1.76e6           # cm3/(mol s)
    k_inc_a: float = 1.0e5        # 1/cm [J]
    b_acc: float = 4.5e-3 / 2.6e-5
    # ---- leveller [J]
    c_l: float = 1.0e-8           # mol/cm3
    Gamma_l: float = 2.0e-10      # mol/cm2
    k_l: float = 1.0e-3           # cm/s
    k_des_l: float = 1.0          # 1/s
    k_inc_l: float = 1.0e5        # 1/cm
    strength_l: float = 1.0
    # ---- reverse pulse [J]
    k_rd: float = 20.0            # 1/s per mA/cm2
    i_ref: float = 1.0e-3         # A/cm2
    block_s: float = 1.0
    block_a: float = 0.5
    block_l: float = 1.0

    # ------------------------------------------------------------------ coverage updates
    def chloride_step(self, theta_cl, dt, c_cl=None, v=0.0):
        """Chloride coverage after ``dt`` (see module function); ``c_cl`` defaults to bulk."""
        return chloride_step(theta_cl, dt, self.c_cl if c_cl is None else c_cl, v=v,
                             k_cl=self.k_cl, Gamma_cl=self.Gamma_cl,
                             k_des=self.k_des_cl, k_inc=self.k_inc_cl)

    def chloride_sink(self, c_cl, theta_cl):
        """Cl- surface flux [mol/(cm2 s)] at concentration ``c_cl``."""
        return chloride_sink(c_cl, theta_cl, self.k_cl)

    def suppressor_step(self, theta_s, dt, theta_cl, c_s=None, v=0.0):
        """Cl-mediated S-NDR suppressor coverage after ``dt``; ``c_s`` defaults to bulk."""
        return suppressor_step(theta_s, dt, self.c_s if c_s is None else c_s, theta_cl, v=v,
                               k_plus=self.k_plus, k_minus=self.k_minus,
                               k_des=self.k_des_s, Gamma_s=self.Gamma_s, n_cl=self.n_cl)

    def suppressor_sink(self, c_s, theta_s, theta_cl):
        """Suppressor molar sink [mol/(cm2 s)] for the transport equation."""
        return suppressor_sink(c_s, theta_s, theta_cl, self.Gamma_s, self.k_plus, n_cl=self.n_cl)

    def suppressor_factor(self, theta_s):
        """Suppression factor on i0 (1 = free surface, i0_s/i0_u = fully covered)."""
        return suppressor_i0_factor(theta_s, self.i0_u, self.i0_s)

    def accelerator_step(self, theta_a, dt, c_a=None, v=0.0, kappa=0.0):
        """Continuum CEAC update on a level-set front."""
        return accelerator_step(theta_a, dt, self.c_a if c_a is None else c_a, v=v,
                                kappa=kappa, k_a=self.k_a, k_inc=self.k_inc_a)

    def leveller_step(self, theta_l, dt, c_l=None, v=0.0):
        """Transport-limited leveller coverage after ``dt``; ``c_l`` defaults to bulk."""
        return leveller_step(theta_l, dt, self.c_l if c_l is None else c_l, v=v,
                             k_l=self.k_l, Gamma_l=self.Gamma_l,
                             k_des=self.k_des_l, k_inc=self.k_inc_l)

    def leveller_factor(self, theta_l):
        """Blocking factor on i0 from leveller coverage."""
        return leveller_i0_factor(theta_l, self.strength_l)

    def reverse_pulse_desorption(self, theta, dt, i_a):
        """Coverage after a reverse pulse of ``dt`` at anodic current ``i_a``."""
        return reverse_pulse_desorption(theta, dt, i_a, k_rd=self.k_rd, i_ref=self.i_ref)

    def reverse_pulse_factor(self, theta_s=0.0, theta_a=0.0, theta_l=0.0):
        """Coverage-dependent anodic dissolution factor on the reverse pulse."""
        return reverse_pulse_dissolution_factor(theta_s, theta_a, theta_l,
                                                self.block_s, self.block_a, self.block_l)

    # ------------------------------------------------------------------ composition
    def combined_i0_factor(self, theta_s, theta_a=0.0, theta_l=0.0):
        """Multiplier on the unsuppressed exchange current density ``i0_u``.

            f = [1 - theta_s (1 - i0_s/i0_u)] (1 + b_acc theta_a) [1 - strength_l theta_l]

        with the CEAC displacement applied first, ``theta_s <= 1 - theta_a``.  All three
        factors are products because they act on independent site populations / mechanisms
        ([J] no cross terms beyond the site displacement).  ``f = 1`` exactly on a clean
        surface; the accelerator raises i0, the suppressor and leveller lower it.
        """
        th_s = displace_suppressor(theta_s, theta_a)
        return np.clip(self.suppressor_factor(th_s)
                       * (1.0 + self.b_acc * np.clip(np.asarray(theta_a, dtype=float), 0.0, 1.0))
                       * self.leveller_factor(theta_l), 0.0, None)
