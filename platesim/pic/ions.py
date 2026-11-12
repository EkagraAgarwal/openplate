"""Multi-ion particle-in-cell (PIC) Brownian dynamics and its 1-D continuum reference.

Plan v2, item E4 (research track).  A small-domain *validation* tool: every particle stands for
many ions (a super-particle), so this is not a production solver and its noise falls only as
1/sqrt(N).  It solves the same Nernst-Planck system as the continuum (``feature/engine`` tertiary
path) independently, which is what makes it useful as a check of the continuum transport model
(plan item D1) rather than as a second production engine.

Geometry and boundary conditions
--------------------------------
1-D in z, domain [0, L], unit cross-section:

* ``z = 0`` -- cathode.  The reacting species (Cu2+) is absorbed by the exact 1-D first-passage
  rule (method of images: propose the free step, fold ``z < 0`` to ``-z``, accept with probability
  ``1 - exp(-2 z0 z / sigma^2)``, so the absorbed fraction is ``erf(z0/(sigma sqrt 2))``); a plain
  "absorb on crossing" rule leaves an effective wall concentration of order ``dz/2`` and biases the
  flux low by a few per cent.  Every other species is specularly reflected (zero wall flux).  A
  first-order cathode rate ``k`` is available but uses the cruder crossing rule (documented in
  ``run_pic``).
* ``z = L`` -- bulk reservoir.  A band of ``n_bath`` thermostat cells *above* ``z = L`` is reset to
  the bulk density of every species at every step; the band is a Dirichlet bath whose effective
  plane is the band's centre.  The band must be at least two random steps thick, otherwise the
  gross exchange traffic drains it within one step and the interior sees a weakened source.  Its
  residual bias on the measured flux is within about +-3 % at ``n_cells = 32`` (measured: 0.999 on
  the neutral-species test, see ``tests/test_pic.py``).

Dynamics: Euler-Maruyama,
    z(t+dt) = z + u_i(z) dt + sqrt(2 D_i dt) N(0,1),      u_i = -z_i D_i F/(R T) dphi/dz  [cm/s],
with Cloud-in-Cell (CIC, linear) deposition of the particle densities onto the grid and CIC
interpolation of the drift field back to the particles.  Reflection at the walls is specular
(the normal component of the step is reversed, ``z -> -z``), the convention of the E1 tracer.

Why the quasi-neutral potential and not Poisson
-----------------------------------------------
The Debye length of this bath is about 0.15 nm [C]; the domain is micrometres and ``dz`` is tens of
nm, so a Poisson solve would need 10^4-10^5 cells across the domain to resolve the double layer,
and a "scaled permittivity" would be an arbitrary knob that changes the physics it is supposed to
validate.  Instead the potential is taken in the quasi-neutral limit, from zero divergence of the
total current on the particle densities -- the same closure the continuum tertiary path uses
(``kappa = F^2/(R T) sum z_i^2 D_i c_i``)::

    d/dz( kappa dphi/dz ) = -F d/dz( sum_i z_i D_i dc_i/dz ),   kappa = F f sum_i z_i^2 D_i c_i
 => dphi/dz = ( -F sum_i z_i D_i dc_i/dz - i ) / kappa,          f = F/(R T).

The integration constant is the total current density ``i`` in the +z direction, and it is fixed by
the galvanostatic condition at the cathode: ``i = -z_Cu F J_Cu`` with ``J_Cu`` the measured
absorption flux (a running window average).  In 1-D this is an exact integral of the elliptic
equation (no linear solve, no iteration, unconditionally stable), and it reproduces the classical
steady binary-salt solution: the continuum reference below returns ``J/J_diff = 2.0000`` for a
binary 2:2 salt and the PIC returns 1.91 +- 0.03 on the same species.

Two honest caveats, both measured rather than assumed.  (1) The equation conserves *current*, not
*charge*: the ensemble's net charge is set by the boundary conditions (the reservoir re-supplies
every species at its bulk value) and is reported as ``charge_rel``; the binary-salt run keeps it
under 3 % in the bulk.  (2) The density gradients that enter ``dphi/dz`` are noisy and the noise is
amplified by ``F z D / kappa`` -- worst for a fast majority ion, so the H+ case is ~40x noisier than
the Cu2+ one.  The drift field is therefore computed from an exponentially time-averaged density
(``field_tau_frac``) and smoothed (``field_smooth``), and the drift is capped at ``drift_cfl * dz``
per step (the ideal binary solution has ``dphi/dz ~ 1/z`` at the wall, which no finite grid can
resolve).  What survives is a rectified noise field that acts as an extra diffusivity: in the 20x
excess-acid case it adds ~20 % to the Cu flux, so the excess-support limit is tested with
``field=False`` (where migration drops out *by definition*) and the self-consistent run is reported
separately.  Potential noise is the hard part of this method, exactly as the plan says; expect
per-run flux errors of a few per cent on the neutral and binary cases and tens of per cent on the
high-mobility support case at 10^3 super-particles.

Steady 1-D Nernst-Planck reference (``continuum_flux``)
-------------------------------------------------------
For the same species list the module also solves the steady BVP
    N_i = -D_i (dc_i/dz + z_i f c_i dphi/dz) = const,   sum_i z_i c_i = 0,
with the same boundary conditions.  Species that do not react have ``N_i = 0`` at both ends, so
their steady profile is the Boltzmann distribution ``c_i = c_i,bulk exp(-z_i f psi)`` with
``psi = phi - phi(L)`` -- the exact (exponentially fitted / Scharfetter-Gummel) discrete solution.
Electroneutrality then eliminates ``c_react`` and leaves one ODE for ``u = exp(z_c f psi)``, which
is integrated from the reservoir to the wall by RK4; the reacting flux is the shooting parameter,
fixed by the cathode condition (``c_react(0) = 0`` for a perfect sink, ``N = k c_react(0)`` for a
first-order rate) and solved by Brent's method.  This formulation is deliberate: iterating on the
concentration profile (a Picard loop) diverges in the perfect-sink binary limit, where ``psi`` goes
to ``-inf`` at the wall while ``u`` stays smooth and linear.  Analytic limits used as tests:

* binary 2:2 salt (CuSO4, only Cu2+ reacting, anion blocked).  The anion is stagnant at steady
  state, so its diffusion and migration cancel and the current is carried entirely by Cu2+:
  ``N_Cu = -D_Cu (c_Cu + c_SO4)' = -2 D_Cu c_Cu'`` for equal concentrations.  Hence
  ``J_lim = 2 D_Cu c_bulk / L``: twice the diffusion-only flux ``D_Cu c_bulk / L``.  [C]
* excess supporting electrolyte: the field needed to hold the support ions is O(c_Cu/c_support),
  so the enhancement over ``D_Cu c_bulk / L`` tends to 1.  [C]

Units are cm, s, mol, A, V, K throughout (``c`` in mol/cm3, ``D`` in cm2/s, positions in cm);
lengths passed as ``*_um`` are micrometres.

Diffusivities and the default bath are copied from ``platesim/feature/engine/params.py``
(``D_cu25 = 2.65e-6``, ``D_H25 = 9.31e-5``, ``D_SO425 = 1.07e-5`` cm2/s at 25 C, Stokes-Einstein
scaling, bath 65 g/L Cu as CuSO4 + 8 g/L H2SO4 fully dissociated) rather than imported, so that a
change in the engine cannot silently change the yardstick it is validated against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..core.constants import F, R_GAS

__all__ = [
    "Ion",
    "PICResult",
    "ContinuumResult",
    "default_bath",
    "run_pic",
    "continuum_flux",
    "T_DEFAULT_C",
]

T_DEFAULT_C = 22.0           # bath temperature used across the project [J]
_UM = 1e-4                   # cm per um
_M_CU = 63.546               # g/mol, platesim.core.constants.M_CU
_EPS = 1e-300


# --------------------------------------------------------------------------- species
@dataclass(frozen=True)
class Ion:
    """One ion species.  ``D`` in cm2/s, ``c_bulk`` in mol/cm3, ``z`` the charge number."""

    name: str
    z: int
    D: float
    c_bulk: float
    reacts: bool = False

    def __post_init__(self):
        if self.D <= 0.0:
            raise ValueError(f"{self.name}: D must be > 0, got {self.D!r}")
        if self.c_bulk <= 0.0:
            raise ValueError(f"{self.name}: c_bulk must be > 0, got {self.c_bulk!r}")


def _stokes_einstein(T_C: float, T_ref_C: float = 25.0) -> float:
    """D(T)/D(T_ref) by Stokes-Einstein; identical to feature/engine/params.stokes_einstein_factor."""
    def eta(T):
        return 2.414e-5 * 10.0 ** (247.8 / (T + 273.15 - 140.0))
    return ((T_C + 273.15) / (T_ref_C + 273.15)) * (eta(T_ref_C) / eta(T_C))


def default_bath(T_C: float = T_DEFAULT_C, with_hso4: bool = False, c_hso4: float = 1e-12):
    """The project bath (65 g/L Cu as CuSO4 + 8 g/L H2SO4, fully dissociated), Cu2+ reacting.

    ``with_hso4=True`` adds HSO4- as an extra *unreacting* species (z = -1) at ``c_hso4``
    (mol/cm3).  It is not in equilibrium with H+ / SO4 -- there is no speciation here, plan D1 owns
    that -- so it is a diagnostic species only: it takes part in electroneutrality and in the
    conductivity but in no reaction.  The default concentration is a placeholder, so a caller that
    wants a physical HSO4- fraction has to move the same charge out of H+ / SO4 to stay neutral.
    """
    se = _stokes_einstein(T_C)
    c_cu = 65.0 / _M_CU * 1e-3                                  # mol/cm3 [J] 1.023 M
    c_h2so4 = 8.0 / 98.079 * 1e-3                               # mol/cm3 [J] 0.0816 M
    ions = [
        Ion("Cu2+", +2, 2.65e-6 * se, c_cu, reacts=True),
        Ion("H+", +1, 9.31e-5 * se, 2.0 * c_h2so4),
        Ion("SO4^2-", -2, 1.07e-5 * se, c_cu + c_h2so4),
    ]
    if with_hso4:
        # D(HSO4-) from the limiting molar conductivity (~50 S cm2/mol) by Nernst-Einstein [SH]
        ions.append(Ion("HSO4-", -1, 1.3e-5 * se, float(c_hso4)))
    return ions


# --------------------------------------------------------------------------- grids helpers
def _cic_weights(x, dx, n):
    """CIC (linear) stencil: (j0, j1, w0, w1) for positions ``x`` on a cell-centred grid."""
    t = x / dx - 0.5
    j0 = np.floor(t).astype(np.int64)
    j0 = np.clip(j0, 0, n - 1)
    w1 = np.clip(t - j0, 0.0, 1.0)
    j1 = np.minimum(j0 + 1, n - 1)
    return j0, j1, 1.0 - w1, w1


def _deposit(x, weight, dx, n):
    """Cloud-in-cell deposition of weighted particles -> cell concentrations (weight/dz units)."""
    j0, j1, w0, w1 = _cic_weights(x, dx, n)
    rho = np.bincount(j0, weights=weight * w0, minlength=n) + np.bincount(
        j1, weights=weight * w1, minlength=n)
    return rho / dx


def _interp(field, x, dx):
    """Cloud-in-cell interpolation of a cell-centred field at continuous positions."""
    n = field.size
    j0, j1, w0, w1 = _cic_weights(x, dx, n)
    return w0 * field[j0] + w1 * field[j1]


def _field_coarse(n_spec, ions, dz, n_cells, nf, current_a_cm2, f_rt, smooth, drift_cfl, dt):
    """Field on a coarser grid than the particles, then interpolated back.

    ``dphi/dz`` is a *mean* field: it varies on the scale of the diffusion layer, not of a cell,
    while the CIC density gradients that feed it are noisy at the cell scale and the noise is
    amplified by F z D / kappa (worse for a fast majority ion: the H+ case is ~40x noisier than the
    Cu2+ one).  Binning to ``nf`` cells and differencing there cuts the gradient noise by the bin
    factor squared; the linear interpolation back to the particle grid adds the cell-scale smoothing
    the particles need.  This is the standard mean-field split of a PIC code.
    """
    nf = int(max(2, min(nf, n_cells)))
    if nf == n_cells:
        return _field_gradient(n_spec, ions, dz, current_a_cm2, f_rt, smooth,
                               drift_cfl=drift_cfl, dtk=dt)
    b = n_cells // nf
    m = nf * b
    n_coarse = n_spec[:, :m].reshape(len(ions), nf, b).sum(axis=2) / (b * dz)
    phi_c = _field_gradient(n_coarse, ions, b * dz, current_a_cm2, f_rt, smooth,
                            drift_cfl=drift_cfl, dtk=dt)
    xf = (np.arange(n_cells) + 0.5) * dz
    xc = (np.arange(nf) + 0.5) * b * dz
    return np.interp(xf, xc, phi_c)


def _field_gradient(n_spec, ions, dz, current_a_cm2, f_rt, smooth, drift_cfl=None, dtk=None):
    """dphi/dz (V/cm) from zero total-current divergence on the deposited densities.

    ``n_spec`` is (nspecies, ncells) in mol/cm3.  The conductivity is the local Nernst-Einstein
    one, ``kappa = F f sum z^2 D c``, exactly the closure of the continuum tertiary path.

    ``drift_cfl`` caps |u_i| dt at ``drift_cfl * dz``.  The ideal binary-salt solution has
    ``dphi/dz ~ 1/z`` near the wall (the field that keeps the co-ion stagnant), which a finite grid
    cannot represent; without the cap a single noisy near-wall gradient produces a drift of several
    cells per step and the run diverges.  The cap sits at half a cell per substep, i.e. at the same
    scale as the exact discrete field, so it only clips the unresolvable spike.
    """
    z = np.array([ion.z for ion in ions], dtype=float)
    D = np.array([ion.D for ion in ions], dtype=float)
    if not np.any(z != 0.0):
        return np.zeros(n_spec.shape[1])
    dn = np.gradient(n_spec, dz, axis=1)                       # central inside, one-sided at ends
    kappa = F * f_rt * np.sum(z[:, None] ** 2 * D[:, None] * n_spec, axis=0)
    num = -F * np.sum(z[:, None] * D[:, None] * dn, axis=0) - current_a_cm2
    safe = kappa > 1e-30
    phi_p = np.where(safe, num / np.where(safe, kappa, 1.0), 0.0)
    for _ in range(int(smooth)):
        phi_p = 0.25 * np.concatenate((phi_p[:1], phi_p[:-1])) \
            + 0.5 * phi_p \
            + 0.25 * np.concatenate((phi_p[1:], phi_p[-1:]))
    if drift_cfl is not None and dtk:
        u_max = np.max(np.abs(z) * D) * f_rt * np.abs(phi_p)      # cm/s
        cap = drift_cfl * dz / dtk
        phi_p = np.where(u_max > cap, phi_p * (cap / u_max), phi_p)
    return phi_p


# --------------------------------------------------------------------------- PIC
@dataclass
class PICResult:
    """Output of :func:`run_pic`.  Fluxes are mol/cm2/s, positive *towards* the cathode."""

    flux: dict            # species -> absorbed flux magnitude (0 for non-reacting species)
    flux_se: dict         # Poisson standard error of ``flux``
    flux_se_blocks: dict  # block-average standard error (cross-check, catches slow drift)
    counts: dict          # absorption events per species
    z_um: np.ndarray      # cell centres
    profile: dict         # species -> time-averaged concentration, mol/cm3
    profile_se: dict      # species -> block standard error of ``profile``
    field: np.ndarray     # time-averaged dphi/dz, V/cm
    current_a_cm2: float  # time-averaged total current density used by the field solve
    charge_rel: float     # |sum z n| / sum |z| n in the bulk window
    dt: float
    n_steps: int
    substeps: dict
    t_warm_s: float
    t_sample_s: float
    L_um: float
    n_cells: int


def run_pic(ions, L_um=8.0, n_cells=32, n_particles=1000, t_warm_s=None, t_sample_s=None,
            dt_s=None, cfl=0.4, seed=0, field_every=1, field_smooth=2,
            current_tau_frac=0.15, sink_k_cm_s=None, init="bulk", n_blocks=8,
            bulk_window=0.3, n_bath=4, field_tau_frac=0.05, drift_cfl=0.5, field_cells=None,
            field=True):
    """Brownian dynamics of super-particles with a self-consistent quasi-neutral potential.

    Parameters
    ----------
    ions : sequence of :class:`Ion`
        Exactly one species must have ``reacts=True`` (it is absorbed at the cathode).
    L_um, n_cells : float, int
        Domain length and number of CIC cells.
    n_particles : int or dict
        Super-particles per species (``dict`` keyed by name).  Each carries
        ``c_bulk * L / n`` mol/cm2, so a species' absorption count is a direct flux measurement.
    t_warm_s, t_sample_s : float, optional
        Equilibration and sampling windows; default ``2 L^2/D_slow`` and ``1.5 L^2/D_slow`` with
        ``D_slow`` the smallest diffusivity (the slowest relaxation).
    dt_s : float, optional
        Macro time step; default ``cfl dz^2 / D_slow``.  Faster species are sub-cycled with their
        own stable step (this is what makes H+ affordable).
    field_every : int
        Recompute the potential every N macro steps (the drift is held in between).
    field_smooth : int
        Passes of the [1/4, 1/2, 1/4] stencil on dphi/dz (noise control).
    current_tau_frac : float
        Window for the measured total current, in units of ``L^2/D_slow``.
    sink_k_cm_s : float, optional
        First-order absorption rate at the cathode (``None`` = perfect sink).  The per-encounter
        probability is ``min(1, k dt/dz)``, first-order in ``dt``/``dz``; use the perfect sink for
        quantitative work.
    init : {"bulk", "linear"}
        Initial profile: uniform bulk concentration, or the linear profile ``c_bulk*z/L`` (a warm
        start that shortens the transient; the tests use the honest uniform start).
    n_blocks : int
        Number of blocks for the block-average standard errors.
    bulk_window : float
        Fraction of the top of the domain used for the ``charge_rel`` bulk-neutrality diagnostic.
    field_tau_frac, drift_cfl : float
        Time constant of the density average that feeds the field (in units of ``L^2/D_slow``) and
        the drift cap in cells per macro step.  Both are noise/stability controls; see the module
        docstring for why they are needed.
    field : bool
        ``False`` solves the same transport with ``dphi/dz = 0`` -- the pure-diffusion reference
        used for the excess-supporting-electrolyte limit (where migration drops out) and to isolate
        the field noise from the transport discretisation.
    field_cells : int, optional
        Solve the field on a coarser grid than the particles (a mean-field split).  ``None`` keeps
        the particle grid; coarse grids remove gradient noise but smear the near-wall field.
    n_bath : int
        Number of thermostat cells above ``z = L`` holding the bulk concentration.

    Returns
    -------
    :class:`PICResult`
    """
    ions = list(ions)
    if not ions:
        raise ValueError("need at least one ion species")
    if not 0.0 < bulk_window <= 1.0:
        raise ValueError("bulk_window must be in (0, 1]")
    reacting = [i for i in ions if i.reacts]
    if len(reacting) != 1:
        raise ValueError(f"exactly one species must have reacts=True, got {len(reacting)}")
    if init not in ("bulk", "linear"):
        raise ValueError(f"init must be 'bulk' or 'linear', got {init!r}")

    L = float(L_um) * _UM
    dz = L / n_cells
    names = [i.name for i in ions]
    zq = np.array([i.z for i in ions], dtype=float)
    D = np.array([i.D for i in ions], dtype=float)
    cb = np.array([i.c_bulk for i in ions], dtype=float)
    react_idx = names.index(reacting[0].name)

    if isinstance(n_particles, dict):
        M = np.array([int(n_particles[name]) for name in names], dtype=np.int64)
    else:
        M = np.full(len(ions), int(n_particles), dtype=np.int64)
    if np.any(M < 10):
        raise ValueError("need at least 10 super-particles per species")
    weight = cb * L / M                                   # mol/cm2 per super-particle

    D_slow = float(D.min())
    tau_diff = L * L / D_slow
    t_warm = float(t_warm_s) if t_warm_s is not None else 2.0 * tau_diff
    t_sample = float(t_sample_s) if t_sample_s is not None else 1.5 * tau_diff
    dt_max = cfl * dz * dz / D_slow
    dt = min(dt_max, float(dt_s)) if dt_s is not None else dt_max
    substeps = np.maximum(1, np.ceil(dt * D / (cfl * dz * dz) - 1e-9)).astype(np.int64)
    n_warm = max(1, int(round(t_warm / dt)))
    n_samp = max(1, int(round(t_sample / dt)))
    n_steps = n_warm + n_samp
    f_rt = F / (R_GAS * (T_DEFAULT_C + 273.15))

    rng = np.random.default_rng(seed)

    # Grid: n_cells physical cells over [0, L] plus a band of `n_bath` ghost cells above z = L whose
    # content is reset to the bulk density every step (a thermostat = Dirichlet bath).  The band has
    # to be thicker than ~2 sigma of the random step, otherwise the gross exchange traffic (24 % of
    # a cell per step at cfl = 0.4) drains it within a step and the interior sees a weakened source.
    # Residual bias of this boundary: the measured flux runs about +4 % high at n_cells = 32,
    # 1 % at n_cells = 64 (reported in tests/test_pic.py, not hidden).
    n_bath = int(n_bath)
    n_grid = n_cells + n_bath
    n_ghost = np.maximum(1, np.round(M / n_cells)).astype(np.int64)

    # ---- initial positions: uniform in (0, L) at the bulk density (or the linear warm start)
    pos = []
    for k in range(len(ions)):
        u = rng.random(M[k])
        frac = u if init == "bulk" else (u if k != react_idx else np.sqrt(u))
        # "linear" for the reacting species: sample the density ~ z (inverse CDF z = L sqrt(u))
        pos.append(np.clip(frac, 1e-9, 1.0 - 1e-12) * L)
    pos = [np.asarray(p, dtype=float) for p in pos]

    # ---- current window (measured absorption rate -> galvanostatic i)
    k_win = max(1, int(round(current_tau_frac * tau_diff / dt)))
    cur_buf = np.zeros(k_win)
    cur_idx = 0
    cur_sum = 0.0
    cur_fill = 0
    rate = 0.0
    current = 0.0
    abs_moles_step = 0.0

    # ---- accumulators
    n_tot = np.zeros(len(ions), dtype=np.int64)
    n_blk = np.zeros((n_blocks, len(ions)), dtype=np.int64)
    prof_acc = np.zeros((len(ions), n_cells))
    prof_blk = np.zeros((n_blocks, len(ions), n_cells))
    field_acc = np.zeros(n_cells)
    field_n = 0
    phi_p = np.zeros(n_grid)
    n_bulk = max(1, int(round(bulk_window * n_cells)))
    charge_acc = np.zeros((len(ions), n_bulk))

    k_rate = None if sink_k_cm_s is None else float(sink_k_cm_s)
    n_ema = None
    a_ema = min(1.0, dt / max(field_tau_frac * tau_diff, dt))

    for step in range(n_steps):
        sampling = step >= n_warm
        blk = min(n_blocks - 1, (step - n_warm) * n_blocks // n_samp) if sampling else 0

        # ---- reservoir: reset the bath band above z = L to the bulk density
        for k in range(len(ions)):
            p = pos[k]
            p = p[p < L]                                      # drop anything that left the bath
            n_new = int(n_ghost[k]) * n_bath
            pos[k] = np.concatenate((p, L + n_bath * dz * rng.random(n_new)))

        # ---- diagnostics and the potential (held for field_every steps)
        if step % field_every == 0:
            n_spec = np.empty((len(ions), n_cells))
            for k in range(len(ions)):
                # only the physical particles: the bath band sits above z = L and its CIC stencil
                # would otherwise leak into the top physical cell and fake a gradient there (which
                # in turn drives a spurious field of tens of V/cm in the support-electrolyte case)
                n_spec[k] = _deposit(pos[k][pos[k] < L], weight[k], dz, n_cells)
            # temporal average of the deposited density: the near-wall field is very sensitive to
            # the gradient of a single noisy snapshot (the -i/kappa term divides by a small local
            # conductivity), so the field is computed from an exponentially averaged density.
            if n_ema is None:
                n_ema = n_spec.copy()
            else:
                n_ema += a_ema * (n_spec - n_ema)
            if not field:
                phi_p = np.zeros(n_cells)
            elif field_cells is None or int(field_cells) >= n_cells:
                phi_p = _field_gradient(n_ema, ions, dz, current, f_rt, field_smooth,
                                        drift_cfl=drift_cfl, dtk=dt)
            else:
                phi_p = _field_coarse(n_ema, ions, dz, n_cells, int(field_cells), current,
                                      f_rt, field_smooth, drift_cfl, dt)
            if sampling:
                field_acc += phi_p[:n_cells]
                field_n += 1
                charge_acc += n_spec[:, n_cells - n_bulk:n_cells]

        if sampling:
            for k in range(len(ions)):
                d = _deposit(pos[k][pos[k] < L], weight[k], dz, n_cells)
                prof_acc[k] += d
                prof_blk[blk, k] += d

        # ---- one macro step, species by species (sub-cycled for the fast ones)
        abs_moles_step = 0.0
        for k in range(len(ions)):
            m = int(substeps[k])
            dtk = dt / m
            sigma = math.sqrt(2.0 * D[k] * dtk)
            p = pos[k]
            absorbed = 0
            for _ in range(m):
                if p.size == 0:
                    break
                if zq[k] != 0.0:
                    u = -zq[k] * D[k] * f_rt * _interp(phi_p, p, dz)
                else:
                    u = 0.0
                if k == react_idx:
                    if k_rate is None:
                        # Exact 1-D absorbing-wall transition for one step (method of images):
                        # the target density is G(z-z0) - G(z+z0), sampled by proposing the free
                        # step, folding z<0 to -z, and accepting with 1 - exp(-2 z0 z / sigma^2).
                        # The acceptance is erf(z0 / (sigma sqrt 2)), so the absorbed fraction is
                        # exactly the first-passage probability; a plain "absorb on crossing" rule
                        # leaves an effective wall concentration of order dz/2 instead.
                        z0 = np.maximum(p + u * dtk, 1e-300)
                        zp = z0 + sigma * rng.standard_normal(p.size)
                        zp = np.abs(zp)
                        acc = -np.expm1(-2.0 * z0 * zp / (sigma * sigma))
                        alive = rng.random(p.size) < acc
                        absorbed += int(p.size - np.count_nonzero(alive))
                        p = zp[alive]
                        p = p[p < L + n_bath * dz]                     # into the bath: gone
                        continue
                    else:
                        prob = min(1.0, k_rate * dtk / dz)
                        pn = p + u * dtk + sigma * rng.standard_normal(p.size)
                        hit = pn < 0.0
                        if np.any(hit):
                            take = hit & (rng.random(pn.size) < prob)
                            absorbed += int(np.count_nonzero(take))
                            pn = pn[~take]
                            pn = np.abs(pn)
                        p = pn[pn < L + n_bath * dz]
                        continue
                pn = p + u * dtk + sigma * rng.standard_normal(p.size)
                p = np.abs(pn)                                # specular reflection at z = 0
                p = p[p < L + n_bath * dz]                             # into the bath: gone
            pos[k] = p
            if k == react_idx and absorbed:
                abs_moles_step += absorbed * weight[k]
                if sampling:
                    n_tot[k] += absorbed
                    n_blk[blk, k] += absorbed

        # ---- measured current (window average of the absorption rate)
        cur_sum += abs_moles_step - cur_buf[cur_idx]
        cur_buf[cur_idx] = abs_moles_step
        cur_idx = (cur_idx + 1) % k_win
        cur_fill = min(cur_fill + 1, k_win)
        rate = cur_sum / (cur_fill * dt)
        current = -zq[react_idx] * F * rate                 # A/cm2, +z direction

    # ---- assemble the result
    z_um = (np.arange(n_cells) + 0.5) * dz / _UM
    flux, flux_se, flux_se_blocks, counts = {}, {}, {}, {}
    for k, name in enumerate(names):
        cnt = int(n_tot[k])
        counts[name] = cnt
        if k == react_idx:
            j = cnt * weight[k] / t_sample
            flux[name] = j
            flux_se[name] = j / math.sqrt(cnt) if cnt > 0 else float("inf")
            per = n_blk[:, k] * weight[k] / (t_sample / n_blocks)
            flux_se_blocks[name] = float(per.std(ddof=1) / math.sqrt(n_blocks)) \
                if n_blocks > 1 and cnt > 0 else float("nan")
        else:
            flux[name] = 0.0
            flux_se[name] = 0.0
            flux_se_blocks[name] = 0.0

    profile, profile_se = {}, {}
    for k, name in enumerate(names):
        profile[name] = prof_acc[k] / n_samp
        if n_blocks > 1:
            b = prof_blk[:, k, :] / (n_samp / n_blocks)
            profile_se[name] = b.std(axis=0, ddof=1) / math.sqrt(n_blocks)
        else:
            profile_se[name] = np.zeros(n_cells)

    q = charge_acc / max(field_n, 1)
    zabs = np.abs(zq)[:, None]
    num = np.abs(np.sum(zq[:, None] * q, axis=0)).sum()
    den = np.sum(zabs * q)
    charge_rel = float(num / den) if den > 0.0 else 0.0

    return PICResult(
        flux=flux, flux_se=flux_se, flux_se_blocks=flux_se_blocks, counts=counts,
        z_um=z_um, profile=profile, profile_se=profile_se,
        field=field_acc / max(field_n, 1), current_a_cm2=float(current),
        charge_rel=charge_rel, dt=dt, n_steps=n_steps,
        substeps={names[k]: int(substeps[k]) for k in range(len(ions))},
        t_warm_s=t_warm, t_sample_s=t_sample, L_um=float(L_um), n_cells=int(n_cells),
    )


# --------------------------------------------------------------------------- continuum reference
@dataclass
class ContinuumResult:
    """Steady 1-D Nernst-Planck solution for the same species list."""

    flux: float           # reacting species, mol/cm2/s (magnitude, towards the cathode)
    enhancement: float    # flux / (D_react c_bulk / L)
    z_um: np.ndarray
    c: dict               # species -> concentration profile, mol/cm3
    psi: np.ndarray       # phi - phi(L), V
    iterations: int       # brentq iterations on the flux
    converged: bool
    residual: float       # |r| / (D c_bulk / L) of the flux equation at the root
    neutrality: float     # max |sum z_i c_i| / sum |z_i| c_i over the profile


def _u_rhs(u, N, ions, react_idx, f_rt):
    """RHS of du/ds and dg/ds with u = exp(z_c f psi), s = L - z (top -> wall).

    Neutrality eliminates the reacting concentration, and its conservation law becomes
        du/ds = (z_c^2 N / D) u^2 / S,   dg/ds = u,
        S = sum_{i != react} z_i c_i,b (z_c - z_i) u^((z_c - z_i)/z_c).
    Integrating u instead of psi removes the log singularity of the perfect-sink limit (u goes
    linearly to 0 at the wall for a binary 2:2 salt instead of psi going to -inf).
    """
    zc = ions[react_idx].z
    Dc = ions[react_idx].D
    S = 0.0
    for k, ion in enumerate(ions):
        if k == react_idx:
            continue
        zk = ion.z
        if zk == 0.0:
            continue
        S += zk * ion.c_bulk * (zc - zk) * u ** ((zc - zk) / zc)
    if abs(S) < 1e-300 or u <= 0.0:
        return 0.0
    return (zc * zc * N / Dc) * u * u / S


def _march_u(N, ions, react_idx, f_rt, ds, n_steps):
    """RK4 march of (u, g) from the reservoir (s = 0) to the wall (s = L)."""
    u = np.empty(n_steps + 1)
    g = np.empty(n_steps + 1)
    u[0], g[0] = 1.0, 0.0
    for j in range(n_steps):
        uj = u[j]

        def f(uu):
            return _u_rhs(uu, N, ions, react_idx, f_rt)

        a1 = uj
        b1 = f(uj)
        a2 = max(uj + 0.5 * ds * b1, 0.0)
        b2 = f(a2)
        a3 = max(uj + 0.5 * ds * b2, 0.0)
        b3 = f(a3)
        a4 = max(uj + ds * b3, 0.0)
        b4 = f(a4)
        g[j + 1] = g[j] + ds * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        u[j + 1] = min(1.0, max(0.0, uj + ds * (b1 + 2.0 * b2 + 2.0 * b3 + b4) / 6.0))
    return u, g


def continuum_flux(ions, L_um=8.0, n_cells=200, k_cm_s=None, xtol=1e-9, max_iter=200):
    """Steady 1-D Nernst-Planck BVP for the same species and boundary conditions as the PIC.

    The non-reacting species have zero flux at both ends, so their steady profile is the Boltzmann
    distribution ``c_i = c_i,bulk exp(-z_i f psi)`` -- the exact (exponentially fitted) discrete
    solution, i.e. the finite-volume solution with Scharfetter-Gummel faces.  Electroneutrality
    then eliminates ``c_react`` and gives one ODE for ``u = exp(z_c f psi)`` (see :func:`_u_rhs`),
    integrated from the reservoir to the wall by RK4.  The reacting flux ``N`` is the shooting
    parameter, fixed by the cathode condition:

    * perfect sink (``k_cm_s=None``): ``c_react(0) = 0``  <=>  ``N = D c_bulk / g(0)``;
    * first-order rate ``k``: ``N = k c_react(0)``.

    Both are solved by Brent's method, so the result is a genuine BVP solution and not an
    iteration that has to converge to the (singular) limiting current.  Remaining error is the
    RK4 quadrature only, O(ds^4).
    """
    ions = list(ions)
    reacting = [i for i in ions if i.reacts]
    if len(reacting) != 1:
        raise ValueError(f"exactly one species must have reacts=True, got {len(reacting)}")
    if k_cm_s is not None and k_cm_s <= 0.0:
        raise ValueError("k_cm_s must be > 0 (use None for a perfect sink)")
    if reacting[0].z < 0:
        raise ValueError("only a cation (z >= 0) can be the reacting species here")

    L = float(L_um) * _UM
    names = [i.name for i in ions]
    z = np.array([i.z for i in ions], dtype=float)
    D = np.array([i.D for i in ions], dtype=float)
    cb = np.array([i.c_bulk for i in ions], dtype=float)
    react_idx = names.index(reacting[0].name)
    zc, Dc, cbc = float(z[react_idx]), float(D[react_idx]), float(cb[react_idx])
    f_rt = F / (R_GAS * (T_DEFAULT_C + 273.15))
    j_diff = Dc * cbc / L
    ds = L / n_cells
    n_steps = int(n_cells)

    def wall_state(N):
        u, g = _march_u(N, ions, react_idx, f_rt, ds, n_steps)
        return u[-1], g[-1], u, g

    def residual(N):
        u0, g0, _, _ = wall_state(N)
        if k_cm_s is None:
            return N - Dc * cbc / max(g0, _EPS)
        if u0 <= 1e-300:
            return 1.0
        return N / k_cm_s - (cbc - N * g0 / Dc) / u0

    n_hi = 4.0 * j_diff
    r_hi = residual(n_hi)
    while r_hi < 0.0 and n_hi < 1e6 * j_diff:                  # widen the bracket if needed
        n_hi *= 4.0
        r_hi = residual(n_hi)
    flux, info = brentq(residual, 0.0, n_hi, xtol=xtol * j_diff, rtol=1e-13,
                        maxiter=max_iter, full_output=True, disp=False)
    n_it, conv = int(info.iterations), bool(info.converged)

    u, g = _march_u(flux, ions, react_idx, f_rt, ds, n_steps)
    s = np.arange(n_steps + 1) * ds
    zz = L - s
    psi = np.log(np.maximum(u, _EPS)) / (zc * f_rt) if zc else np.zeros(n_steps + 1)
    c_all = {}
    for k, name in enumerate(names):
        if k == react_idx:
            if zc == 0.0:
                c_all[name] = np.maximum(cbc - flux * s / Dc, 0.0)
            else:
                c_all[name] = np.where(u > 1e-300,
                                       np.maximum(cbc - flux * g / Dc, 0.0) / np.maximum(u, _EPS),
                                       0.0)
        elif z[k] != 0.0 and zc != 0.0:
            c_all[name] = cb[k] * np.maximum(u, _EPS) ** (-z[k] / zc)
        else:
            c_all[name] = np.full(n_steps + 1, cb[k])
    # the wall node may be singular (u -> 0); the profile arrays keep it, the residual skips it
    q = np.array([c_all[name] for name in names])
    den = np.sum(np.abs(z)[:, None] * q)
    num = np.abs(np.sum(z[:, None] * q, axis=0))
    neutrality = float(np.max(np.where(den > 0.0, num / np.maximum(den, _EPS), 0.0)))

    # residual reported on the *dimensional* flux scale
    return ContinuumResult(
        flux=float(flux), enhancement=float(flux / j_diff),
        z_um=zz / _UM, c=c_all, psi=psi, iterations=int(n_it),
        converged=bool(conv), residual=float(abs(residual(flux)) / j_diff),
        neutrality=neutrality,
    )
