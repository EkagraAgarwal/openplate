"""Quasi-steady field solve of one 3-D tile, physics level L1 (no flow) plus optional cell-local
additive coverage (Phase 1 of the 3-D additives port).

Same model as the 2-D engine's additive-free 'enhanced' path (platesim/feature/engine/fields.py,
StepGeom 'diffuse' + newton_potential), written per unit volume on a uniform grid:

  Cu2+ (ct = c / c_bulk):   -div(D_c Hs grad ct) + eff * w * ci(eta) / (2 F c_bulk) * ct = 0
  potential:                -div(kappa Hs grad phi) + w * ct * ci(eta) = 0,   eta = V - phi
  galvanostat:              sum(w * ct * ci) * dx^3 = j_net * Lx * Ly

  Hs = max(0.5 (1 + tanh(psi / eps)), 1e-6),  eps = eps_fac * dx      (smoothed electrolyte indicator)
  w  = |grad Hs|  [1/cm]                                              (interface area per unit volume)
  ci = i0_cell exp(-alpha_u f eta)  [A/cm2]                          (cathodic exchange current)

  i0_cell = i0_u * Fcell(z) * Fcov(theta_s, theta_cl)                 (default: the scalar i0_u)

Fcell is an optional prescribed per-cell multiplier (``kinetics_factor``) and Fcov the Cl-gated
suppressor factor of :mod:`platesim.tile3d.additives3d` (``cov``); both are applied to the cathodic
kinetics, the galvanostatic V root-find, the multigrid shape term and the pulse phase solves.  With
both ``None`` every expression is the original scalar one (bit-identical).

Boundary conditions: ct = 1 and phi = 0 on the bulk plane z = z_plane (at a distance d_top in
[0.5, 1.5) dx above the last solved cell layer, as in 2-D); no flux through the bottom; lateral sides
periodic or no flux.  Units: cm, s, mol, A, V.  Pulse-reverse enters only through the net current.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, splu

from ..core.constants import F
from ..feature.engine.params import Params, derive
from . import additives3d, operators
from .backend import ls_numpy

DIRECT_MAX = 40000      # unknowns up to which a sparse LU is used; above it AMG-preconditioned CG
LU_FALLBACK_MAX = 600000  # advection systems up to this size may fall back to sparse LU at high Peclet
PE_ILU = 10.0             # cell Peclet above which the Cu advection solve starts with ILU-GMRES


class _Precond:
    """AMG preconditioner shared by a family of systems A0 + diag(q) (q >= 0, changing between calls).

    The hierarchy is built for the first matrix of the family and reused; it is rebuilt when a solve
    needs more than `grow` times the iterations of the first solve after the last rebuild (the
    diagonal has drifted far, e.g. the Cu sink near the limiting current)."""

    def __init__(self, grow=2.5, floor=25, info=None):
        self.M, self.base, self.grow, self.floor, self.builds = None, None, grow, floor, 0
        self.lu, self.info = None, info

    def get(self, A):
        if self.M is None:
            import pyamg
            self.M = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=500).aspreconditioner()
            self.base, self.builds = None, self.builds + 1
        return self.M

    def report(self, iterations):
        if self.base is None:
            self.base = max(iterations, 1)
        elif iterations > max(self.floor, self.grow * self.base):
            self.M = None                              # rebuild on the next system


class _MGFamily:
    """Matrix-free geometric multigrid CG (backend/mgcg_numba.MGCG) for a family A0 + diag(q): the
    hierarchy is built once from the face coefficients and only the diagonal changes per system."""

    def __init__(self, faces, dx, active, dirichlet, periodic, tol=1e-8):
        from .backend.mgcg_numba import MGCG
        self.shape, self.tol, self.iterations = active.shape, tol, 0
        # A little more smoothing saves expensive Krylov/transfer work on thin
        # trench tiles (tools/bench_tile3d.py --field-micro).
        self.mg = MGCG(*faces, dx, reaction=None, active=active, dirichlet=dirichlet,
                       periodic=periodic, pre=3, post=3)
        self.coarsest = int(np.prod(self.mg.hierarchy[-1].mask.shape))

    def system(self, q):
        self.iterations = 0                            # callers add .iterations after each system
        self.mg.set_reaction(np.asarray(q, dtype=np.float64).reshape(self.shape))
        return self

    def solve(self, b, x0=None):
        x, info = self.mg.solve(b, x0=x0, tol=self.tol, maxiter=300)
        self.iterations += info["iterations"]
        if not info["converged"]:
            raise RuntimeError(f"MGCG did not converge (rel. residual {info['rel_residual']:.2e})")
        return np.asarray(x, dtype=np.float64).ravel()


class _AdvLinear:
    """Non-symmetric Cu2+ system with advection. Direct with a shared cache:
    factor the first system, then use that LU as a GMRES preconditioner for
    changing sink diagonals. Without a cache, direct remains exact sparse LU.
    Otherwise GMRES preconditioned by
    the symmetric part (same faces and sink, no advection): one MGCG V-cycle (fam: _MGFamily) or the
    reused AMG hierarchy of A_sym + diag(q).  Fine while the cell Peclet number is O(1)."""

    # The symmetric-part preconditioner degrades as the cell Peclet number grows (slow, not failing:
    # ~1000 GMRES iterations per system on the wide validation trench).  The SG operator is an
    # M-matrix, so an incomplete LU of the full matrix is the robust preconditioner there; one factor
    # is reused across the systems of a field solve (only the sink diagonal changes) and rebuilt when
    # its iteration count grows.  Exact LU is the last resort.
    MG_RESTARTS = 4        # budget of the symmetric-preconditioned first attempt (x 60 iterations)
    SLOW_ITS = 150         # converged but slower than this -> later systems start with ILU
    LU_REFRESH_ITS = 20    # refresh a drifted exact factor; one restart bounds a stale attempt

    def __init__(self, A, q, direct, fam=None, A_sym=None, pc=None, tol=1e-8, start=None, ilu_cache=None):
        # start: None, "ilu" or "lu" - begin with a fallback that an earlier system already needed
        from scipy.sparse.linalg import LinearOperator
        self.A, self.tol, self.iterations, self.fallback = A, tol, 0, None
        self._cache = ilu_cache if ilu_cache is not None else {}
        if start == "lu" and not direct:
            direct, self.fallback = True, "lu"
        self._reuse_lu = direct and ilu_cache is not None
        self._lu = None
        self._M = None
        if self._reuse_lu:
            if "lu" not in self._cache:
                self._factor_lu()
            else:
                self._M = LinearOperator(A.shape, matvec=self._cache["lu"].solve, dtype=np.float64)
        elif direct:
            self._lu = splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A")
        if not direct and start == "ilu":
            self.fallback = "ilu"
        elif not direct:
            n = A.shape[0]
            if fam is not None:
                fam.system(q)
                self._M = LinearOperator((n, n), matvec=fam.mg.vcycle, dtype=np.float64)
            else:
                S = A_sym + sparse.diags(q)
                self._M = pc.get(S)

    def _factor_lu(self):
        if self.A.shape[0] > LU_FALLBACK_MAX:
            raise RuntimeError(f"Cu advection cached LU exceeds fallback limit (n={self.A.shape[0]})")
        self._lu = splu(self.A.tocsc(), permc_spec="MMD_AT_PLUS_A")
        self._cache["lu"] = self._lu
        self._cache["lu_builds"] = self._cache.get("lu_builds", 0) + 1

    def _solve_lu(self, b, x0):
        x, info, its = self._gmres(b, x0, self._M, 1)
        # SciPy checks the true residual too; keep an explicit gate here so
        # no stale-factor solve can weaken the requested inner tolerance.
        valid = np.isfinite(x).all() and np.linalg.norm(b-self.A@x) <= self.tol*np.linalg.norm(b)
        if info != 0 or not valid:
            # The original high-Peclet escape hatch remains exact LU of A.
            self._factor_lu()
            return self._lu.solve(b)
        if its > self.LU_REFRESH_ITS:
            self._cache.pop("lu", None)
        return x

    def _ilu(self, rebuild=False):
        from scipy.sparse.linalg import LinearOperator, spilu
        if rebuild or "M" not in self._cache:
            n = self.A.shape[0]
            ilu = spilu(self.A.tocsc(), drop_tol=1e-5, fill_factor=15)
            self._cache.update(M=LinearOperator((n, n), matvec=ilu.solve, dtype=np.float64), base=None)
            self._cache["builds"] = self._cache.get("builds", 0) + 1
        return self._cache["M"]

    def _gmres(self, b, x0, M, restarts):
        from scipy.sparse.linalg import gmres
        count = [0]
        x, info = gmres(self.A, b, x0=x0, rtol=self.tol, atol=0.0, M=M, restart=60, maxiter=restarts,
                        callback=lambda _: count.__setitem__(0, count[0] + 1), callback_type="pr_norm")
        self.iterations += count[0]
        return x, info, count[0]

    def _lu_or_raise(self, b, why):
        if self.A.shape[0] > LU_FALLBACK_MAX:
            raise RuntimeError(f"Cu advection solve: {why} (n={self.A.shape[0]}); cell Peclet number too "
                               "high for this grid?")
        self._lu, self.fallback = splu(self.A.tocsc(), permc_spec="MMD_AT_PLUS_A"), "lu"
        return self._lu.solve(b)

    def _solve_ilu(self, b, x0):
        try:
            x, info, its = self._gmres(b, x0, self._ilu(), 20)
            if info != 0:                         # a stale reused factor: rebuild once and retry
                x, info, its = self._gmres(b, x, self._ilu(rebuild=True), 20)
        except RuntimeError:                      # singular incomplete factor
            return self._lu_or_raise(b, "ILU factorization failed")
        if info != 0:
            return self._lu_or_raise(b, f"GMRES+ILU did not converge (info={info})")
        base = self._cache.get("base")
        if base is None:
            self._cache["base"] = its
        elif its > max(30, 3 * base):             # factor drifted from the current sink: refresh
            self._cache.pop("M", None)
        return x

    def solve(self, b, x0=None):
        if self._lu is not None:
            return self._lu.solve(b)
        if self._reuse_lu:
            return self._solve_lu(b, x0)
        if self.fallback == "ilu":
            return self._solve_ilu(b, x0)
        x, info, its = self._gmres(b, x0, self._M, self.MG_RESTARTS)
        if info == 0:
            if its > self.SLOW_ITS:
                self.fallback = "ilu"             # sticky: later systems skip the slow path
            return x
        self.fallback = "ilu"
        return self._solve_ilu(b, x)


class _Sticky:
    """Remember an _AdvLinear fallback on the FieldSolver so later systems (next root-find step, next
    time step) start with it, and report it in info["linear_cu"]."""

    def __init__(self, lin, owner, info):
        self.lin, self.owner, self.info = lin, owner, info
        if lin._reuse_lu and lin.fallback is None:
            info["linear_cu"] = "gmres+lu"

    @property
    def iterations(self):
        return self.lin.iterations

    def solve(self, b, x0=None):
        x = self.lin.solve(b, x0=x0)
        if self.lin.fallback and self.lin.fallback != self.owner._cu_fallback:
            self.owner._cu_fallback = self.lin.fallback
        if self.lin.fallback and not self.info["linear_cu"].endswith("->" + self.lin.fallback):
            self.info["linear_cu"] = self.info["linear_cu"].split("->")[0] + "->" + self.lin.fallback
        return x


def _mg_available():
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


class _Linear:
    """SPD family: auto-small reuses one LU as a CG preconditioner across
    changing diagonals; explicit direct stays exact. Large systems use AMG.
    A missing shared precond also keeps the original exact-direct behaviour.
    """

    def __init__(self, A, tol=1e-8, precond=None, mode="auto"):
        self.A, self.tol, self.iterations = A, tol, 0
        self._pc = precond if precond is not None else _Precond()
        direct = mode == "direct" or (mode != "amg" and A.shape[0] <= DIRECT_MAX)
        self._cached_direct = direct and mode != "direct" and precond is not None
        if self._cached_direct and self._pc.lu is not None:
            from scipy.sparse.linalg import LinearOperator
            self._lu = None
            self._M = LinearOperator(A.shape, matvec=self._pc.lu.solve, dtype=np.float64)
        elif direct:
            self._factor()
        else:
            self._lu, self._M = None, self._pc.get(A)
        if self._cached_direct and self._pc.info is not None:
            self._pc.info["linear"] = "cg+lu"

    def _factor(self):
        self._lu, self._M = splu(self.A.tocsc(), permc_spec="MMD_AT_PLUS_A"), None
        if self._cached_direct:
            self._pc.lu = self._lu

    def solve(self, b, x0=None):
        if self._lu is not None:
            return self._lu.solve(b)
        count = [0]
        x, info = cg(self.A, b, x0=x0, rtol=self.tol, atol=0.0, M=self._M,
                     maxiter=60 if self._cached_direct else 400,
                     callback=lambda _: count.__setitem__(0, count[0] + 1))
        self.iterations += count[0]
        if self._cached_direct:
            valid = np.isfinite(x).all() and np.linalg.norm(b-self.A@x) <= self.tol*np.linalg.norm(b)
            if info != 0 or not valid:
                self._factor()
                return self._lu.solve(b)
            if count[0] > 20:
                self._pc.lu = None
        else:
            self._pc.report(count[0])
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
        return x


def _faces(H, periodic, top_scale=None):
    """Arithmetic face means of the cell field H; outer faces 0 unless periodic.  top_scale (nx, ny)
    replaces the outer z_hi face (used for the Dirichlet plane at distance d_top)."""
    out = []
    for axis in range(3):
        v = np.moveaxis(H, axis, 0)
        f = np.zeros((v.shape[0] + 1, *v.shape[1:]))
        f[1:-1] = 0.5 * (v[:-1] + v[1:])
        if periodic[axis]:
            f[0] = f[-1] = 0.5 * (v[0] + v[-1])
        out.append(np.moveaxis(f, 0, axis).copy())
    if top_scale is not None:
        out[2][:, :, -1] = top_scale
    return out


def _broadcast_factor(value, shape, nz, band):
    """Per-cell prescribed kinetics factor on the solved block.

    ``value`` is broadcastable to ``shape = (nx, ny, kt + 1)``.  A 1-D array of length ``nz``
    (the full-grid z centres) or a full ``(nx, ny, nz)`` array is trimmed to the solved block, so a
    caller may build the factor from ``Grid3D.centers()[2]`` without knowing ``kt``.  Values outside
    the interface band ``band`` (``w > 0``) are set to 1: they multiply ``w = 0`` anyway, but the
    diagnostics stay well defined.  Returns a float64 array on the solved block.
    """
    a = np.asarray(value, dtype=np.float64)
    if a.ndim == 1 and a.shape[0] == nz:
        a = a[:shape[2]]
    elif a.ndim == 3 and a.shape[2] == nz:
        a = a[:, :, :shape[2]]
    try:
        a = np.broadcast_to(a, shape)
    except ValueError as exc:
        raise ValueError(f"kinetics_factor must broadcast to the solved block {shape}, got {a.shape}") from exc
    if not np.all(np.isfinite(a)):
        raise ValueError("kinetics_factor must be finite")
    if np.any(a <= 0.0):
        raise ValueError("kinetics_factor must be positive (it multiplies the exchange current)")
    return np.where(band, a, 1.0)


class FieldSolver:
    """L1 field solve.  Construct once per run, call solve() once per front position."""

    def __init__(self, params: Params | None = None, T_C: float = 22.0, linear: str = "auto", pulse=None):
        """linear: "auto" (sparse LU up to DIRECT_MAX unknowns, else matrix-free multigrid CG when numba
        is available, else AMG-preconditioned CG), "direct", "amg" or "mg".
        pulse: optional run_tile waveform dict. Resolved pulses use pulse_alpha_a
        (not the legacy 2-D alpha_a), shared i0, and optional C_dl_F_cm2.
        """
        if linear not in ("auto", "direct", "amg", "mg"):
            raise ValueError("linear must be auto, direct, amg or mg")
        self.p = Params(suppressor=False) if params is None else params
        self.ph = derive(self.p, T_C)
        self.T_C = T_C
        self.linear = linear
        self._cu_fallback = None   # "ilu" / "lu" once a high-Peclet Cu system needed it (sticky)
        self.pulse = pulse
        if not np.isfinite(self.p.C_dl_F_cm2) or self.p.C_dl_F_cm2 < 0:
            raise ValueError('C_dl_F_cm2 must be finite and nonnegative')
        if not np.isfinite(self.p.pulse_alpha_a) or self.p.pulse_alpha_a <= 0:
            raise ValueError('pulse_alpha_a must be finite and positive')

    def solve(self, psi, dx, z0, z_plane, j_net_A_cm2, periodic=(True, True, False), state=None,
              tol=2e-4, maxit=30, flow_faces=None, kinetics_factor=None, cov=None):
        """psi: (nx, ny, nz) level set in cm (positive in electrolyte); z0: z of the low domain face;
        z_plane: height of the bulk plane (cm); j_net_A_cm2: net current per projected area.
        state: dict returned by a previous call (warm start) or None.
        flow_faces: optional (ux, uy, uz) MAC face velocities on the full grid, cm/s, shaped (nx+1, ny, nz),
        (nx, ny+1, nz), (nx, ny, nz+1) (platesim.flow3d.solve_stokes3d(...)["faces"]); they advect Cu2+.
        kinetics_factor: optional prescribed multiplier on ``i0_u`` per cell, broadcastable to the
        solved (nx, ny, kt + 1) block or given on the full (nx, ny, nz) grid (see
        :func:`_broadcast_factor`).  It is an *effective suppression map* (uncalibrated): 0.85 below
        a depth stands in for a floor-depleted accelerator, for example.  None -> the legacy scalar.
        cov: optional coverage state (:func:`platesim.tile3d.additives3d.coverage_init`), updated
        in place with the quasi-steady Phase-1 theta_cl/theta_s of every interface cell; its
        suppressor factor multiplies the prescribed one.  None -> no additives.  ``additives='full'``
        (bulk transport, CEAC accelerator, leveller: plan phases 2-3) raises NotImplementedError.
        Returns dict(ct, phi, i, speed, V, w, active, kt, info, state, i0_factor) on the full grid;
        i is the interface current density (A/cm2) of each cell, speed = Mfac * i, and i0_factor is
        the effective exchange-current multiplier actually used (1 everywhere in the legacy path).
        In resolved mode i/speed/phi are cycle means; ct is the outer mean.
        info['pulse'] reports phase voltages, currents, diffusion length, charging
        times and cycle Faraday error. Off-time C*|delta eta|/j is undefined for
        applied j=0; differential RC relaxation time is reported separately.
        info['additives'] (cov only) reports theta ranges at the front, the coverage change of the
        last pass and the NDR/hysteresis flag.
        Charging is an area-mean periodic RC/BV circuit, not a full spatial
        transient potential solve. Its phase Faradaic charges feed the 3-D
        phase distributions, using BV for off-time relaxation near equilibrium.
        """
        p, ph = self.p, self.ph
        from .pulse import Waveform, surface_concentration, capacitive_cycle
        wave = (None if self.pulse is None or self.pulse.get('model') == 'net'
                else Waveform.parse(self.pulse, j_net_A_cm2))
        resolved = wave is not None and wave.model == 'resolved' and not wave.dc
        psi = np.asarray(psi, dtype=np.float64)
        nx, ny, nz = psi.shape
        zc = z0 + (np.arange(nz) + 0.5) * dx
        kt = int(np.floor((z_plane - 0.5 * dx - z0) / dx - 0.5))
        if kt < 2:
            raise ValueError("bulk plane is below the grid")
        if kt > nz - 1:
            raise ValueError("bulk plane is above the grid: enlarge the domain in z "
                             f"(needs z up to {z_plane * 1e4:.1f} um)")
        d_top = z_plane - zc[kt]
        sl = np.s_[:, :, :kt + 1]
        eps = p.eps_fac * dx
        x = np.clip(psi / eps, -30, 30)
        Hs_full = np.maximum(0.5 * (1 + np.tanh(x)), 1e-6)
        gm = np.sqrt(sum(((ls_numpy.shifted(psi, 1, ax, periodic)
                           - ls_numpy.shifted(psi, -1, ax, periodic)) / (2 * dx)) ** 2 for ax in range(3)))
        gHs = 0.5 / eps / np.cosh(x) ** 2 * gm
        Hs = Hs_full[sl]
        active = psi[sl] > -p.n_active_eps * eps
        w = np.where(active, gHs[sl], 0.0)
        if not w.any():
            raise ValueError("no plating front below the bulk plane")
        shape = Hs.shape
        top = Hs[:, :, -1] * dx / (2.0 * d_top)          # so that 2 k / dx^2 = k Hs / (dx d_top)
        hx, hy, hz = _faces(Hs, periodic, top_scale=top)
        per = tuple(bool(v) for v in periodic)

        def system(coef, value):
            return operators.assemble(coef * hx, coef * hy, coef * hz, dx, active=active,
                                      dirichlet={"z_hi": value}, periodic=per)

        A_c, b_c = system(ph.D_c, 1.0)
        A_p, _ = system(ph.kappa, 0.0)
        wf = w.ravel()
        act = active.ravel()
        It = j_net_A_cm2 * nx * ny / dx                 # target of sum(w * i) (volume-normalised)
        af = p.alpha_u * ph.f_RT
        sink = p.efficiency / (2.0 * F * p.c_bulk)

        # ---- per-cell exchange-current factor (additives_3d_plan.md Phase 1) -------------------
        # i0v[0] is the current per-cell i0 on the solved block, or None for the legacy scalar
        # ph.i0_u.  The None branch keeps every arithmetic expression below identical.
        fac = None if kinetics_factor is None else _broadcast_factor(kinetics_factor, shape, nz, w > 0)
        if cov is not None:
            if p.additives == "full":
                raise NotImplementedError(
                    "additives='full' has no 3-D implementation: bulk suppressor/leveller transport, "
                    "the CEAC accelerator and surface coverage transport are phases 2-3 of "
                    "the 3-D additives port.  Use additives='cl_ndr' (Phase 1).")
            if p.alpha_s != p.alpha_u:
                raise NotImplementedError(
                    "the 3-D additive coupling factorises i0 with one Tafel slope: alpha_s must equal "
                    f"alpha_u (got alpha_s={p.alpha_s}, alpha_u={p.alpha_u})")
            if not isinstance(cov, dict) or set(additives3d.COV_KEYS) - set(cov):
                raise ValueError(f"cov must be a dict with the keys {additives3d.COV_KEYS}")
            if np.shape(cov["theta_s"]) != psi.shape:
                raise ValueError(f"coverage arrays must have the psi shape {psi.shape}, "
                                 f"got {np.shape(cov['theta_s'])}")
            if p.additives == "legacy":
                raise ValueError("cov was supplied with additives='legacy'; set additives='cl_ndr' "
                                 "or pass cov=None")
        i0v = [None]

        def i0_cell_factor():
            """Fcov * Fac on the solved block (1 outside the interface band), raveled."""
            value = 1.0 if fac is None else fac
            if cov is not None:
                value = additives3d.coverage_factor(cov["theta_s"][sl], p) * value
            return np.where(active, value, 1.0).ravel()

        if fac is not None or cov is not None:
            i0v[0] = ph.i0_u * i0_cell_factor()

        def ci_of(eta):
            b = ph.i0_u if i0v[0] is None else i0v[0]
            return b * np.exp(np.clip(-af * eta, -60, 60))

        def V_fixed_phi(ct, phi):
            b = ph.i0_u if i0v[0] is None else i0v[0]
            s = np.dot(wf * ct, b * np.exp(np.clip(af * phi, -60, 60)))
            return -np.log(It / (s + 1e-300)) / af

        n = wf.size
        if state is not None and state.get("shape") == shape:
            ct, phi = state["ct"].copy(), state["phi"].copy()
        else:
            ct, phi = np.where(act, 1.0, 0.0), np.zeros(n)
        ct = np.where(act, np.clip(ct, 1e-6, None), 0.0)
        V = V_fixed_phi(ct, phi)
        info = dict(outer=0, newton=0, cg=0)
        pc_p, pc_c = _Precond(info=info), _Precond(info=info)  # one AMG hierarchy per family, reused
        use_mg = self.linear == "mg" or (self.linear == "auto" and n > DIRECT_MAX and _mg_available())
        mode = self.linear if self.linear in ("direct", "amg") else "auto"
        info["linear"] = "mgcg" if use_mg else (
            "direct" if mode == "direct" or (mode == "auto" and n <= DIRECT_MAX) else "amg")
        if use_mg:
            fam_p = _MGFamily((ph.kappa * hx, ph.kappa * hy, ph.kappa * hz), dx, active, {"z_hi": 0.0}, per)
            fam_c = _MGFamily((ph.D_c * hx, ph.D_c * hy, ph.D_c * hz), dx, active, {"z_hi": 1.0}, per)
            lin_p, lin_c_of = fam_p.system, fam_c.system
        else:
            def lin_p(q):
                return _Linear(A_p + sparse.diags(q), precond=pc_p, mode=mode)

            def lin_c_of(q):
                return _Linear(A_c + sparse.diags(q), precond=pc_c, mode=mode)
        if flow_faces is not None:
            # Cu2+ with advection: non-symmetric exponentially fitted operator.  Small systems: sparse LU;
            # large: GMRES preconditioned by the symmetric part (diffusion + the same sink), one MGCG V-cycle
            # (or the reused AMG hierarchy without numba).  The potential equation has no advection.
            fx, fy, fz = (np.asarray(f, dtype=np.float64) for f in flow_faces)
            slab = (fx[:, :, :kt + 1], fy[:, :, :kt + 1], fz[:, :, :kt + 2])
            A_c, b_c = operators.assemble_advdiff(ph.D_c * hx, ph.D_c * hy, ph.D_c * hz, dx, slab,
                                                  active=active, dirichlet={"z_hi": 1.0}, periodic=per)
            F_top = fz[:, :, kt + 1] / dx * active[:, :, -1]        # > 0 leaves through the bulk plane
            gmax = float(np.max([np.max(np.abs(f[:, :, :kt + 1])) for f in slab[:2]] + [np.max(np.abs(slab[2]))]))
            info["cell_peclet_max"] = gmax * dx / ph.D_c
            direct_c = self.linear == "direct" or (self.linear == "auto" and n <= DIRECT_MAX)
            info["linear_cu"] = "direct" if direct_c else ("gmres+mgcg" if use_mg else "gmres+amg")

            A_sym_c = system(ph.D_c, 1.0)[0] if not (use_mg or direct_c) else None
            start_c = self._cu_fallback or ("ilu" if info["cell_peclet_max"] > PE_ILU else None)
            ilu_cache = {}                       # one incomplete factor per field solve (A_c fixed)

            def lin_c_of(q):
                lin = _AdvLinear(A_c + sparse.diags(q), q, direct_c, fam_c if use_mg else None,
                                 A_sym=A_sym_c, pc=pc_c, start=start_c,
                                 ilu_cache=ilu_cache if self.linear != "direct" else None)
                return _Sticky(lin, self, info)
        else:
            F_top = None
        relax = 1.0
        last_change = np.inf
        cov_change = np.inf                    # Phase-1 coverage change of the last outer pass
        last_cov_change = np.inf
        cov_beta = p.cov_relax
        if cov is not None:
            maxit = max(int(maxit), 40)        # theta_s needs the extra passes (plan 2.3)
        for outer in range(maxit):
            # ---- potential and electrode potential V for fixed ct (bordered Newton, as in 2-D)
            for _ in range(14):
                eta = V - phi
                ci = ci_of(eta)
                I = ct * ci
                q = wf * af * I                          # d(w I)/d(phi) > 0
                R = A_p @ phi + wf * I
                g = np.dot(wf, I) - It
                lin = lin_p(q)
                x1, x2 = lin.solve(-R), lin.solve(q)
                info["cg"] += lin.iterations
                dV = (g + np.dot(q, x1)) / (q.sum() - np.dot(q, x2) + 1e-300)
                dphi = x1 + x2 * dV
                m = float(np.max(np.abs(dV - dphi)))
                lam = min(1.0, 0.15 / (m + 1e-300))
                phi, V = phi + lam * dphi, V + lam * dV
                info["newton"] += 1
                if lam == 1.0 and m < 2e-7:
                    break
            # ---- Cu2+ for fixed phi.  For a given electrode potential the problem is linear in ct (implicit
            # sink), and the total current is a monotone function of u = -af V, so V is found by a scalar
            # root-find with one linear solve per evaluation.  This stays stable up to the limiting current,
            # where alternating "ct at fixed eta" / "V at fixed ct" oscillates.
            b_i0 = ph.i0_u if i0v[0] is None else i0v[0]
            shape_k = wf * b_i0 * np.exp(np.clip(af * phi, -60, 60))     # w * ci / exp(u)
            guess = [ct]

            def total(u):
                lin_c = lin_c_of(sink * np.exp(u) * shape_k)
                c = np.where(act, np.clip(lin_c.solve(b_c, x0=guess[0]), 1e-12, None), 0.0)
                info["cg"] += lin_c.iterations
                guess[0] = c
                return np.log(np.dot(shape_k, c) + 1e-300) + u - np.log(It), c

            u1 = -af * V
            f1, new = total(u1)
            u0, f0 = u1 - 0.5 * np.sign(f1 if f1 != 0 else 1.0), None
            for _ in range(40):
                if abs(f1) < 1e-10:
                    break
                if f0 is None:
                    f0, _ = total(u0)
                slope = (f1 - f0) / (u1 - u0) if u1 != u0 else 1.0
                slope = slope if slope > 1e-6 else 1e-6        # f is increasing in u; flat near the limit
                step = float(np.clip(-f1 / slope, -4.0, 4.0))
                u0, f0 = u1, f1
                u1 = u1 + step
                f1, new = total(u1)
            V = -u1 / af
            info["limited"] = bool(abs(f1) > 1e-6)       # set current above the transport limit
            ct_change = float(np.max(np.abs(new - ct)))
            change = ct_change
            last_change = change
            ct = new
            if cov is not None:
                # ---- Phase-1 quasi-steady coverages of the current iterate (plan 2.2-2.3) -----
                # v = Mfac * i_local, cathodic positive (cycle mean in a resolved pulse).  Only the
                # front band is updated: theta lives on the interface, everything else keeps theta0.
                i_loc = ct * ci_of(V - phi)
                th_cl_ss, th_s_ss = additives3d.steady_state(np.maximum(i_loc, 0.0), p, ph.Mfac)
                front = wf > 0.0
                # Convergence is measured where the interface area actually is: the tanh band's
                # outer tails carry < 1e-3 of w and < 0.1 % of the current, but their theta creeps
                # towards 1 on the slow (burial-free) branch and would otherwise mask convergence.
                crit = wf > 1e-3 * wf.max()
                th_cl_old = cov["theta_cl"][sl].ravel()
                th_s_old = cov["theta_s"][sl].ravel()
                th_cl_new = np.where(front, th_cl_old + cov_beta * (th_cl_ss - th_cl_old), th_cl_old)
                th_s_new = np.where(front, th_s_old + cov_beta * (th_s_ss - th_s_old), th_s_old)
                cov_change = float(max(np.max(np.abs((th_cl_new - th_cl_old)[crit])),
                                       np.max(np.abs((th_s_new - th_s_old)[crit]))))
                cov["theta_cl"][sl] = th_cl_new.reshape(shape)
                cov["theta_s"][sl] = th_s_new.reshape(shape)
                if cov_change > last_cov_change * 1.2:      # pulse-branch heuristic (TP/fields.py:528)
                    cov_beta = max(0.2, cov_beta * 0.5)
                last_cov_change = cov_change
                i0v[0] = ph.i0_u * i0_cell_factor()
                change = max(change, cov_change)
            info["outer"] = outer + 1
            if change < tol:
                break
        info["amg_builds"] = pc_p.builds + pc_c.builds
        info["converged"] = bool(change < tol)
        info["ct_change"] = ct_change
        if cov is not None:
            front_info = wf > 1e-3 * wf.max()               # same measure as the convergence test
            th_s = cov["theta_s"][sl].ravel()[front_info]
            th_cl = cov["theta_cl"][sl].ravel()[front_info]
            info["additives"] = dict(
                model="cl_ndr Phase 1: quasi-steady theta_cl/theta_s, no bulk additive transport",
                cov_change=float(cov_change), relax=float(cov_beta),
                converged=bool(cov_change < tol),
                # A limit cycle of the S-NDR Picard map is reported, not hidden (plan 2.3 keeps the
                # monotone-from-below branch; no separate bistability search is performed).
                ndr_hysteresis=bool(cov_change >= tol),
                theta_s=(float(np.min(th_s)), float(np.max(th_s)), float(np.mean(th_s))) if th_s.size else None,
                theta_cl=(float(np.min(th_cl)), float(np.max(th_cl)), float(np.mean(th_cl)))
                if th_cl.size else None,
                uncalibrated=True,
                note="effective suppression map; constants are the 2-D engine defaults [J]")
        I = ct * ci_of(V - phi)
        if resolved:
            # The outer Cu solve sees the cycle-mean sink, never the instantaneous
            # forward current. Its fixed point is coupled to three frozen-front
            # potential distributions and a periodic, independent-normal diffusion layer.
            weights = wave.weights
            targets = np.array(wave.currents) * nx * ny / dx
            phase_i = np.array([I * v / j_net_A_cm2 for v in wave.currents])
            phase_i[weights == 0] = 0.
            phase_phi, phase_V = phi.copy(), V
            eta_mean = np.zeros(3)
            area_ratio = wf.sum() * dx / (nx * ny)
            pulse_change = np.inf
            pulse_relax = 1.

            def phase_solve(concentration, target, relaxation=False):
                nonlocal phase_phi, phase_V
                bv = relaxation and p.C_dl_F_cm2 > 0
                if abs(target) < 1e-30 and not bv:
                    phase_phi, phase_V = np.zeros(n), 0.
                    return np.zeros(n), 0., 0.
                sign = float(np.sign(target))
                a = (p.alpha_u if sign > 0 else p.pulse_alpha_a) * ph.f_RT
                # Shared exchange current; anodic Cu activity is one (solid Cu),
                # not the electrolyte Cu2+ concentration used by the cathodic branch.
                # The per-cell additive/prescribed factor multiplies both branches (the same
                # surface sites gate the cathodic and the anodic exchange).
                b0 = ph.i0_u if i0v[0] is None else i0v[0]
                pref = b0 * (concentration if sign > 0 else np.ones(n))
                pref = np.where(act, pref, 0.)
                def kinetics(eta):
                    if bv:
                        cath = b0 * concentration * np.exp(np.clip(-af * eta, -60, 60))
                        aa = p.pulse_alpha_a * ph.f_RT
                        anod = b0 * np.exp(np.clip(aa * eta, -60, 60))
                        return np.where(act, cath - anod, 0.), np.where(act, af*cath + aa*anod, 0.)
                    current = sign * pref * np.exp(np.clip(-sign * a * eta, -60, 60))
                    return current, a * np.abs(current)
                if bv:
                    from scipy.optimize import brentq
                    phase_V = brentq(lambda v: np.dot(wf, kinetics(v - phase_phi)[0]) - target, -3., 3.)
                else:
                    phase_V = -sign / a * np.log(abs(target) / max(
                        np.dot(wf, pref * np.exp(np.clip(sign * a * phase_phi, -60, 60))), 1e-300))
                for iteration in range(60):
                    current, derivative = kinetics(phase_V - phase_phi)
                    q = wf * derivative
                    R = A_p @ phase_phi + wf * current
                    g = np.dot(wf, current) - target
                    lin = lin_p(q)
                    x1, x2 = lin.solve(-R), lin.solve(q)
                    info['cg'] += lin.iterations
                    dV = (g + np.dot(q, x1)) / (q.sum() - np.dot(q, x2) + 1e-300)
                    dphi = x1 + x2 * dV
                    m = float(np.max(np.abs(dV - dphi)))
                    lam = min(1., .15 / (m + 1e-300))
                    phase_phi += lam * dphi
                    phase_V += lam * dV
                    info['newton'] += 1
                    if m < 2e-8:
                        break
                else:
                    raise RuntimeError('pulse potential Newton iteration did not converge')
                current = kinetics(phase_V - phase_phi)[0]
                return current, float(phase_V), float(np.dot(wf, phase_V - phase_phi) / wf.sum())

            def i0_area():
                """Area-mean exchange current of the front, for the lumped RC circuit (it is scalar).

                Exactly ``ph.i0_u`` in the legacy path so the arithmetic there is unchanged.
                """
                if i0v[0] is None:
                    return ph.i0_u
                return float(np.dot(wf, i0v[0]) / wf.sum())

            for coupling in range(40):
                cs = surface_concentration(ct, phase_i, wave.durations, ph.D_c, p.c_bulk, p.efficiency)
                front = wf > 1e-6 * wf.max()
                if np.min(cs[:, front][weights > 0]) <= 0:
                    raise ValueError('pulse diffusion layer depleted: reduce phase current or period')
                cs = np.maximum(cs, 1e-12)
                old_i = phase_i.copy()
                volts = np.zeros(3)
                mean_phi = np.zeros(n)
                effective = targets.copy()
                if p.C_dl_F_cm2:
                    effective_j, charging = capacitive_cycle(wave.durations, wave.currents,
                        p.C_dl_F_cm2, i0_area(), af, p.pulse_alpha_a * ph.f_RT,
                        cs @ wf / wf.sum(), area_ratio)
                    effective = effective_j * nx * ny / dx
                for k, dt_phase in enumerate(wave.durations):
                    if dt_phase == 0:
                        continue
                    phase_i[k], volts[k], eta_mean[k] = phase_solve(cs[k], effective[k], relaxation=k == 2)
                    mean_phi += weights[k] * phase_phi
                Iavg = weights @ phase_i
                # Implicit positive sink, explicit local dissolution source.
                q_c = sink * wf * np.maximum(Iavg, 0.) / np.maximum(ct, 1e-12)
                lin = lin_c_of(q_c)
                new_ct = lin.solve(b_c - sink * wf * np.minimum(Iavg, 0.), x0=ct)
                info['cg'] += lin.iterations
                new_ct = np.where(act, new_ct, 0.)
                previous_change = pulse_change
                pulse_change = max(float(np.max(np.abs(new_ct - ct))),
                    float(np.max(np.abs(phase_i - old_i))) / max(np.max(np.abs(phase_i)), 1e-12))
                if pulse_change > previous_change * 1.2:
                    pulse_relax = max(.2, pulse_relax * .5)
                ct += pulse_relax * (new_ct - ct)
                if pulse_change < min(tol, 2e-5):
                    break
            I = weights @ phase_i
            phi, V = mean_phi, float(np.dot(weights, volts))
            if not p.C_dl_F_cm2:
                _, charging = capacitive_cycle(wave.durations, wave.currents, 0.,
                    i0_area(), af, p.pulse_alpha_a * ph.f_RT, cs @ wf / wf.sum(), area_ratio)
            info['pulse'] = dict(model='resolved', coupling_iterations=coupling + 1,
                converged=bool(pulse_change < min(tol, 2e-5)), change=pulse_change,
                phase_V=volts.tolist(), phase_eta_V=eta_mean.tolist(),
                phase_current_A_cm2=(phase_i @ wf * dx / (nx * ny)).tolist(),
                diffusion_length_um=float(np.sqrt(ph.D_c * sum(wave.durations)) * 1e4),
                faraday_error_rel=float(np.dot(wf, I) / It - 1.),
                alpha_a=p.pulse_alpha_a, kinetics='cathodic/anodic Tafel; i0 shared; uncalibrated',
                alpha_a_reference='literature fitting range 1.0-1.5 (Cu/acid sulfate); uncalibrated')
            info['pulse'].update(charging)
            if p.C_dl_F_cm2:
                info['pulse']['kinetics'] = 'Tafel forward/reverse; BV lumped RC and off-time relaxation; i0 shared'
            info['pulse']['tau_off_note'] = 'No finite C*|delta eta|/j when applied j=0; null denotes undefined.'
            info['pulse']['phase_surface_ct_min'] = [float(np.min(cs[k, front]))
                if weights[k] else None for k in range(3)]
            info['converged'] = info['pulse']['converged']
            info['amg_builds'] = pc_p.builds + pc_c.builds
            info['current_error'] = info['pulse']['faraday_error_rel']
        elif wave is not None and wave.model == 'resolved':
            # A periodic DC waveform has no transitions or oscillatory diffusion.
            # Keep every field bit-identical while still reporting charging data.
            info['pulse'] = dict(model='resolved', dc_limit=True, converged=info['converged'],
                phase_current_A_cm2=[j_net_A_cm2, 0., 0.], phase_V=[float(V), None, None],
                tau_s=[0., None, None], C_dl_report_F_cm2=p.C_dl_F_cm2 or 30e-6,
                capacitive_charge_C_cm2=[0., 0., 0.],
                faraday_error_rel=float(np.dot(wf, I) / It - 1.), alpha_a=p.pulse_alpha_a)
        info["current_error"] = float(np.dot(wf, I) / It - 1.0)
        # Cu2+ entering through the bulk plane vs the deposition sink (both mol/s per tile)
        ct3 = ct.reshape(shape)
        influx = float(np.sum(ph.D_c * Hs[:, :, -1] / (dx * d_top) * (1.0 - ct3[:, :, -1])
                              * active[:, :, -1]) * p.c_bulk * dx ** 3)
        if F_top is not None:                            # upwinded advective flux through the bulk plane
            adv = -np.where(F_top < 0, F_top * 1.0, F_top * ct3[:, :, -1])
            influx += float(np.sum(adv) * p.c_bulk * dx ** 3)
        need = j_net_A_cm2 * nx * ny * dx * dx * p.efficiency / (2.0 * F)
        info["cu_balance_rel"] = influx / need - 1.0
        info["ct_min_front"] = float(np.min(ct[wf > 0.25 * wf.max()]))

        def full(a, fill):
            out = np.full(psi.shape, fill, dtype=np.float64)
            out[sl] = a.reshape(shape)
            return out

        i_full = full(np.where(act, I, 0.0), 0.0)
        # Effective exchange-current map actually used: Fcov * prescribed factor, 1 in the bulk and
        # outside the front band.  Exactly 1 everywhere in the legacy path.
        i0_factor = full(np.ones(n) if i0v[0] is None else i0v[0] / ph.i0_u, 1.0)
        return dict(ct=full(np.where(act, ct, 0.0), 1.0), phi=full(phi, 0.0), i=i_full,
                    speed=ph.Mfac * i_full, V=float(V), w=full(wf, 0.0), active=full(act, False).astype(bool),
                    kt=kt, info=info, state=dict(shape=shape, ct=ct, phi=phi),
                    i0_factor=i0_factor)
