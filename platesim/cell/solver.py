r"""Pseudo-tertiary current-distribution solver for the beaker coupon cell.

Secondary current distribution (Laplace in the electrolyte) with a Nernst-layer
mass-transfer correction on the cathode:

    div(kappa grad phi) = 0            in the electrolyte
    kappa dphi/dn = -j_c               on cathode facets (n = out of electrolyte)
    kappa dphi/dn = +j_a               on anode facets
    insulating walls (natural BC) elsewhere

Butler-Volmer fluxes (eta = phi_electrode - phi_solution):

    cathode:  j = j0 (1 - j/jlim) exp(-ac f eta) - j0 exp(aa f eta)
    anode:    j = j0 [exp(aa f eta) - exp(-ac f eta)]

The cathode equation is implicit in j; it has the closed form

    j = j0 (Ec - Ea) / (1 + j0 Ec / jlim),   Ec = exp(-ac f eta), Ea = exp(aa f eta)

Galvanostatic: total cathode current is pinned to I_total by keeping the cathode
potential as an unknown (gauge phi_cathode = 0, solve phi_anode = V_cell) with the
extra Newton row  \int_cathode j dA = I_total.  Charge neutrality then gives
\int_anode j dA = I_total automatically.

Optional thin-seed sheet resistance (terminal effect), OFF by default
--------------------------------------------------------------------
The base model treats the coupon metal as equipotential (phi_cathode = 0).  With
`seed_R_sq = R_s` (ohm/sq) the PVD Cu seed on the coupon front face carries an
in-plane metal potential V_m(s) obeying the thin-film (sheet) equation

    div_s( (1/R_s) grad_s V_m ) = -j(s),      V_m = 0 on the clip edge,
                                              dV_m/dn = 0 on the other edges

where j is the local cathodic current density (A/cm2, positive into the metal).
SIGN CONVENTION (this is the easy place to get it wrong): the cathodic current
enters the film over the whole face and leaves through the clip, so V_m rises
away from the clip and is non-negative; for uniform j0 on a strip of length L
fed at one end, V_m(x) = R_s j0 x (2L - x) / 2, i.e. V_m(L) = R_s j0 L^2 / 2 > 0.
The local cathode overpotential therefore becomes

    eta_c = V_m(s) - phi_l          (base model: eta_c = 0 - phi_l, V_m = 0)

so the far-from-clip region is less cathodic and plates thinner, as observed
early in plating.  V_m is a nodal offset on the coupon front nodes only; it is
solved by a fixed-point loop around the unchanged Newton solve (inner solve ->
j on the front face -> 2-D sheet Poisson on a structured grid -> relax -> repeat).
With seed_R_sq = None no offset is applied and the result is bit-identical to the
equipotential model.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from skfem import BilinearForm, CellBasis, ElementTetP1, FacetBasis, Functional, LinearForm, asm

from ..core.bath import Bath
from ..core.cell import CellGeometry
from ..core.constants import F, thickness_um
from .mesh import COUPON_T_CM, CellMesh, build_cell_mesh, build_duct_mesh

GRID_N = 41                     # output j(x, y) grid resolution [spec]
JLIM_UNLIMITED = 1e30           # "no mass-transfer limit" sentinel (must stay finite:
                                # skfem field interpolation does 0 * x, and 0 * inf = nan)
_EXP_CLIP = 40.0                # exponent guard for BV (e^40 ~ 2e17, plenty of headroom)

ASSUMPTIONS = [
    "beaker 9 cm ID x 8 cm fill, gap 4 cm, anode 5x5 cm, coupon centred at z = 4 cm: "
    "cell defaults are guesses until the team measures the cell [J, CellGeometry doc]",
    "coupon front face lies in the plane through the beaker axis (centre on axis), "
    "anode parallel at x = gap; anode wider than the wall chord is clipped by the beaker [J]",
    "electrodes are thin plates (Si 0.05 cm) at uniform potential; only the facing anode "
    "face is active, anode back/sides insulating [J]",
    "taped = True: only the coupon front face is active; taped = False: back + 4 sidewalls "
    "also plate (stray_area adds one lumped patch above the top edge, front face only) [J]",
    "baffles, stir bar and clip hardware omitted (insulating / not measured) [J]",
    "isothermal bath, steady state, homogeneous kappa; convection enters only through the "
    "Nernst layer delta (constant delta_um unless a delta_map is supplied) [J]",
    "H2/O2 evolution neglected; Faraday efficiency applied a-posteriori via `efficiency` [J]",
    "V_cell decomposed as V_ohm + eta_c + eta_a using area-weighted mean overpotentials [J]",
]


# ---------------------------------------------------------------- Butler-Volmer (vectorised)

def _cath_j(eta, jlim, i0, ac, aa, f):
    """Net cathodic current density A/cm2 (positive = deposition), Nernst-layer corrected."""
    Ec = np.exp(np.clip(-ac * f * eta, -_EXP_CLIP, _EXP_CLIP))
    Ea = np.exp(np.clip(aa * f * eta, -_EXP_CLIP, _EXP_CLIP))
    num = i0 * (Ec - Ea)
    den = 1.0 + i0 * Ec / jlim          # jlim = inf -> 0 (no mass-transfer limitation)
    return num / den


def _cath_dj(eta, jlim, i0, ac, aa, f):
    """d j_cath / d eta  (negative: cathodic current grows as eta becomes more negative)."""
    Ec = np.exp(np.clip(-ac * f * eta, -_EXP_CLIP, _EXP_CLIP))
    Ea = np.exp(np.clip(aa * f * eta, -_EXP_CLIP, _EXP_CLIP))
    num = i0 * (Ec - Ea)
    dnum = i0 * (-ac * f * Ec - aa * f * Ea)
    den = 1.0 + i0 * Ec / jlim
    dden = i0 * (-ac * f * Ec) / jlim
    return (dnum * den - num * dden) / den ** 2


def _anode_j(eta, i0, ac, aa, f):
    """Anodic Cu dissolution current density A/cm2 (positive = into electrolyte)."""
    Ea = np.exp(np.clip(aa * f * eta, -_EXP_CLIP, _EXP_CLIP))
    Ec = np.exp(np.clip(-ac * f * eta, -_EXP_CLIP, _EXP_CLIP))
    return i0 * (Ea - Ec)


def _anode_dj(eta, i0, ac, aa, f):
    Ea = np.exp(np.clip(aa * f * eta, -_EXP_CLIP, _EXP_CLIP))
    Ec = np.exp(np.clip(-ac * f * eta, -_EXP_CLIP, _EXP_CLIP))
    return i0 * (aa * f * Ea + ac * f * Ec)


# ------------------------------------------------------------------------- forms

@BilinearForm
def _lap(u, v, w):
    from skfem.helpers import dot, grad
    return dot(grad(u), grad(v))


@LinearForm
def _cath_flux(v, w):
    return _cath_j(w.eta, w.jlim, w.i0, w.ac, w.aa, w.fnum) * v


@BilinearForm
def _cath_mass(u, v, w):
    return _cath_dj(w.eta, w.jlim, w.i0, w.ac, w.aa, w.fnum) * u * v


@LinearForm
def _anode_flux(v, w):
    return _anode_j(w.eta, w.i0, w.ac, w.aa, w.fnum) * v


@BilinearForm
def _anode_mass(u, v, w):
    return _anode_dj(w.eta, w.i0, w.ac, w.aa, w.fnum) * u * v


@Functional
def _area(w):
    return 0.0 * w.x[0] + 1.0


@Functional
def _mean_eta(w):
    return w.eta


# --------------------------------------------------------------------- Newton core

def _v_hint(I: float, gap: float, kappa: float, area: float, i0: float,
            ac: float, aa: float, f: float) -> float:
    """Initial cell-voltage guess: ohmic + Tafel-ish kinetic terms [J]."""
    j = max(I / area, 1e-9)
    v_ohm = I * gap / (kappa * area) if kappa > 0 else 1.0
    if j > i0:
        eta_c = np.log(j / i0) / (ac * f)
        eta_a = np.log(j / i0) / (aa * f)
    else:
        eta_c = j / (i0 * ac * f)
        eta_a = j / (i0 * aa * f)
    return float(max(v_ohm + eta_c + eta_a, 0.01))


def _solve_core(cm: CellMesh, kappa: float, I_total: float, i0: float,
                ac: float, aa: float, f: float, jlim: np.ndarray, gap: float,
                taped: bool = True, on_iter=None, tol: float = 1e-7,
                max_iter: int = 60, vm: np.ndarray | None = None,
                phi0: tuple[np.ndarray, float] | None = None) -> SimpleNamespace:
    """Newton solve of the augmented (phi_l, V_cell) system. phi_cathode fixed at 0.

    `vm` (optional, nodal, V) is the seed metal potential on the coupon front face:
    it replaces the cathode overpotential eta_c = -phi_l by eta_c = vm - phi_l
    (vm is zero off the front face).  `vm=None` reproduces the equipotential model
    bit-for-bit.  d eta_c / d phi_l is still -1, so the Jacobian is unchanged.
    `phi0=(phi, phi_a)` warm-starts the Newton iteration.
    """
    if I_total <= 0:
        raise ValueError("I_total must be positive")
    m = cm.mesh
    elem = ElementTetP1()
    basis = CellBasis(m, elem)
    N = basis.N
    bnd = m.boundary_facets()
    g = cm.groups
    # taped: only the coupon front face plates; the wet-clip stray patch is always active
    cath_mask = g["cath_front"] | g["stray"]
    if not taped:
        cath_mask = cath_mask | g["cath_back"] | g["cath_edge"]
    if not np.any(cath_mask):
        raise RuntimeError("no cathode facets found - geometry/mesh classification failed")
    if not np.any(g["anode"]):
        raise RuntimeError("no anode facets found - geometry/mesh classification failed")
    fb_c = FacetBasis(m, elem, facets=bnd[cath_mask], intorder=4)
    fb_f = FacetBasis(m, elem, facets=bnd[g["cath_front"]], intorder=4)
    fb_a = FacetBasis(m, elem, facets=bnd[g["anode"]], intorder=4)
    A_cath = asm(_area, fb_c)
    A_front = asm(_area, fb_f)
    A_anode = asm(_area, fb_a)

    K = kappa * asm(_lap, basis)
    e = np.ones(N)
    kw = dict(i0=i0, ac=ac, aa=aa, fnum=f)

    x0 = np.clip(m.p[0], 0.0, gap)
    phi = (_v_hint(I_total, gap, kappa, A_cath, i0, ac, aa, f)) * x0 / max(gap, 1e-12)
    phi_a = float(_v_hint(I_total, gap, kappa, A_cath, i0, ac, aa, f))
    if phi0 is not None:
        phi, phi_a = np.array(phi0[0], dtype=float), float(phi0[1])

    if vm is None:
        def _eta_c(phi_v):
            return -phi_v
    else:
        vm_v = np.asarray(vm, dtype=float)

        def _eta_c(phi_v):
            return vm_v - phi_v           # eta_c = V_m - phi_l on the front face

    def residual(phi_v, pa_v):
        eta_c = _eta_c(phi_v)
        eta_a = pa_v - phi_v
        Fc = asm(_cath_flux, fb_c, eta=eta_c, jlim=jlim, **kw)
        Fa = asm(_anode_flux, fb_a, eta=eta_a, **kw)
        Rl = K @ phi_v + Fc - Fa
        Rc = float(Fc.sum() - I_total)
        return Rl, Rc, Fc, Fa

    scale = max(I_total, 1e-30)
    Rl, Rc, _, _ = residual(phi, phi_a)
    Rn = max(np.abs(Rl).max(), abs(Rc)) / scale
    n_iter = 0
    converged = False
    for it in range(1, max_iter + 1):
        n_iter = it
        if on_iter is not None:
            on_iter(it, Rn)
        if Rn < tol:
            converged = True
            break
        eta_c = _eta_c(phi)
        eta_a = phi_a - phi
        Gc = asm(_cath_mass, fb_c, eta=eta_c, jlim=jlim, **kw)
        Ga = asm(_anode_mass, fb_a, eta=eta_a, **kw)
        sc = Gc @ e                      # d(constraint)/dphi_l = -sc, see module doc
        sa = Ga @ e                      # d(-F_a)/dphi_a = -sa
        J = bmat([[K - Gc + Ga, -sa[:, None]],
                  [-sc[None, :], None]], format="csc")
        rhs = -np.concatenate([Rl, [Rc]])
        d = spsolve(J, rhs)
        if not np.all(np.isfinite(d)):
            raise RuntimeError("Newton Jacobian solve produced non-finite steps")
        dphi, dpa = d[:-1], d[-1]
        step_max = max(np.abs(dphi).max(), abs(dpa))
        if step_max > 0.25:              # cap per-iteration potential move (BV safety)
            dphi = dphi * (0.25 / step_max)
            dpa = dpa * (0.25 / step_max)
        alpha, accepted = 1.0, False
        while alpha >= 1.0 / 64.0:
            phi_t = phi + alpha * dphi
            pa_t = phi_a + alpha * dpa
            Rl_t, Rc_t, _, _ = residual(phi_t, pa_t)
            Rn_t = max(np.abs(Rl_t).max(), abs(Rc_t)) / scale
            if np.isfinite(Rn_t) and Rn_t < Rn:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            raise RuntimeError(f"Newton line search failed at iteration {it} (residual {Rn:.3e})")
        phi, phi_a, Rn = phi_t, pa_t, Rn_t
        Rl, Rc = Rl_t, Rc_t          # keep the right-hand side in sync with the state
    if not converged and Rn >= tol:
        raise RuntimeError(f"Newton did not converge in {max_iter} iterations (residual {Rn:.3e})")

    # ---- converged state
    eta_c = _eta_c(phi)
    eta_a = phi_a - phi
    Fc = asm(_cath_flux, fb_c, eta=eta_c, jlim=jlim, **kw)
    Fa = asm(_anode_flux, fb_a, eta=eta_a, **kw)
    I_front = float(asm(_cath_flux, fb_f, eta=eta_c, jlim=jlim, **kw).sum())
    jc_nodes = _cath_j(eta_c, jlim, i0, ac, aa, f)          # A/cm2 at every node
    eta_bar_c = float(asm(_mean_eta, fb_c, eta=eta_c) / A_cath)
    eta_bar_a = float(asm(_mean_eta, fb_a, eta=eta_a) / A_anode)
    jlim_cath_mean = _nodal_mean(cm, jlim, cath_mask)
    return SimpleNamespace(
        phi=phi, phi_a=phi_a, jc_nodes=jc_nodes, eta_c=eta_c, eta_a=eta_a,
        I_cath=float(Fc.sum()), I_anode=-float(Fa.sum()), I_front=I_front,
        A_cath=A_cath, A_front=A_front, A_anode=A_anode,
        eta_bar_c=eta_bar_c, eta_bar_a=eta_bar_a,
        jlim_cath_mean=jlim_cath_mean, n_iter=n_iter, residual=Rn,
        basis=basis, fb_c=fb_c, fb_f=fb_f, fb_a=fb_a, bnd=bnd,
    )


def _nodal_mean(cm: CellMesh, jlim: np.ndarray, cath_mask: np.ndarray) -> float:
    """Area-proxy (unweighted) mean of nodal jlim over cathode facets' nodes."""
    m = cm.mesh
    nodes = np.unique(m.facets[:, cm.mesh.boundary_facets()[cath_mask]])
    vals = jlim[nodes]
    return float(np.mean(vals))


# ------------------------------------------------------------------- grid + metrics

def _grid_interp(xy_nodes: np.ndarray, values: np.ndarray,
                 gx: np.ndarray, gz: np.ndarray) -> np.ndarray:
    """Interpolate nodal values over a face onto a regular (len(gz), len(gx)) grid.

    Uses a Delaunay triangulation of the face's node set (scipy, already a project
    dependency); points outside the hull fall back to the nearest node value.
    """
    from scipy.spatial import Delaunay

    uu, zz = np.meshgrid(gx, gz)
    pts = np.column_stack([uu.ravel(), zz.ravel()])
    deln = Delaunay(xy_nodes)
    ti = deln.find_simplex(pts)
    out = np.empty(len(pts))
    ok = ti >= 0
    if np.any(ok):
        simp = deln.simplices[ti[ok]]
        T = deln.transform[ti[ok]]                          # (m, ndim+1, ndim)
        b = np.einsum("mij,mj->mi", T[:, :2, :], pts[ok] - T[:, 2, :])
        l1, l2 = b[:, 0], b[:, 1]
        l0 = 1.0 - l1 - l2
        out[ok] = (l0 * values[simp[:, 0]] + l1 * values[simp[:, 1]]
                   + l2 * values[simp[:, 2]])
    if np.any(~ok):                    # points outside triangulation (boundary): nearest node
        d2 = ((xy_nodes[None, :, 0] - pts[~ok, None, 0]) ** 2
              + (xy_nodes[None, :, 1] - pts[~ok, None, 1]) ** 2)
        out[~ok] = values[np.argmin(d2, axis=1)]
    return out.reshape(uu.shape)


def _uniformity(j: np.ndarray, half_w: float, half_h: float) -> dict:
    """Edge/centre, spread and corner L/R metrics on the regular j grid (mA/cm2)."""
    nz, nx = j.shape
    gx = np.linspace(-half_w, half_w, nx)
    gz = np.linspace(-half_h, half_h, nz)
    su = np.abs(gx) / half_w
    sv = np.abs(gz) / half_h
    SU, SV = np.meshgrid(su, sv)
    edge = (SU >= 0.85) | (SV >= 0.85)
    centre = (SU <= 0.40) & (SV <= 0.40)
    j_mean = float(j.mean())
    j_edge = float(j[edge].mean()) if np.any(edge) else np.nan
    j_centre = float(j[centre].mean()) if np.any(centre) else np.nan
    corners = (SU >= 0.80) & (SV >= 0.80)
    UU, _ = np.meshgrid(gx, gz)
    left = corners & (UU < 0)
    right = corners & (UU > 0)
    j_left = float(j[left].mean()) if np.any(left) else np.nan
    j_right = float(j[right].mean()) if np.any(right) else np.nan
    return {
        "j_mean": j_mean,
        "j_max": float(j.max()),
        "j_min": float(j.min()),
        "std_over_mean": float(j.std() / j_mean) if j_mean else float("nan"),
        "edge_center_ratio": float(j_edge / j_centre) if j_centre else float("nan"),
        "corner_LR_ratio": float(j_left / j_right) if j_right else float("nan"),
        "corner_LR_grad_pct": float(100.0 * (j_left - j_right) / j_mean) if j_mean else float("nan"),
    }


# --------------------------------------------------- seed sheet resistance (terminal effect)

_CLIP_EDGES = ("top", "bottom", "left", "right")


def _sheet_poisson(j_grid: np.ndarray, R_sq: float, hx: float, hz: float,
                   clip_edge: str = "top") -> np.ndarray:
    """Metal potential V_m(s) on the coupon front-face grid, in volts.

    Solves the thin-film equation  div_s((1/R_s) grad_s V_m) = -j  on the
    rectangle with V_m = 0 on the clip edge and zero flux on the other three
    edges (finite differences on the structured (nz, nx) grid, direct sparse
    solve; `j_grid` is the cathodic current density into the metal, A/cm2).

    Sign: current enters the film everywhere and leaves through the clip, so
    V_m >= 0 and grows with distance from the clip (max at the far edge/corner).
    """
    if clip_edge not in _CLIP_EDGES:
        raise ValueError(f"clip_edge must be one of {_CLIP_EDGES}, got {clip_edge!r}")
    nz, nx = j_grid.shape
    n = nz * nx
    ihx2, ihz2 = 1.0 / (hx * hx), 1.0 / (hz * hz)
    clip_x = clip_edge in ("left", "right")
    clip_z = clip_edge in ("top", "bottom")
    clip_lo = clip_edge in ("bottom", "left")
    rhs = -R_sq * np.maximum(j_grid, 0.0).ravel().copy()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def idx(i, j):
        return i * nx + j

    for i in range(nz):
        for j in range(nx):
            k = idx(i, j)
            on_clip = ((clip_x and j == (0 if clip_lo else nx - 1))
                       or (clip_z and i == (0 if clip_lo else nz - 1)))
            if on_clip:                        # Dirichlet V_m = 0 at the clip edge
                rows.append(k)
                cols.append(k)
                vals.append(1.0)
                rhs[k] = 0.0
                continue
            off: dict[int, float] = {}
            if j > 0:
                off[idx(i, j - 1)] = ihx2
            if j < nx - 1:
                off[idx(i, j + 1)] = ihx2
            if i > 0:
                off[idx(i - 1, j)] = ihz2
            if i < nz - 1:
                off[idx(i + 1, j)] = ihz2
            if j == 0 and nx > 1:              # mirror ghost node = zero flux
                off[idx(i, 1)] = off.get(idx(i, 1), 0.0) + ihx2
            if j == nx - 1 and nx > 1:
                off[idx(i, nx - 2)] = off.get(idx(i, nx - 2), 0.0) + ihx2
            if i == 0 and nz > 1:
                off[idx(1, j)] = off.get(idx(1, j), 0.0) + ihz2
            if i == nz - 1 and nz > 1:
                off[idx(nz - 2, j)] = off.get(idx(nz - 2, j), 0.0) + ihz2
            rows.append(k)
            cols.append(k)
            vals.append(-float(sum(off.values())))
            for k2, a in off.items():
                rows.append(k)
                cols.append(k2)
                vals.append(a)

    from scipy.sparse import csc_matrix as _csc
    A = _csc((vals, (rows, cols)), shape=(n, n))
    return spsolve(A, rhs).reshape(nz, nx)


def _grid_to_nodes(xy_nodes: np.ndarray, v_grid: np.ndarray,
                   gx: np.ndarray, gz: np.ndarray) -> np.ndarray:
    """Interpolate a (nz, nx) face grid back to (y, z) front-face node coords."""
    itp = RegularGridInterpolator((gz, gx), v_grid, bounds_error=False,
                                  fill_value=None)       # linear extrapolation
    return itp(np.column_stack([xy_nodes[:, 1], xy_nodes[:, 0]]))


def _delta_to_jlim(cm: CellMesh, bath: Bath, delta_um: float, delta_map,
                   ) -> tuple[np.ndarray, list[str]]:
    """Nodal limiting-current-density field (A/cm2, inf = no limitation)."""
    warnings: list[str] = []
    m = cm.mesh
    N = m.p.shape[1]
    delta_cm = float(delta_um) * 1e-4
    const = bath.j_lim_cu(delta_cm) if delta_cm > 0 else JLIM_UNLIMITED
    jlim = np.full(N, const)
    if delta_map is not None:
        xg = np.asarray(delta_map["x"], float)
        yg = np.asarray(delta_map["y"], float)
        dmap = np.asarray(delta_map["delta_um"], float)
        if dmap.shape == (len(xg), len(yg)) and dmap.shape[0] != dmap.shape[1]:
            dmap = dmap.T                       # tolerate (nx, ny) layout
        if dmap.shape != (len(yg), len(xg)):
            raise ValueError(f"delta_map['delta_um'] shape {dmap.shape} != "
                             f"(len(y), len(x)) = {(len(yg), len(xg))}")
        order = np.argsort(xg), np.argsort(yg)
        xg_s, yg_s = xg[order[0]], yg[order[1]]
        dmap_s = dmap[np.ix_(order[1], order[0])]
        itp = RegularGridInterpolator((yg_s, xg_s), dmap_s,
                                      bounds_error=False, fill_value=None)
        bnd = m.boundary_facets()
        fverts = m.facets[:, bnd[cm.groups["cath_front"]]]
        front_nodes = np.unique(fverts)
        # caller (u, v) -> (x=horizontal -> beaker y, y=vertical -> beaker z)
        d_um = itp(np.column_stack([m.p[2, front_nodes], m.p[1, front_nodes]]))
        d_cm = np.maximum(d_um, 0.0) * 1e-4
        with np.errstate(divide="ignore"):
            jlim[front_nodes] = np.where(d_cm > 0,
                                         2 * F * bath.D_cu_eff * bath.c_cu / d_cm,
                                         JLIM_UNLIMITED)
        warnings.append("delta_map supplied: Nernst layer varies over the coupon face "
                        "(back/edges keep the constant delta_um)")
    return jlim, warnings


# --------------------------------------------------------------------- public API

@dataclass
class CellResult:
    x_cm: np.ndarray                 # horizontal in-face coord (beaker y), len GRID_N
    z_cm: np.ndarray                 # absolute height (beaker z), len GRID_N
    j_mA_cm2: np.ndarray             # (GRID_N, GRID_N) rows = z
    thickness_um: np.ndarray         # Faraday thickness after t_s at `efficiency`
    V_cell_V: float
    V_ohm_V: float
    eta_c_V: float                   # -mean(eta_c) > 0
    eta_a_V: float                   # mean(eta_a) > 0
    mean_eta_c_V: float              # signed (negative)
    mean_eta_a_V: float
    Wa: float                        # kappa * deta/dj / coupon_w at the mean operating point
    j_mean_mA_cm2: float
    j_max_mA_cm2: float
    j_min_mA_cm2: float
    std_over_mean: float
    edge_center_ratio: float
    corner_LR_ratio: float
    corner_LR_grad_pct: float
    I_cathode_A: float
    I_anode_A: float                 # signed: should equal -I_total
    front_fraction: float            # share of I_total landing on the coupon front face
    loss_fraction: float             # back + edges + stray share (untaped leakage)
    anode_j_mA_cm2: float
    front_area_cm2: float
    cathode_area_cm2: float
    anode_area_cm2: float
    n_nodes: int
    n_elems: int
    n_newton: int
    seed_R_sq: float | None         # sheet resistance used (ohm/sq); None = equipotential
    vm_max_mV: float                # largest metal-potential drop on the front face
    n_seed_iter: int                # outer fixed-point iterations (0 when disabled)
    mesh_time_s: float
    solve_time_s: float
    domain: str
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=lambda: list(ASSUMPTIONS))

    def to_dict(self) -> dict:
        def _clean(v):
            if isinstance(v, float) and not np.isfinite(v):
                return None
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            if isinstance(v, float) and not np.isfinite(v):
                return None
            return v

        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = np.where(np.isfinite(v), v, np.nan).tolist()
            elif isinstance(v, list):
                d[k] = v
            else:
                d[k] = _clean(v)
        return d


_MESH_CACHE: dict[str, CellMesh] = {}


def _cached_mesh(geom: CellGeometry, resolution: str) -> CellMesh:
    key = json.dumps({"g": geom.to_dict(), "r": resolution}, sort_keys=True)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = build_cell_mesh(geom, resolution)
    return _MESH_CACHE[key]


def solve_cell(geom: CellGeometry, bath: Bath, I_total_A: float, t_s: float = 3600,
               efficiency: float = 1.0, delta_um: float = 60.0,
               delta_map: dict | None = None, suppressed: bool = True,
               resolution: str = "coarse", *, seed_R_sq: float | None = None,
               clip_edge: str = "top", clip_R_ohm: float = 0.0,
               progress=None) -> CellResult:
    """Solve the 3-D beaker cell at galvanostatic total current `I_total_A`.

    `seed_R_sq` (ohm/sq) enables the thin-seed terminal-effect correction: the
    coupon front face gets a metal potential V_m(s) from the sheet equation
    div_s((1/R_s) grad_s V_m) = -j, V_m = 0 on the `clip_edge` ("top" | "bottom"
    | "left" | "right") and zero flux elsewhere; the front-face overpotential
    becomes eta_c = V_m - phi_l.  None (default) keeps the metal equipotential.
    `clip_R_ohm` adds I_total * clip_R_ohm to V_cell (lumped clip contact, no
    effect on the current distribution).
    """
    t_start = time.perf_counter()
    if progress:
        progress(0.05, f"meshing beaker cell ({resolution})")
    cm = _cached_mesh(geom, resolution)
    mesh_time = cm.mesh_time_s
    if progress:
        progress(0.15, f"mesh ready: {cm.n_nodes} nodes, {cm.n_elems} tets")

    i0 = bath.i0_cu_supp if suppressed else bath.i0_cu
    jlim, warns = _delta_to_jlim(cm, bath, delta_um, delta_map)
    warnings = list(cm.warnings) + warns

    def on_iter(it, rn):
        if progress:
            progress(0.15 + 0.6 * min(it, 40) / 40, f"Newton iteration {it} (residual {rn:.2e})")

    # ---- front-face grid + node set (also used by the seed fixed point below)
    m = cm.mesh
    bnd = m.boundary_facets()
    fverts = m.facets[:, bnd[cm.groups["cath_front"]]]
    front_nodes = np.unique(fverts)
    xy = np.column_stack([m.p[1, front_nodes], m.p[2, front_nodes]])   # (beaker y, z)
    x_grid = np.linspace(-geom.coupon_w / 2, geom.coupon_w / 2, GRID_N)
    z_grid = np.linspace(geom.coupon_center_z - geom.coupon_h / 2,
                         geom.coupon_center_z + geom.coupon_h / 2, GRID_N)

    t_solve = time.perf_counter()
    if seed_R_sq is None:
        core = _solve_core(cm, bath.kappa, I_total_A, i0, bath.alpha_c, bath.alpha_a,
                           bath.f_RT, jlim, geom.gap, taped=geom.taped,
                           on_iter=on_iter)
        vm_nodes, n_seed, vm_max_V = None, 0, 0.0
    else:
        if clip_edge not in _CLIP_EDGES:
            raise ValueError(f"clip_edge must be one of {_CLIP_EDGES}, got {clip_edge!r}")
        if seed_R_sq <= 0:
            raise ValueError("seed_R_sq must be > 0 ohm/sq (use None for equipotential)")
        hx = float(x_grid[1] - x_grid[0])
        hz = float(z_grid[1] - z_grid[0])
        vm_nodes = np.zeros(m.p.shape[1])
        phi0, n_seed, vm_max_V = None, 0, 0.0
        alpha, prev_step = 1.0, np.inf
        for it in range(1, 31):
            core = _solve_core(cm, bath.kappa, I_total_A, i0, bath.alpha_c,
                               bath.alpha_a, bath.f_RT, jlim, geom.gap,
                               taped=geom.taped, on_iter=on_iter,
                               vm=vm_nodes, phi0=phi0)
            phi0 = (core.phi, core.phi_a)
            j_faces = np.maximum(_grid_interp(xy, core.jc_nodes[front_nodes],
                                              x_grid, z_grid), 0.0)       # A/cm2
            vm_new = _grid_to_nodes(xy, _sheet_poisson(j_faces, seed_R_sq, hx, hz,
                                                       clip_edge), x_grid, z_grid)
            vm_old = vm_nodes[front_nodes].copy()
            step = float(np.abs(vm_new - vm_old).max()) if vm_old.size else 0.0
            if step > prev_step:                 # oscillating -> under-relax
                alpha = max(0.25, 0.5 * alpha)
            prev_step = step
            vm_nodes[front_nodes] = vm_old + alpha * (vm_new - vm_old)
            dvm = float(np.abs(vm_nodes[front_nodes] - vm_old).max()) if vm_old.size else 0.0
            n_seed = it
            if progress:
                progress(0.15 + 0.6 * min(it, 20) / 20,
                         f"seed fixed point {it} (d V_m {dvm:.2e} V)")
            if dvm < 1e-6:
                break
        else:
            warnings.append("seed sheet-resistance fixed point did not reach 1e-6 V "
                            "in 30 iterations")
        vm_max_V = float(np.max(vm_nodes[front_nodes])) if front_nodes.size else 0.0
        warnings.append(f"seed terminal effect: R_s = {seed_R_sq:g} ohm/sq, clip on the "
                        f"{clip_edge} edge, max metal drop {vm_max_V * 1e3:.2f} mV")
        if not geom.taped:
            warnings.append("seed_R_sq is applied as a nodal metal-potential offset on the "
                            "coupon front face only; the untaped back/sidewall facets share "
                            "their perimeter nodes and inherit part of that offset")
    solve_time = time.perf_counter() - t_solve
    if progress:
        progress(0.85, "post-processing: j map on coupon face")

    # ---- regular grid j(x, y) on the coupon front face (corrected j when seeded)
    j_nodes_mA = core.jc_nodes[front_nodes] * 1e3
    j_grid = _grid_interp(xy, j_nodes_mA, x_grid, z_grid)
    j_grid = np.maximum(j_grid, 0.0)
    th_grid = thickness_um(j_grid, t_s, efficiency)

    metrics = _uniformity(j_grid, geom.coupon_w / 2, geom.coupon_h / 2)

    # ---- overpotentials, ohmic drop, Wagner number
    eta_bar_c, eta_bar_a = core.eta_bar_c, core.eta_bar_a
    V_cell = float(core.phi_a) + I_total_A * clip_R_ohm
    V_ohm = V_cell - eta_bar_a + eta_bar_c
    g_mean = abs(float(_cath_dj(np.array([eta_bar_c]), np.array([core.jlim_cath_mean]),
                                i0, bath.alpha_c, bath.alpha_a, bath.f_RT)[0]))
    Wa = bath.kappa / (g_mean * geom.coupon_w) if g_mean > 0 else float("inf")

    front_fraction = core.I_front / I_total_A
    loss_fraction = 1.0 - front_fraction
    if geom.taped and geom.stray_area <= 0:
        loss_fraction = 0.0            # nothing but the front face is active by construction
        front_fraction = 1.0
    if abs(core.I_cath - I_total_A) / I_total_A > 1e-4:
        warnings.append(f"cathode charge balance off by "
                        f"{abs(core.I_cath - I_total_A) / I_total_A:.2e} (solver residual)")
    if not geom.taped:
        warnings.append(f"untaped: {100 * loss_fraction:.1f} % of I_total plates the back/"
                        f"sidewalls/stray patch instead of the front face")
    if core.n_iter >= 60:
        warnings.append("Newton needed many iterations - check mesh/parameters")

    if progress:
        progress(1.0, "done")
    return CellResult(
        x_cm=x_grid, z_cm=z_grid, j_mA_cm2=j_grid, thickness_um=th_grid,
        V_cell_V=V_cell, V_ohm_V=float(V_ohm), eta_c_V=float(-eta_bar_c),
        eta_a_V=float(eta_bar_a), mean_eta_c_V=float(eta_bar_c),
        mean_eta_a_V=float(eta_bar_a), Wa=float(Wa),
        j_mean_mA_cm2=metrics["j_mean"], j_max_mA_cm2=metrics["j_max"],
        j_min_mA_cm2=metrics["j_min"], std_over_mean=metrics["std_over_mean"],
        edge_center_ratio=metrics["edge_center_ratio"],
        corner_LR_ratio=metrics["corner_LR_ratio"],
        corner_LR_grad_pct=metrics["corner_LR_grad_pct"],
        I_cathode_A=core.I_cath, I_anode_A=core.I_anode,
        front_fraction=float(front_fraction), loss_fraction=float(loss_fraction),
        anode_j_mA_cm2=-core.I_anode * 1e3 / core.A_anode,
        front_area_cm2=core.A_front, cathode_area_cm2=core.A_cath,
        anode_area_cm2=core.A_anode,
        n_nodes=cm.n_nodes, n_elems=cm.n_elems, n_newton=core.n_iter,
        seed_R_sq=seed_R_sq, vm_max_mV=float(vm_max_V * 1e3), n_seed_iter=n_seed,
        mesh_time_s=mesh_time, solve_time_s=solve_time, domain=cm.domain,
        warnings=warnings,
    )


def solve_duct(gap: float, w: float, h: float, I_total_A: float, bath: Bath,
               suppressed: bool = True, delta_um: float = 0.0,
               n_xyz=(14, 10, 10)) -> dict:
    """1-D sanity case: electrodes spanning a rectangular duct (walls insulating).

    Pure ohmic + kinetics; with fast kinetics V_cell must equal I*gap/(kappa*A).
    """
    cm = build_duct_mesh(gap, w, h, n_xyz)
    i0 = bath.i0_cu_supp if suppressed else bath.i0_cu
    N = cm.n_nodes
    delta_cm = float(delta_um) * 1e-4
    jlim = np.full(N, bath.j_lim_cu(delta_cm) if delta_cm > 0 else JLIM_UNLIMITED)
    core = _solve_core(cm, bath.kappa, I_total_A, i0, bath.alpha_c, bath.alpha_a,
                       bath.f_RT, jlim, gap)
    j_mA = core.I_cath / (w * h) * 1e3
    front_nodes = np.unique(cm.mesh.facets[:, core.bnd[cm.groups["cath_front"]]])
    j_front = core.jc_nodes[front_nodes]
    return {
        "V_cell_V": float(core.phi_a),
        "V_ohm_V": float(core.phi_a - core.eta_bar_a + core.eta_bar_c),
        "eta_c_V": float(-core.eta_bar_c),
        "eta_a_V": float(core.eta_bar_a),
        "j_mean_mA_cm2": float(j_mA),
        "I_cathode_A": core.I_cath,
        "I_anode_A": core.I_anode,
        "j_std_over_mean": float(np.std(j_front) / max(np.mean(j_front), 1e-30)),
        "n_newton": core.n_iter,
        "domain": cm.domain,
    }
