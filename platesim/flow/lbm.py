"""2-D D2Q9 lattice-Boltzmann flow in a horizontal slice of the stirred beaker at coupon height, giving the wall
shear rate along the coupon front face and, through the Leveque relation, the local Nernst layer delta(x).

Model choices (all [J] unless tagged):
- The stir bar sits on the beaker floor, below the coupon, so it is NOT in this slice. Its effect at coupon height
  is the swirl it drives through the depth; we model that as a relaxation body force towards solid-body rotation
  u_target = k_swirl * omega_bar x r (k_swirl ~ 0.3: the forced-vortex core of an unbaffled stirred vessel turns
  at a fraction of the impeller speed; Nagata / Busciglio, KB report 02 Topic 2 [SH]). Baffles, the coupon and
  the anode are no-slip solids that the swirl has to flow around; that is what makes delta non-uniform.
- Layout copies platesim.cell: coupon front plane through the beaker axis (x = 0, spanning the coupon width),
  anode parallel at x = gap, width clipped to the beaker chord.
- Re_real = N L_bar^2 / nu (~7000 at 300 rpm) is far above what BGK runs on a laptop grid. We run at Re_sim,
  measure the dimensionless wall shear G = gamma / omega_bar, and scale with the laminar boundary-layer law
  gamma ~ U^1.5 at fixed nu and geometry, i.e. gamma_real = G * omega_real * sqrt(Re_real / Re_sim).
- Leveque (developing concentration layer in linear shear): k(x) = 0.538 D^(2/3) gamma^(1/3) x^(-1/3) [FT, standard
  result, e.g. Bard & Faulkner; Newman Electrochemical Systems], delta = D/k, x from the leading edge of the face.
Units: lattice units inside the solver; cm, s outside.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..core.bath import Bath, _eta_water
from ..core.cell import CellGeometry

# D2Q9: velocities, weights, opposite directions
C = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
RES = {"coarse": 96, "medium": 160}
K_SWIRL = 0.3          # swirl core speed / bar speed [J]
LEVEQUE = 0.538        # Sh_x = 0.538 (gamma x^2 / D)^(1/3) [FT]


_CX = C[:, 0, None, None].astype(np.float32)
_CY = C[:, 1, None, None].astype(np.float32)
_W = W[:, None, None].astype(np.float32)


def equilibrium(rho, ux, uy):
    cu = 3.0 * (_CX * ux + _CY * uy)
    usq = 1.5 * (ux * ux + uy * uy)
    return _W * rho * (1.0 + cu + 0.5 * cu * cu - usq)


def step(f, solid, tau, beta=0.0, ut=None):
    """One BGK collide + stream + full-way bounce-back step. Optional relaxation body force
    F = beta rho (u_target - u) by the velocity-shift method. Returns (f_new, ux, uy) with the pre-step velocity."""
    rho = f.sum(0)
    ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
    uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
    if ut is not None:
        ux_e = ux + tau * beta * (ut[0] - ux)
        uy_e = uy + tau * beta * (ut[1] - uy)
    else:
        ux_e, uy_e = ux, uy
    fpost = f - (f - equilibrium(rho, ux_e, uy_e)) / tau
    fpost[:, solid] = f[OPP][:, solid]                     # bounce back on solid nodes
    for i in range(1, 9):
        fpost[i] = np.roll(fpost[i], (C[i, 0], C[i, 1]), axis=(0, 1))
    return fpost, ux, uy


def macro(f):
    rho = f.sum(0)
    return rho, (C[:, 0, None, None] * f).sum(0) / rho, (C[:, 1, None, None] * f).sum(0) / rho


def leveque_delta_cm(x_cm, gamma_s, D):
    """Local Nernst layer (cm) of a developing layer in shear gamma (1/s) at distance x from the leading edge."""
    x = np.maximum(np.asarray(x_cm, float), 0.02)
    k = LEVEQUE * D ** (2 / 3) * np.maximum(np.asarray(gamma_s, float), 1e-9) ** (1 / 3) * x ** (-1 / 3)
    return D / k


def _geometry(geom: CellGeometry, N: int):
    """Solid mask and coupon-face indices on an N x N grid covering the beaker (index i -> x, j -> y)."""
    R = geom.beaker_id / 2
    dx = geom.beaker_id / N
    xc = (np.arange(N) + 0.5) * dx - R
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    solid = X ** 2 + Y ** 2 >= (R - dx) ** 2
    # baffles: radial strips at the wall, avoiding the +x / -x electrode axis
    for k in range(geom.n_baffles):
        th = math.pi / 4 + k * 2 * math.pi / max(geom.n_baffles, 1)
        e = np.array([math.cos(th), math.sin(th)])
        along = X * e[0] + Y * e[1]
        across = -X * e[1] + Y * e[0]
        solid |= (along >= R - geom.baffle_width) & (np.abs(across) <= dx)
    # coupon: 2 cells thick, front face (facing +x, the anode) at x = 0
    i_c = int(np.searchsorted(xc, 0.0))
    half_c = geom.coupon_w / 2
    jc = np.abs(xc) <= half_c
    solid[i_c - 2:i_c, jc] = True
    # anode: at x = gap, width clipped to the chord
    i_a = int(np.searchsorted(xc, geom.gap))
    chord = 2 * math.sqrt(max(R ** 2 - geom.gap ** 2, 0.0)) - 4 * dx
    half_a = min(geom.anode_w, chord) / 2
    solid[i_a:i_a + 2, np.abs(xc) <= half_a] = True
    return solid, xc, dx, i_c, jc, X, Y


def run_lbm(geom: CellGeometry, bath: Bath, rpm: float, resolution: str = "coarse", max_seconds: float = 45,
            tau: float = 0.52, u_tip: float = 0.05, n_spin: float = 4.0) -> dict:
    N = RES.get(resolution, 160)
    solid, xc, dx, i_c, jc, X, Y = _geometry(geom, N)
    fluid = ~solid
    L_cells = geom.stir_bar_len / dx
    omega = u_tip / (L_cells / 2)                       # bar angular speed, rad/step
    nu = (tau - 0.5) / 3
    Re_sim = omega / (2 * math.pi) * L_cells ** 2 / nu
    nu_real = 1e-6 * _eta_water(bath.T_C) / _eta_water(20.0) * 1e4   # cm2/s
    omega_real = 2 * math.pi * max(rpm, 1e-6) / 60
    Re_real = rpm / 60 * geom.stir_bar_len ** 2 / nu_real
    # relaxation forcing towards solid-body swirl, time constant one bar revolution
    beta = omega / (2 * math.pi)
    Xc, Yc = X / dx, Y / dx
    ut_x, ut_y = -K_SWIRL * omega * Yc, K_SWIRL * omega * Xc
    ut = (np.where(fluid, ut_x, 0.0).astype(np.float32), np.where(fluid, ut_y, 0.0).astype(np.float32))
    n_steps = int(n_spin / beta)                         # n_spin forcing time constants (spin-up ~ 1/beta)
    f = equilibrium(np.ones((N, N), np.float32), np.zeros((N, N), np.float32), np.zeros((N, N), np.float32))
    t0, acc, n_acc, steps = time.time(), np.zeros((2, N)), 0, 0
    avg_from = int(0.6 * n_steps)
    warnings: list[str] = []
    for steps in range(1, n_steps + 1):
        f, ux, uy = step(f, solid, tau, beta, ut)
        if steps >= avg_from and steps % 5 == 0:
            acc += uy[i_c:i_c + 2]                        # tangential velocity, first two fluid columns
            n_acc += 1
        if time.time() - t0 > max_seconds:
            warnings.append(f"stopped at max_seconds after {steps}/{n_steps} steps; flow may not be fully spun up")
            break
    if not np.isfinite(f).all():
        raise FloatingPointError("LBM diverged; increase tau or lower u_tip")
    if n_acc == 0:                                         # too short to reach the averaging window
        _, _, uy = macro(f.astype(np.float64))
        acc, n_acc = uy[i_c:i_c + 2].copy(), 1
    v = acc / n_acc                                        # shape (2, N): distance 0.5 and 1.5 cells from wall
    gamma_lat = (0.5 * v[0] + 1.5 * v[1]) / (0.25 + 2.25)  # least squares v = gamma d through the wall
    vt = gamma_lat[jc]
    G = np.abs(vt) / omega                                 # dimensionless shear
    gamma_real = G * omega_real * math.sqrt(max(Re_real, 1e-9) / Re_sim)
    # leading edge: flow runs along +y if mean tangential velocity > 0
    y = xc[jc]
    x_from_edge = (y - y.min()) if vt.mean() >= 0 else (y.max() - y)
    D = bath.D_cu_eff
    delta_cm = leveque_delta_cm(x_from_edge + dx / 2, gamma_real, D)
    delta_um = delta_cm * 1e4
    mean_delta = float(np.mean(delta_um))
    rho, ux, uy = macro(f.astype(np.float64))
    speed = np.where(fluid, np.hypot(ux, uy), np.nan) / u_tip
    stride = max(1, N // 120)
    sp = speed[::stride, ::stride]
    axis = (xc[::stride]).tolist()
    summary = {"delta_mean_um": round(mean_delta, 2), "k_m_cm_s": float(f"{D / (mean_delta * 1e-4):.4g}"),
               "Re_real": round(Re_real, 1), "Re_sim": round(Re_sim, 1), "steps": steps,
               "G_mean": float(f"{G.mean():.4g}"), "flow_dir_along_coupon": "+" if vt.mean() >= 0 else "-",
               "model": "lbm", "rpm": rpm}
    return {
        "x_cm": y.tolist(), "delta_um": delta_um.tolist(), "gamma_s": gamma_real.tolist(),
        "mean_delta_um": mean_delta, "k_m_cm_s": summary["k_m_cm_s"], "Re_real": Re_real, "Re_sim": Re_sim,
        "steps": steps,
        "plots": [
            {"kind": "heatmap", "title": "|u| / bar tip speed (slice at coupon height, sim Re)",
             "x": axis, "y": axis, "z": np.where(np.isnan(sp.T), None, np.round(sp.T, 4)).tolist(),
             "xlabel": "x (cm)", "ylabel": "y (cm)", "zlabel": "|u|/u_tip"},
            {"kind": "line", "title": "Nernst layer along coupon width (LBM + Leveque)", "x": y.tolist(),
             "series": [{"name": "delta (um)", "y": np.round(delta_um, 2).tolist()}],
             "xlabel": "position across coupon (cm)", "ylabel": "delta (um)"},
        ],
        "summary": summary,
        "assumptions": [
            "2-D horizontal slice at coupon height; the floor stir bar acts through a relaxation force towards "
            f"solid-body swirl at {K_SWIRL} x bar speed [J]",
            f"Re_sim = {Re_sim:.0f} vs Re_real = {Re_real:.0f}; wall shear scaled by sqrt(Re_real/Re_sim) "
            "(laminar boundary layer, gamma ~ U^1.5) [J] - order-of-magnitude only",
            "Leveque developing-layer mass transfer, k = 0.538 D^(2/3) gamma^(1/3) x^(-1/3) [FT]",
            "coupon front plane through the beaker axis, anode at x = gap (same layout as platesim.cell)",
        ],
        "warnings": warnings,
    }


def delta_map_lbm(geom: CellGeometry, bath: Bath, rpm: float, nx: int = 41, ny: int = 41, **kw) -> dict:
    """delta on the coupon face grid in the format platesim.cell.solve_cell(delta_map=...) expects
    (x across the width, uniform along the height in this 2-D model)."""
    r = run_lbm(geom, bath, rpm, **kw)
    x = np.linspace(-geom.coupon_w / 2, geom.coupon_w / 2, nx)
    y = np.linspace(-geom.coupon_h / 2, geom.coupon_h / 2, ny)
    d = np.interp(x, r["x_cm"], r["delta_um"])
    return {"x": x.tolist(), "y": y.tolist(), "delta_um": np.tile(d, (ny, 1)).tolist(),
            "mean_um": float(d.mean()), "k_m_cm_s": bath.D_cu_eff / (float(d.mean()) * 1e-4)}
