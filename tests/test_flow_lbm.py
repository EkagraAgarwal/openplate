import math

import numpy as np

from platesim.core.bath import Bath
from platesim.core.cell import CellGeometry
from platesim.flow import lbm


def test_poiseuille_profile():
    """Body-force driven channel, walls at j = 0 and j = ny-1 (full-way bounce-back -> wall halfway)."""
    nx, ny, tau, g = 4, 34, 0.8, 1e-6
    solid = np.zeros((nx, ny), bool)
    solid[:, 0] = solid[:, -1] = True
    f = lbm.equilibrium(np.ones((nx, ny), np.float32), np.zeros((nx, ny), np.float32),
                        np.zeros((nx, ny), np.float32)).astype(np.float64)
    # constant body force as a relaxation towards a huge target velocity is awkward; apply it directly
    for _ in range(12000):
        rho, ux, uy = lbm.macro(f)
        feq = lbm.equilibrium(rho, ux + tau * g / rho, uy)
        fpost = f - (f - feq) / tau
        fpost[:, solid] = f[lbm.OPP][:, solid]
        for i in range(1, 9):
            fpost[i] = np.roll(fpost[i], (lbm.C[i, 0], lbm.C[i, 1]), axis=(0, 1))
        f = fpost
    _, ux, _ = lbm.macro(f)
    nu = (tau - 0.5) / 3
    H = ny - 2                                     # walls halfway between node 0/1 and ny-2/ny-1
    y = np.arange(1, ny - 1) - 0.5
    exact = g / (2 * nu) * y * (H - y)
    sim = ux[1, 1:-1]
    assert np.max(np.abs(sim - exact)) / exact.max() < 0.02


def test_leveque_constant_shear():
    D, gamma, x = 4e-6, 50.0, 0.5
    k = 0.538 * D ** (2 / 3) * gamma ** (1 / 3) * x ** (-1 / 3)
    assert math.isclose(lbm.leveque_delta_cm(x, gamma, D), D / k, rel_tol=1e-12)
    # thicker downstream, thinner at higher shear
    assert lbm.leveque_delta_cm(1.0, gamma, D) > lbm.leveque_delta_cm(0.2, gamma, D)
    assert lbm.leveque_delta_cm(0.5, 8 * gamma, D) < lbm.leveque_delta_cm(0.5, gamma, D)


def test_run_lbm_short():
    r = lbm.run_lbm(CellGeometry(), Bath(), 300, max_seconds=10)
    d = np.asarray(r["delta_um"])
    assert d.size == len(r["x_cm"]) > 10
    assert np.isfinite(d).all() and (d > 0).all()
    assert r["Re_real"] > r["Re_sim"] > 10
    m = lbm.delta_map_lbm(CellGeometry(), Bath(), 300, max_seconds=5)
    assert np.asarray(m["delta_um"]).shape == (41, 41)
