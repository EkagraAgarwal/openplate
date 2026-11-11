"""Tests for platesim.trenchflow.transport (steady advection-diffusion-reaction of Cu2+)."""
import numpy as np

from platesim.trenchflow.geometry import couette_flow, flat_grid, trench_grid
from platesim.trenchflow.transport import region_means, sherwood_scan, solve_concentration

D = 5.6e-6
H_UM = 200.0
H = H_UM * 1e-4                                  # cm
K = 1e-3


def _flat():
    return flat_grid(pitch_um=80.0, height_um=H_UM, dx_um=4.0)


def test_pure_diffusion_flat_wall():
    """1-D resistance chain 1/k + H/D over a flat wall; profile linear in z."""
    grid = _flat()
    res = solve_concentration(grid, flow=None, D_cm2_s=D, k_cm_s=K)
    exact = 1.0 / (1.0 / K + H / D)
    got = np.mean(res["surf_flux_density"])
    assert abs(got - exact) / exact < 0.01, (got, exact, abs(got - exact) / exact)

    c = res["c"]
    col = c[:, 0]
    col = col[np.isfinite(col)]                              # the two bottom rows are solid
    assert np.max(np.abs(np.diff(col, 2))) < 1e-9            # linear in z
    assert np.isnan(c[0, 0]) and np.isnan(c[1, 0]) and np.isfinite(c[2, 0])
    assert 0.0 < col[0] < col[-1] <= 1.0
    assert res["balance_error"] < 1e-10


def test_pure_diffusion_mass_transfer_limit():
    """k = inf: flux density = D/H (c_wall = 0)."""
    grid = _flat()
    res = solve_concentration(grid, flow=None, D_cm2_s=D, k_cm_s=np.inf)
    exact = D / H
    got = np.mean(res["surf_flux_density"])
    assert abs(got - exact) / exact < 0.01, (got, exact)
    assert res["c"][2, 0] < 0.02                             # wall cell close to 0
    assert res["delta_eff_um"] > 0.0


def test_couette_over_flat_wall_matches_diffusion():
    """Couette flow is parallel to the wall and x-uniform -> must reproduce the 1-D solution."""
    grid = _flat()
    ref = 1.0 / (1.0 / K + H / D)
    res = solve_concentration(grid, flow=couette_flow(grid, 80.0), D_cm2_s=D, k_cm_s=K)
    got = np.mean(res["surf_flux_density"])
    assert abs(got - ref) / ref < 0.005, (got, ref, abs(got - ref) / ref)
    assert np.ptp(res["surf_flux_density"]) / ref < 1e-6     # independent of x


def _trench():
    return trench_grid(160.0, 80.0, height_um=160.0, dx_um=8.0)


def test_conservation_and_boundedness_in_trench():
    grid = _trench()
    flow = couette_flow(grid, 50.0)
    res = solve_concentration(grid, flow=flow, D_cm2_s=D, k_cm_s=K)
    assert res["balance_error"] < 1e-6, res["balance_error"]
    c = res["c"]
    fin = c[np.isfinite(c)]
    assert fin.min() >= -1e-12 and fin.max() <= 1.0 + 1e-12
    rm = region_means(res, grid)
    assert rm["n_field"] > 0 and rm["n_floor"] > 0 and rm["n_wall"] > 0
    assert rm["floor"] < rm["field"], (rm["floor"], rm["field"])


def test_no_reaction_gives_uniform_concentration():
    """k = 0: c = 1 everywhere, even with the (non-physical inside the trench) Couette field."""
    grid = _trench()
    res = solve_concentration(grid, flow=couette_flow(grid, 50.0), D_cm2_s=D, k_cm_s=0.0)
    c = res["c"]
    fin = c[np.isfinite(c)]
    assert np.max(np.abs(fin - 1.0)) < 1e-9
    assert abs(res["total_wall_flux"]) < 1e-20
    assert abs(res["top_flux"]) < 1e-15                      # roundoff only (2D ~ 1e-5)


def test_mass_transfer_limit_floor_over_field():
    """Diffusion only: the floor is deeper than the field, and the contrast grows with k."""
    grid = _trench()
    r_low = region_means(solve_concentration(grid, None, D_cm2_s=D, k_cm_s=1e-4), grid)
    r_high = region_means(solve_concentration(grid, None, D_cm2_s=D, k_cm_s=1e-1), grid)
    assert r_low["floor_over_field"] < 1.0, r_low["floor_over_field"]
    assert r_high["floor_over_field"] < r_low["floor_over_field"], (r_high, r_low)
    assert r_high["wall_over_field"] < r_low["wall_over_field"]


def test_sherwood_scan_shapes():
    grid = _trench()
    rows = sherwood_scan(grid, None, D, [1e-4, 1e-2, np.inf])
    assert len(rows) == 3
    for row in rows:
        assert row["n_field"] > 0
        assert np.isfinite(row["floor_over_field"])
    assert rows[2]["k_cm_s"] == np.inf
    assert rows[2]["floor_over_field"] < rows[0]["floor_over_field"]
