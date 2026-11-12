"""Tests for platesim.trenchflow.stokes (direct steady Stokes / Oseen MAC solver).

Small grids, analytic references: plane Couette over a flat wall, discrete divergence of the face
fluxes, shear sign symmetry, trench eddies, the Stokes lid-driven cavity and a loose LBM comparison.
"""
import time
import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning

from platesim.trenchflow.geometry import CavityGrid, flat_grid, trench_grid
from platesim.trenchflow.stokes import solve_stokes


def _box_cavity(n=40, dx=1.0):
    """Lid-driven square cavity: solid bottom row and solid left/right columns (x stays periodic but
    the two solid columns close the box), 40 x 40 fluid cells, lid at the top face."""
    nx, nz = n + 2, n + 1
    x = (np.arange(nx) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    solid = np.zeros((nz, nx), bool)
    solid[0, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    return CavityGrid(dx_um=dx, x_um=x, z_um=z, solid=solid, width_um=n * dx, depth_um=0.0,
                      pitch_um=nx * dx, height_um=nz * dx)


def test_couette_linear_profile():
    """Flat wall + moving lid: ux(z) is the exact linear Couette profile, uz = 0."""
    g = flat_grid(pitch_um=80, height_um=160, dx_um=8)
    shear = 50.0
    flow = solve_stokes(g, shear)
    u_top = shear * g.height_um * 1e-4
    assert flow.converged
    assert flow.ux.shape == g.shape and np.all(flow.ux[g.solid] == 0.0)
    fluid = ~g.solid[:, 0]
    z = g.z_um[fluid]
    exact = shear * z * 1e-4
    sim = flow.ux[fluid, 3]
    err = float(np.abs(sim - exact).max() / u_top)
    assert err < 1e-10, f"Couette relative error {err:.3e}"
    assert np.abs(flow.uz).max() / u_top < 1e-12
    assert np.abs(flow.info["u_face_z"]).max() / u_top < 1e-12
    assert flow.info["method"] == "stokes_mac"
    print(f"\nCouette rel. error = {err:.3e}, max|uz|/u_top = {np.abs(flow.uz).max() / u_top:.3e}")


def test_face_divergence_free():
    """The staggered face field satisfies the discrete continuity equation exactly in the open cells."""
    g = trench_grid(width_um=60, depth_um=30, dx_um=4)
    for shear in (40.0, -120.0):
        flow = solve_stokes(g, shear)
        uf, wf = flow.info["u_face_x"], flow.info["u_face_z"]
        assert uf.shape == g.shape and wf.shape == (g.shape[0] + 1, g.shape[1])
        div = (np.roll(uf, -1, axis=1) - uf) + (wf[1:] - wf[:-1])
        ref = max(np.abs(uf).max(), np.abs(wf).max())
        rel = float(np.abs(div[~g.solid]).max() / ref)
        assert rel < 1e-10, f"divergence {rel:.3e} of the face flux"
        assert abs(rel - flow.info["div_max_rel"]) < 1e-14
        print(f"\ndiv_rel(G={shear:+.0f}) = {rel:.3e}")


def test_shear_sign_symmetry():
    """A negative shear rate is the exact mirror image of the positive one for a symmetric trench."""
    g = trench_grid(width_um=60, depth_um=30, dx_um=4)
    assert np.array_equal(g.solid, g.solid[:, ::-1]), "test geometry must be x-symmetric"
    pos = solve_stokes(g, 80.0)
    neg = solve_stokes(g, -80.0)
    scale = np.abs(pos.ux).max()
    e1 = np.abs(neg.ux + pos.ux[:, ::-1]).max() / scale
    e2 = np.abs(neg.uz - pos.uz[:, ::-1]).max() / scale
    assert e1 < 1e-10 and e2 < 1e-10, (e1, e2)
    # vector field under x -> -x: u_face[j] -> -u_face[nx-j], w_face[j] -> +w_face[nx-1-j]
    nx = g.shape[1]
    jj = np.arange(nx)
    uf_p, uf_n = pos.info["u_face_x"], neg.info["u_face_x"]
    e3 = np.abs(uf_n + uf_p[:, (nx - jj) % nx]).max() / np.abs(uf_p).max()
    assert e3 < 1e-10, e3
    wf_p, wf_n = pos.info["u_face_z"], neg.info["u_face_z"]
    e4 = np.abs(wf_n - wf_p[:, ::-1]).max() / max(np.abs(wf_p).max(), 1e-30)
    assert e4 < 1e-10, e4
    print(f"\nmirror errors: ux {e1:.2e}, uz {e2:.2e}, u_face {e3:.2e}, w_face {e4:.2e}")


def test_trench_recirculation():
    """Deep narrow trench: reversed flow below z = 0 on the centre column. Shallow wide: follows the lid."""
    deep = trench_grid(width_um=60, depth_um=60, dx_um=4)
    f = solve_stokes(deep, 100.0)
    ix = int(np.argmin(np.abs(deep.x_um - 0.5 * deep.pitch_um)))
    u_top = 100.0 * deep.height_um * 1e-4
    below = (deep.z_um < 0.0) & (~deep.solid[:, ix])
    assert below.any()
    assert f.ux[below, ix].min() < -1e-3 * u_top, "no closed eddy in the deep trench"
    print(f"\ndeep trench: min ux on the centre column = {f.ux[below, ix].min():.4g} cm/s "
          f"({f.ux[below, ix].min() / u_top:+.3f} u_top)")

    shallow = trench_grid(width_um=200, depth_um=30, dx_um=4)
    f = solve_stokes(shallow, 100.0)
    ix = int(np.argmin(np.abs(shallow.x_um - 0.5 * shallow.pitch_um)))
    u_ref = 100.0 * shallow.depth_um * 1e-4            # shear * depth: the cavity velocity scale
    iz = int(np.argmin(np.abs(shallow.z_um + 0.5 * shallow.depth_um)))
    assert not shallow.solid[iz, ix]
    assert f.ux[iz, ix] > 0.1 * u_ref, f"flow does not follow the lid at mid depth: {f.ux[iz, ix]:.4g}"
    print(f"shallow trench: ux at mid depth = {f.ux[iz, ix] / u_ref:+.3f} (G*depth), "
          f"u_top = {100.0 * shallow.height_um * 1e-4:.3f} cm/s")


def test_stokes_cavity_centreline():
    """Lid-driven square cavity in the Stokes limit: min u/U on the vertical centreline ~ -0.21."""
    g = _box_cavity(40, dx=1.0)
    U = 50.0 * g.height_um * 1e-4
    flow = solve_stokes(g, 50.0)
    ix = int(np.argmin(np.abs(g.x_um - 0.5 * g.width_um)))
    col = flow.ux[:, ix] / U
    fluid = ~g.solid[:, ix]
    umin = float(col[fluid].min())
    iz = int(np.argmin(np.where(fluid, col, np.inf)))
    assert -0.22 <= umin <= -0.19, f"Stokes cavity min u/U = {umin:.4f}"
    print(f"\ncavity: min u/U on the vertical centreline = {umin:.4f} at z/H = "
          f"{(g.z_um[iz] - 0.5 * g.dx_um) / g.height_um:.3f}")


def test_against_lbm():
    """Same small trench: the direct Stokes field vs the D2Q9 lattice field (trench interior, z < 0).

    The LBM clamps tau for Reynolds matching and is itself 5-25 % off, so only a loose bound is
    asserted; the numbers are printed for the record.
    """
    from platesim.trenchflow.cavity_lbm import solve_flow

    g = trench_grid(width_um=60, depth_um=30, height_um=120, dx_um=6)
    st = solve_stokes(g, 80.0)
    lbm = solve_flow(g, 80.0)
    below = (g.z_um[:, None] < 0.0) & (~g.solid)
    assert below.sum() > 20
    ref = max(np.abs(st.ux[below]).max(), np.abs(lbm.ux[below]).max())
    d = np.abs(st.ux[below] - lbm.ux[below]) / ref
    print(f"\nStokes vs LBM (trench, {below.sum()} cells): mean rel. diff = {d.mean():.3f}, "
          f"max = {d.max():.3f}; max|ux| stokes = {np.abs(st.ux[below]).max():.4g} cm/s, "
          f"lbm = {np.abs(lbm.ux[below]).max():.4g} cm/s")
    assert d.mean() < 0.3, f"mean relative difference {d.mean():.2f}"


def test_timing_large_grid():
    """Feature-scale grid (142 x 46 cells, dx 8 um): one direct solve in well under a second."""
    g = trench_grid(width_um=227, depth_um=88, pitch_um=1135, dx_um=8)
    assert g.shape == (46, 142), g.shape
    t0 = time.perf_counter()
    flow = solve_stokes(g, 80.0)
    dt = time.perf_counter() - t0
    assert flow.converged
    assert flow.info["div_max_rel"] < 1e-10
    print(f"\n{g.shape[1]} x {g.shape[0]} cells: {dt * 1e3:.0f} ms, "
          f"div_max_rel = {flow.info['div_max_rel']:.2e}, wall_s = {flow.info['wall_s']:.3f}")
    assert dt < 1.0, f"solve took {dt:.2f} s"


def _pocket_grid(seam=False, bubble=True, npad=4):
    """60 x 30 um trench (dx 4) with 4 extra all-solid rows under the floor.

    The extra rows put a closed 3 x 3-cell fluid hole (the "void pocket" the feature engine can leave
    inside the deposit) fully inside Si: padded rows 1-3 are 42-50 um below the wafer plane, i.e.
    12-20 um below the trench floor, with solid rows 0 and 4-6 around them. Rows 7+ are the trench.
    ``bubble=False`` returns the identical grid with the hole filled solid (one fluid region).
    ``seam=True`` moves the hole onto the periodic x seam (columns nx-2 .. 1).
    """
    g = trench_grid(width_um=60, depth_um=30, height_um=120, dx_um=4)
    nx = g.shape[1]
    solid = np.vstack([np.ones((npad, nx), bool), g.solid])
    z = np.concatenate([g.z_um[0] - g.dx_um * np.arange(npad, 0, -1), g.z_um])
    hole = solid.copy()
    if bubble:
        if seam:
            cols = [nx - 2, nx - 1, 0, 1]
        else:
            ixc = int(np.argmin(np.abs(g.x_um - 0.5 * g.pitch_um)))
            cols = list(range(ixc - 1, ixc + 2))
        hole[1:4, cols] = False
    grid = CavityGrid(dx_um=g.dx_um, x_um=g.x_um, z_um=z, solid=hole, width_um=g.width_um,
                      depth_um=g.depth_um, pitch_um=g.pitch_um, height_um=g.height_um)
    return grid, solid


def test_isolated_pocket_is_a_second_region():
    """A sealed fluid bubble below the trench floor must not make the saddle-point matrix singular.

    The pocket has no lid contact and no driving, so its velocity is exactly zero to round-off, and the
    main flow is the same as on the identical grid with the pocket filled solid.
    """
    gb, solid = _pocket_grid()
    gr, _ = _pocket_grid(bubble=False)
    pocket = (~gb.solid) & solid
    assert pocket.sum() == 9, "test geometry must contain the 3 x 3 hole"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fb = solve_stokes(gb, 80.0)
        fr = solve_stokes(gr, 80.0)
    rank = [w for w in caught if issubclass(w.category, MatrixRankWarning)]
    assert not rank, [str(w.message) for w in rank]

    u_top = 80.0 * gb.height_um * 1e-4
    assert fb.info["n_fluid_regions"] == 2 and fr.info["n_fluid_regions"] == 1
    assert len(fb.info["pin_cells"]) == 2
    assert np.isfinite(fb.ux).all() and np.isfinite(fb.uz).all()

    inside = np.hypot(fb.ux, fb.uz)[pocket].max() / u_top
    assert inside < 1e-12, f"pocket is not at rest: max|u| = {inside:.3e} u_top"
    scale = max(np.abs(fr.ux).max(), np.abs(fr.uz).max())
    du = np.abs(fb.ux - fr.ux).max() / scale
    dw = np.abs(fb.uz - fr.uz).max() / scale
    assert max(du, dw) < 1e-10, f"pocket changed the main flow: du {du:.3e}, dw {dw:.3e}"
    assert fb.converged
    print(f"\npocket: {pocket.sum()} cells at rest (max|u|/u_top = {inside:.1e}); main flow vs filled "
          f"solid: du = {du:.2e}, dw = {dw:.2e}; regions = {fb.info['n_fluid_regions']}, "
          f"pins = {fb.info['pin_cells']}")


def test_pocket_across_the_x_seam_is_one_region():
    """A pocket spanning columns nx-1 and 0 is ONE region (4-connectivity is periodic in x), so it needs
    one gauge, not two: the grid must still report two regions in total and solve cleanly."""
    gs, solid = _pocket_grid(seam=True)
    pocket = (~gs.solid) & solid
    assert pocket.sum() == 12
    assert np.any(pocket[:, 0]) and np.any(pocket[:, -1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flow = solve_stokes(gs, 80.0)
    rank = [w for w in caught if issubclass(w.category, MatrixRankWarning)]
    assert not rank, [str(w.message) for w in rank]
    assert flow.info["n_fluid_regions"] == 2, flow.info["n_fluid_regions"]
    assert len(flow.info["pin_cells"]) == 2
    assert np.isfinite(flow.ux).all() and np.isfinite(flow.uz).all()
    u_top = 80.0 * gs.height_um * 1e-4
    inside = np.hypot(flow.ux, flow.uz)[pocket].max() / u_top
    assert inside < 1e-12, inside
    print(f"\nseam pocket: {pocket.sum()} cells, regions = {flow.info['n_fluid_regions']}, "
          f"max|u|/u_top in the pocket = {inside:.1e}, max|ux|/u_top = "
          f"{np.abs(flow.ux).max() / u_top:.3f}")

