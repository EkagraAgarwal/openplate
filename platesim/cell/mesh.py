"""3-D mesh builders for the beaker plating cell.

Two backends:
- gmsh tetrahedral mesh of the actual beaker (cylinder). The thin coupon / anode /
  stray-patch solids are built as boxes that lie INSIDE the cylinder, their surfaces are
  *embedded* in the cylinder volume (gmsh makes the volume mesh conformal to them), and
  the tets inside the electrode boxes are deleted afterwards - the exposed conformal
  surface facets become the electrode boundaries. This avoids OpenCASCADE boolean cuts
  entirely, which are broken in the gmsh Windows wheels (BOPAlgo_AlertBOPNotAllowed /
  silently-empty results, 4.11-4.15) [M, tested 2026-10-28].
- structured tensor-product tetrahedra ("duct" for the 1-D sanity check, and a box
  approximation of the beaker used only if gmsh is unavailable).

Each builder returns a CellMesh: facet_groups maps "cath_front" / "cath_back" /
"cath_edge" / "stray" / "anode" to boolean masks over mesh.boundary_facets();
everything not in a group is insulating (natural BC).

Geometry convention (assumption [J]): the coupon front face lies in the plane x = 0
(its centre is on the beaker axis), the coupon extends x in [-t, 0] with t = 0.05 cm
Si thickness [spec], the anode front face is the plane x = gap facing it; both are
centred in y = 0 at height coupon_center_z; the beaker axis is z. Because a rectangle
cannot follow the curved wall, the anode is narrowed to fit inside the beaker with a
small wall margin (warning emitted when that happens).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from skfem import MeshTet

COUPON_T_CM = 0.05     # Si coupon thickness, plated sidewalls/back when untaped [spec]
ANODE_T_CM = 0.05      # anode plate thickness [J]
WALL_MARGIN_CM = 0.05  # keep electrodes this far inside the beaker wall (mesh quality) [J]
STRAY_STANDOFF = 0.10  # cm gap between coupon top edge and the lumped wet-clip patch [J]

# resolution -> (h_far, h_cathode, h_anode, reach_cathode, reach_anode) in cm [J]
_RESOLUTIONS = {
    "coarse": (1.20, 0.13, 0.30, 1.8, 1.2),
    "medium": (0.80, 0.09, 0.22, 2.2, 1.6),
    "fine": (0.50, 0.06, 0.16, 2.6, 2.0),
}


@dataclass
class CellMesh:
    mesh: MeshTet
    groups: dict[str, np.ndarray]          # name -> bool mask over boundary_facets()
    domain: str                            # "gmsh_beaker" | "structured_box" | "duct"
    warnings: list[str] = field(default_factory=list)
    mesh_time_s: float = 0.0

    @property
    def n_nodes(self) -> int:
        return self.mesh.p.shape[1]

    @property
    def n_elems(self) -> int:
        return self.mesh.t.shape[1]


# --------------------------------------------------------------------------- helpers

def _facet_coords(mesh: MeshTet, bnd: np.ndarray) -> np.ndarray:
    """Coordinates of boundary facets: (n_bnd, 3 vertices, 3 coords)."""
    facets = np.asarray(mesh.facets)          # built by MeshTet.__post_init__
    return mesh.p[:, facets[:, bnd]].transpose(2, 1, 0)


def _on_plane(c: np.ndarray, axis: int, val: float, tol: float = 1e-7) -> np.ndarray:
    """All 3 vertices of a facet within tol of the plane coord[axis] = val."""
    return np.all(np.abs(c[:, :, axis] - val) <= tol, axis=1)


def _within(c: np.ndarray, axis: int, lo: float, hi: float, tol: float = 1e-7) -> np.ndarray:
    """All 3 vertices within [lo, hi] (slightly padded) on one axis."""
    return np.all((c[:, :, axis] >= lo - tol) & (c[:, :, axis] <= hi + tol), axis=1)


def _orient_positive(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Flip tets with negative signed volume so skfem sees positively oriented cells."""
    t = t.copy()
    a, b, cc, d = p[:, t[0]], p[:, t[1]], p[:, t[2]], p[:, t[3]]
    vol = np.einsum("ij,ij->j",
                    np.cross(b - a, cc - a, axisa=0, axisb=0, axisc=0), d - a)
    neg = vol < 0
    if np.any(neg):
        t[[1, 2], neg] = t[[2, 1], neg]
    return t


# -------------------------------------------------------------- shared geometry logic

def _anode_width(geom) -> tuple[float, str | None]:
    """Effective anode width: narrowed until the plate fits inside the beaker wall."""
    R = geom.beaker_id / 2
    back_x = geom.gap + ANODE_T_CM
    avail = (R - WALL_MARGIN_CM) ** 2 - back_x ** 2
    if avail <= 0.0:
        raise ValueError(f"anode at gap {geom.gap:g} cm does not fit a "
                         f"{geom.beaker_id:g} cm ID beaker at all - reduce gap/anode")
    w_fit = 2.0 * math.sqrt(avail)
    if geom.anode_w <= w_fit:
        return geom.anode_w, None
    warn = (f"anode {geom.anode_w:g} cm wide at gap {geom.gap:g} cm does not fit the "
            f"{geom.beaker_id:g} ID beaker - narrowed to {w_fit:.2f} cm (active area "
            f"{w_fit * geom.anode_h:.1f} cm2 instead of "
            f"{geom.anode_w * geom.anode_h:.1f} cm2)")
    return w_fit, warn


def _stray_z_range(geom, H: float) -> tuple[float, float] | None:
    """(z0, z1) of the lumped wet-clip patch, or None if it cannot be placed [J]."""
    if geom.stray_area <= 0:
        return None
    hs = geom.stray_area / geom.coupon_w                 # patch width = coupon width [J]
    s0 = geom.coupon_center_z + geom.coupon_h / 2 + STRAY_STANDOFF
    if s0 + hs > H - 0.05:                               # slide down under the free surface
        s0 = H - 0.05 - hs
    if s0 <= 0.05:
        return None
    return s0, s0 + hs


def _z_clamped(z_lo: float, z_hi: float, H: float) -> tuple[float, float, bool]:
    """Keep a plate inside (0, H); returns (z_lo, z_hi, clamped?)."""
    lo, hi = max(z_lo, 0.05), min(z_hi, H - 0.05)
    return lo, hi, (lo, hi) != (z_lo, z_hi)


# ------------------------------------------------------------------- facet classification

def _classify_electrode_span(mesh: MeshTet, x0: float, x1: float,
                             tol: float = 1e-7) -> dict[str, np.ndarray]:
    """Duct: cathode = every facet on plane x = x0, anode = every facet on plane x = x1."""
    bnd = mesh.boundary_facets()
    c = _facet_coords(mesh, bnd)
    return {"cath_front": _on_plane(c, 0, x0, tol),
            "cath_back": np.zeros(len(bnd), bool),
            "cath_edge": np.zeros(len(bnd), bool),
            "stray": np.zeros(len(bnd), bool),
            "anode": _on_plane(c, 0, x1, tol)}


def _classify_box_patches(mesh: MeshTet, x1: float, geom, tol: float = 1e-7) -> dict[str, np.ndarray]:
    """gmsh-failure fallback: wall-mounted coupon patch on x = 0, anode patch on x = gap."""
    bnd = mesh.boundary_facets()
    c = _facet_coords(mesh, bnd)
    w, h, zc = geom.coupon_w, geom.coupon_h, geom.coupon_center_z
    aw, ah = geom.anode_w, geom.anode_h
    cath = (_on_plane(c, 0, 0.0, tol)
            & _within(c, 1, -w / 2, w / 2, tol)
            & _within(c, 2, zc - h / 2, zc + h / 2, tol))
    anode = (_on_plane(c, 0, x1, tol)
             & _within(c, 1, -aw / 2, aw / 2, tol)
             & _within(c, 2, zc - ah / 2, zc + ah / 2, tol))
    return {"cath_front": cath,
            "cath_back": np.zeros(len(bnd), bool),
            "cath_edge": np.zeros(len(bnd), bool),
            "stray": np.zeros(len(bnd), bool),
            "anode": anode}


def _classify_beaker(mesh: MeshTet, geom) -> dict[str, np.ndarray]:
    """Classify boundary facets of the beaker domain by their exact CAD planes."""
    bnd = mesh.boundary_facets()
    c = _facet_coords(mesh, bnd)
    t = COUPON_T_CM
    w, h, zc = geom.coupon_w, geom.coupon_h, geom.coupon_center_z
    z0, z1 = zc - h / 2, zc + h / 2
    aw_eff, _ = _anode_width(geom)
    az0, az1, _ = _z_clamped(zc - geom.anode_h / 2, zc + geom.anode_h / 2, geom.fill_height)
    tol = 1e-7

    x_front = _on_plane(c, 0, 0.0, tol)
    x_back = _on_plane(c, 0, -t, tol)
    in_y = _within(c, 1, -w / 2, w / 2, tol)
    in_z = _within(c, 2, z0, z1, tol)

    front = x_front & in_y & in_z

    # untaped sidewalls/back of the coupon: x spans [-t, 0], on a side plane
    span_x = _within(c, 0, -t, 0.0, tol) & ~x_front & ~x_back
    side_y = np.all(np.abs(np.abs(c[:, :, 1]) - w / 2) <= tol, axis=1)
    side_z = (np.all(np.abs(c[:, :, 2] - z0) <= tol, axis=1)
              | np.all(np.abs(c[:, :, 2] - z1) <= tol, axis=1))
    back = x_back & in_y & in_z
    edge = span_x & ((side_y & in_z) | (side_z & in_y))

    stray = np.zeros(len(bnd), bool)
    sz = _stray_z_range(geom, geom.fill_height)
    if sz is not None:
        stray = x_front & in_y & _within(c, 2, sz[0], sz[1], tol)

    anode = (_on_plane(c, 0, geom.gap, tol)
             & _within(c, 1, -aw_eff / 2, aw_eff / 2, tol)
             & _within(c, 2, az0, az1, tol))

    return {"cath_front": front, "cath_back": back, "cath_edge": edge,
            "stray": stray, "anode": anode}


# ------------------------------------------------------------------------- gmsh backend

def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


def _entities_within(bbox: tuple, dim: int) -> list[tuple[int, int]]:
    """gmsh entities of `dim` whose bounding box is contained in `bbox`.

    gmsh's getEntitiesInBoundingBox rejects entities whose bbox merely *touches* the
    query box, so the query is inflated slightly and a strict containment test keeps
    only entities genuinely inside `bbox` (this also filters out big entities such as
    the cylinder wall, whose bbox intersects everything).
    """
    import gmsh
    out = []
    q = 1e-3                                    # query inflation (cm)
    lo = [bbox[i] - q for i in range(3)]
    hi = [bbox[i + 3] + q for i in range(3)]
    for e in gmsh.model.getEntitiesInBoundingBox(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], dim):
        eb = gmsh.model.getBoundingBox(*e)
        if (eb[0] >= bbox[0] - 1e-6 and eb[1] >= bbox[1] - 1e-6
                and eb[2] >= bbox[2] - 1e-6 and eb[3] <= bbox[3] + 1e-6
                and eb[4] <= bbox[4] + 1e-6 and eb[5] <= bbox[5] + 1e-6):
            out.append(e)
    return out


def _build_beaker_gmsh(geom, resolution: str) -> CellMesh:
    import gmsh

    h_far, h_cath, h_anode, r_cath, r_anode = _RESOLUTIONS[resolution]
    R, H = geom.beaker_id / 2, geom.fill_height
    t = COUPON_T_CM
    warnings: list[str] = []

    # geometry feasibility + clamps (shared logic also used by facet classification)
    w = geom.coupon_w
    if w / 2 > R - WALL_MARGIN_CM:
        w_new = 2 * (R - WALL_MARGIN_CM)
        warnings.append(f"coupon {w:g} cm wide exceeds the beaker - narrowed to {w_new:g} cm")
        w = w_new
    z0, z1, cl = _z_clamped(geom.coupon_center_z - geom.coupon_h / 2,
                            geom.coupon_center_z + geom.coupon_h / 2, H)
    if cl:
        warnings.append("coupon plate does not fit the fill height - z extent clamped")
    aw, aw_warn = _anode_width(geom)
    if aw_warn:
        warnings.append(aw_warn)
    az0, az1, cl = _z_clamped(geom.coupon_center_z - geom.anode_h / 2,
                              geom.coupon_center_z + geom.anode_h / 2, H)
    if cl:
        warnings.append("anode plate does not fit the fill height - z extent clamped")
    stray_z = _stray_z_range(geom, H)
    if geom.stray_area > 0 and stray_z is None:
        warnings.append("stray_area patch does not fit above the coupon - stray patch skipped")

    t_start = time.perf_counter()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("beaker_cell")
        occ = gmsh.model.occ
        cyl = occ.addCylinder(0, 0, 0, 0, 0, H, R)
        coupon = occ.addBox(-t, -w / 2, z0, t, w, z1 - z0)
        anode = occ.addBox(geom.gap, -aw / 2, az0, ANODE_T_CM, aw, az1 - az0)
        boxes = [coupon, anode]
        if stray_z is not None:
            boxes.append(occ.addBox(-t, -w / 2, stray_z[0], t, w, stray_z[1] - stray_z[0]))
        occ.synchronize()
        # The electrode boxes are OCC solids, i.e. model volumes. Nested volumes inside
        # the cylinder break gmsh's 3-D region marking ("No elements in volume 1"), so
        # drop the solids while keeping their faces - those become free surfaces that we
        # embed into the cylinder below.
        gmsh.model.removeEntities([(3, tag) for tag in boxes], recursive=False)
        vols = gmsh.model.getEntities(3)
        if len(vols) != 1:
            raise RuntimeError(f"expected a single electrolyte volume, got {vols}")
        vol_tag = vols[0][1]

        # ---- electrode boxes: bbox (in cm) for entity queries
        box_bboxes = [
            (-t, -w / 2, z0, 0.0, w / 2, z1),
            (geom.gap, -aw / 2, az0, geom.gap + ANODE_T_CM, aw / 2, az1),
        ]
        if stray_z is not None:
            box_bboxes.append((-t, -w / 2, stray_z[0], 0.0, w / 2, stray_z[1]))

        # ---- embed the box surfaces in the cylinder volume so the volume mesh is
        #      conformal to the electrode faces (no OCC booleans needed)
        embed_surfs: list[int] = []
        for bb in box_bboxes:
            embed_surfs += [tag for d, tag in _entities_within(bb, 2) if d == 2]
        if not embed_surfs:
            raise RuntimeError("failed to locate electrode surfaces to embed")
        gmsh.model.mesh.embed(2, embed_surfs, 3, vol_tag)

        # ---- size fields: refine on / near the electrodes, coarse elsewhere
        f = gmsh.model.mesh.field
        ids = []
        for bb, h_min, reach in ((box_bboxes[0], h_cath, r_cath),
                                 (box_bboxes[1], h_anode, r_anode)):
            ents = _entities_within(bb, 1) + _entities_within(bb, 2)
            curves = [e for e in ents if e[0] == 1]
            surfaces = [e for e in ents if e[0] == 2]
            if not curves and not surfaces:
                continue
            d = f.add("Distance")
            if curves:
                f.setNumbers(d, "CurvesList", [e[1] for e in curves])
            if surfaces:
                f.setNumbers(d, "SurfacesList", [e[1] for e in surfaces])
            th = f.add("Threshold")
            f.setNumber(th, "InField", d)          # wire the distance field in!
            f.setNumber(th, "SizeMin", h_min)
            f.setNumber(th, "SizeMax", h_far)
            f.setNumber(th, "DistMin", 0.0)
            f.setNumber(th, "DistMax", reach)
            f.setNumber(th, "LcMin", h_min)
            f.setNumber(th, "LcMax", h_far)
            ids.append(th)
        if ids:
            if len(ids) > 1:
                mn = f.add("Min")
                f.setNumbers(mn, "FieldsList", ids)
                bg = mn
            else:
                bg = ids[0]
            f.setAsBackgroundMesh(bg)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        gmsh.model.mesh.generate(3)

        # ---- extract nodes + tets
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        ntags = np.asarray(ntags, dtype=np.int64)
        order = np.argsort(ntags)
        sorted_tags = ntags[order]
        coords = np.asarray(ncoords, dtype=float).reshape(-1, 3)[order]

        tet_nodes = None
        for etype, etags, ents_nodes in zip(*gmsh.model.mesh.getElements(3)):
            if int(etype) == 4:  # 4-node linear tetrahedra
                tet_nodes = np.asarray(ents_nodes, dtype=np.int64).reshape(-1, 4)
                break
        if tet_nodes is None:
            raise RuntimeError("gmsh produced no linear tetrahedra")
        idx = np.searchsorted(sorted_tags, tet_nodes)          # (n_tet, 4)
        if idx.max() >= coords.shape[0] or idx.min() < 0:
            raise RuntimeError("node tag mapping failed")

        # ---- delete tets inside the electrode solids (conformal embed => clean split)
        cent = coords[idx].mean(axis=1)                        # (n_tet, 3)
        inside = np.zeros(len(idx), bool)
        for bb in box_bboxes:
            inside |= np.all((cent >= np.array(bb[:3]) - 1e-12)
                             & (cent <= np.array(bb[3:]) + 1e-12), axis=1)
        idx = idx[~inside]
        if len(idx) == 0:
            raise RuntimeError("all tets were removed - electrode boxes engulf the domain?")

        # ---- compact to used nodes only
        uniq, inv = np.unique(idx, return_inverse=True)
        p = np.ascontiguousarray(coords[uniq].T)             # (3, N_used)
        tets = _orient_positive(p, inv.reshape(-1, 4).T.astype(np.int64))
    finally:
        gmsh.finalize()

    mesh = MeshTet(p, tets)
    groups = _classify_beaker(mesh, geom)
    return CellMesh(mesh=mesh, groups=groups, domain="gmsh_beaker",
                    warnings=warnings, mesh_time_s=time.perf_counter() - t_start)


# --------------------------------------------------------------- structured backends

def _axis_points(lo: float, hi: float, h_far: float, clusters: list[float],
                 h_near: float, band: float) -> np.ndarray:
    """Structured 1-D nodes: coarse base + refinement bands around cluster positions."""
    base = np.arange(lo, hi + 0.5 * h_far, h_far)
    pts = [base[(base >= lo) & (base <= hi)]]
    n = max(int(round(2 * band / h_near)), 2)
    for cpos in clusters:
        seg = np.linspace(max(cpos - band, lo), min(cpos + band, hi), n + 1)
        pts.append(seg)
    pts.append(np.array([lo, hi], float))
    return np.unique(np.clip(np.concatenate(pts), lo, hi))


def _build_duct(gap: float, w: float, h: float, n_xyz=(14, 10, 10)) -> CellMesh:
    """Rectangular duct: cathode spans the x = 0 face, anode the x = gap face, walls insulating."""
    x = np.linspace(0.0, gap, n_xyz[0] + 1)
    y = np.linspace(-w / 2, w / 2, n_xyz[1] + 1)
    z = np.linspace(0.0, h, n_xyz[2] + 1)
    mesh = MeshTet.init_tensor(x, y, z)
    groups = _classify_electrode_span(mesh, 0.0, gap)
    return CellMesh(mesh=mesh, groups=groups, domain="duct")


def _build_box_fallback(geom) -> CellMesh:
    """Structured box approximation of the beaker (used only if gmsh is unavailable).

    Electrodes are wall patches on the x = 0 / x = gap planes of a box of the beaker's
    diameter; the floating-plate back/edges do not exist in this approximation.
    """
    R, H = geom.beaker_id / 2, geom.fill_height
    gap = geom.gap
    w, h, zc = geom.coupon_w, geom.coupon_h, geom.coupon_center_z
    aw, ah = geom.anode_w, geom.anode_h
    x = _axis_points(0.0, gap, 1.0, [0.0, gap], 0.18, 0.30)
    y = _axis_points(-R, R, 1.0, [-w / 2, w / 2, -aw / 2, aw / 2], 0.18, 0.30)
    z = _axis_points(0.0, H, 1.0, [zc - h / 2, zc + h / 2, zc - ah / 2, zc + ah / 2],
                     0.18, 0.30)
    mesh = MeshTet.init_tensor(x, y, z)
    groups = _classify_box_patches(mesh, gap, geom)
    return CellMesh(mesh=mesh, groups=groups, domain="structured_box",
                    warnings=["gmsh unavailable - structured BOX approximation of the beaker: "
                              "coupon/anode are wall patches, untaped back/edges/stray not modelled"])


# ----------------------------------------------------------------------------- public

def build_cell_mesh(geom, resolution: str = "coarse") -> CellMesh:
    """Beaker mesh (gmsh), falling back to a structured box if gmsh cannot be imported."""
    if resolution not in _RESOLUTIONS:
        raise ValueError(f"resolution must be one of {sorted(_RESOLUTIONS)}, got {resolution!r}")
    if _gmsh_available():
        return _build_beaker_gmsh(geom, resolution)
    return _build_box_fallback(geom)


def build_duct_mesh(gap: float, w: float, h: float, n_xyz=(14, 10, 10)) -> CellMesh:
    return _build_duct(gap, w, h, n_xyz)
