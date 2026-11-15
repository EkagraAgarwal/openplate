"""Six-face electrolyte connectivity, with periodic x/y label merging."""
import numpy as np
from platesim.tile3d.levelset3d import label_voids

from .maps import _check


def _regions(psi, grid):
    psi = _check(psi, grid)
    bulk = np.zeros(psi.shape, dtype=bool)
    bulk[..., -1] = True
    labels, count = label_voids(psi, bulk, periodic=grid.periodic)
    # feature_report uses positive, non-trapped labels to find bulk access.
    # Keep that convention without a second connectivity implementation.
    labels[(psi > 0) & (labels == 0)] = count + 1
    return labels, np.arange(1, count + 1)


def void_report(psi, grid):
    """Volumes use cell occupancy; periodic centroids use circular means.

    seam flags a pocket with a transverse span at most two cells [J], not
    a physical classification of a sub-grid seam.
    """
    labels, ids = _regions(psi, grid)
    volumes, centroids, seams = [], [], []
    for ident in ids:
        cells = np.argwhere(labels == ident)
        volumes.append(len(cells) * grid.cell_volume * 1e12)
        center = []
        for a in range(3):
            q = cells[:, a] + .5
            if a < 2 and grid.periodic[a]:
                angle = np.angle(np.mean(np.exp(2j * np.pi * q / grid.shape[a])))
                qmean = (angle % (2 * np.pi)) * grid.shape[a] / (2 * np.pi)
            else:
                qmean = np.mean(q)
            center.append((grid.origin[a] + qmean * grid.dx) * 1e4)
        centroids.append(center)
        seams.append(bool(min(len(np.unique(cells[:, a])) for a in (0, 1)) <= 2))
    return dict(count=len(ids), volumes_um3=np.asarray(volumes),
                centroids_um=np.asarray(centroids).reshape(-1, 3),
                seam=any(seams), seam_flags=seams)
