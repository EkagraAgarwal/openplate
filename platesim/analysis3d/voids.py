"""Six-face electrolyte connectivity, with periodic x/y label merging."""
import numpy as np
from scipy.ndimage import label

from .maps import _check


def _regions(psi, grid):
    labels, count = label(_check(psi, grid) > 0)
    parent = np.arange(count + 1)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for axis in (0, 1):
        if not grid.periodic[axis]:
            continue
        lo, hi = np.take(labels, 0, axis), np.take(labels, -1, axis)
        pairs = np.unique(np.column_stack((lo.ravel(), hi.ravel())), axis=0)
        for a, b in pairs:
            if a and b:
                parent[root(b)] = root(a)
    lookup = np.array([root(i) for i in range(count + 1)])
    labels = lookup[labels]
    bulk = np.unique(labels[..., -1])
    return labels, np.setdiff1d(np.unique(labels), np.r_[0, bulk])


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
