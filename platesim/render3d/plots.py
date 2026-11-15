"""Agg-only plots; no GUI backend or pyplot global state."""
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ._geom import COPPER, SILICON


def figure(size):
    fig = Figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    FigureCanvasAgg(fig)
    return fig


def save(fig, path, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, format='png', dpi=100, facecolor='white')
    finally:
        fig.clear()
    return dict(kind='png', path=str(path), title=title)


def render_height_map(height_um, x_um, y_um, out_png, title=None,
                      vmin=None, vmax=None, *, size=(1280, 960)):
    title = title or 'Top surface height'
    fig = figure(size)
    ax = fig.add_subplot()
    im = ax.pcolormesh(x_um, y_um, np.asarray(height_um).T, shading='auto',
                       cmap='copper', vmin=vmin, vmax=vmax)
    ax.set(xlabel='x (µm)', ylabel='y (µm)', title=title, aspect='equal')
    fig.colorbar(im, ax=ax, label='Height (µm)')
    fig.tight_layout()
    return save(fig, out_png, title)


def render_cross_section(section, out_png, sem_overlay=None, *, size=(1280, 960)):
    s, z = section['s_um'], section['z_um']
    psi, bare = np.asarray(section['psi']), np.asarray(section['psi_si'])
    fig = figure(size)
    ax = fig.add_subplot()
    # Both solids share the exact interpolated zero contours used below.
    # Si is painted last, subtracting it from Cu (psi <= 0, psi_si > 0).
    # Avoid contouring max(psi, -bare): that shifts thin-layer intersections.
    ax.contourf(s, z, psi.T, levels=[-np.inf, 0], colors=[COPPER])
    ax.contourf(s, z, bare.T, levels=[-np.inf, 0], colors=[SILICON])
    for a, color in ((bare, '#454545'), (psi, '#754120')):
        if a.min() < 0 < a.max():
            ax.contour(s, z, a.T, levels=[0], colors=[color], linewidths=.8)
    if sem_overlay is not None and len(sem_overlay):
        points = np.asarray(sem_overlay)
        ax.plot(points[:, 0], points[:, 1], 'c.-', label='Outline overlay')
        ax.legend()
    ax.set(xlabel='s (µm)', ylabel='z (µm)', title='Vertical section', aspect='equal')
    fig.tight_layout()
    return save(fig, out_png, 'Vertical section')
