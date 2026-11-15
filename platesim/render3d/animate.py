"""Streaming GIF creation, one frame per saved simulation time."""
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as iio
import numpy as np

from .surface import render_surface


def render_growth_animation(result, out_path, camera='iso', fps=4, *,
                            size=(1280, 960), renderer='auto', **surface_options):
    """Write a GIF; temporary PNGs are removed even on failure.

    GIF only (not mp4). Auto falls back once for the remaining frames when an
    OpenGL context fails. Fixed cameras use the grid, never changing mesh bounds.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    if not len(result.times_s):
        raise ValueError('animation requires at least one saved time')
    path = Path(out_path)
    if path.suffix.lower() != '.gif':
        raise ValueError('growth animation supports .gif output')
    path.parent.mkdir(parents=True, exist_ok=True)
    used, reasons = [], []
    with TemporaryDirectory(prefix='.gif_', dir=path.resolve().parent) as temp:
        with iio.get_writer(path, mode='I', duration=1000/fps, loop=0) as writer:
            for index in range(len(result.times_s)):
                p = Path(temp)/'frame.png'
                artifact = render_surface(result, index=index, out_png=p, camera=camera,
                                          size=size, renderer=renderer, **surface_options)
                used.append(artifact['renderer'])
                if 'fallback_reason' in artifact:
                    reasons.append(artifact['fallback_reason'])
                    renderer = 'matplotlib'
                writer.append_data(iio.imread(p))
    return dict(kind='gif', path=str(path), title='Tile growth',
                frames=len(result.times_s), fps=float(fps), camera=artifact['camera'],
                renderer=used[0] if len(set(used)) == 1 else 'mixed',
                fallback_reasons=reasons)
