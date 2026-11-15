"""Uniform cell-centred grid. All stored geometry is in centimetres."""
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np


def _default_bc():
    return dict(x_lo='periodic', x_hi='periodic', y_lo='periodic',
                y_hi='periodic', z_lo='bottom', z_hi='bulk')


@dataclass(frozen=True)
class Grid3D:
    shape: tuple[int, int, int]
    dx: float
    origin: tuple[float, float, float] = (0., 0., 0.)
    bc: dict[str, str] = field(default_factory=_default_bc)

    def __post_init__(self):
        if len(self.shape) != 3 or any(isinstance(n, bool) or
                not isinstance(n, Integral) or n <= 0 for n in self.shape):
            raise ValueError('shape must contain three positive integers')
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError('dx must be finite and positive (cm)')
        if len(self.origin) != 3 or not np.all(np.isfinite(self.origin)):
            raise ValueError('origin must contain three finite coordinates (cm)')
        bc = dict(self.bc)
        if set(bc) != set(_default_bc()):
            raise ValueError('bc keys must be exactly x_lo, x_hi, y_lo, y_hi, z_lo, z_hi')
        for key, value in bc.items():
            allowed = {'periodic', 'symmetry', 'halo'}
            if key.startswith('z_'):
                allowed |= {'bottom', 'bulk'}
            if value not in allowed:
                raise ValueError(f'bc[{key!r}]={value!r}; allowed: {sorted(allowed)}')
        for axis in 'xyz':
            if (bc[f'{axis}_lo'] == 'periodic') != (bc[f'{axis}_hi'] == 'periodic'):
                raise ValueError(f'bc periodic requires both sides of {axis}')
        object.__setattr__(self, 'shape', tuple(int(n) for n in self.shape))
        object.__setattr__(self, 'dx', float(self.dx))
        object.__setattr__(self, 'origin', tuple(float(v) for v in self.origin))
        object.__setattr__(self, 'bc', bc)

    @classmethod
    def from_um(cls, size_um, dx_um, origin_um=(0, 0, 0), bc=None):
        size_um = np.asarray(size_um, dtype=float)
        if size_um.shape != (3,) or not np.all(np.isfinite(size_um)) or np.any(size_um <= 0):
            raise ValueError('size_um must contain three finite positive lengths')
        if not np.isfinite(dx_um) or dx_um <= 0:
            raise ValueError('dx_um must be finite and positive')
        cells = size_um / dx_um
        if not np.allclose(cells, np.rint(cells), rtol=0, atol=1e-9):
            raise ValueError('size_um must be an integer multiple of dx_um on every axis')
        return cls(tuple(int(n) for n in np.rint(cells)), float(dx_um) * 1e-4,
                   tuple(float(v) * 1e-4 for v in origin_um),
                   _default_bc() if bc is None else bc)

    @property
    def periodic(self):
        return tuple(self.bc[f'{axis}_lo'] == 'periodic' for axis in 'xyz')

    def centers(self):
        return tuple(o + (np.arange(n, dtype=np.float64) + .5) * self.dx
                     for o, n in zip(self.origin, self.shape))

    def meshgrid(self):
        return tuple(np.meshgrid(*self.centers(), indexing='ij'))

    @property
    def cell_volume(self):
        return self.dx ** 3

    def to_dict(self):
        return dict(shape=list(self.shape), dx=self.dx, origin=list(self.origin), bc=dict(self.bc))

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d['shape']), d['dx'], tuple(d['origin']), d['bc'])
