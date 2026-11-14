"""Streaming NPZ tile results; only scalar time series are retained in memory."""
import json
from pathlib import Path
import re

import numpy as np

from .grid import Grid3D


def _write_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
        raise ValueError(f'invalid field/scalar name: {name!r}')
    if name == 't_s':
        raise ValueError('t_s is reserved for frame times')
    return name


class TileResultWriter:
    def __init__(self, run_dir, grid: Grid3D, psi_si: np.ndarray, meta: dict | None = None):
        self.run_dir = Path(run_dir)
        self.grid = grid
        bare = self._array(psi_si)
        self._manifest = dict(meta or {})
        json.dumps(self._manifest, allow_nan=False)
        if any((self.run_dir / p).exists() for p in
               ('grid.json', 'static.npz', 'frames', 'fields', 'series.json', 'manifest.json')):
            raise FileExistsError(f'result already exists in {self.run_dir}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'frames').mkdir()
        (self.run_dir / 'fields').mkdir()
        _write_json(self.run_dir / 'grid.json', grid.to_dict())
        np.savez_compressed(self.run_dir / 'static.npz', psi_si=bare)
        self._series = {'t_s': []}
        self._closed = False
        _write_json(self.run_dir / 'manifest.json', self._manifest)
        _write_json(self.run_dir / 'series.json', self._series)

    def _array(self, value):
        array = np.asarray(value, dtype=np.float32, order='C')
        if array.shape != self.grid.shape or not np.all(np.isfinite(array)):
            raise ValueError(f'field must be finite with shape {self.grid.shape}')
        return array

    def add_frame(self, t_s: float, psi: np.ndarray,
                  fields: dict[str, np.ndarray] | None = None,
                  scalars: dict[str, float] | None = None) -> int:
        if self._closed:
            raise RuntimeError('writer is closed')
        t_s = float(t_s)
        times = self._series['t_s']
        if not np.isfinite(t_s) or t_s < 0 or (times and t_s <= times[-1]):
            raise ValueError('t_s must be finite, nonnegative and strictly increasing')
        psi = self._array(psi)
        fields = {_name(k): self._array(v) for k, v in (fields or {}).items()}
        scalars = {_name(k): float(v) for k, v in (scalars or {}).items()}
        if not all(np.isfinite(v) for v in scalars.values()):
            raise ValueError('scalars must be finite')
        index = len(times)
        stamp = np.float64(t_s)
        np.savez_compressed(self.run_dir / 'frames' / f'psi_{index:04d}.npz', psi=psi, t_s=stamp)
        for name, array in fields.items():
            np.savez_compressed(self.run_dir / 'fields' / f'{name}_{index:04d}.npz',
                                **{name: array, 't_s': stamp})
        for name in scalars:
            if name not in self._series:
                self._series[name] = [None] * index
        for name, values in self._series.items():
            values.append(t_s if name == 't_s' else scalars.get(name))
        _write_json(self.run_dir / 'series.json', self._series)
        return index

    def close(self, manifest: dict | None = None) -> None:
        combined = {**self._manifest, **(manifest or {})}
        _write_json(self.run_dir / 'manifest.json', combined)
        _write_json(self.run_dir / 'series.json', self._series)
        self._manifest = combined
        self._closed = True


class TileResult:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.grid = Grid3D.from_dict(self._json('grid.json'))
        self.series = self._json('series.json')
        self.manifest = self._json('manifest.json')
        self.times_s = np.asarray(self.series['t_s'], dtype=np.float64)
        if (self.times_s.ndim != 1 or not np.all(np.isfinite(self.times_s)) or
                np.any(self.times_s < 0) or np.any(np.diff(self.times_s) <= 0)):
            raise ValueError('invalid series t_s')
        if any(len(values) != len(self.times_s) for values in self.series.values()):
            raise ValueError('series lengths must match t_s')
        with np.load(self.run_dir / 'static.npz', allow_pickle=False) as saved:
            self.psi_si = self._check(saved['psi_si'])

    def _json(self, name):
        return json.loads((self.run_dir / name).read_text(encoding='utf-8'))

    def _check(self, array):
        if array.shape != self.grid.shape or array.dtype != np.float32:
            raise ValueError('stored field must be float32 with grid shape')
        return np.ascontiguousarray(array)

    def _index(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError('frame index must be an integer')
        if index < 0:
            index += len(self.times_s)
        if not 0 <= index < len(self.times_s):
            raise IndexError('frame index out of range')
        return index

    def _load(self, path, key, index):
        with np.load(path, allow_pickle=False) as saved:
            if saved['t_s'].shape != () or float(saved['t_s']) != self.times_s[index]:
                raise ValueError('frame timestamp does not match series')
            return self._check(saved[key])

    def psi(self, index: int = -1) -> np.ndarray:
        index = self._index(index)
        return self._load(self.run_dir / 'frames' / f'psi_{index:04d}.npz', 'psi', index)

    def field(self, name: str, index: int = -1) -> np.ndarray | None:
        name = _name(name)
        index = self._index(index)
        path = self.run_dir / 'fields' / f'{name}_{index:04d}.npz'
        return self._load(path, name, index) if path.exists() else None
