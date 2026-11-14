"""Portable surface/volume artifacts. Coordinates and scalar psi stay in cm."""
import base64
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from skimage.measure import marching_cubes

from .maps import _check


def export_surface(psi, grid, path, fmt='stl'):
    fmt = fmt.lower()
    if fmt not in ('stl', 'vtp', 'gltf'):
        raise ValueError('fmt must be stl, vtp or gltf')
    package = None
    if fmt != 'stl':
        try:
            if fmt == 'vtp':
                import pyvista as package
            else:
                import trimesh as package
        except ImportError as exc:
            name = 'pyvista' if fmt == 'vtp' else 'trimesh'
            raise ImportError(f'{fmt} export requires optional package {name}') from exc
    vertices, faces, _, _ = marching_cubes(_check(psi, grid), level=0,
                                          spacing=(grid.dx,) * 3)
    vertices += np.asarray(grid.origin) + grid.dx / 2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'stl':
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.where(lengths > 0, lengths, 1)[:, None]
        records = np.zeros(len(faces), dtype=[('normal', '<f4', (3,)),
                           ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
        records['normal'], records['vertices'] = normals, triangles
        with path.open('wb') as stream:
            stream.write(b'platesim surface; coordinates cm'.ljust(80, b'\0'))
            stream.write(np.asarray(len(faces), dtype='<u4').tobytes())
            stream.write(records.tobytes())
    elif fmt == 'vtp':
        package.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces))).save(path)
    else:
        # Embedded buffers avoid untracked .bin sidecars.
        from trimesh.exchange.gltf import export_gltf
        mesh = package.Trimesh(vertices=vertices, faces=faces, process=False)
        files = export_gltf(package.Scene(mesh), embed_buffers=True)
        path.write_bytes(files['model.gltf'])
    return dict(kind=fmt, path=str(path), title='Tile surface (cm)')


def _vti(result, index, path):
    grid = result.grid
    extent = ' '.join(f'0 {n}' for n in grid.shape)
    root = ET.Element('VTKFile', type='ImageData', version='1.0',
                      byte_order='LittleEndian', header_type='UInt32')
    image = ET.SubElement(root, 'ImageData', WholeExtent=extent,
                          Origin=' '.join(map(str, grid.origin)),
                          Spacing=' '.join([str(grid.dx)] * 3))
    piece = ET.SubElement(image, 'Piece', Extent=extent)
    cells = ET.SubElement(piece, 'CellData', Scalars='psi')
    fields = {'psi': result.psi(index), 'psi_si': result.psi_si}
    # TileResult has no field-name listing API; discover its optional NPZ fields.
    suffix = f'_{index:04d}.npz'
    for file in sorted((result.run_dir / 'fields').glob(f'*{suffix}')):
        name = file.name[:-len(suffix)]
        value = result.field(name, index)
        if value is not None:
            fields[name] = value
    for name, value in fields.items():
        # VTK's x index varies fastest, unlike the stored C-order (x,y,z).
        raw = np.asarray(value, dtype='<f4').tobytes(order='F')
        encoded = base64.b64encode(np.asarray(len(raw), dtype='<u4').tobytes() + raw).decode('ascii')
        node = ET.SubElement(cells, 'DataArray', type='Float32', Name=name,
                             NumberOfComponents='1', format='binary')
        node.text = encoded
    data = ET.SubElement(piece, 'FieldData')
    ET.SubElement(data, 'DataArray', type='Float64', Name='TimeValue',
                  NumberOfTuples='1', format='ascii').text = str(result.times_s[index])
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def export_volume(result, path):
    """One VTI per frame; path names the final frame, earlier files gain _NNNN.

    Returns the final artifact plus a frames list of all artifact dictionaries.
    """
    path = Path(path)
    if not len(result.times_s):
        raise ValueError('export_volume requires at least one frame')
    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for index in range(len(result.times_s)):
        target = path if index == len(result.times_s) - 1 else path.with_name(f'{path.stem}_{index:04d}.vti')
        _vti(result, index, target)
        artifacts.append(dict(kind='vti', path=str(target), title=f'Tile volume at {result.times_s[index]:g} s (cm)'))
    return {**artifacts[-1], 'frames': artifacts}
