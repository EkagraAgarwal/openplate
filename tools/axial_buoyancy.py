"""Uncoupled buoyancy estimate; run from the root with PYTHONPATH=.

Prescribed ct is 1 at/above the mouth, decreasing linearly to .75 at the
floor. This is an estimate, not a self-consistent transport/flow solution.
"""
import numpy as np

from platesim.flow3d import buoyancy_force, solve_stokes3d
from platesim.tile3d.fields import FieldSolver
from platesim.trenchflow.geometry import trench_grid


def main():
    width, depth, margin, delta, dx_um, ny = 227., 88., 100., 64., 4., 4
    grid = trench_grid(width, depth, pitch_um=width + 2 * margin,
                       height_um=delta, dx_um=dx_um)
    dx = dx_um * 1e-4
    psi = np.repeat(np.where(grid.solid.T[:, None, :], -.5, .5), ny, axis=1) * dx
    ct = np.broadcast_to(1. - .25 * np.clip(-grid.z_um[None, None, :] / depth, 0., 1.), psi.shape)
    diffusivity = float(FieldSolver().ph.D_c)
    print('Nominal trench: 227 x 88 um; field margin 100 um; lid 64 um above mouth; dx 4 um; ny 4')
    print('ct: prescribed linear 1 (mouth) -> 0.75 (floor); c_bulk=0.63 mol/L; g=981 cm/s^2; no shear')
    print(f'D_c={diffusivity:.8g} cm^2/s (FieldSolver default derived parameters); nu=0.0114 cm^2/s')
    print('Pe_y=max|uy|*depth/D_c; Pe_axis=max|u_axis|*depth/D_c; speeds are cell-centred')
    print('| beta (L/mol) | gravity | max abs uy (um/s) | Pe_y | max abs u_axis (um/s) | Pe_axis |')
    print('| ---: | :--- | ---: | ---: | ---: | ---: |')
    for beta in (.06, .07, .08):
        for label, direction, component in (('-y (axial)', (0, -1, 0), 'uy'),
                                             ('-x (across)', (-1, 0, 0), 'ux')):
            force = buoyancy_force(ct, .63, beta, direction)
            result = solve_stokes3d(psi, dx, (grid.z_um[0] - dx_um / 2) * 1e-4,
                                    delta * 1e-4, body_force=force)
            uy = float(np.max(np.abs(result['uy'])))
            speed = float(np.max(np.abs(result[component])))
            pe_y, pe_axis = (v * depth * 1e-4 / diffusivity for v in (uy, speed))
            print(f'| {beta:.2f} | {label} | {uy * 1e4:.6g} | {pe_y:.6g} | {speed * 1e4:.6g} | {pe_axis:.6g} |')


if __name__ == '__main__':
    main()
