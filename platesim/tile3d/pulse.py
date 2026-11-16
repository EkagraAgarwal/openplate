"""Frozen-front pulse transport (cm, s, mol, A).

Only the oscillatory flux is resolved here: the DC component belongs to the
outer 3-D diffusion problem. For D c_x(0,t)=J(t), the semi-infinite solution is
 c_n(0) = -J_n / sqrt(i omega_n D), n != 0.
Analytic rectangular-wave Fourier coefficients and phase integrals avoid a
time grid, start-up cycles, and arrays of size cells x harmonics. Phase means
converge as n**(-5/2); 512 harmonics are inexpensive (a 3 x 3 matrix).
The independent-normal-column approximation requires sqrt(D T) small compared
with lateral curvature scales. It is not a grain/nucleation model.
Pulse-diffusion background: Cheh, J. Electrochem. Soc. 118 (1971);
Ibl, Surface Technology (1980). The Fourier/Abel equations above
are derived directly from the diffusion PDE, not fitted to a roughness outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class Waveform:
    durations: tuple[float, float, float]
    currents: tuple[float, float, float]
    model: str = 'resolved'

    @classmethod
    def parse(cls, value, net_A_cm2):
        allowed = {'t_f_ms', 't_r_ms', 't_off_ms', 'j_f_mA_cm2', 'j_r_mA_cm2', 'model'}
        if set(value) - allowed:
            raise ValueError('unknown pulse keys: ' + str(set(value) - allowed))
        model = value.get('model', 'resolved')
        if model not in ('net', 'resolved'):
            raise ValueError('pulse model must be net or resolved')
        ts = tuple(float(value[k]) * .001 for k in ('t_f_ms', 't_r_ms', 't_off_ms'))
        jf, jr = (float(value[k]) * .001 for k in ('j_f_mA_cm2', 'j_r_mA_cm2'))
        if not np.isfinite((*ts, jf, jr)).all() or ts[0] <= 0 or min(ts) < 0 or jf <= 0 or jr < 0:
            raise ValueError('finite nonnegative pulse times/currents and positive forward phase required')
        wave = cls(ts, (jf, -jr, 0.), model)
        if wave.net <= 0 or not np.isclose(wave.net, net_A_cm2, rtol=1e-9, atol=1e-14):
            raise ValueError('pulse average must equal the supplied positive net current')
        return wave

    @property
    def weights(self):
        return np.array(self.durations) / sum(self.durations)

    @property
    def net(self):
        return float(np.dot(self.weights, self.currents))

    @property
    def dc(self):
        return self.durations[1] == self.durations[2] == 0


@lru_cache(maxsize=32)
def diffusion_response(durations, D, harmonics=512):
    """Map piecewise molar fluxes to phase-mean surface c deviations (mol/cm3).

    The weighted mean correction is exactly zero. Zero-length phases have zero
    rows/columns. No clipping: negative concentrations mean the model is invalid.
    """
    dt = np.array(durations)
    T = dt.sum()
    edges = np.r_[0., np.cumsum(dt)]
    omega = 2 * np.pi * np.arange(1, harmonics + 1) / T
    coeff = (np.exp(-1j * omega[:, None] * edges[:-1])
             - np.exp(-1j * omega[:, None] * edges[1:])) / (1j * omega[:, None] * T)
    mean = np.zeros_like(coeff)
    valid = dt > 0
    mean[:, valid] = (np.exp(1j * omega[:, None] * edges[1:][valid])
                     - np.exp(1j * omega[:, None] * edges[:-1][valid])) / (1j * omega[:, None] * dt[valid])
    response = -2 * np.real(mean.T @ (coeff / np.sqrt(1j * omega[:, None] * D)))
    response -= (dt / T) @ response  # remove roundoff in the DC component
    response[~valid] = 0
    response.setflags(write=False)
    return response


def surface_concentration(mean, currents, durations, D, c_bulk, efficiency=1.):
    """currents: signed cathodic-positive A/cm2, shape (3, ...); mean is c/c_bulk."""
    from ..core.constants import F
    response = diffusion_response(tuple(durations), D)
    return np.asarray(mean)[None, ...] + np.einsum('kl,l...->k...', response, currents) * (
        efficiency / (2 * F * c_bulk))


def surface_at(times, currents, durations, D, harmonics=8192):
    """Oscillatory surface c at specified times, for scalar molar flux per phase.

    Useful for analytic validation; excludes the mean flux's outer DC depletion.
    """
    T = sum(durations)
    edges = np.r_[0., np.cumsum(durations)]
    times = np.asarray(times)
    def partial(count):
        omega = 2 * np.pi * np.arange(1, count + 1) / T
        coeff = ((np.exp(-1j * omega[:, None] * edges[:-1])
                  - np.exp(-1j * omega[:, None] * edges[1:])) @ np.asarray(currents)) / (1j * omega * T)
        return -2 * np.real(np.exp(1j * times[..., None] * omega)
                            @ (coeff / np.sqrt(1j * omega * D)))
    value = partial(harmonics)
    # At a flux discontinuity c(t) has a square-root cusp: the unaveraged
    # Fourier tail is O(N**-1/2), unlike the fast-converging phase means.
    # Richardson at N,4N cancels that leading Abel tail, leaving O(N**-3/2).
    endpoint = np.any(np.isclose(np.mod(times[..., None], T), edges,
                                 atol=1e-12*T, rtol=0), axis=-1)
    if np.any(endpoint):
        value = np.where(endpoint, 2 * partial(4 * harmonics) - value, value)
    return value


def capacitive_cycle(durations, currents, C_dl, i0, alpha_c_f, alpha_a_f,
                     concentrations, area_ratio=1., rtol=2e-7, method='LSODA'):
    """Periodic lumped interfacial RC/BV solve, not an instantaneous eta jump.

    Cathodic-positive convention: j_app = j_F - C_dl*deta/dt, with
      j_F = i0 [ct exp(-alpha_c_f eta) - exp(alpha_a_f eta)].
    Currents/charges are per projected area; C_dl is per wetted area.
    Each phase holds its area-mean ct fixed. This circuit captures charging and
    off-time Faradaic relaxation, but not spatially resolved transient RC modes.
    Full BV is essential here: a single Tafel branch cannot relax to equilibrium.
    Phase Faradaic charges follow exactly from endpoint capacitor storage; no
    clipping or rescaling of reverse charge, and no loss of net cycle charge.
    Cdl=0 computes the same *diagnostic* transient at 30 uF/cm2 while leaving
    returned Faradaic currents identical to the applied waveform.

    Cdl range/context: literature 25-40 uF/cm2 for Cu in acid sulfate (~30% of the
    reverse charge is an estimate, not an enforced calibration target).
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    from math import exp
    dt = np.asarray(durations, dtype=float)
    applied = np.asarray(currents, dtype=float) / area_ratio
    ct = np.asarray(concentrations, dtype=float)
    C = C_dl or 30e-6
    def faradaic(eta, k):
        cath = i0 * ct[k] * exp(min(max(-alpha_c_f * eta, -60.), 60.))
        anod = i0 * exp(min(max(alpha_a_f * eta, -60.), 60.))
        return cath - anod, alpha_c_f * cath + alpha_a_f * anod
    live = np.flatnonzero(dt > 0)
    start = brentq(lambda e: faradaic(e, live[0])[0] - applied[live[0]], -3., 3.)
    ends, starts = np.zeros(3), np.zeros(3)
    for cycle in range(40):
        previous = start
        for k in live:
            starts[k] = start
            solution = solve_ivp(lambda t, y: np.array([(faradaic(y[0], k)[0] - applied[k]) / C]),
                (0., dt[k]), [start], method=method, rtol=rtol, atol=rtol * 1e-3,
                jac=lambda t, y: np.array([[-faradaic(y[0], k)[1] / C]]))
            if not solution.success:
                raise RuntimeError('double-layer integration failed: ' + solution.message)
            start = ends[k] = float(solution.y[0, -1])
        if abs(start - previous) < 1e-10:
            break
    else:
        raise RuntimeError('double-layer circuit did not reach a periodic steady state')
    q_cap = -C * area_ratio * (ends - starts)
    q_cap[dt == 0] = 0.
    faradaic_currents = np.asarray(currents, dtype=float).copy()
    if C_dl:
        faradaic_currents[live] -= q_cap[live] / dt[live]
    charge_tau = [float(abs(q_cap[k]) / abs(currents[k]))
                  if dt[k] and currents[k] else None for k in range(3)]
    relaxation_tau = [float(C / faradaic(ends[k], k)[1]) if dt[k] else None for k in range(3)]
    reverse_charge = abs(currents[1]) * dt[1]
    report = dict(C_dl_F_cm2=C_dl, C_dl_report_F_cm2=C, eta_start_V=starts.tolist(), eta_end_V=ends.tolist(),
        tau_s=charge_tau, differential_tau_s=relaxation_tau,
        capacitive_charge_C_cm2=(q_cap if C_dl else np.zeros(3)).tolist(),
        diagnostic_capacitive_charge_C_cm2=q_cap.tolist(),
        reverse_capacitive_fraction=float(q_cap[1] / (currents[1] * dt[1])) if reverse_charge and C_dl else 0.,
        diagnostic_reverse_capacitive_fraction=float(q_cap[1] / (currents[1] * dt[1])) if reverse_charge else 0.,
        periodic_eta_error_V=abs(start - previous),
        cycle_charge_error_C_cm2=float(np.dot(faradaic_currents - currents, dt)),
        approximation='lumped area-mean periodic RC/BV; frozen phase concentration; not spatial transient RC')
    return faradaic_currents, report
