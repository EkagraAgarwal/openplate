"""Plating recipes: lists of steps.

A step is a dict-like dataclass:
    duration_s : length of the step
    j_mA_cm2   : galvanostatic set current, mA per cm2 of PROJECTED unit-cell area (net, i.e. time-averaged
                 over a pulse-reverse cycle), or None for a potentiostatic step
    V_mV       : potentiostatic step: metal potential vs the reference at the delta plane (cathodic negative)
    pr         : None or {'t_f_ms', 't_r_ms', 't_off_ms', 'rr'} pulse-reverse; period
                 t_f_ms + t_r_ms + t_off_ms ('t_off_ms' optional, default 0.0 = no off time), the reverse
                 current is rr * forward current.  Duty-averaged: no sub-ms transient [J].
    stir       : None or {'delta_factor', 'k_mix_factor'} and/or {'delta_um', 'k_mix'} absolute overrides
Constructors: dc, pulse_reverse, team_staged, plan_ramp, plan_ramp_pr, potentiostatic, scaled."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from typing import Optional


@dataclass
class Step:
    duration_s: float
    j_mA_cm2: Optional[float] = None
    V_mV: Optional[float] = None
    pr: Optional[dict] = None
    stir: Optional[dict] = None
    label: str = ''

    def __post_init__(self):
        if (self.j_mA_cm2 is None) == (self.V_mV is None):
            raise ValueError('a step needs exactly one of j_mA_cm2 (galvanostatic) or V_mV (potentiostatic)')


@dataclass
class Recipe:
    name: str
    steps: list = field(default_factory=list)

    @property
    def duration_s(self):
        return sum(s.duration_s for s in self.steps)

    def to_dict(self):
        return {'name': self.name, 'steps': [asdict(s) for s in self.steps]}


# --------------------------------------------------------------------------- pulse reverse control
def pr_forward(j_net, pr):
    """Duty-averaged pulse-reverse control from a NET (time-averaged) current density j_net mA/cm2 and a
    pr dict {'t_f_ms', 't_r_ms', ['t_off_ms'], 'rr'}.

    Returns (j_f, duty_f, duty_r): instantaneous forward current density and the forward / reverse duty
    fractions over the period P = t_f + t_r + t_off.  With toff = 0 this is exactly
    j*(tf+tr)/(tf-rr*tr), tf/(tf+tr), tr/(tf+tr).  [J] duty-averaged model: no sub-millisecond transient
    and no double-layer charging."""
    tf, tr, rr = float(pr['t_f_ms']), float(pr['t_r_ms']), float(pr['rr'])
    toff = float(pr.get('t_off_ms', 0.0))
    P = tf + tr + toff
    return j_net * P / (tf - rr * tr), tf / P, tr / P


# --------------------------------------------------------------------------- constructors
def dc(j, t_max_h):
    """Single DC step at j mA/cm2 for up to t_max_h hours (the run stops earlier once the via is filled)."""
    return Recipe(f'dc_{j:g}mA_{t_max_h:g}h', [Step(t_max_h * 3600.0, j, label='dc')])


def pulse_reverse(j_net_mA_cm2, t_max_h, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0) -> Recipe:
    """Single pulse-reverse step with an OFF time, at a NET current density j_net_mA_cm2 for up to t_max_h
    hours (the run stops earlier once the feature is filled).

    Waveform: forward t_f_ms at j_f, reverse t_r_ms at rr * j_f, off t_off_ms at 0, repeated with period
    P = t_f + t_r + t_off.  j_f = j_net * P / (t_f - rr * t_r) so the cycle average is j_net.  Defaults are
    the team waveform (9 / 0.5 / 0.5 ms, rr = 2).  [J] The model is duty-averaged: no sub-millisecond
    transient and no double-layer charging (the interface speed uses the cycle-averaged current only)."""
    if (t_f_ms - rr * t_r_ms) <= 0:
        raise ValueError(f'net current not cathodic: t_f_ms ({t_f_ms:g}) - rr*t_r_ms '
                         f'({rr:g}*{t_r_ms:g}) <= 0')
    pr = {'t_f_ms': t_f_ms, 't_r_ms': t_r_ms, 't_off_ms': t_off_ms, 'rr': rr}
    return Recipe(f'pr_{j_net_mA_cm2:g}mA_{t_max_h:g}h', [Step(t_max_h * 3600.0, j_net_mA_cm2, pr=pr,
                                                              label='pr')])


def team_staged():
    """Original team recipe: 1 mA/cm2 for 30 s, 5 mA/cm2 for 3 h, 15 mA/cm2 for 3 h."""
    return Recipe('team_staged', [Step(30.0, 1.0, label='entry'), Step(3 * 3600.0, 5.0, label='5mA'),
                                  Step(3 * 3600.0, 15.0, label='15mA')])


_RAMP = {
    40: ([3.10, 3.35, 3.85, 4.75, 8.58], [80, 63, 40, 26, 24]),
    25: ([2.60, 2.70, 3.00, 3.58, 6.18], [114, 91, 57, 38, 35]),
}


def _ramp_tables(T):
    key = 40 if T >= 32 else 25          # 22 and 25 C share the same table
    return _RAMP[key]


def plan_ramp(T):
    """Fill-plan ramp (KB section 5): 10 min entry at 0.5 mA/cm2, then five rising steps tracking the front.
    The last step halves the stirring (delta * 1.4, k_mix * 0.7)."""
    js, ts = _ramp_tables(T)
    steps = [Step(600.0, 0.5, label='entry')]
    for i, (j, t) in enumerate(zip(js, ts)):
        stir = {'delta_factor': 1.4, 'k_mix_factor': 0.7} if i == len(js) - 1 else None
        steps.append(Step(t * 60.0, j, stir=stir, label=f'ramp{i + 1}'))
    return Recipe(f'plan_ramp_{T:g}C', steps)


def plan_ramp_pr(T, t_f_ms=20.0, t_r_ms=1.0, rr=3.0, dc_first_min=30.0):
    """Same NET currents as plan_ramp, but pulse-reverse (20 ms forward / 1 ms reverse at 3x forward) after the
    first 30 min of DC (10 min entry + first 20 min of ramp step 1)."""
    base = plan_ramp(T)
    pr = {'t_f_ms': t_f_ms, 't_r_ms': t_r_ms, 'rr': rr}
    steps, t_acc = [], 0.0
    for s in base.steps:
        t0, t1 = t_acc, t_acc + s.duration_s
        t_acc = t1
        cut = dc_first_min * 60.0
        if t1 <= cut or s.label == 'entry':
            steps.append(replace(s))
        elif t0 >= cut:
            steps.append(replace(s, pr=dict(pr)))
        else:                                       # straddles the 30 min mark: split
            steps.append(replace(s, duration_s=cut - t0, stir=None, label=s.label + '_dc'))
            steps.append(replace(s, duration_s=t1 - cut, pr=dict(pr), label=s.label + '_pr'))
    return Recipe(f'plan_ramp_pr_{T:g}C', steps)


def potentiostatic(steps):
    """Potentiostatic recipe from [(duration_s, V_mV), ...]; V is the metal potential vs the reference at the
    delta plane (cathodic negative, e.g. -150)."""
    return Recipe('potentiostatic', [Step(float(t), None, V_mV=float(v), label='pstat') for t, v in steps])


def scaled(recipe, current_factor=1.0, time_factor=1.0):
    """Copy of a recipe with every galvanostatic current multiplied by current_factor and every duration by
    time_factor (potentiostatic set potentials are left unchanged).  Used to design faster recipes."""
    out = []
    for s in recipe.steps:
        j = None if s.j_mA_cm2 is None else s.j_mA_cm2 * current_factor
        out.append(replace(s, j_mA_cm2=j, duration_s=s.duration_s * time_factor,
                           pr=None if s.pr is None else dict(s.pr),
                           stir=None if s.stir is None else dict(s.stir)))
    return Recipe(f'{recipe.name}_x{current_factor:g}i_x{time_factor:g}t', out)


def by_name(name, T=40.0, **kw):
    """CLI helper: 'dc', 'pulse_reverse', 'team_staged', 'plan_ramp', 'plan_ramp_pr', 'potentiostatic'."""
    if name == 'plan_ramp':
        return plan_ramp(T)
    if name == 'plan_ramp_pr':
        return plan_ramp_pr(T)
    if name == 'team_staged':
        return team_staged()
    if name == 'dc':
        return dc(kw.get('j', 3.0), kw.get('t_max_h', 6.0))
    if name == 'pulse_reverse':
        return pulse_reverse(kw.get('j', 3.0), kw.get('t_max_h', 6.0))
    raise ValueError(f'unknown recipe {name}')
