"""platesim.feature: feature-scale Cu fill model for vias AND trenches (level-set engine).

Model id: "feature_fill".  The engine (platesim.feature.engine/) is a verbatim port of the research
reference research/via_fill_sim (read-only) with one addition: Params.geometry
  'via'     - 2-D axisymmetric (r, z), numerics identical to the research engine;
  'trench'  - planar 2-D (x, z) per unit length: long trench of width w = d, depth H, half-domain
              x in [0, R] (symmetry at x = 0 and x = R), R = a / f_open (open fraction of the trench
              array = a / R); every axisymmetric r-weighting (face areas ~ r, cell volumes ~ r dr dz,
              galvanostat weights, set current pi R^2) replaced by weight 1 / R per unit length.
  'trench_full' - planar 2-D (x, z) per unit length over the FULL pitch x in [0, pitch], pitch = d /
              f_open: two walls (no mirror symmetry), periodic in x, resolved LBM shear convection.
Validation: platesim/feature/validate_trench.py, tests/test_feature.py.
"""
from __future__ import annotations

import math

from .engine import Params, dc, plan_ramp, plan_ramp_pr, pulse_reverse, run_case, scaled, team_staged
from .engine.levelset import GEOMETRIES
from .engine.run_case import decode_cfield

MODEL_SPEC = {
    "id": "feature_fill",
    "title": "Feature fill: via / trench (level set)",
    "description": (
        "Galvanostatic (or pulse-reverse) Cu fill of one blind feature + its share of surrounding "
        "field: axisymmetric via or planar 2-D trench (per unit length). Level-set interface, "
        "smoothed-boundary finite-volume Cu2+/suppressor transport, ohmic potential, S-NDR suppressor "
        "kinetics (Wheeler/Moffat/Josell/Braun type), void detection and seam/keyhole classification."
    ),
    "params": [
        {"name": "geometry", "type": "enum", "choices": ["via", "trench", "trench_full"], "default": "via",
         "help": "via = axisymmetric (r, z) unit cell; trench = planar 2-D (x, z) half-domain of a "
                 "long trench, quantities per unit length, R = a/f_open; trench_full = planar 2-D full "
                 "pitch x in [0, d/f_open], two walls, periodic in x, resolved cavity shear flow."},
        {"name": "d_um", "type": "float", "default": 30.0, "min": 5.0, "max": 1000.0, "unit": "um",
         "help": "via diameter / trench full width w (= 2a). Common choices: 30, 50, 75, 100, 150, "
                 "250, 500, 1000 um."},
        {"name": "H_um", "type": "float", "default": 300.0, "min": 10.0, "max": 1000.0, "unit": "um",
         "help": "feature depth (via depth / trench depth)."},
        {"name": "T_C", "type": "float", "default": 40.0, "min": 15.0, "max": 60.0, "unit": "degC",
         "help": "bath temperature (D, kappa, i0 follow Arrhenius / Stokes-Einstein)."},
        {"name": "recipe", "type": "enum",
         "choices": ["dc", "pulse_reverse", "plan_ramp", "plan_ramp_pr", "team_staged", "fast_x1.4"],
         "default": "dc",
         "help": "dc: constant j for tmax_h. pulse_reverse: duty-averaged PR (9/0.5/0.5 ms, rr 2) at "
                 "j_mA_cm2 NET for tmax_h. plan_ramp / plan_ramp_pr: team fill-plan ramp (40/25 C "
                 "tables, ~4 h). team_staged: 1/5/15 mA steps (6 h). fast_x1.4: plan_ramp with "
                 "currents x1.4. dc duration = tmax_h; the other recipes run their full plan."},
        {"name": "j_mA_cm2", "type": "float", "default": 2.5, "min": 0.05, "max": 50.0,
         "unit": "mA/cm2", "help": "galvanostatic current density (dc recipe only; projected unit-cell "
                                   "area - pi R^2 for the via, R per unit length for the trench)."},
        {"name": "tmax_h", "type": "float", "default": 4.0, "min": 0.05, "max": 48.0, "unit": "h",
         "help": "dc recipe: max duration (run stops earlier when the feature is filled)."},
        {"name": "t_f_ms", "type": "float", "default": 9.0, "min": 0.1, "max": 1000.0, "unit": "ms",
         "help": "pulse_reverse recipe only: forward (cathodic) pulse length."},
        {"name": "t_r_ms", "type": "float", "default": 0.5, "min": 0.0, "max": 100.0, "unit": "ms",
         "help": "pulse_reverse recipe only: reverse (anodic) pulse length."},
        {"name": "t_off_ms", "type": "float", "default": 0.5, "min": 0.0, "max": 1000.0, "unit": "ms",
         "help": "pulse_reverse recipe only: off (zero current) time in the cycle."},
        {"name": "rr", "type": "float", "default": 2.0, "min": 0.0, "max": 20.0,
         "help": "pulse_reverse recipe only: reverse current as a multiple of the forward current."},
        {"name": "f_open", "type": "float", "default": 0.05, "min": 0.01, "max": 1.0,
         "help": "open area fraction of the unit cell: R = a/sqrt(f_open) (via), R = a/f_open (trench)."},
        {"name": "k_minus", "type": "float", "default": 1.0e6, "min": 1.0e4, "max": 1.0e6,
         "unit": "1/cm", "help": "suppressor incorporation (deactivation) rate per growth velocity; "
                                 "TUNED knob (literature 1e5 does not break down; README V5/6)."},
        {"name": "delta_um", "type": "float", "default": 20.0, "min": 2.0, "max": 200.0, "unit": "um",
         "help": "diffusion boundary layer above the field (Dirichlet c = c_bulk plane)."},
        {"name": "k_mix", "type": "float", "default": 1.0, "min": 0.0, "max": 1.6,
         "help": "well-mixed eddy depth inside the feature, in feature diameters (0 = no mixing)."},
        {"name": "efficiency", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0,
         "help": "current efficiency for the growth velocity and for Cu2+ consumption (Cu2+ sink and "
                 "reverse source scaled by it; 1.0 = every electron deposits Cu)."},
        {"name": "D_cu25", "type": "float", "default": 2.65e-6, "min": 1.0e-6, "max": 1.0e-5,
         "unit": "cm2/s", "help": "Cu2+ diffusivity at 25 C (Stokes-Einstein scaled with T; NIST/Braun)."},
        {"name": "suppressor", "type": "bool", "default": True,
         "help": "False = no suppressor (conformal growth; used by the Faraday and seam validations)."},
    ],
}


def _build_recipe(name: str, T_C: float, j_mA_cm2: float, tmax_h: float,
                  t_f_ms: float = 9.0, t_r_ms: float = 0.5, t_off_ms: float = 0.5, rr: float = 2.0):
    if name == "dc":
        return dc(j_mA_cm2, tmax_h)
    if name == "pulse_reverse":
        return pulse_reverse(j_mA_cm2, tmax_h, t_f_ms=t_f_ms, t_r_ms=t_r_ms, t_off_ms=t_off_ms, rr=rr)
    if name == "plan_ramp":
        return plan_ramp(T_C)
    if name == "plan_ramp_pr":
        return plan_ramp_pr(T_C)
    if name == "team_staged":
        return team_staged()
    if name == "fast_x1.4":
        return scaled(plan_ramp(T_C), current_factor=1.4, time_factor=1.0)
    raise ValueError(f"unknown recipe {name!r}; one of "
                     "['dc', 'pulse_reverse', 'plan_ramp', 'plan_ramp_pr', 'team_staged', 'fast_x1.4']")


def _line(title: str, x: list, series: list, xlabel: str, ylabel: str) -> dict:
    return {"kind": "line", "title": title, "x": x, "series": series,
            "xlabel": xlabel, "ylabel": ylabel}


def _to_contract(res: dict, geometry: str) -> dict:
    """Convert the run_case JSON (meta/series/frames/result) into the MODEL_SPEC output contract."""
    series, frames, result = res["series"], res["frames"], res["result"]
    via = geometry == "via"
    xlab = "r (um)" if via else "x (um)"
    t_h = [round(t / 3600.0, 4) for t in series["t_s"] if t is not None]
    n = len(t_h)
    V_mV = [None if v is None else round(v * 1e3, 2) for v in series["V"][:n]]

    plots = [
        _line("Fill fraction", t_h,
              [{"name": "fill_fraction", "y": series["fill_fraction"][:n]}],
              "t (h)", "fill fraction"),
        _line("Metal potential", t_h, [{"name": "V", "y": V_mV}],
              "t (h)", "V (mV, Tafel-intercept reference)"),
        _line("Suppressor coverage", t_h,
              [{"name": "theta_floor", "y": series["theta_floor"][:n]},
               {"name": "theta_face", "y": series["theta_face"][:n]}],
              "t (h)", "theta (-)"),
        _line("Cu2+ at the floor", t_h, [{"name": "c_floor", "y": series["c_floor"][:n]}],
              "t (h)", "c/c_bulk at floor"),
        {"kind": "contours",
         "title": f"Interface front and trapped voids ({geometry}, um)",
         "frames": [{"t": f["t_s"], "lines": list(f["contour"]) + list(f["voids"])} for f in frames],
         "xlabel": xlab, "ylabel": "z (um)"},
    ]

    # final c/c_bulk field heatmap (decode_cfield: 64 z-rows x 48 cols, NaN in metal)
    fr = frames[-1]
    cfield = decode_cfield(fr)
    x0, x1, z0, z1 = fr["extent"]
    nz_c, nx_c = cfield.shape
    xs = [round(x0 + (i + 0.5) * (x1 - x0) / nx_c, 3) for i in range(nx_c)]
    ys = [round(z0 + (j + 0.5) * (z1 - z0) / nz_c, 3) for j in range(nz_c)]
    z_rows = [[None if not math.isfinite(v) else round(float(v), 4) for v in row] for row in cfield]
    plots.append({"kind": "heatmap", "title": "Final Cu2+ field c/c_bulk",
                  "x": xs, "y": ys, "z": z_rows, "zlabel": "c/c_bulk",
                  "xlabel": xlab, "ylabel": "z (um)"})

    summary = {
        "outcome": result["outcome"],                 # filled_void_free | void | seam | keyhole | incomplete
        "fill_time": result["fill_time_s"],           # s, None unless filled to 99.5 %
        "fill_fraction_final": result["fill_fraction_final"],
        "overburden": result["overburden_um"],        # um of overburden on the field
        "void_volume_frac": result["void_volume_frac"],
    }
    warnings = [
        "Uncalibrated parameters: suppressor identity/concentration, Ea and kappa are literature "
        "placeholders, not measurements for this bath (research README sec. 4).",
    ]
    if result.get("stop_reason") == "overburden_exceeded":
        warnings.append("run stopped at the domain top: overburden limit reached "
                        "(reduce tmax_h / j or accept the truncated ramp).")
    assumptions = [
        "quasi-steady Cu2+ and electrolyte potential each step; only the suppressor equation carries "
        "a storage term; migration as constant enhancement E (dilute theory).",
        "level set: first-order Godunov upwind + FMM reinitialisation -> Faraday (deposit vs charge) "
        "error typically 0.2-0.8 %.",
        "metal is equipotential (no seed resistance); efficiency scales the growth velocity and the "
        "Cu2+ consumption/source; no H2/O2 side reactions.",
        "summary units: fill_time [s], overburden [um]; tmax_h/j_mA_cm2 apply to the dc recipe only "
        "(other recipes run their full plan); quantities are per unit length for the trench.",
    ]
    return {"summary": summary, "plots": plots, "warnings": warnings, "assumptions": assumptions}


def run(params: dict, progress=lambda frac, msg="": None) -> dict:
    """Run one feature-fill simulation and return the JSON-serialisable MODEL_SPEC output.

    progress(frac, msg) is called with frac = t / recipe duration (0..1; for dc that is t / tmax_h)."""
    geometry = str(params.get("geometry", "via"))
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {list(GEOMETRIES)}, got {geometry!r}")
    d_um = float(params.get("d_um", 30.0))
    T_C = float(params.get("T_C", 40.0))
    recipe_name = str(params.get("recipe", "dc"))
    j_mA_cm2 = float(params.get("j_mA_cm2", 2.5))
    tmax_h = float(params.get("tmax_h", 4.0))
    f_open = float(params.get("f_open", 0.05))
    suppressor = bool(params.get("suppressor", True))
    t_f_ms = float(params.get("t_f_ms", 9.0))
    t_r_ms = float(params.get("t_r_ms", 0.5))
    t_off_ms = float(params.get("t_off_ms", 0.5))
    rr = float(params.get("rr", 2.0))

    eng = Params(
        geometry=geometry,
        H_um=float(params.get("H_um", 300.0)),
        k_minus=float(params.get("k_minus", 1.0e6)),
        delta_um=float(params.get("delta_um", 20.0)),
        k_mix=float(params.get("k_mix", 1.0)),
        efficiency=float(params.get("efficiency", 1.0)),
        D_cu25=float(params.get("D_cu25", 2.65e-6)),
        suppressor=suppressor,
    )
    rec = _build_recipe(recipe_name, T_C, j_mA_cm2, tmax_h,
                        t_f_ms=t_f_ms, t_r_ms=t_r_ms, t_off_ms=t_off_ms, rr=rr)
    label = f"{geometry}_{d_um:g}um_T{T_C:g}_{rec.name}"

    res = run_case(d_um, T_C, rec, eng, f_open=f_open, label=label, out=None,
                   progress=lambda frac, msg="": progress(min(max(float(frac), 0.0), 1.0), msg))

    out = _to_contract(res, geometry)
    if suppressor:
        out["warnings"].append(
            f"k_minus = {eng.k_minus:g} 1/cm is a TUNED calibration knob at the top of its allowed "
            "window (literature 1e5 never breaks down; research README sec. 5/6).")
    return out
