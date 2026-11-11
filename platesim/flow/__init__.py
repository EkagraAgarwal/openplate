"""platesim.flow: mass-transfer (Nernst-layer) model for the stirred beaker cell.

Model id "flow_mass_transfer". model="correlation" (default) uses the
rpm -> delta correlation in correlation.py; model="lbm" calls
platesim.flow.lbm.run_lbm when that module is importable (written separately).
"""
from __future__ import annotations

from ..core.bath import Bath
from ..core.cell import CellGeometry
from .correlation import (
    MassTransferParams, TOUABI_RPM, TOUABI_KM, delta_mean_um, delta_map,
    natural_convection,
)

MODEL_SPEC = {
    "id": "flow_mass_transfer",
    "title": "Stirred-beaker mass transfer (Nernst layer)",
    "description": (
        "Nernst-layer thickness on the coupon vs stirrer speed: forced convection "
        "delta0*(rpm/rpm_ref)^(-n) [FT Avdeef 2022] harmonically combined with the "
        "unstirred natural-convection floor [SH Fenech 1959], optional Lévéque "
        "x^(1/3) profile along the flow direction, baffles thin the forced part. "
        "Returns delta map on the coupon, k_m, and Cu/O2 limiting current densities."
    ),
    "params": [
        {"name": "model", "type": "enum", "choices": ["correlation", "lbm"],
         "default": "correlation",
         "help": "correlation: analytic Nernst-layer model. lbm: lattice-Boltzmann "
                 "flow solver (platesim.flow.lbm, if available)."},
        {"name": "rpm", "type": "float", "default": 300.0, "min": 0.0, "max": 1500.0,
         "unit": "rpm", "help": "stirrer speed (0 = unstirred, natural convection only)."},
        {"name": "T_C", "type": "float", "default": 22.0, "min": 10.0, "max": 45.0,
         "unit": "degC", "help": "bath temperature (scales D_cu in k_m / j_lim)."},
        {"name": "baffles", "type": "bool", "default": True,
         "help": "baffles present: multiply the forced-convection part by 0.8 [J]."},
        {"name": "delta0_um", "type": "float", "default": 40.0, "min": 5.0, "max": 200.0,
         "unit": "um", "help": "forced Nernst layer at 300 rpm [FT Avdeef 2022: 33-47 um]."},
        {"name": "n_exp", "type": "float", "default": 0.5, "min": 0.1, "max": 1.0,
         "unit": "-", "help": "rpm exponent [FT Avdeef 2022: 0.505]."},
        {"name": "delta_nat_um", "type": "float", "default": 80.0, "min": 10.0, "max": 300.0,
         "unit": "um", "help": "unstirred natural-convection floor [SH Fenech 1959: 70-100 um]."},
        {"name": "stir_bar_len_cm", "type": "float", "default": 3.8, "min": 1.0, "max": 6.0,
         "unit": "cm", "help": "stir bar length (lbm model only) [J]."},
        {"name": "beaker_id_cm", "type": "float", "default": 9.0, "min": 3.0, "max": 20.0,
         "unit": "cm", "help": "beaker inner diameter (lbm model only) [J]."},
        {"name": "resolution", "type": "enum", "choices": ["coarse", "medium"],
         "default": "coarse", "help": "lbm grid resolution (ignored by correlation)."},
    ],
}

_ASSUMPTIONS = [
    "forced delta = delta0*(rpm/300)^(-n), delta0 = 40 um, n = 0.5 [FT Avdeef 2022 "
    "ADMET&DMPK 10:249 doi 10.5599/admet.1568: 33-47 um at 300 rpm, exponent 0.505]",
    "natural-convection floor 80 um [SH Fenech 1959 via KB; 70-100 um unstirred]",
    "1/delta = 1/forced + 1/delta_nat harmonic combine [J]; baffles x0.8 on forced part only [J]",
    "Lévéque delta(x) ~ (x - x0 + 0.05*w)^(1/3) along +x, area-mean normalised [J]",
    "k_m = D_cu_eff / delta_mean (dilute theory); Touabi 2015 points are a different "
    "cell/bath, plotting reference only [FT Int. J. Electrochem. Sci. 10:7227 (2015)]",
]


def run(params: dict, progress=lambda frac, msg="": None) -> dict:
    p = params or {}
    model = str(p.get("model", "correlation"))
    rpm = float(p.get("rpm", 300.0))
    T_C = float(p.get("T_C", 22.0))
    baffles = bool(p.get("baffles", True))
    resolution = str(p.get("resolution", "coarse"))

    geom = CellGeometry(
        stir_bar_len=float(p.get("stir_bar_len_cm", 3.8)),
        beaker_id=float(p.get("beaker_id_cm", 9.0)),
    )
    bath = Bath().at(T_C)
    mtp = MassTransferParams(
        delta_nat_um=float(p.get("delta_nat_um", 80.0)),
        delta0_um=float(p.get("delta0_um", 40.0)),
        n=float(p.get("n_exp", 0.5)),
        use_natural=bool(p.get("use_natural", False)),
    )
    warnings: list[str] = []

    if model == "lbm":
        progress(0.05, "lbm flow solve")
        try:
            # lazy: platesim.flow.lbm is written by another worker and may be absent
            import importlib
            lbm = importlib.import_module("platesim.flow.lbm")
        except ImportError:
            warnings.append("model='lbm' requested but platesim.flow.lbm is not "
                            "importable yet; falling back to the correlation model")
            model = "correlation"
        else:
            res = lbm.run_lbm(geom, bath, rpm, resolution)
            # merge lbm outputs but keep the contract keys present
            out_summary = dict(res.get("summary", {}))
            out_summary.setdefault("delta_mean_um", out_summary.get("delta_mean_um"))
            return {
                "summary": out_summary,
                "plots": list(res.get("plots", [])),
                "warnings": list(res.get("warnings", [])),
                "assumptions": list(res.get("assumptions", [])) or _ASSUMPTIONS,
            }

    # ---- correlation model
    progress(0.2, "delta map on coupon")
    m = delta_map(geom, bath, rpm, mtp, baffles=baffles)

    progress(0.5, "rpm curves")
    rpm_grid = [i * 50.0 for i in range(31)]              # 0..1500 rpm
    mean_curve = [delta_mean_um(r, mtp, baffles=baffles, bath=bath,
                                height_cm=geom.coupon_h) for r in rpm_grid]
    km_curve = [bath.D_cu_eff / (d * 1e-4) for d in mean_curve]

    x_c = m["x"]                                          # across width (cm, centred 0)
    delta_x = m["delta_um"][len(m["y"]) // 2]             # centre-height row

    plots = [
        {"kind": "heatmap", "title": f"Nernst layer on coupon (rpm = {rpm:g})",
         "x": [round(v, 4) for v in x_c], "y": [round(v, 4) for v in m["y"]],
         "z": [[round(v, 3) for v in row] for row in m["delta_um"]],
         "zlabel": "delta (um)"},
        {"kind": "line", "title": "delta vs x at coupon centre height",
         "x": [round(v, 4) for v in x_c],
         "series": [{"name": "delta(x)", "y": [round(v, 3) for v in delta_x]}],
         "xlabel": "x (cm, flow +x from leading edge)", "ylabel": "delta (um)"},
        {"kind": "line", "title": "mean Nernst layer vs stirrer speed",
         "x": rpm_grid,
         "series": [{"name": "delta_mean (um)", "y": [round(v, 3) for v in mean_curve]}],
         "xlabel": "rpm", "ylabel": "delta_mean (um)"},
        {"kind": "line", "title": "mass-transfer coefficient vs rpm "
                                  "(Touabi points: dilute 0.05 M, different cell)",
         "x": rpm_grid,
         "series": [
             {"name": "k_m model (cm/s)", "y": [f"{v:.3e}" for v in km_curve]},
             {"name": "Touabi 2015 (cm/s)", "x": TOUABI_RPM,
              "y": [f"{v:.3e}" for v in TOUABI_KM], "mode": "markers"},
         ],
         "xlabel": "rpm", "ylabel": "k_m (cm/s)"},
    ]
    warnings.append("Touabi 2015 k_m points are for dilute 0.05 M Cu in a different "
                    "cell geometry - qualitative reference only, not a fit target")

    mean_um = m["mean_um"]
    delta_cm = mean_um * 1e-4
    # Natural convection on the vertical coupon at its limiting current: plain
    # bath.D_cu is used (migration enhancement E_mig is NOT included here, so
    # these numbers are a pure diffusive/convective natural-convection estimate).
    nat = natural_convection(geom.coupon_h, bath.c_cu * 1e3, bath.D_cu)
    summary = {
        "delta_mean_um": round(mean_um, 3),
        "k_m_cm_s": float(f"{m['k_m_cm_s']:.4g}"),
        "j_lim_cu_mA_cm2": round(bath.j_lim_cu(delta_cm) * 1e3, 4),
        "j_lim_o2_mA_cm2": round(bath.j_lim_o2(delta_cm) * 1e3, 6),
        "model": "correlation",
        "rpm": rpm,
        "natural_convection": {
            "Gr": float(f"{nat['Gr']:.4g}"),
            "Sc": float(f"{nat['Sc']:.4g}"),
            "Ra": float(f"{nat['Ra']:.4g}"),
            "regime": nat["regime"],
            "delta_mean_um": round(nat["delta_mean_um"], 3),
            "i_lim_mA_cm2": round(nat["i_lim_mA_cm2"], 4),
            "k_mean_cm_s": float(f"{nat['k_mean_cm_s']:.4g}"),
            "height_cm": nat["height_cm"],
            "beta_L_mol": nat["beta_L_mol"],
            "nu_cm2_s": nat["nu_cm2_s"],
            "note": "vertical-plate laminar natural convection, Sh=0.67 Ra^0.25 "
                    "[not verified]; beta is [J] literature order, not measured; "
                    "plain D_cu (no migration factor)",
        },
    }
    progress(1.0, "done")
    return {"summary": summary, "plots": plots, "warnings": warnings,
            "assumptions": _ASSUMPTIONS}


__all__ = ["MODEL_SPEC", "run", "MassTransferParams", "delta_mean_um", "delta_map",
           "TOUABI_RPM", "TOUABI_KM"]
