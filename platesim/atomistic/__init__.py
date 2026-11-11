"""platesim.atomistic: MACE-MP-0 adsorption energies of bath species on Cu (exploratory, trends only).

The heavy code (adsorption.py) runs in platesim/.venv-atomistic via a subprocess, so this module only needs the
standard library. Cached suite results: results.json (python -m platesim.atomistic.adsorption --suite).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENV_PY = ROOT / ".venv-atomistic" / "Scripts" / "python.exe"
RESULTS = HERE / "results.json"

MODEL_SPEC = {
    "id": "atomistic_adsorption",
    "title": "Additive adsorption on Cu (MACE-MP-0)",
    "description": "Relaxes Cl, an S-CH3 thiolate (accelerator anchor proxy), H2O or dimethyl ether (PEG ether-O proxy) on "
                   "a 4x4x4 Cu(111)/(100) slab with the MACE-MP-0 universal potential and reports the adsorption energy. "
                   "Gas phase, no solvent or electrode potential: relative trends only.",
    "params": [
        {"name": "surface", "type": "enum", "choices": ["100", "111"], "default": "100", "help": "Cu facet"},
        {"name": "adsorbate", "type": "enum", "choices": ["Cl", "SCH3", "H2O", "CH3OCH3"], "default": "Cl",
         "help": "Cl vs 1/2 Cl2; SCH3 vs CH3SH - 1/2 H2; H2O; CH3OCH3 (PEG ether proxy)"},
        {"name": "site", "type": "enum", "choices": ["ontop", "bridge", "hollow", "fcc", "hcp"], "default": "hollow",
         "help": "hollow only on (100); fcc/hcp only on (111)"},
        {"name": "n_ads", "type": "enum", "choices": ["1", "2", "4", "8"], "default": "1",
         "help": "adsorbates per 4x4 cell (coverage n/16 ML; 8 on (100) = c(2x2))"},
        {"name": "fmax", "type": "float", "default": 0.05, "min": 0.01, "max": 0.2, "unit": "eV/A",
         "help": "force convergence"},
        {"name": "model", "type": "enum", "choices": ["medium", "small"], "default": "medium", "help": "MACE-MP-0 size"},
        {"name": "use_cache", "type": "bool", "default": True,
         "help": "show the cached suite (results.json) instead of computing when this case is in it"},
    ],
}

ASSUMPTIONS = [
    "MACE-MP-0 (Batatia et al. 2023, arXiv:2401.00096) trained on PBE MPtrj bulk/surface data; not on solvated or charged interfaces",
    "gas-phase references, no solvent, no electrode potential, no Cu(I) complexes; adsorption energies are relative trends",
    "4x4 slab, 4 layers, bottom 2 fixed, 15 A vacuum, a0 from MACE (3.626 A vs 3.615 exp)",
    "SCH3 stands in for the SPS/MPS thiolate anchor; CH3OCH3 for one PEG ether oxygen",
]


def _cached(p: dict) -> dict | None:
    if not RESULTS.exists():
        return None
    data = json.loads(RESULTS.read_text())
    for r in data["results"]:
        if (r["surface"] == f"Cu({p['surface']})" and r["adsorbate"] == p["adsorbate"] and r["site"] == p["site"]
                and abs(r["coverage_ML"] - int(p["n_ads"]) / 16) < 1e-9):
            return r
    return None


def _suite_plots() -> list[dict]:
    if not RESULTS.exists():
        return []
    rows = json.loads(RESULTS.read_text())["results"]
    low = [r for r in rows if r["coverage_ML"] == 0.0625]
    cats = [f"{r['surface']} {r['adsorbate']} {r['site']}" for r in low]
    cov = [r for r in rows if r["adsorbate"] == "Cl" and r["surface"] == "Cu(100)" and r["site"] == "hollow"]
    cov.sort(key=lambda r: r["coverage_ML"])
    return [
        {"kind": "bar", "title": "Adsorption energy, 1/16 ML (MACE-MP-0)", "categories": cats,
         "series": [{"name": "E_ads (eV)", "y": [r["E_ads_eV"] for r in low]}], "ylabel": "E_ads (eV)"},
        {"kind": "line", "title": "Cl on Cu(100) hollow vs coverage", "x": [r["coverage_ML"] for r in cov],
         "series": [{"name": "E_ads per Cl (eV)", "y": [r["E_ads_eV"] for r in cov], "mode": "lines+markers"}],
         "xlabel": "coverage (ML)", "ylabel": "E_ads per Cl (eV)"},
    ]


def run(params: dict, progress=lambda frac, msg="": None) -> dict:
    p = {s["name"]: params.get(s["name"], s["default"]) for s in MODEL_SPEC["params"]}
    warnings: list[str] = []
    valid = {"100": ["ontop", "bridge", "hollow"], "111": ["ontop", "bridge", "fcc", "hcp"]}
    if p["site"] not in valid[p["surface"]]:
        raise ValueError(f"site {p['site']} not available on Cu({p['surface']}); choose {valid[p['surface']]}")
    r = _cached(p) if p["use_cache"] else None
    plot_atoms = []
    if r is None:
        if not VENV_PY.exists():
            raise RuntimeError(f"atomistic venv missing: {VENV_PY} (see platesim/atomistic/NOTES.md)")
        progress(0.1, "relaxing with MACE-MP-0 (venv subprocess)")
        arg = json.dumps({k: p[k] for k in ("surface", "adsorbate", "site", "n_ads", "fmax", "model")})
        out = subprocess.run([str(VENV_PY), "-m", "platesim.atomistic.adsorption", "--single", arg], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=1800)
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("PLATESIM_JSON ")]
        if not line:
            raise RuntimeError("adsorption subprocess failed:\n" + out.stderr[-2000:])
        r = json.loads(line[-1][len("PLATESIM_JSON "):])
        xy, z, sym = r.pop("positions"), r.pop("z"), r.pop("symbols")
        zmax = max(zz for zz, s in zip(z, sym) if s == "Cu")
        for el in sorted(set(sym)):
            pts = [(x, y) for (x, y), zz, s in zip(xy, z, sym) if s == el and (s != "Cu" or zz > zmax - 0.5)]
            plot_atoms.append({"name": el if el != "Cu" else "Cu (top layer)", "x": [q[0] for q in pts],
                               "y": [q[1] for q in pts], "mode": "markers"})
    else:
        warnings.append("shown from the cached suite (results.json); set use_cache=false to recompute")
    progress(0.95, "done")
    plots = _suite_plots()
    if plot_atoms:
        plots.insert(0, {"kind": "line", "title": "Relaxed structure, top view (A)", "x": plot_atoms[0]["x"],
                         "series": [{"name": a["name"], "x": a["x"], "y": a["y"], "mode": "markers"} for a in plot_atoms],
                         "xlabel": "x (A)", "ylabel": "y (A)"})
    return {"summary": r, "plots": plots, "warnings": warnings, "assumptions": ASSUMPTIONS}
