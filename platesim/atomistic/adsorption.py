"""Adsorption energies of plating-bath species on Cu surfaces with the MACE-MP-0 universal potential.

RUNS ONLY IN THE SEPARATE VENV (platesim/.venv-atomistic: ase 3.29, mace-torch 0.3.17, torch 2.14 cpu); the main
server calls it through a subprocess (see __init__.py) so the global environment never imports torch/mace.

  .venv-atomistic/Scripts/python.exe -m platesim.atomistic.adsorption --suite        -> results.json
  .venv-atomistic/Scripts/python.exe -m platesim.atomistic.adsorption --single '{"surface":"100","adsorbate":"Cl","site":"hollow"}'

E_ads = E(slab + ads) - E(slab) - E(ref), references: Cl -> 1/2 Cl2, SCH3 (thiolate, MPS/SPS anchor proxy) ->
CH3SH - 1/2 H2, H2O -> H2O, CH3OCH3 (ether-oxygen proxy for a PEG segment) -> CH3OCH3. Gas-phase, no solvent, no
field, no charge: MACE-MP-0 is trained on PBE bulk/surface data (MPtrj) [FT Batatia et al. 2023, arXiv:2401.00096],
so values are trends, not interfacial free energies (see NOTES.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc100, fcc111, molecule
from ase.constraints import FixAtoms
from ase.filters import ExpCellFilter
from ase.optimize import BFGS

SITES = {"111": ["ontop", "bridge", "fcc", "hcp"], "100": ["ontop", "bridge", "hollow"]}
HEIGHT = {"Cl": 1.9, "SCH3": 1.9, "H2O": 2.3, "CH3OCH3": 2.4}   # initial anchor-atom height, A [J]
_CALC = {}


def calculator(model: str = "medium", dtype: str = "float64"):
    if (model, dtype) not in _CALC:
        from mace.calculators import mace_mp
        _CALC[(model, dtype)] = mace_mp(model=model, default_dtype=dtype, device="cpu")
    return _CALC[(model, dtype)]


def lattice_constant(calc) -> float:
    a = bulk("Cu", "fcc", a=3.62)
    a.calc = calc
    BFGS(ExpCellFilter(a), logfile=None).run(fmax=1e-3)
    return float(a.cell.cellpar()[0] * 2 ** 0.5)


def build_slab(surface: str, a0: float, size=(4, 4, 4), vacuum: float = 7.5) -> Atoms:
    """4 layers, bottom 2 fixed, 2 x vacuum = 15 A between images."""
    slab = (fcc111 if surface == "111" else fcc100)("Cu", size=size, a=a0, vacuum=vacuum)
    slab.set_constraint(FixAtoms(mask=slab.get_tags() >= 3))
    return slab


def adsorbate(name: str) -> tuple[Atoms, int]:
    """Adsorbate with the anchor atom at the origin pointing down (-z); returns (atoms, anchor index)."""
    if name == "Cl":
        return Atoms("Cl", positions=[(0, 0, 0)]), 0
    if name == "SCH3":   # S-CH3 thiolate, C straight above S
        h = [(1.03 * np.cos(t), 1.03 * np.sin(t), 1.82 + 0.36) for t in (0, 2.094, 4.189)]
        return Atoms("SCH3", positions=[(0, 0, 0), (0, 0, 1.82)] + h), 0
    if name in ("H2O", "CH3OCH3"):
        m = molecule(name)
        o = [i for i, s in enumerate(m.get_chemical_symbols()) if s == "O"][0]
        m.translate(-m.positions[o])
        if name == "H2O":   # flat-ish: rotate so H atoms point sideways and slightly up
            m.rotate(90, "x")
        else:               # C-O-C plane vertical, methyls up
            com = m.get_center_of_mass()
            m.rotate(com, (0, 0, 1), center=(0, 0, 0))
        return m, o
    raise ValueError(f"unknown adsorbate {name}")


def reference_energy(name: str, calc) -> float:
    def relaxed(m):
        m = m.copy()
        m.center(vacuum=6.0)
        m.calc = calc
        BFGS(m, logfile=None).run(fmax=0.02, steps=300)
        return m.get_potential_energy()
    if name == "Cl":
        return 0.5 * relaxed(Atoms("Cl2", positions=[(0, 0, 0), (0, 0, 1.99)]))
    if name == "SCH3":
        return relaxed(molecule("CH3SH")) - 0.5 * relaxed(molecule("H2"))
    return relaxed(molecule(name))


def relax(atoms: Atoms, calc, fmax: float, steps: int = 400) -> tuple[float, int]:
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return float(atoms.get_potential_energy()), int(opt.nsteps)


def adsorption_energy(surface: str, name: str, site: str, n_ads: int = 1, fmax: float = 0.05,
                      model: str = "medium", a0: float | None = None, cache: dict | None = None) -> dict:
    """n_ads adsorbates on a 4x4 slab (coverage n_ads/16 ML). n_ads > 1 places them on a regular sub-lattice."""
    calc = calculator(model)
    cache = cache if cache is not None else {}
    a0 = a0 or cache.setdefault("a0", lattice_constant(calc))
    key = f"slab{surface}"
    if key not in cache:
        s = build_slab(surface, a0)
        cache[key] = relax(s, calc, fmax)[0]
    rkey = f"ref{name}"
    if rkey not in cache:
        cache[rkey] = reference_energy(name, calc)
    slab = build_slab(surface, a0)
    mol, anchor = adsorbate(name)
    step = {1: [(0, 0)], 2: [(0, 0), (2, 2)], 4: [(0, 0), (2, 0), (0, 2), (2, 2)],
            8: [(i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0]}[n_ads]
    cell = slab.cell
    for (i, j) in step:
        off = (i * cell[0, :2] + j * cell[1, :2]) / 4
        add_adsorbate(slab, mol, HEIGHT[name], position=site, offset=None, mol_index=anchor)
        slab.positions[-len(mol):, :2] += off
    t0 = time.time()
    e, nsteps = relax(slab, calc, fmax)
    e_ads = (e - cache[key] - n_ads * cache[rkey]) / n_ads
    top = slab.positions[slab.get_tags() == 1]
    ads = slab.positions[-len(mol) * n_ads:]
    anchor_z = float(ads[anchor, 2] - top[:, 2].max())
    return {"surface": f"Cu({surface})", "adsorbate": name, "site": site, "coverage_ML": n_ads / 16,
            "E_ads_eV": round(float(e_ads), 3), "anchor_height_A": round(anchor_z, 2), "relax_steps": nsteps,
            "fmax": fmax, "model": f"mace-mp-0-{model}", "a0_A": round(a0, 4), "seconds": round(time.time() - t0, 1),
            "positions": slab.positions[:, :2].round(3).tolist(), "symbols": slab.get_chemical_symbols(),
            "z": slab.positions[:, 2].round(3).tolist()}


def suite(fmax: float = 0.05, model: str = "medium") -> dict:
    cache: dict = {}
    rows = []
    jobs = [(s, a, site, 1) for s in ("111", "100") for a in ("Cl", "SCH3") for site in SITES[s]]
    jobs += [(s, a, "ontop", 1) for s in ("111", "100") for a in ("H2O", "CH3OCH3")]
    jobs += [("100", "Cl", "hollow", n) for n in (2, 4, 8)]          # 1/8, 1/4, 1/2 ML (c(2x2) = 1/2 ML)
    for k, (s, a, site, n) in enumerate(jobs):
        r = adsorption_energy(s, a, site, n, fmax, model, cache=cache)
        r.pop("positions"); r.pop("symbols"); r.pop("z")
        rows.append(r)
        print(f"[{k + 1}/{len(jobs)}] {r['surface']} {a} {site} {r['coverage_ML']} ML: {r['E_ads_eV']} eV "
              f"({r['seconds']} s)", file=sys.stderr, flush=True)
    return {"model": f"mace-mp-0-{model}", "a0_A": cache["a0"], "a0_exp_A": 3.615, "fmax": fmax, "results": rows,
            "references": {k: v for k, v in cache.items() if k.startswith("ref")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--single", type=str, help="JSON: surface, adsorbate, site, n_ads, fmax, model")
    ap.add_argument("--out", type=str, default=str(Path(__file__).with_name("results.json")))
    args = ap.parse_args()
    if args.suite:
        res = suite()
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(json.dumps({"written": args.out, "n": len(res["results"])}))
    else:
        p = json.loads(args.single)
        r = adsorption_energy(p.get("surface", "100"), p.get("adsorbate", "Cl"), p.get("site", "hollow"),
                              int(p.get("n_ads", 1)), float(p.get("fmax", 0.05)), p.get("model", "medium"))
        print("PLATESIM_JSON " + json.dumps(r))
