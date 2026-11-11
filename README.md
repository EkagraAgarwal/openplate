# OpenPlate

**An open-source, scriptable simulator for electrodeposition (electroplating), inspired by COMSOL's electroplating module.**

OpenPlate models copper electroplating from the beaker scale down to a single via or trench. Every model is a
plug-in with the same small interface, so you can run them from Python, from a FastAPI server, or from the
desktop app (Electron + React + Plotly).

> Status: research code. Several models are validated against experiments; others are early. See the table below.

## What it models

| id | package | what it does | status |
|---|---|---|---|
| `cell_current` | `platesim.cell` | 3-D beaker current distribution (gmsh + scikit-fem, Butler-Volmer + Nernst layer, galvanostatic). Coupon current density and thickness maps, overpotential decomposition, Wagner number, current lost to the back and edges | charge conservation to 1e-7; duct potential checked against analytic solution |
| `flow_mass_transfer` | `platesim.flow` | Nernst layer on the coupon: rotation correlation, or a 2-D lattice-Boltzmann slice + Leveque | LBM Poiseuille error < 2 %; Reynolds-number extrapolation is order-of-magnitude only |
| `planar_efficiency` | `platesim.efficiency` | lumped loss model (stray current, CC to CV fallback, O2 reduction, start-up loss) with MAP/MCMC calibration of competing hypotheses | calibrated to a planar DOE (data not included) |
| `coupon_pipeline` | `platesim.coupon` | flow, current and efficiency coupled and calibrated jointly to DOE and SEM points | see `platesim/coupon/validation.json` |
| `feature_fill` | `platesim.feature` | level-set superfilling (S-NDR) fill of one via (axisymmetric) or trench (2-D Cartesian) | Faraday limit within 0.03 % in trench test |
| `sem_thickness` | `platesim.imaging` | pixel size from SEM image metadata and automatic Cu thickness measurement | within 3-5 % of manual measurements |
| `atomistic_adsorption` | `platesim.atomistic` | MACE-MP-0 adsorption energies for Cl, thiolate, H2O and ethers on Cu(100)/(111); separate venv | trends only (see `platesim/atomistic/NOTES.md`) |
| `trenchflow` | `platesim.trenchflow` | trench fill transport with LBM cavity flow and particle transport | convergence studies in `platesim/trenchflow/` |
| `demo_faraday` | `platesim.demo` | Faraday thickness, end-to-end test model | |

Literature parameter sets (digitised and cited, DOIs checked) live in `platesim/data/literature/`.
Each number in the code is tagged with its source: `[FT]` full text, `[AB]` abstract, `[SH]` secondhand,
`[C]` calculated, `[J]` judgement or calibration target, `[M]` measured.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/EkagraAgarwal/openplate.git
cd openplate
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

## Run

Start the simulation server and open the built-in web UI:

```bash
python -m platesim.server --port 8765
# then open http://127.0.0.1:8765/
```

Desktop app:

```bash
cd app && npm install && npm run build && npm start
```

The desktop app starts its own backend. `npm run dev:mock` runs the UI without Python.

## Tests

```bash
python -m pytest tests -m "not slow"      # fast suite
python -m pytest tests                    # includes long validation runs (about 3 min+)
cd app && npm test                        # desktop UI tests
```

Some tests and calibrations read experimental datasets (team DOE and SEM measurements). Those files are not in
this repository, so the affected tests will fail until you supply your own data in the same format.

## Writing a model plug-in

Each model package exposes a `MODEL_SPEC` dictionary and a `run(params, progress)` function. The server discovers
them automatically and the UI builds the input form from `params`. See [CONTRACTS.md](CONTRACTS.md) for the
full interface, units convention and output schema.

```python
MODEL_SPEC = {
    "id": "my_model",
    "title": "My model",
    "description": "...",
    "params": [{"name": "I_total_mA", "type": "float", "default": 48.0, "min": 0.1, "max": 500, "unit": "mA"}],
}

def run(params: dict, progress=lambda frac, msg="": None) -> dict:
    return {"summary": {...}, "plots": [...], "warnings": [], "assumptions": []}
```

Units inside the code are cm, s, mol, A, V and K. Suffixes such as `_um`, `_mA_cm2` and `_min` mark other units.

## Configuration

- `PLATESIM_RESEARCH_DIR` (optional): path to a private reference engine used by one regression check in
  `platesim/feature/validate_trench.py`. Unset means that check is skipped.

## Repository layout

```
platesim/      Python package: models, server, shared data
app/           Desktop UI (Electron, Vite, React, Plotly)
tests/         pytest suites for the Python models and server
CONTRACTS.md   Model plug-in interface and conventions
```

## Contributing

Issues and pull requests are welcome. For a new model, follow the plug-in contract above and add tests in
`tests/test_<module>.py`. Please tag every literature value with its source and do not commit personal data.

## License

MIT. See [LICENSE](LICENSE).
