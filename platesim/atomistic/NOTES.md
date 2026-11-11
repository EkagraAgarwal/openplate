# platesim.atomistic: notes

## Setup
- Separate venv: `platesim/.venv-atomistic` (Python 3.11, ase 3.29.0, mace-torch 0.3.17, torch 2.14.1+cpu). The global
  interpreter never imports torch/mace; `__init__.run()` calls `adsorption.py` via subprocess.
- MACE-MP-0 weights are cached in `~/.cache/mace/` (medium = `20231203mace128L1_epoch199model`).
- UMA / fairchem not installed (needs a gated HuggingFace model licence).
- Suite: `.venv-atomistic\Scripts\python.exe -m platesim.atomistic.adsorption --suite` -> `results.json` (about 10-20 min CPU).

## What the numbers can and cannot mean
- MACE-MP-0 reproduces PBE (trained on MPtrj, Batatia et al. 2023, arXiv:2401.00096). PBE itself mis-orders some molecule-metal
  adsorption energies (no dispersion in the training data, so physisorbed H2O/ether energies are underestimated).
- No solvent, no electrode potential, no Cl- charge, no Cu(I) complexes. The real suppressor is a PEG-Cu(I)-Cl film on a
  Cl-covered surface, and the accelerator is a Cu(I)-thiolate. These calculations do not contain those species.
- Useful only as relative trends:
  - Cl binds strongly in the 4-fold hollow of Cu(100), about -2.4 eV vs 1/2 Cl2. A c(2x2) Cl layer is the expected
    substrate for PEG adsorption (Moffat/Josell picture).
  - Thiolate (SCH3) vs Cl on the same site: which displaces which is the direction behind CEAC accelerator accumulation.
  - The ether O (PEG proxy) and water physisorb weakly on bare Cu. That is consistent with PEG needing Cl- (and Cu+) to
    suppress at all.
- Mapping to the continuum model (`platesim.feature`) is qualitative only. A strong Cl / weak ether contrast supports
  making suppression depend on Cl coverage, and thiolate vs Cl competition supports turning CEAC on. None of these energies
  sets k_plus, k_minus or Gamma; those stay calibration parameters (Braun 2020 high-k set + P2 measurements).
