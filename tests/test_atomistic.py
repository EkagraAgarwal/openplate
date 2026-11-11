import json
import subprocess

import pytest

from platesim import atomistic

pytestmark = pytest.mark.skipif(not atomistic.VENV_PY.exists(), reason="atomistic venv missing")


def _venv(code: str) -> str:
    out = subprocess.run([str(atomistic.VENV_PY), "-c", code], cwd=str(atomistic.ROOT), capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    return out.stdout.strip().splitlines()[-1]


def test_lattice_constant():
    a0 = float(_venv("from platesim.atomistic.adsorption import calculator, lattice_constant;"
                     "print(lattice_constant(calculator('small')))"))
    assert abs(a0 - 3.615) / 3.615 < 0.015


def test_cl_hollow_binds():
    r = atomistic.run({"surface": "100", "adsorbate": "Cl", "site": "hollow", "model": "small", "use_cache": False})
    assert r["summary"]["E_ads_eV"] < 0
    assert 1.2 < r["summary"]["anchor_height_A"] < 2.2
    json.dumps(r)


def test_bad_site_rejected():
    with pytest.raises(ValueError):
        atomistic.run({"surface": "111", "site": "hollow"})
