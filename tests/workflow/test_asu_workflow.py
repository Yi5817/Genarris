"""
End-to-end test of a workflow with the asu_generation task, launched under
mpirun on the two molecules of the BEDQAG co-crystal.

Run with:  pytest -m integration
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from ase.io.jsonio import read_json

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not found"),
]

CONFIG = """[master]
name = BEDQAG
molecule_path = ["./BEDQAG_mol1.xyz", "./BEDQAG_mol2.xyz"]

[workflow]
tasks = ['asu_generation']

[asu_generation]
stoichiometry = [1, 1]
num_asus = 6
"""


CRYSTAL_CONFIG = """[master]
name = BEDQAG
molecule_path = ["./BEDQAG_mol1.xyz", "./BEDQAG_mol2.xyz"]
z = 2

[workflow]
tasks = ['asu_generation', 'crystal_generation']

[asu_generation]
stoichiometry = [1, 1]
num_asus = 2

[crystal_generation]
stoichiometry = [1, 1]
num_structures_per_spg = 2
spg_distribution_type = [2, 4]
"""


def write_inputs(run_dir: Path, config: str) -> None:
    for molecule in (Path(__file__).parents[1] / "data").glob("BEDQAG_mol*.xyz"):
        shutil.copy(molecule, run_dir)
    (run_dir / "ui.conf").write_text(config)


def test_crystal_generation_from_asus(
    tmp_path: Path, run_gnrs: Callable[..., subprocess.CompletedProcess]
) -> None:
    write_inputs(tmp_path, CRYSTAL_CONFIG)

    proc = run_gnrs(tmp_path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    asus = read_json(tmp_path / "structures" / "asu_generation" / "structures.json")
    xtals = read_json(
        tmp_path / "structures" / "crystal_generation" / "structures.json"
    )
    # 2 crystals for each of 2 ASUs in 2 space groups, with 2 ASUs per cell
    assert len(xtals) == 8
    for xtal in xtals.values():
        assert xtal.pbc.all()
        assert xtal.info["spg"] in (2, 4)
        assert (
            xtal.get_chemical_symbols()
            == 2 * next(iter(asus.values())).get_chemical_symbols()
        )
    for name in asus:
        assert (
            (tmp_path / "tmp" / "crystal_generation" / name / "geometry.out")
            .stat()
            .st_size
        )


def test_asu_generation_workflow(
    tmp_path: Path, run_gnrs: Callable[..., subprocess.CompletedProcess]
) -> None:
    write_inputs(tmp_path, CONFIG)

    proc = run_gnrs(tmp_path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    asus = read_json(tmp_path / "structures" / "asu_generation" / "structures.json")
    assert len(asus) == 6

    # No ASU fits this window: the rerun must fail, not keep the pool above
    (tmp_path / "ui.conf").write_text(
        CONFIG + "sr_min = 50\nsr_max = 51\nmax_attempts_per_asu = 10\n"
    )
    proc = run_gnrs(tmp_path, "--overwrite")
    assert proc.returncode != 0
    assert "generated no structures" in proc.stdout + proc.stderr
