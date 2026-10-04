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


def test_asu_generation_workflow(
    tmp_path: Path, run_gnrs: Callable[..., subprocess.CompletedProcess]
) -> None:
    for molecule in (Path(__file__).parents[1] / "data").glob("BEDQAG_mol*.xyz"):
        shutil.copy(molecule, tmp_path)
    (tmp_path / "ui.conf").write_text(CONFIG)

    proc = run_gnrs(tmp_path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    asus = read_json(tmp_path / "structures" / "asu_generation" / "structures.json")
    assert len(asus) == 6
