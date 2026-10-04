"""
The ASU generator builds asymmetric units from the two molecules of the
BEDQAG co-crystal. Run in a single process (no mpirun).
"""
from __future__ import annotations

from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms
from mpi4py import MPI

import gnrs.parallel as gp
from gnrs.generation.asu import ASUGenerator
from gnrs.parallel.io import read_geometry_out

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() != 1, reason="single-process unit test"
)

DATA = Path(__file__).parents[1] / "data"
MOLECULE_PATHS = [str(DATA / "BEDQAG_mol1.xyz"), str(DATA / "BEDQAG_mol2.xyz")]
SETTINGS = {
    "stoichiometry": [1, 1],
    "num_asus": 5,
    "sr_min": 0.75,
    "sr_max": 1.3,
    "max_attempts_per_asu": 100000,
    "seed": 42,
}


def generate_asus(run_dir: Path, **settings) -> list[Atoms]:
    gp.init_parallel(MPI.COMM_WORLD)
    generator = ASUGenerator(
        MPI.COMM_WORLD,
        {"master": {"molecule_path": MOLECULE_PATHS}},
        {"molecule_path": MOLECULE_PATHS},
    )
    task_set = generator.pack_settings({**SETTINGS, **settings})
    generator.generate(task_set, str(run_dir))
    return list(read_geometry_out(str(run_dir / "geometry.out")).values())


def test_asus_hold_both_molecules_rigid_in_input_order(tmp_path: Path) -> None:
    mol1, mol2 = (ase.io.read(path, parallel=False) for path in MOLECULE_PATHS)
    asus = generate_asus(tmp_path)
    assert len(asus) == SETTINGS["num_asus"]
    for asu in asus:
        assert asu.get_chemical_symbols() == list((mol1 + mol2).symbols)
        assert not asu.pbc.any()
        for placed, mol in ((asu[: len(mol1)], mol1), (asu[len(mol1) :], mol2)):
            np.testing.assert_allclose(
                placed.get_all_distances(), mol.get_all_distances(), atol=1e-4
            )


def test_invalid_sr_window_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="could not generate asymmetric units"):
        generate_asus(tmp_path, sr_min=1.3, sr_max=0.75)
