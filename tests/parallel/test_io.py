"""
Structure file IO, with or without mpirun.

Run with:  pytest tests/parallel
           mpirun -np 3 python -m pytest tests/parallel
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io.jsonio import read_json
from mpi4py import MPI

import gnrs.parallel as gp
from gnrs.parallel.io import read_parallel, write_parallel

comm = MPI.COMM_WORLD


def test_structures_json_round_trip(tmp_path: Path) -> None:
    gp.init_parallel(comm)
    path = comm.bcast(str(tmp_path / "structures.json"), root=0)
    # Uneven pool: rank r holds r + 1 structures
    structs = {
        f"r{gp.rank}s{i}": Atoms("OH", positions=[[0, 0, 0], [0, 0, 1 + i]])
        for i in range(gp.rank + 1)
    }
    written = {k: v for part in comm.allgather(structs) for k, v in part.items()}

    write_parallel(path, structs)
    scattered = comm.allgather(read_parallel(path))

    assert list(read_json(path)) == list(written)
    assert {k: v for part in scattered for k, v in part.items()} == written
    sizes = [len(part) for part in scattered]
    assert max(sizes) - min(sizes) <= 1
