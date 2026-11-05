"""
Log setup, with or without mpirun.

Run with:  pytest tests/core
           mpirun -np 3 python -m pytest tests/core
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from mpi4py import MPI

from gnrs.core.logging import GenarrisLogger

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
logger = logging.getLogger("gnrs.test")


def test_one_line_per_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(comm.bcast(tmp_path, root=0))
    genlogger = GenarrisLogger(comm)
    logger.info("info on every rank")
    logger.warning("warning on every rank")
    if rank == size - 1:
        logger.warning("warning on the last rank")
    genlogger.sync()
    logging.getLogger().removeHandler(genlogger.handler)

    text = comm.bcast(Path("Genarris.log").read_text() if rank == 0 else None)
    assert text.count("info on every rank") == 1
    assert text.count("warning on the last rank") == 1
    # Rank 0 writes its own line; the other ranks share one
    assert text.count("warning on every rank") == min(size, 2)
    if size > 2:
        assert f"[{size - 1} ranks] warning on every rank" in text
        assert f"[rank {size - 1}] warning on the last rank" in text
