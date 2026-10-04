"""
This module provides parallel processing utilities for Genarris.

This source code is licensed under the BSD-3 license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import logging

from mpi4py import MPI
from mpi4py.util import pkl5

logger = logging.getLogger(__name__)

comm = None
rank = None
size = None
is_master = None
base_seed = 42


def init_parallel(comm_in: MPI.Comm) -> None:
    """
    Initialize parallel environment with MPI communicator.

    ``gnrs.parallel.comm`` is the package communicator; pass it to tasks.
    It has no 2 GiB limit on gather, scatter and bcast, unlike plain mpi4py.

    Args:
        comm_in: MPI communicator object
    """
    global comm, rank, size, is_master

    comm = pkl5.Intracomm(comm_in)
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_master = rank == 0


init_parallel(MPI.COMM_WORLD)
