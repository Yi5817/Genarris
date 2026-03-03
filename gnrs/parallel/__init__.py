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
import random

import numpy as np
from mpi4py import MPI

logger = logging.getLogger("parallel")

comm = None
rank = None
size = None
is_master = None
base_seed = None


def init_parallel(comm_in: MPI.Comm, seed: int = 42) -> None:
    """
    Initialize parallel environment with MPI communicator.
    
    Args:
        comm_in: MPI communicator object
        seed: Random seed
    """
    global comm, rank, size, is_master, base_seed
    
    comm = comm_in
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_master = rank == 0
    
    base_seed = seed
    rank_seed = base_seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
