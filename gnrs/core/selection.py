"""
Abstract base class for crystal structure selection.

This module provides the base class for implementing crystal structure selection.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import abc
from mpi4py import MPI

class SelectionABC(abc.ABC):
    """
    Abstract base class for crystal structure selection algorithms.
    
    This class defines the interface for selecting crystal structures from a pool.
    All selection implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, comm: MPI.Comm, settings: dict) -> None:
        """Initialize the selection algorithm.
        
        Args:
            comm: MPI communicator for parallel computation
            settings: Selection settings
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_master = self.rank == 0
        self.settings = settings

    def run(self, struct_dict: dict) -> None:
        """
        Run the selection workflow.
        
        1. Initialize
        2. Perform selection
        3. Finalize
        
        Args:
            struct_dict: Crystal structures
        """
        self.initialize()
        self.select(struct_dict)
        self.finalize()

    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initialize for selection.
        """
        pass

    @abc.abstractmethod
    def select(self, struct_dict: dict) -> None:
        """
        Perform selection on the provided structures.
        
        Args:
            struct_dict: Crystal structures
        """
        pass

    @abc.abstractmethod
    def finalize(self) -> None:
        """
        Finalize the selection.
        """
        pass
