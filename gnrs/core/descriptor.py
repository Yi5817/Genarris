"""
Abstract base class for crystal structure descriptors.

This module provides the base class for implementing crystal structure descriptors.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import abc

from mpi4py import MPI
from ase import Atoms

class DescriptorABC(abc.ABC):
    """
    Abstract base class for crystal structure descriptors.
    
    This class defines the interface for computing descriptors for crystal structures.
    All descriptor implementations should inherit from this class and implement
    the abstract methods.
    """

    def __init__(self, comm: MPI.Comm, task_settings: dict) -> None:
        """Initialize the descriptor calculator.
        
        Args:
            comm: MPI communicator
            task_settings: Task settings
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_master = self.rank == 0
        self.tsk_set = task_settings

    def run(self, xtal: Atoms) -> None:
        """
        Run the descriptor computation workflow.
        
        1. Initialize
        2. Compute descriptor
        3. Finalize
        
        Args:
            xtal: Crystal structure
        """
        self.initialize()
        self.compute(xtal)
        self.finalize()

    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initialize for descriptor computation.
        """
        pass

    @abc.abstractmethod
    def compute(self, xtal: Atoms) -> None:
        """
        Compute descriptor for a crystal structure.
        """
        pass

    @abc.abstractmethod
    def finalize(self) -> None:
        pass
