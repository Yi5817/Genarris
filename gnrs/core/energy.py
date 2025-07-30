"""
Abstract base class for energy calculators.

This module provides the base class for implementing energy calculators.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import abc
from typing import Any
from mpi4py import MPI
from ase import Atoms

class EnergyCalculatorABC(abc.ABC):
    """
    Abstract base class for energy calculators.
    
    This class defines the interface for computing energies for crystal structures.
    All energy calculator implementations should inherit from this class and implement
    the abstract methods.
    """
    def __init__(self, comm: MPI.Comm, task_settings: dict, energy_name: str) -> None:
        """
        Initialize the energy calculations.

        Args:
            comm: MPI communicator
            task_settings: Task settings
            energy_name: Energy name
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_master = True if self.rank == 0 else False
        self.tsk_set = task_settings
        self.energy_name = energy_name

        # Stores energy calculator
        self.calc = None

    def run(self, xtal: Atoms) -> None:
        """
        Run the energy calculations.

        Args:
            xtal: Crystal structure
        """
        if self.energy_name in xtal.info:
            return

        self.initialize()
        self.compute(xtal)
        self.finalize()

    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initialize the energy calculations.
        """
        pass

    def get_calculator(self) -> Any:
        """
        Returns the calculator.
        """
        return self.calc

    @abc.abstractmethod
    def compute(self, xtal: Atoms) -> None:
        """
        Compute the energy.

        Args:
            xtal: Crystal structure
        """
        pass

    @abc.abstractmethod
    def finalize(self) -> None:
        """
        Finalize the energy calculations.
        """
        pass
