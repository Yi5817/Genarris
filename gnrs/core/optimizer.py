"""
Abstract base class for geometry optimization.

This module provides the base class for implementing geometry optimization.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import abc

from mpi4py import MPI
from ase.atoms import Atoms

class GeometryOptimizerABC(abc.ABC):
    """
    Abstract class for Geometry optimization methods.
    
    All optimizers should inherit this class.
    """

    def __init__(
        self, 
        comm: MPI.Comm, 
        task_set: dict,
        opt_name: str = "relax",
        energy_method: str | None = None,
        energy_calc: any | None = None,
    ) -> None:
        """
        Initialize the geometry optimizer.
        
        Args:
            comm: MPI communicator for parallel computation
            task_set: Optimization settings
            opt_name: Optimizer
            energy_method: Energy calculation method
        """
        self.opt_name = opt_name
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_master = self.rank == 0
        self.tsk_set = task_set
        self.energy_method = energy_method
        self.energy_calc = energy_calc
        self.converged = False

    def run(self, xtal: Atoms) -> None:
        """
        Run the optimization workflow.
        
        1. Initialize
        2. Perform optimization
        3. Update structure information
        4. Finalize
        
        Args:
            xtal: ASE Atoms object representing the crystal structure
        """

        self.initialize()
        self.optimize(xtal)
        self.update(xtal)
        self.finalize(xtal)

    def initialize(self) -> None:
        """
        Initialize for optimization.
        """
        pass

    @abc.abstractmethod
    def optimize(self, xtal: Atoms) -> None:
        """
        Perform optimization.
        
        Args:
            xtal: ASE Atoms object
        """
        pass

    @abc.abstractmethod
    def update(self, xtal: Atoms) -> None:
        """
        Update the geometry and add energy information.
        
        Args:
            xtal: ASE Atoms object
        """
        pass

    def finalize(self, xtal: Atoms) -> None:
        """
        Finalize the optimization and clean up.
        
        Args:
            xtal: ASE Atoms object
        """
        xtal.calc = None
