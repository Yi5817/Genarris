"""
This module provides a wrapper around the BFGS implementation from ASE.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

from ase.atoms import Atoms
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry

from gnrs.core.optimizer import GeometryOptimizerABC


class BFGSOptimizer(GeometryOptimizerABC):
    """
    Limited-memory BFGS optimization using ASE's BFGS implementation.
    
    Attributes:
        fmax: Maximum force tolerance for convergence criterion
        steps: Maximum number of optimization steps to perform
        fix_sym: Whether to fix the symmetry of the structure
        cell_opt: Whether to optimize the cell parameters as well
        opt_name: Name of the optimizer for storing in the crystal info
        converged: Whether the optimization successfully converged
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.opt_name = "bfgs"
        self.fmax = self.tsk_set.pop("fmax")
        self.steps = self.tsk_set.pop("steps")
        self.fix_sym = self.tsk_set.pop("fix_sym")
        self.cell_opt = self.tsk_set.pop("cell_opt")

    def optimize(self, xtal: Atoms) -> None:
        """
        Performs geometry optimization using BFGS algorithm.
        
        Args:
            xtal: ASE Atoms object
        """
        # Assign the calculator to the structure
        xtal.calc = self.energy_calc
        if self.fix_sym:
            xtal.set_constraint(FixSymmetry(xtal))
        if self.cell_opt:
            ecf = FrechetCellFilter(xtal)
            dyn = BFGS(ecf, master=True, logfile="bfgs.log", **self.tsk_set)
        else:
            dyn = BFGS(
                xtal, master=True, logfile="bfgs.log", **self.tsk_set
            )

        try:
            self.converged = dyn.run(fmax=self.fmax, steps=self.steps)
        except:
            self.converged = False
        # Remove the constraints
        if self.fix_sym:
            del xtal.constraints

    def update(self, xtal: Atoms) -> None:
        """
        Update the optimizer with the new structure.
        """
        super().update(xtal)
        try:
            xtal.info[f"{self.opt_name}_{self.energy_method}"] = xtal.get_potential_energy()
        except:
            xtal.info[f"{self.opt_name}_{self.energy_method}"] = 0
        xtal.info[self.opt_name] = "converged" if self.converged else "unconverged"
