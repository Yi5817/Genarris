"""
This module provides a wrapper around the rigid_press C code.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

from ase import Atoms
from gnrs.cgenarris.src.rpack.rigid_press import optimize_structure

from gnrs.core.optimizer import GeometryOptimizerABC

class RIGID_PRESSOptimizer(GeometryOptimizerABC):
    """
    Optimizes structure using rigid_press C code.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.opt_name = "rigid_press"
        self.maxiter = self.tsk_set["maxiter"]
        self.debug_flag = self.tsk_set["debug_flag"]
        self.cutoff_matrix = self.tsk_set["cutoff_matrix"]
        self.z = self.tsk_set["z"]

    def initialize(self) -> None:
        """
        Initialize the optimizer.
        """
        pass

    def optimize(self, xtal: Atoms) -> None:
        """
        Optimize the structure using rigid_press C code.
        """
        if self.debug_flag:
            xtal.write("geometry.in", parallel=False)
        self.status = False

        self.status = optimize_structure(
                xtal,
                self.z,
                self.cutoff_matrix,
                max_iter=self.maxiter,
            )
            
    def update(self, xtal: Atoms) -> None:
        """
        Update the optimizer with the new structure.
        """
        if self.status:
            xtal.info[self.opt_name] = "converged"
        else:
            xtal.info[self.opt_name] = "unconverged"
