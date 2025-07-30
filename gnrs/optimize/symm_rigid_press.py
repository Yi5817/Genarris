"""
This module provides a wrapper around the Rigid Press implementation with symmetry constraints.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang","Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

from ase.io import read
from ase import Atoms

from gnrs.core.optimizer import GeometryOptimizerABC
from gnrs.optimize.rpress_symm_impl import RigidPressSymm

class SYMM_RIGID_PRESSOptimizer(GeometryOptimizerABC):
    """
    Optimizes crystal structures using rigid_press algorithm with symmetry constraints.

    This optimizer uses van der Waals cutoff matrices for 
    tight molecular packing and preserves symmetry during optimization.

    Attributes:
        opt_name: Name identifier for the optimizer
        int_scale: Interaction scale factor
        method: Optimization method to use
        tol: Convergence tolerance
        maxiter: Maximum number of iterations
        vol_tol: Volume tolerance
        debug_flag: Whether to enable debug output
    """

    def __init__(self, *args):
        """
        Initialize the symmetry-preserving rigid press optimizer.
        """
        super().__init__(*args)
        self.opt_name = "symm_rigid_press"
        self.int_scale = self.tsk_set["int_scale"]
        self.method = self.tsk_set["method"]
        self.tol = self.tsk_set["tol"]
        self.maxiter = self.tsk_set["maxiter"]
        self.vol_tol = self.tsk_set["vol_tol"]
        self.debug_flag = self.tsk_set["debug_flag"]
        self.mol_path = self.tsk_set["mol_path"]
        self.cutoff_matrix = self.tsk_set["cutoff_matrix"]

    def initialize(self) -> None:
        """
        Initialize the optimizer.
        """
        pass

    def optimize(self, xtal: Atoms) -> None:
        """
        Perform structure optimization using the RigidPressSymm algorithm.

        Args:
            xtal: ASE Atoms object
        """
        if self.debug_flag:
            xtal.write("geometry.in", parallel=False)
        mol = read(self.mol_path, parallel=False)
        opt = RigidPressSymm(
            mol=mol,
            xtal=xtal,
            radius=self.cutoff_matrix,
            int_scale=self.int_scale,
            method=self.method,
            tol=self.tol,
            maxiter=self.maxiter,
            vol_tol=self.vol_tol,
            debug_flag=self.debug_flag,
            rank=self.rank,
        )
        self.status = opt.run(xtal)

    def update(self, xtal: Atoms) -> None:
        """
        Update crystal info with optimization status.

        Args:
            xtal: ASE Atoms object
        """
        xtal.info[self.opt_name] = "converged" if self.status else "unconverged"
