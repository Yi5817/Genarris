"""
This module provides a wrapper around the Rigid Press implementation with symmetry constraints.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import logging
from collections.abc import Callable, Collection

import torch
from ase import Atoms
from ase.io import read

import gnrs.output as gout
from gnrs.core.optimizer import GeometryOptimizerABC
from gnrs.gnrsutil.core import check_no_changes_in_covalent_matrix
from gnrs.optimize.symm_rigid_press.batch import RigidPressBatch
from gnrs.optimize.symm_rigid_press.optimizer import RigidPressSymm

logger = logging.getLogger(__name__)


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
        debug_flag: Whether to enable debug output
        symprec: spglib symmetry tolerance in Angstrom
        batch_size: Crystals optimized at once on the GPU (0: scipy, one at a time)
    """

    def __init__(self, *args):
        """
        Initialize the symmetry-preserving rigid press optimizer.
        """
        super().__init__(*args)
        self.opt_name = "symm_rigid_press"
        self.int_scale = self.tsk_set["int_scale"]
        self.method = self.tsk_set["method"]
        if self.method == "L-BFGS-B":
            raise ValueError(
                "symm_rigid_press: L-BFGS-B stops at the starting structure; use BFGS"
            )
        self.tol = self.tsk_set["tol"]
        self.maxiter = self.tsk_set["maxiter"]
        self.debug_flag = self.tsk_set["debug_flag"]
        self.symprec = self.tsk_set["symprec"]
        self.mol_path = self.tsk_set["mol_path"]
        self.stoichiometry = self.tsk_set["stoichiometry"]
        self.cutoff_matrix = self.tsk_set["cutoff_matrix"]
        self.batch_size = self.tsk_set["batch_size"]
        if self.batch_size and not torch.cuda.is_available():
            raise ValueError("symm_rigid_press: batch_size > 0 needs a GPU on rank 0")

    def initialize(self) -> None:
        """
        Initialize the optimizer.
        """

    def prepare(self, xtal: Atoms) -> RigidPressSymm:
        """
        Analyse the symmetry of a crystal and set up its energy.
        """
        mols = [read(path, parallel=False) for path in self.mol_path]
        return RigidPressSymm(
            xtal,
            mols,
            self.cutoff_matrix,
            stoic=self.stoichiometry,
            symprec=self.symprec,
            int_scale=self.int_scale,
            method=self.method,
            tol=self.tol,
            maxiter=self.maxiter,
            debug_flag=self.debug_flag,
            rank=self.rank,
        )

    def optimize(self, xtal: Atoms) -> None:
        """
        Perform structure optimization using the RigidPressSymm algorithm.

        Args:
            xtal: ASE Atoms object
        """
        if self.debug_flag:
            xtal.write("geometry.in", parallel=False)
        self.status = self.prepare(xtal).run(xtal)

    def run_batch(
        self,
        structs: dict[str, Atoms],
        on_structure_done: Callable[[str, Atoms], None] | None = None,
        done: Collection[str] = (),
    ) -> None:
        """
        Optimize the structures of this rank. With batch_size > 0, rank 0
        collects the structures of all ranks and optimizes them in one torch
        batch on its GPU.
        """
        if not self.batch_size:
            super().run_batch(structs, on_structure_done, done)
            return
        todo = {name: xtal for name, xtal in structs.items() if name not in done}
        initial = {name: xtal.copy() for name, xtal in todo.items()}
        parts = self.comm.gather(todo, root=0)
        if self.is_master:
            pool = {name: xtal for part in parts for name, xtal in part.items()}
            self.optimize_batch(pool)
            parts = [
                {name: pool[name] for name in part if name in pool} for part in parts
            ]
        mine = self.comm.scatter(parts, root=0)
        for name in todo.keys() - mine.keys():
            del structs[name]
        for name, xtal in mine.items():
            structs[name] = xtal
            if not self.skip_bond_check:
                # Every rank checks its own structures: the check is slow
                xtal.info[f"{self.opt_name}_bonds_intact"] = (
                    check_no_changes_in_covalent_matrix(initial[name], xtal)
                )
            if on_structure_done is not None:
                on_structure_done(name, xtal)
        n_failed = self.comm.allreduce(len(todo) - len(mine))
        if n_failed:
            gout.warning(
                f"{n_failed} structure(s) failed optimization and were "
                "removed. Genarris.log lists them."
            )

    def optimize_batch(self, structs: dict[str, Atoms]) -> None:
        """
        Optimize structures in place with the batched torch backend; drop
        those whose symmetry analysis fails.
        """
        opts = {}
        for name, xtal in list(structs.items()):
            try:
                opts[name] = self.prepare(xtal)
            except (ValueError, RuntimeError) as e:
                logger.warning("Optimization failed for %s: %s", name, e)
                del structs[name]
        if not opts:
            return
        logger.info("Optimizing %d structures on the GPU", len(opts))
        converged = RigidPressBatch(list(opts.values())).run(
            [structs[name] for name in opts], self.tol, self.maxiter, self.batch_size
        )
        for name, status in zip(opts, converged):
            structs[name].info[self.opt_name] = "converged" if status else "unconverged"

    def update(self, xtal: Atoms) -> None:
        """
        Update crystal info with optimization status.

        Args:
            xtal: ASE Atoms object
        """
        xtal.info[self.opt_name] = "converged" if self.status else "unconverged"
