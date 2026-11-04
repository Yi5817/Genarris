"""
This module implements the RIGID_PRESS algorithm for
optimizing crystal structures under symmetry constraints.

RIGID_PRESS algorithm is adapted from:
[rigid-press](https://github.com/godotalgorithm/rigid-press)

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom", "Jonathan Moussa"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import logging
from itertools import combinations, pairwise

import numpy as np
from ase import Atoms
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from gnrs.optimize.symm_rigid_press.coordinates import RigidCoordinates
from gnrs.optimize.symm_rigid_press.energy import ContactEnergy

logger = logging.getLogger(__name__)


class RigidPressSymm:
    """
    Symmetry-preserving rigid press: minimize cell volume plus contact energy
    over the rigid-body state of the crystal with scipy.
    """

    def __init__(
        self,
        xtal: Atoms,
        mols: list[Atoms],
        radius: np.ndarray,
        stoic: list[int] = (1,),
        symprec: float = 1e-3,
        int_scale: float = 0.1,
        method: str = "BFGS",
        tol: float = 0.01,
        maxiter: int = 5000,
        debug_flag: bool = False,
        rank: int = 0,
    ):
        """
        Initialize the optimization class.

        Args:
            xtal: Initial crystal structure
            mols: One conformer per component
            radius: Interaction radius matrix
            stoic: Copies of each component in the asymmetric unit
            symprec: spglib symmetry tolerance in Angstrom
            int_scale: Interaction energy scaling factor
            method: Optimization method for scipy.optimize.minimize
            tol: Convergence tolerance for optimization
            maxiter: Maximum number of iterations
            debug_flag: Whether to print debug information
            rank: Process rank for parallel computation
        """
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.debug_flag = debug_flag
        self.rank = rank
        self.radius = np.asarray(radius, dtype=float)

        self.set_asu_layout(stoic, [len(mol) for mol in mols])
        if self.n_asu_mols > 1 and self.asu_overlaps(xtal):
            logger.debug("Overlap in asymmetric unit: pressing it as one body")
            self.set_asu_layout([1], [self.n_asu_atoms])

        self.coords = RigidCoordinates(xtal, self.asu_mol_atoms, symprec)
        xtal.spg = self.coords.spg

        # Interaction distance of a molecule pair = half lengths + cutoffs
        half = [
            np.linalg.norm(site.ref_pos, axis=1).max() + np.max(self.radius[mol, mol])
            for site, mol in zip(self.coords.sites, self.asu_mol_atoms)
        ]
        z = self.coords.z
        self.contacts = ContactEnergy(
            self.coords.cell_mol_atoms,
            self.asu_mol_atoms * z,
            self.radius,
            np.tile(np.add.outer(half, half), (z, z)),
            int_scale / self.n_asu_atoms**2,
        )

    def set_asu_layout(self, stoic: list[int], sizes: list[int]) -> None:
        """
        Set the atom range of each molecule in the asymmetric unit.
        """
        mol_types = [t for t, n in enumerate(stoic) for _ in range(n)]
        self.n_asu_mols = len(mol_types)
        ends = np.cumsum([0, *(sizes[t] for t in mol_types)])
        self.asu_mol_atoms = [
            slice(int(start), int(stop)) for start, stop in pairwise(ends)
        ]
        self.n_asu_atoms = int(ends[-1])

    def asu_overlaps(self, xtal: Atoms) -> bool:
        """
        Whether two molecules of the asymmetric unit are closer than the cutoff.
        """
        pos = xtal.positions[: self.n_asu_atoms]
        return any(
            np.any(cdist(pos[mol1], pos[mol2]) < self.radius[mol1, mol2])
            for mol1, mol2 in combinations(self.asu_mol_atoms, 2)
        )

    def evaluate(self, state: np.ndarray) -> tuple[float, np.ndarray, float]:
        """
        Total energy (volume + contacts), its gradient and the smallest atom
        distance minus cutoff.
        """
        lattice, frac = self.coords.positions(state)
        energy, grad_pos, grad_lat, min_gap = self.contacts(lattice, frac)
        if np.isinf(energy):
            return np.inf, np.zeros(len(state)), min_gap
        volume = abs(np.linalg.det(lattice))
        grad_lat = grad_lat + volume * np.linalg.inv(lattice).T
        return (
            volume + energy,
            self.coords.gradient(state, grad_pos, grad_lat),
            min_gap,
        )

    def run(self, xtal: Atoms) -> bool:
        """
        Minimize the energy in place; return whether the optimization converged.
        """
        state = self.coords.to_state(xtal)
        if self.evaluate(state)[2] < 0:
            logger.debug("Rank %d: atoms overlap, structure not pressed", self.rank)
            return False

        res = minimize(
            lambda s: self.evaluate(s)[:2],
            state,
            jac=True,
            method=self.method,
            tol=self.tol,
            options={"disp": self.debug_flag, "maxiter": self.maxiter},
        )
        if self.debug_flag:
            logger.debug("Rank %d: %s", self.rank, res.message)

        xtal_opt = self.coords.to_crystal(res.x)
        xtal.set_cell(xtal_opt.cell)
        xtal.set_positions(xtal_opt.positions)
        return res.success and self.evaluate(res.x)[2] >= 0
