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
import warnings
from itertools import combinations, pairwise

import numpy as np
from ase import Atoms
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from gnrs.optimize.symm_rigid_press.symmetry import (
    find_site,
    find_symm,
    get_lattice_type,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)
# Maximum number of unit cells to sum over before deciding a crystal is too tightly packed
MAX_CELL_SUM = 2000


class RigidPressSymm:
    """
    Class to optimize a crystal structure under symmetry constraints
    using the RIGID_PRESS algorithm.

    This class implements the constrained optimization of crystal structures
    while preserving the space group symmetry.
    """

    def __init__(
        self,
        mols: list[Atoms],
        xtal: Atoms,
        radius: np.ndarray,
        stoic: list[int] = (1,),
        int_scale: float = 0.1,
        method: str = "BFGS",
        tol: float = 0.01,
        maxiter: int = 5000,
        vol_tol: float = 10,
        debug_flag: bool = False,
        rank: int = 0,
    ):
        """
        Initialize the optimization class.

        Args:
            mols: Reference conformer of each component
            xtal: Initial crystal structure
            radius: Interaction radius matrix
            stoic: Copies of each component in the asymmetric unit
            int_scale: Interaction energy scaling factor
            method: Optimization method for scipy.optimize.minimize
            tol: Convergence tolerance for optimization
            maxiter: Maximum number of iterations
            vol_tol: Minimum allowed volume in callback function
            debug_flag: Whether to print debug information
            rank: Process rank for parallel computation
        """
        self.logger = logger
        self.int_scale = int_scale
        self.method = method
        self.tol = tol
        self.vol_tol = vol_tol
        self.maxiter = maxiter
        self.debug_flag = debug_flag
        self.rank = rank
        self.radius = radius

        self.ref_pos = [mol.positions - mol.positions.mean(axis=0) for mol in mols]
        self.stoic = list(stoic)
        self._set_asu_layout()
        if self.n_asu_mols > 1 and self._asu_overlaps(xtal):
            self.logger.debug("Overlap in asymmetric unit: pressing it as one body")
            pos = xtal.positions[: self.n_asu_atoms]
            self.ref_pos = [pos - pos.mean(axis=0)]
            self.stoic = [1]
            self._set_asu_layout()

        # One operation per asymmetric unit in the cell
        # all_ops is the full space group
        self.asu_rot, self.asu_trans, self.spg, all_ops = find_symm(
            xtal, self.n_asu_atoms
        )
        self.z = len(self.asu_rot)
        self.n_cell_mols = self.z * self.n_asu_mols
        xtal.spg = self.spg

        # Wyckoff site of every molecule of the asymmetric unit (None if general)
        self.sites = [
            find_site(*all_ops, xtal.positions[mol], xtal.numbers[mol], xtal.cell.array)
            for mol in self.asu_mol_atoms
        ]
        # State parameters of every molecule centre, then of every orientation
        self.dofs = [3 if s is None else s.directions.shape[1] for s in self.sites] + [
            3 if s is None else s.axes.shape[1] for s in self.sites
        ]

        # Atom range of every molecule in the cell, asymmetric unit by asymmetric unit
        self.cell_mol_atoms = [
            slice(i * self.n_asu_atoms + mol.start, i * self.n_asu_atoms + mol.stop)
            for i in range(self.z)
            for mol in self.asu_mol_atoms
        ]
        self.D_cell = np.tile(self.D, (self.z, self.z))

    def _set_asu_layout(self) -> None:
        """
        Set the atom range of each molecule in the asymmetric unit
        and the interaction distance of each pair of them.
        """
        self.mol_types = [t for t, n in enumerate(self.stoic) for _ in range(n)]
        self.n_asu_mols = len(self.mol_types)
        sizes = [len(self.ref_pos[t]) for t in self.mol_types]
        ends = np.cumsum([0, *sizes])
        self.asu_mol_atoms = [
            slice(int(start), int(stop)) for start, stop in pairwise(ends)
        ]
        self.n_asu_atoms = int(ends[-1])
        # Interaction distance = mean mol length + max cutoff of each molecule
        half = [
            np.linalg.norm(self.ref_pos[t], axis=1).max()
            + np.max(self.radius[mol, mol])
            for t, mol in zip(self.mol_types, self.asu_mol_atoms)
        ]
        self.D = np.add.outer(half, half)

    def _asu_overlaps(self, xtal: Atoms) -> bool:
        """
        Whether two molecules of the asymmetric unit are closer than the cutoff.

        Args:
            xtal: Crystal structure

        Returns:
            True if any pair of atoms is within its cutoff
        """
        pos = xtal.positions[: self.n_asu_atoms]
        return any(
            np.any(cdist(pos[mol1], pos[mol2]) < self.radius[mol1, mol2])
            for mol1, mol2 in combinations(self.asu_mol_atoms, 2)
        )

    def find_pairs(self, xtal: Atoms) -> dict:
        """
        Find the molecule pairs that are within the interaction distance of the central cell.

        Args:
            xtal: ASE Atoms object

        Returns:
            A dict with key as mol indices pair and value as lattice
            displacements that are within interaction distance.
        """
        # Calculate centres of geometry of each molecule in fractional and cartesian
        pos = xtal.get_scaled_positions(wrap=False)
        lattice = xtal.cell.array
        cog_f = np.array([np.mean(pos[mol], axis=0) for mol in self.cell_mol_atoms])
        cog_c = cog_f @ lattice

        # The bounds of the super cell considered
        cell_lengths = xtal.cell.lengths()
        max_lat = np.ceil(np.max(self.D) / cell_lengths + 2).astype(int)
        min_lat = -max_lat

        # Test if a crystal is too packed to continue
        if np.prod(max_lat) > MAX_CELL_SUM or np.any(max_lat <= 0):
            return {"inf": True}

        # Create combinations of integer displacements in fractional space
        lat_ranges = [np.arange(min_lat[i], max_lat[i]) for i in range(3)]
        comb_lat = np.array(np.meshgrid(*lat_ranges)).T.reshape(-1, 3)

        # Calculate possible displacements in real space
        lattice_dis = comb_lat @ lattice

        interact_pairs = {}
        # Cross pairs within range, (0,0,0) are molecules within the cell
        for mol1 in range(self.n_cell_mols):
            for mol2 in range(mol1 + 1, self.n_cell_mols):
                disp = cog_c[mol1] - (cog_c[mol2] + lattice_dis)
                dist = np.linalg.norm(disp, axis=1)
                interact_pairs[(mol1, mol2)] = comb_lat[dist < self.D_cell[mol1, mol2]]

        # Periodic pairs within range
        # Molecule will interact with its own mirror image under periodic boundaries
        # (0,0,0) is the same molecule hence excluded
        comb_lat_non_zero = comb_lat[~np.all(comb_lat == 0, axis=1)]
        non_zero_disp = comb_lat_non_zero @ lattice
        dist = np.linalg.norm(non_zero_disp, axis=1)
        for mol1 in range(self.n_cell_mols):
            interact_pairs[(mol1, mol1)] = comb_lat_non_zero[
                dist < self.D_cell[mol1, mol1]
            ]
        return interact_pairs

    def interaction_kernel(
        self, dist: np.ndarray, radius: np.ndarray, weight: float, D: float
    ) -> float:
        """
        Define how atoms of different molecules interact.

        Args:
            dist: Array of distances between atoms
            radius: Array of cutoff radii
            weight: Scaling factor for interaction energy
            D: Interaction distance of the molecule pair

        Returns:
            Total interaction energy
        """
        energy = np.zeros(len(dist))
        energy[dist < radius] = np.inf
        energy[dist > D] = 0
        energy[(dist > radius) & (dist < D)] = (weight * (D - dist) / (dist - radius))[
            (dist > radius) & (dist < D)
        ]
        return energy.sum()

    def pair_energy(
        self, xtal: Atoms, mol1: int, mol2: int, weight: float, interact_pairs: dict
    ) -> float:
        """
        Compute the interaction energy between a pair of molecules.

        Args:
            xtal: ASE Atoms object
            mol1: Index of first molecule
            mol2: Index of second molecule
            weight: Scaling factor for interaction energy
            interact_pairs: Dictionary of interacting pairs and their lattice displacements

        Returns:
            Total energy of the pair
        """
        lattice = xtal.cell.array
        pairs = interact_pairs[(mol1, mol2)]
        pos1 = xtal.positions[self.cell_mol_atoms[mol1]]
        # Combine all possible lattice displacements of mol2 into a single vector
        pos2 = (
            xtal.positions[self.cell_mol_atoms[mol2]][:, :, np.newaxis]
            + (lattice.T @ pairs.T)[np.newaxis, :, :]
        )
        pos2 = pos2.transpose(2, 0, 1).reshape(-1, 3)
        dist = cdist(pos1, pos2).T.reshape(-1)
        # dist runs over atoms of mol2 first, so the cutoff block is transposed
        radius = self.radius[
            self.asu_mol_atoms[mol1 % self.n_asu_mols],
            self.asu_mol_atoms[mol2 % self.n_asu_mols],
        ]
        stretched_radius = np.resize(radius.T.ravel(), dist.shape)
        return self.interaction_kernel(
            dist, stretched_radius, weight, self.D_cell[mol1, mol2]
        )

    def total_energy(self, state: np.ndarray) -> float:
        """
        Calculate the total energy of the crystal structure.

        The total energy is the sum of:
        1. Interaction energy between molecules
        2. Unit cell volume

        Args:
            state: State vector containing lattice parameters
            and molecular position/orientation

        Returns:
            Total energy (sum of volume and interaction energy)
        """
        # Convert state vector to a standardized form
        state_copy = np.array(state, dtype=float)
        state_std = self.standardize_state(state_copy)

        # Create crystal structure from state vector
        xtal = self.create_xtal(state_std)

        # Find interacting molecule pairs
        interact_pairs = self.find_pairs(xtal)
        if "inf" in interact_pairs:
            return np.inf

        # Calculate interaction energy
        energy = 0.0
        w_pair = self.int_scale / (self.n_asu_atoms * self.n_asu_atoms)

        for key in interact_pairs:
            mol1, mol2 = key
            energy += self.pair_energy(xtal, mol1, mol2, w_pair, interact_pairs)

        return energy + xtal.get_volume()

    def generate_pos(self, asu_pos: np.ndarray, lattice: np.ndarray) -> np.ndarray:
        """
        Generate full crystal coordinates from asymmetric unit and symmetry operations

        Args:
            asu_pos: Cartesian positions of the asymmetric unit
            lattice: Lattice vectors as rows

        Returns:
            Fractional coordinates of the whole cell
        """

        rot, trans = self.asu_rot, self.asu_trans
        pos = np.zeros((len(asu_pos) * len(rot), 3))
        asu_frac = np.linalg.solve(lattice.T, asu_pos.T).T
        cog_frac = np.mean(asu_frac, axis=0)

        for i, (r, t) in enumerate(zip(rot, trans)):
            # Find the shift of cog of mol from the central cell
            new_cog = (r @ cog_frac.T).T + t
            shift = np.divmod(new_cog, 1)[1] - new_cog

            unshifted_pos = (r @ asu_frac.T).T + t
            pos[i * self.n_asu_atoms : (i + 1) * self.n_asu_atoms, :] = (
                unshifted_pos + shift
            )  # Move inside central cell

        return pos

    def create_xtal(self, state: np.ndarray) -> Atoms:
        """
        Create an ASE Atoms object from the state vector

        Args:
            state: State vector

        Returns:
            ASE Atoms object
        """

        lattice_type = get_lattice_type(self.spg)

        if lattice_type == "triclinic":
            lattice = [
                [state[0], 0, 0],
                [state[1], state[2], 0],
                [state[3], state[4], state[5]],
            ]

        elif lattice_type == "monoclinic":
            lattice = [[state[0], 0, 0], [0, state[1], 0], [state[2], 0, state[3]]]

        elif lattice_type == "orthorhombic":
            lattice = [[state[0], 0, 0], [0, state[1], 0], [0, 0, state[2]]]

        elif lattice_type == "tetragonal":
            lattice = [[state[0], 0, 0], [0, state[0], 0], [0, 0, state[1]]]

        elif lattice_type in ["hexagonal", "trigonal"]:
            gamma = 2 * np.pi / 3  # in radians
            lattice = [
                [state[0], 0, 0],
                [state[0] * np.cos(gamma), state[0] * np.sin(gamma), 0],
                [0, 0, state[1]],
            ]

        elif lattice_type == "cubic":
            lattice = [[state[0], 0, 0], [0, state[0], 0], [0, 0, state[0]]]

        else:
            raise Exception("Invalid spg")

        lattice = np.array(lattice)
        state = np.asarray(state, dtype=float)
        n = self.n_asu_mols
        params = np.split(
            state[len(state) - sum(self.dofs) :], np.cumsum(self.dofs)[:-1]
        )

        # create asymm unit from mol positions and angles
        asu_pos = np.concatenate(
            [
                self._mol_positions(m, cog, angles, lattice)
                for m, (cog, angles) in enumerate(zip(params[:n], params[n:]))
            ]
        )

        # generate full crystal using symmetry
        pos = self.generate_pos(asu_pos, lattice)
        xtal = Atoms(cell=lattice, scaled_positions=pos, pbc=True)
        return xtal

    def _mol_positions(
        self, m: int, cog: np.ndarray, angles: np.ndarray, lattice: np.ndarray
    ) -> np.ndarray:
        """
        Place one molecule of the asymmetric unit from its state parameters.

        Args:
            m: Index of the molecule in the asymmetric unit
            cog: Cartesian centre, or free fractional coordinates on the site
            angles: Euler angles, or the angle about the free axis of the site
            lattice: Lattice vectors as rows

        Returns:
            Cartesian positions of the molecule
        """
        site = self.sites[m]
        if site is None:
            ref = self.ref_pos[self.mol_types[m]]
            return Rotation.from_euler("ZYX", angles).apply(ref) + cog
        cog = (site.origin + site.directions @ cog) @ lattice
        return Rotation.from_rotvec(site.axes @ angles).apply(site.ref_pos) + cog

    def create_state(self, xtal: Atoms) -> np.ndarray:
        """
        Create an optimization state vector from ASE Atoms object.
        State vector is the lattice vector + position of asymm molecule
        + orientation of asymmetric molecule

        Args:
            xtal: ASE Atoms object

        Returns:
            State vector
        """

        lattice = xtal.cell.array
        asu_cogs, asu_angles = [], []
        for t, mol, site in zip(self.mol_types, self.asu_mol_atoms, self.sites):
            pos = xtal.positions[mol]
            cog = np.mean(pos, axis=0)
            if site is None:
                # Compute the orientation of each molecule using Kabsch method
                R, rmsd = Rotation.align_vectors(pos - cog, self.ref_pos[t])
                if (rmsd) > 0.1:
                    raise Exception("Can not create rotation")
                asu_cogs.append(cog)
                asu_angles.append(R.as_euler("ZYX"))
                continue
            cog_frac = np.linalg.solve(lattice.T, cog)
            asu_cogs.append(site.directions.T @ (cog_frac - site.origin))
            R, _ = Rotation.align_vectors(pos - cog, site.ref_pos)
            asu_angles.append(R.as_rotvec() @ site.axes)

        # Create state vector with independent lattice params
        lattice_type = get_lattice_type(self.spg)
        # All six lattice params can vary
        if lattice_type == "triclinic":
            state = [
                lattice[0][0],
                lattice[1][0],
                lattice[1][1],
                lattice[2][0],
                lattice[2][1],
                lattice[2][2],
            ]

        elif lattice_type == "monoclinic":
            state = [lattice[0][0], lattice[1][1], lattice[2][0], lattice[2][2]]

        elif lattice_type == "orthorhombic":
            state = [lattice[0][0], lattice[1][1], lattice[2][2]]

        elif lattice_type in ["tetragonal", "hexagonal", "trigonal"]:
            state = [lattice[0][0], lattice[2][2]]

        elif lattice_type == "cubic":
            state = [lattice[0][0]]

        else:
            raise Exception("Invalid spg")

        state += np.concatenate(asu_cogs).tolist()
        state += np.concatenate(asu_angles).tolist()

        return state

    def standardize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Standardize the state vector

        Args:
            state: State vector

        Returns:
            Standardized state vector
        """
        lattice_type = get_lattice_type(self.spg)
        if lattice_type == "triclinic":
            q = int(state[4] / state[2])
            state[3] = state[3] - q * state[1]
            state[4] = state[4] - q * state[2]

            q = int(state[1] / state[0])
            state[1] = state[1] - q * state[0]

            q = int(state[3] / state[0])
            state[3] = state[3] - q * state[0]

        if lattice_type == "monoclinic":
            q = int(state[2] / state[0])
            if abs(q) > 2:
                state[2] = state[2] - q * state[0]

        return state

    def callback(self, xk: np.ndarray) -> None:
        """
        Callback function for the optimizer

        Args:
            xk: Current state vector
        """
        # Avoid volume converge to small values
        if self.objective_function(xk) < self.vol_tol:
            if self.debug_flag:
                self.logger.debug(f"Failed optimization in {self.rank}")
            raise StopIteration("Custom stopping condition met")
        return self.standardize_state(xk)

    def objective_function(self, state: np.ndarray) -> float:
        """
        Function to be minimized.

        Args:
            state: State vector

        Returns:
            Total energy
        """

        return self.total_energy(state)

    def run(self, xtal: Atoms) -> bool:
        """
        Minimize the energy using Scipy local optimizer.

        Args:
            xtal: Initial crystal structure

        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Create initial state vector
            state = self.create_state(xtal)
            state_std = self.standardize_state(state)

            # Check if initial state is valid
            energy = self.total_energy(state_std)
            if np.isinf(energy):
                return False

            # Run optimization
            res = minimize(
                self.objective_function,
                state,
                method=self.method,
                tol=self.tol,
                options={"disp": self.debug_flag, "maxiter": self.maxiter},
                callback=self.callback,
            )

            # Update structure with optimized parameters
            xtal_opt = self.create_xtal(res.x)
            xtal.set_cell(xtal_opt.cell)
            xtal.set_positions(xtal_opt.positions)
            return res.success
        except (StopIteration, RuntimeError):
            return False
