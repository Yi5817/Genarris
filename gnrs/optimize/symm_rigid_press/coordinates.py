"""
Rigid-body coordinates of a crystal whose molecules sit on Wyckoff sites.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom", "Jonathan Moussa"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation

from gnrs.optimize.symm_rigid_press.lattice import (
    LATTICE_BASIS,
    get_lattice_type,
    lattice_from_params,
    params_from_lattice,
)
from gnrs.optimize.symm_rigid_press.symmetry import find_site, find_symm


def rotation_jacobian(rotvec: np.ndarray) -> np.ndarray:
    """
    Left Jacobian of SO(3): exp([r + dr]) = exp([J dr]) exp([r]) to first order.
    """
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return np.eye(3)
    x, y, z = rotvec
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.eye(3)
        + (1 - np.cos(angle)) / angle**2 * skew
        + (angle - np.sin(angle)) / angle**3 * skew @ skew
    )


class RigidCoordinates:
    """
    Map between a state vector and a crystal of rigid molecules.

    State = free lattice parameters of the crystal family, then for every
    molecule of the asymmetric unit its centre along the free directions of
    its Wyckoff site (Angstrom), then its rotation vector along the site axes.
    """

    def __init__(
        self, xtal: Atoms, asu_mol_atoms: list[slice], symprec: float = 1e-3
    ) -> None:
        """
        Args:
            xtal: Crystal structure, rotated in place to a lower-triangular cell
            asu_mol_atoms: Atom range of every molecule of the asymmetric unit
            symprec: spglib symmetry tolerance in Angstrom

        Raises:
            ValueError: If the cell is not in the standard setting of the space group
        """
        cell, rotation = xtal.cell.standard_form()
        xtal.set_cell(cell)
        xtal.set_positions(xtal.positions @ rotation.T)
        lattice = xtal.cell.array

        self.asu_mol_atoms = list(asu_mol_atoms)
        self.n_asu_atoms = self.asu_mol_atoms[-1].stop
        self.asu_rot, self.asu_trans, self.spg, all_ops = find_symm(
            xtal, self.n_asu_atoms, symprec
        )
        self.z = len(self.asu_rot)
        self.basis = LATTICE_BASIS[get_lattice_type(self.spg)]
        if not np.allclose(
            lattice_from_params(self.basis, params_from_lattice(self.basis, lattice)),
            lattice,
        ):
            raise ValueError(
                f"The cell is not in the standard setting of space group {self.spg}"
            )

        self.sites = [
            find_site(*all_ops, xtal.positions[mol], xtal.numbers[mol], lattice)
            for mol in self.asu_mol_atoms
        ]
        self.dofs = [s.directions.shape[1] for s in self.sites] + [
            s.axes.shape[1] for s in self.sites
        ]
        # Centre parameters are in Angstrom
        self.cog_scale = [
            np.linalg.norm(s.directions.T @ lattice, axis=1) for s in self.sites
        ]
        self.cell_mol_atoms = [
            slice(i * self.n_asu_atoms + mol.start, i * self.n_asu_atoms + mol.stop)
            for i in range(self.z)
            for mol in self.asu_mol_atoms
        ]

    def unpack(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """
        Lattice, fractional centres and rotation vectors of a state.
        """
        state = np.asarray(state, dtype=float)
        n_lat = len(self.basis)
        params = np.split(state[n_lat:], np.cumsum(self.dofs)[:-1])
        n = len(self.sites)
        cogs = [
            s.origin + s.directions @ (u / scale)
            for s, u, scale in zip(self.sites, params[:n], self.cog_scale)
        ]
        rotvecs = [s.axes @ angles for s, angles in zip(self.sites, params[n:])]
        return lattice_from_params(self.basis, state[:n_lat]), cogs, rotvecs

    def offsets(self, rotvecs: list[np.ndarray]) -> np.ndarray:
        """
        Cartesian offsets of the asymmetric unit atoms from their molecule centres.
        """
        return np.concatenate(
            [
                Rotation.from_rotvec(rv).apply(site.ref_pos)
                for site, rv in zip(self.sites, rotvecs)
            ]
        )

    def cell_positions(self, asu_frac: np.ndarray) -> np.ndarray:
        """
        Fractional coordinates of the cell, every asymmetric unit centred in it.
        """
        cog = asu_frac.mean(axis=0)
        pos = np.einsum("zij,aj->zai", self.asu_rot, asu_frac) + self.asu_trans[:, None]
        new_cog = self.asu_rot @ cog + self.asu_trans
        shift = np.divmod(new_cog, 1)[1] - new_cog
        return (pos + shift[:, None]).reshape(-1, 3)

    def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Lattice and fractional positions of the cell.
        """
        lattice, cogs, rotvecs = self.unpack(state)
        offsets = self.offsets(rotvecs)
        sizes = [mol.stop - mol.start for mol in self.asu_mol_atoms]
        asu_frac = offsets @ np.linalg.inv(lattice) + np.repeat(cogs, sizes, axis=0)
        return lattice, self.cell_positions(asu_frac)

    def to_crystal(self, state: np.ndarray) -> Atoms:
        """
        Crystal of a state (cell and positions only).
        """
        lattice, frac = self.positions(state)
        return Atoms(cell=lattice, scaled_positions=frac, pbc=True)

    def to_state(self, xtal: Atoms) -> np.ndarray:
        """
        State of a crystal.
        """
        lattice = xtal.cell.array
        cogs, angles = [], []
        for mol, site, scale in zip(self.asu_mol_atoms, self.sites, self.cog_scale):
            pos = xtal.positions[mol]
            cog = pos.mean(axis=0)
            cog_frac = np.linalg.solve(lattice.T, cog)
            cogs.append(scale * (site.directions.T @ (cog_frac - site.origin)))
            rot, _ = Rotation.align_vectors(pos - cog, site.ref_pos)
            angles.append(rot.as_rotvec() @ site.axes)
        return np.concatenate(
            [params_from_lattice(self.basis, lattice), *cogs, *angles]
        )

    def gradient(
        self, state: np.ndarray, grad_pos: np.ndarray, grad_lat: np.ndarray
    ) -> np.ndarray:
        """
        Gradient on the state from the gradient on the Cartesian positions of
        the cell and on the lattice at fixed fractional positions.
        """
        lattice, _, rotvecs = self.unpack(state)
        inv_lattice = np.linalg.inv(lattice)
        offsets = self.offsets(rotvecs)
        # Pull back to the asymmetric unit through the Cartesian operations
        rot_cart = lattice.T @ self.asu_rot @ inv_lattice.T
        grad_asu = np.einsum(
            "zax,zxy->ay", grad_pos.reshape(self.z, self.n_asu_atoms, 3), rot_cart
        )
        # Atoms keep their Cartesian, not fractional, offset from the centre
        grad_lat = grad_lat - (offsets @ inv_lattice).T @ grad_asu
        grad_cog, grad_rot = [], []
        for site, mol, rv, scale in zip(
            self.sites, self.asu_mol_atoms, rotvecs, self.cog_scale
        ):
            force = grad_asu[mol]
            grad_cog.append(force.sum(axis=0) @ lattice.T @ site.directions / scale)
            torque = np.cross(offsets[mol], force).sum(axis=0)
            grad_rot.append(torque @ rotation_jacobian(rv) @ site.axes)
        return np.concatenate(
            [np.einsum("kij,ij->k", self.basis, grad_lat), *grad_cog, *grad_rot]
        )
