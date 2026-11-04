"""
Regularized contact energy of rigid molecules in a periodic cell.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom", "Jonathan Moussa"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

from itertools import combinations_with_replacement

import numpy as np

# Maximum number of unit cells to sum over before deciding a crystal is too tightly packed
MAX_CELL_SUM = 2000
# Below r + WALL_BAND the kernel continues linearly, so the energy stays finite
WALL_BAND = 1e-3
# Fractional tolerance when deciding whether a molecule centre is inside the cell
CENTRE_TOL = 1e-12


def pair_energy(
    pos1: np.ndarray,
    pos2: np.ndarray,
    radius: np.ndarray,
    D: float,
    weight: float,
    shifts: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Contact energy weight * (D - d) / (d - r) between a molecule and the images
    of another, with the kernel continued linearly below d = r + WALL_BAND.

    Args:
        pos1: Cartesian positions of the first molecule, (n1, 3)
        pos2: Cartesian positions of the second molecule, (n2, 3)
        radius: Cutoff radii of the atom pairs, (n1, n2)
        D: Interaction distance of the molecule pair
        weight: Scaling factor for interaction energy
        shifts: Cartesian lattice displacements of the second molecule, (k, 3)

    Returns:
        Energy, gradient on the first molecule (n1, 3), gradient on the second
        molecule (n2, 3), gradient summed per image (k, 3), smallest d - r
    """
    diff = pos1[:, None, None] - pos2[None, None] - shifts[None, :, None]
    dist = np.linalg.norm(diff, axis=-1)
    radius = radius[:, None]
    gap = dist - radius
    active = dist < D
    band = gap < WALL_BAND
    gap_safe = np.maximum(gap, WALL_BAND)
    deriv = -weight * (D - radius) / gap_safe**2 * active
    energy = np.where(
        band,
        weight * (D - radius - WALL_BAND) / WALL_BAND + deriv * (gap - WALL_BAND),
        weight * (D - dist) / gap_safe,
    )
    energy *= active
    coeff = deriv / dist
    grad1 = np.einsum("ikb,ikbx->ix", coeff, diff)
    grad2 = -np.einsum("ikb,ikbx->bx", coeff, diff)
    grad_img = -np.einsum("ikb,ikbx->kx", coeff, diff)
    return energy.sum(), grad1, grad2, grad_img, gap.min()


class ContactEnergy:
    """
    Contact energy of all molecule pairs of a cell and their periodic images
    whose centres are closer than the pair's interaction distance.
    """

    def __init__(
        self,
        mol_atoms: list[slice],
        asu_atoms: list[slice],
        radius: np.ndarray,
        D_cell: np.ndarray,
        weight: float,
    ) -> None:
        """
        Args:
            mol_atoms: Atom range of every molecule of the cell
            asu_atoms: Atom range of every molecule of the cell in the asymmetric unit
            radius: Cutoff radii of the atom pairs of the asymmetric unit
            D_cell: Interaction distance of every pair of molecules
            weight: Scaling factor for interaction energy
        """
        self.mol_atoms = list(mol_atoms)
        self.asu_atoms = list(asu_atoms)
        self.radius = radius
        self.D_cell = D_cell
        self.weight = weight

    def __call__(
        self, lattice: np.ndarray, frac: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, float]:
        """
        Energy (infinite if the cell is too small to sum over), gradient on the
        Cartesian positions, gradient on the lattice at fixed fractional
        positions and the smallest d - r (negative: overlap) of a cell.
        """
        inv_lattice = np.linalg.inv(lattice)
        # Every molecule inside the cell, so that the image bound below is complete
        # (a centre within CENTRE_TOL of the cell is not moved)
        frac = frac.copy()
        for mol in self.mol_atoms:
            cog = frac[mol].mean(axis=0)
            inside = (cog > -CENTRE_TOL) & (cog < 1 + CENTRE_TOL)
            frac[mol] -= np.where(inside, 0.0, np.floor(cog))
        pos = frac @ lattice
        cog = np.array([pos[mol].mean(axis=0) for mol in self.mol_atoms])

        # Lattice displacements reaching every image within range
        bound = np.linalg.norm(inv_lattice, axis=0) * self.D_cell.max() + 1
        if not np.all(np.isfinite(bound)) or np.prod(bound) > MAX_CELL_SUM:
            return np.inf, np.zeros_like(pos), np.zeros((3, 3)), -np.inf
        max_lat = np.floor(bound).astype(int)
        shifts_frac = np.stack(
            np.meshgrid(*(np.arange(-m, m + 1) for m in max_lat), indexing="ij"), -1
        ).reshape(-1, 3)
        shifts = shifts_frac @ lattice
        nonzero = np.any(shifts_frac != 0, axis=1)

        energy = 0.0
        min_gap = np.inf
        grad_pos = np.zeros_like(pos)
        grad_lat = np.zeros((3, 3))
        for mol1, mol2 in combinations_with_replacement(range(len(self.mol_atoms)), 2):
            dist = np.linalg.norm(cog[mol1] - cog[mol2] - shifts, axis=1)
            select = dist < self.D_cell[mol1, mol2]
            if mol1 == mol2:
                # +L and -L are the same image pair: count each once
                select &= nonzero
                weight = 0.5 * self.weight
            else:
                weight = self.weight
            if not select.any():
                continue
            atoms1, atoms2 = self.mol_atoms[mol1], self.mol_atoms[mol2]
            e, grad1, grad2, grad_img, gap = pair_energy(
                pos[atoms1],
                pos[atoms2],
                self.radius[self.asu_atoms[mol1], self.asu_atoms[mol2]],
                self.D_cell[mol1, mol2],
                weight,
                shifts[select],
            )
            energy += e
            min_gap = min(min_gap, gap)
            grad_pos[atoms1] += grad1
            grad_pos[atoms2] += grad2
            grad_lat += shifts_frac[select].T @ grad_img
        grad_lat += frac.T @ grad_pos
        return energy, grad_pos, grad_lat, min_gap
