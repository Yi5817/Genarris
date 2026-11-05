"""
Symmetry helpers of the symmetry-preserving rigid press.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom", "Jonathan Moussa"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

from typing import NamedTuple

import numpy as np
from ase import Atoms
from scipy.linalg import null_space
from scipy.spatial.distance import cdist
from spglib import get_symmetry_dataset


def is_duplicate(cog1: np.ndarray, cog2: np.ndarray, tol: float = 0.001) -> bool:
    """
    Check if the difference between two centers of geometry is close to an integer.

    Args:
        cog1: First center of geometry
        cog2: Second center of geometry
        tol: Tolerance for comparison

    Returns:
        True if the difference is close to an integer, False otherwise
    """
    intval = np.round(cog1 - cog2)
    is_dup = np.allclose(intval, cog1 - cog2, atol=tol, rtol=0)
    return is_dup


def find_symm(
    xtal: Atoms, n_asu_atoms: int, symprec: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, int, tuple[np.ndarray, np.ndarray]]:
    """
    Find the operations that generate the Z asymmetric units of the cell.

    Args:
        xtal: Crystal structure
        n_asu_atoms: Number of atoms in the asymmetric unit
        symprec: spglib symmetry tolerance in Angstrom

    Returns:
        Z rotation matrices and translation vectors (one per asymmetric unit),
        the space group number, and all rotation matrices and translation
        vectors of the space group
    """
    symm = get_symmetry_dataset(
        (xtal.get_cell(), xtal.get_scaled_positions(), xtal.get_atomic_numbers()),
        symprec=symprec,
    )
    if symm is None:
        # spglib failed to determine the symmetry of this structure
        raise ValueError("spglib get_symmetry_dataset returned None for this crystal ")
    spg = symm.number
    rot, trans = symm.rotations, symm.translations

    # Check if the operations are unique
    uniq_rot, uniq_trans = [], []
    pos = xtal.get_scaled_positions(wrap=False)
    asu_frac = pos[:n_asu_atoms]
    cog_frac = np.mean(asu_frac, axis=0)
    uniq_cog = []

    for i, (r, t) in enumerate(zip(rot, trans)):
        # Find the shift of cog of mol to bring it back to central cell
        new_cog = (r @ cog_frac.T).T + t
        new_cog = np.divmod(new_cog, 1)[1]

        # Check if the pos is unique
        unique = True
        for cog in uniq_cog:
            if is_duplicate(cog, new_cog):
                unique = False
                continue

        if unique:
            uniq_cog.append(new_cog)
            uniq_rot.append(r)
            uniq_trans.append(t)

    if n_asu_atoms * len(uniq_rot) != len(xtal):
        raise RuntimeError("Unable to find the unique symmetry operations")

    return np.array(uniq_rot), np.array(uniq_trans), spg, (rot, trans)


class Site(NamedTuple):
    """
    Freedom of a rigid molecule on a special Wyckoff position.

    Attributes:
        origin: Fractional point of the site closest to the molecule centre
        directions: Orthonormal free fractional directions of the centre, (3, k)
        axes: Orthonormal rotation axes that keep the site symmetry, (3, dof)
        ref_pos: Site-symmetrised positions of the molecule, centred on the site
    """

    origin: np.ndarray
    directions: np.ndarray
    axes: np.ndarray
    ref_pos: np.ndarray


def find_site(
    rot: np.ndarray,
    trans: np.ndarray,
    pos: np.ndarray,
    numbers: np.ndarray,
    lattice: np.ndarray,
    tol: float = 1e-3,
) -> Site:
    """
    Find the Wyckoff site of a molecule from the space group operations.

    Args:
        rot: Rotation matrices of the space group, shape (n, 3, 3)
        trans: Translation vectors of the space group, shape (n, 3)
        pos: Cartesian positions of the molecule
        numbers: Atomic numbers of the molecule
        lattice: Lattice vectors as rows
        tol: Tolerance on the fractional image of the molecule centre (fractional)

    Returns:
        The Site of the molecule; on a general position every direction and
        axis is free and the reference is the molecule itself

    Raises:
        RuntimeError: If a site operation does not map the molecule onto itself
    """
    cog = pos.mean(axis=0)
    point = np.linalg.solve(lattice.T, cog)
    # Operations fixing the centre; translations shifted so it is fixed exactly
    image = rot @ point + trans
    shift = np.round(image - point)
    keep = np.all(np.abs(image - shift - point) < tol, axis=1)
    if keep.sum() == 1:
        return Site(point, np.eye(3), np.eye(3), pos - cog)
    rot, trans = rot[keep], (trans - shift)[keep]
    # Fixed-point set origin + directions @ u of the site group
    a = np.concatenate(rot - np.eye(3))
    origin = point - np.linalg.pinv(a) @ (a @ point + np.concatenate(trans))
    directions = null_space(a)
    # Rotation axes commuting with every Cartesian site operation
    rot_c = lattice.T @ rot @ np.linalg.inv(lattice).T
    det = np.linalg.det(rot_c)
    axes = null_space(np.concatenate(det[:, None, None] * rot_c - np.eye(3)))
    # Average the molecule over the site group so that it is exactly invariant
    centred = pos - cog
    ref_pos = np.zeros_like(pos)
    for r in rot_c:
        image = centred @ r.T
        dist = cdist(image, centred)
        dist[numbers[:, None] != numbers[None, :]] = np.inf
        perm = dist.argmin(axis=1)
        if len(set(perm)) < len(pos) or dist[np.arange(len(pos)), perm].max() > 0.1:
            raise RuntimeError("A site operation does not map the molecule onto itself")
        ref_pos[perm] += image
    return Site(origin, directions, axes, ref_pos / len(rot_c))
