"""
Symmetry helpers of the symmetry-preserving rigid press.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom", "Jonathan Moussa"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import numpy as np
from ase import Atoms
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


def find_symm(xtal: Atoms, natoms: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Find  unique symmetry operations

    Args:
        xtal: Crystal structure
        natoms: Number of atoms in the molecule

    Returns:
        Tuple of unique rotation matrices, translation vectors, and space group number
    """
    symm = get_symmetry_dataset(
        (xtal.get_cell(), xtal.get_scaled_positions(), xtal.get_atomic_numbers()),
        symprec=1e-3,
    )
    if symm is None:
        # spglib failed to determine the symmetry of this structure
        raise ValueError("spglib get_symmetry_dataset returned None for this crystal ")
    spg = symm.number
    rot, trans = symm.rotations, symm.translations

    # Check if the operations are unique
    uniq_rot, uniq_trans = [], []
    pos = xtal.get_scaled_positions(wrap=False)
    asym_frac = pos[:natoms]
    cog_frac = np.mean(asym_frac, axis=0)
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

    if natoms * len(uniq_rot) != len(xtal):
        raise RuntimeError("Unable to find the unique symmetry operations")

    return np.array(uniq_rot), np.array(uniq_trans), spg


def get_lattice_type(spg: int) -> str:
    """
    Get cell type from spg

    Args:
        spg: Space group number

    Returns:
        Cell type
    """
    if spg <= 0 or spg >= 230:
        return "invalid"
    if spg <= 2:
        return "triclinic"
    elif spg <= 15:
        return "monoclinic"
    elif spg <= 74:
        return "orthorhombic"
    elif spg <= 142:
        return "tetragonal"
    elif spg <= 167:
        return "trigonal"
    elif spg <= 194:
        return "hexagonal"
    else:
        return "cubic"
