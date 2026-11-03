"""
Molecule helpers of the symmetry-preserving rigid press.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom", "Jonathan Moussa"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import numpy as np
from ase import Atoms
from scipy.spatial.distance import cdist


def standardize_mol(mol: Atoms) -> Atoms:
    """
    Standardize a molecule by moving its center of geometry to origin
    and aligning with principal axes of rotation.

    Args:
        mol: The molecule to standardize

    Returns:
        Standardized molecule
    """
    an = mol.get_atomic_numbers()
    mol.set_atomic_numbers([1 for _ in range(len(mol))])
    com = mol.get_center_of_mass()
    pos = mol.get_positions()
    pos -= com
    mol.set_positions(pos)
    mol.set_atomic_numbers(an)
    return mol


def get_mol_len(ref_mol: Atoms) -> float:
    """
    Calculate the diameter of the smallest sphere that can enclose a molecule.

    Args:
        ref_mol: The reference molecule

    Returns:
        Diameter of the enclosing sphere
    """
    pos = ref_mol.positions
    cog = np.mean(pos, axis=0)
    dist = cdist(pos, [cog])
    return 2 * dist.max()
