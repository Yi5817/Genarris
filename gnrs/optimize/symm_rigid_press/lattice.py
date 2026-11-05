"""
Lattice parametrization of the crystal families.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom", "Jonathan Moussa"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import numpy as np

FAMILIES = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)


def get_lattice_type(spg: int) -> str:
    """
    Crystal family of a space group number ("invalid" outside 1-230).
    """
    if not 0 < spg <= 230:
        return "invalid"
    return FAMILIES[np.searchsorted((2, 15, 74, 142, 167, 194), spg)]


def basis_matrix(*entries: tuple[int, int, float]) -> np.ndarray:
    """
    Build one lattice basis matrix from (row, column, value) entries.
    """
    matrix = np.zeros((3, 3), dtype=float)
    for i, j, value in entries:
        matrix[i, j] = value
    return matrix


HEXAGONAL_GAMMA = 2 * np.pi / 3
# Lattice = sum_k params_k * basis_k for a lower-triangular cell of each family
LATTICE_BASIS = {
    "triclinic": [
        basis_matrix((0, 0, 1)),
        basis_matrix((1, 0, 1)),
        basis_matrix((1, 1, 1)),
        basis_matrix((2, 0, 1)),
        basis_matrix((2, 1, 1)),
        basis_matrix((2, 2, 1)),
    ],
    "monoclinic": [
        basis_matrix((0, 0, 1)),
        basis_matrix((1, 1, 1)),
        basis_matrix((2, 0, 1)),
        basis_matrix((2, 2, 1)),
    ],
    "orthorhombic": [
        basis_matrix((0, 0, 1)),
        basis_matrix((1, 1, 1)),
        basis_matrix((2, 2, 1)),
    ],
    "tetragonal": [basis_matrix((0, 0, 1), (1, 1, 1)), basis_matrix((2, 2, 1))],
    "trigonal": [
        basis_matrix(
            (0, 0, 1), (1, 0, np.cos(HEXAGONAL_GAMMA)), (1, 1, np.sin(HEXAGONAL_GAMMA))
        ),
        basis_matrix((2, 2, 1)),
    ],
    "cubic": [basis_matrix((0, 0, 1), (1, 1, 1), (2, 2, 1))],
}
LATTICE_BASIS["hexagonal"] = LATTICE_BASIS["trigonal"]
LATTICE_BASIS = {k: np.array(v) for k, v in LATTICE_BASIS.items()}


def lattice_from_params(basis: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Lattice vectors (rows) from the free lattice parameters of a family.
    """
    return np.einsum("kij,k->ij", basis, params)


def params_from_lattice(basis: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Free lattice parameters of a lattice (projection on the family basis).
    """
    return np.einsum("kij,ij->k", basis, lattice) / np.einsum(
        "kij,kij->k", basis, basis
    )
