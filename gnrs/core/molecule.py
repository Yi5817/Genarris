"""
This module provides functions for handling molecules.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

class Molecule(Atoms):
    """
    A class inherited from ase for hanlding molecules.
    probably will NOT be used much.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the molecule.

        Args:
            *args: Arguments
            **kwargs: Keyword arguments
        """
        super().__init__(*args, **kwargs)

    @classmethod
    def read(cls, file_path: str, *args, **kwargs) -> Molecule:
        """
        Read a molecule from a file.

        Args:
            file_path: File path
            *args: Arguments
            **kwargs: Keyword arguments
        """
        ase_atoms = ase_read(file_path, *args, **kwargs)
        return cls(ase_atoms)

    def standardize_orientation(self) -> None:
        """
        Orient molecule s.t. pricipal axes align with cartesian basis.
        """
        com = self.get_center_of_mass()
        self.translate(-1 * com)
        eig_val, eig_vec = self.get_moments_of_inertia(vectors=True)
        pos = self.get_positions()
        pos = np.dot(eig_vec, pos.transpose())
        self.set_positions(pos.transpose())
