"""
Utility functions for Genarris.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os

import numpy as np
from ase import Atoms
from ase.io import read

import gnrs.output as gout
from gnrs.parallel.io import read_parallel
from gnrs.parallel.structs import DistributedStructs


def eV2kJ(e: float) -> float:
    """
    Convert energy from eV to kJ/mol.
    """
    from ase.units import eV, kJ, mol

    return e * eV / kJ * mol


def check_no_changes_in_covalent_matrix(
    initial_atoms: Atoms, final_atoms: Atoms
) -> bool:
    """
    Check that a relaxation neither broke nor formed covalent bonds.

    Args:
        initial_atoms: Crystal before the relaxation.
        final_atoms: Crystal after the relaxation.

    Returns:
        True if every atom has the same JmolNN neighbors in both crystals.
    """
    from pymatgen.analysis.local_env import JmolNN
    from pymatgen.io.ase import AseAtomsAdaptor

    if not initial_atoms.pbc.any():
        # for non-periodic
        initial_atoms, final_atoms = initial_atoms.copy(), final_atoms.copy()
        initial_atoms.center(vacuum=10.0)
        final_atoms.center(vacuum=10.0)

    # Detect reconstructions by comparing the covalent bonded network
    # before and after relaxation. If any bonds are broken or formed,
    # it's probably a reconstruction of some sort or related to bad inputs
    # like overlapping atoms

    # Grab all the NN info for the first structure, then fill in the
    # adjacency matrix
    initial_structure = AseAtomsAdaptor.get_structure(initial_atoms)
    final_structure = AseAtomsAdaptor.get_structure(final_atoms)

    initial_nn_info = JmolNN().get_all_nn_info(initial_structure)
    initial_nn_matrix = np.zeros((len(initial_nn_info), len(initial_nn_info)))
    for i in range(len(initial_nn_info)):
        for j in range(len(initial_nn_info[i])):
            initial_nn_matrix[i, initial_nn_info[i][j]["site_index"]] = 1

    # Grab all the NN info for the final output structure, then fill in the
    # adjacency matrix
    final_nn_info = JmolNN().get_all_nn_info(final_structure)
    final_nn_matrix = np.zeros((len(final_nn_info), len(final_nn_info)))
    for i in range(len(final_nn_info)):
        for j in range(len(final_nn_info[i])):
            final_nn_matrix[i, final_nn_info[i][j]["site_index"]] = 1

    # Check that both matrices are the same
    return (initial_nn_matrix == final_nn_matrix).all()


def check_if_exp_found(config: dict, gnrs_info: dict):
    """
    Check if experimental structure is found within the generated pool.

    Args:
        config: Configuration dictionary
        gnrs_info: Dictionary containing information about the Genarris run
    """
    if "experimental_structure" not in config:
        gout.emit("Passing experimental structure check...")
        return

    exp_path = config["experimental_structure"].get("path", None)
    if exp_path is None:
        raise ValueError("Experimental structure path not found in config")
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"Experimental structure file not found: {exp_path}")
    exp = read(exp_path, parallel=False)
    gout.emit("Searching for Experimental structure within the pool...")
    structs = read_parallel(gnrs_info["last_struct_path"])
    match_list = DistributedStructs(structs).find_matches(
        exp, settings=config["experimental_structure"].get("settings", None)
    )
    if match_list:
        gout.emit("Found Experimental structure within the pool.")
    else:
        gout.emit("Experimental structure not found within the pool.")
