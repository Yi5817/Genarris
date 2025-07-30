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
from ase.io import read
import gnrs.output as gout
from gnrs.parallel.io import read_parallel
from gnrs.parallel.structs import DistributedStructs

def eV2kJ(e: float) -> float:
    """
    Convert energy from eV to kJ/mol.
    """
    from ase.units import kJ, mol, eV
    return e * eV / kJ * mol

def check_if_exp_found(config: dict, gnrs_info: dict):
    """
    Check if experimental structure is found within the generated pool.

    Args:
        config: Configuration dictionary
        gnrs_info: Dictionary containing information about the Genarris run
    """
    exp_path = config["experimental_structure"].get("path", None)
    if exp_path is None:
        raise ValueError("Experimental structure path not found in config")
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"Experimental structure file not found: {exp_path}")
    exp = read(exp_path, parallel=False)
    gout.emit("Searching for Experimental structure within the pool...")
    structs = read_parallel(gnrs_info["last_struct_path"])
    match_list = DistributedStructs(structs).find_matches(exp, settings=config["experimental_structure"].get("settings", None))
    if match_list:
        gout.emit("Found Experimental structure within the pool.")
    else:
        gout.emit("Experimental structure not found within the pool.")