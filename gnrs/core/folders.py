"""
This module provides functions for managing folder.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import logging

from gnrs.core.molecule import Molecule


is_master = False
logger = logging.getLogger("folders")


def init_folders(is_master_in: bool) -> None:
    """
    Initialize folder management.
    """
    global is_master
    is_master = is_master_in


def mkdir(dir_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    """
    if is_master:
        os.makedirs(dir_path, exist_ok=True)


def rmdir(dir_path: str) -> None:
    """
    Remove a directory if it exists.
    """
    if is_master and os.path.exists(dir_path):
        os.remove(dir_path)


def setup_main_folders(gnrs_info: dict) -> None:
    """
    Setup Genarris folder structure by creating tmp and structure directories.
    """
    mkdir(gnrs_info["struct_dir"])
    mkdir(gnrs_info["tmp_dir"])


def copy_molecule(config: dict, gnrs_info: dict) -> None:
    """
    Copy molecule to tmp directory and standardize orientation.
    """
    tmp_dir = gnrs_info["tmp_dir"]
    mol_path = config["master"]["molecule_path"]
    mol_tmp_dir = os.path.join(tmp_dir, "molecule")
    mkdir(mol_tmp_dir)
    logger.debug("Reading molecule")

    gnrs_info["molecule_path"] = []
    for i, mpth in enumerate(mol_path):
        mol = Molecule.read(mpth)
        mol.standardize_orientation()
        mol_tmp_path = os.path.join(mol_tmp_dir, f"geometry_{i}.in")
        mol.write(mol_tmp_path, parallel=False)
        gnrs_info["molecule_path"].append(mol_tmp_path)

    logger.debug("Wrote molecule to tmp/molecule")
