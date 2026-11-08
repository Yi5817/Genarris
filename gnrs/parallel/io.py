"""
This module provides functionality for reading and writing parallel data.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import hashlib
import logging

from ase import Atoms
from ase.io.jsonio import decode, encode

import gnrs.parallel as gp

logger = logging.getLogger(__name__)


def read_geometry_out(file_path: str) -> dict:
    """
    Master process reads geometry file and scatters data to other processes.

    Args:
        file_path: Path to the geometry output file

    Returns:
        Dictionary mapping IDs to Atoms objects.
    """
    if gp.is_master:
        with open(file_path) as gfile:
            str_data = gfile.read()
        str_data = str_data.split("#######  END  STRUCTURE #######")
        str_data = str_data[:-1]
        str_data = _make_scatterable_form(str_data)
    else:
        str_data = None

    str_data = gp.comm.scatter(str_data, root=0)
    struct_dict = {}
    for str_geo in str_data:
        xtal = str2atoms(str_geo.split("\n"))
        if xtal is not None:
            name = hashlib.blake2b(str_geo.encode(), digest_size=8).hexdigest()
            struct_dict[name] = xtal

    return struct_dict


def str2atoms(geometry_str: list) -> Atoms | None:
    """
    Constructs Atoms object from aims geometry format.

    Args:
        geometry_string: List of strings containing geometry data

    Returns:
        ASE Atoms object representing the crystal structure
    """
    species, cell, pos, spg = [], [], [], None

    for line in geometry_str:
        sline = line.split()
        if not sline:
            continue

        if "lattice_vector" in line:
            cell.append([float(x) for x in sline[1:4]])
        elif sline[0] == "atom":
            pos.append([float(x) for x in sline[1:4]])
            species.append(sline[4])
        elif "SPGLIB_detected_spacegroup" in line:
            spg = int(sline[-1])
            if spg == 0:
                return None

    # Asymmetric units have no lattice vectors
    xtal = Atoms(symbols=species, positions=pos, cell=cell or None, pbc=bool(cell))
    if spg is not None:
        xtal.info["spg"] = spg

    return xtal


def encode_struct(name: str, xtal: Atoms) -> str:
    """
    Serialize one structure as a line of a structures.json file.

    Args:
        name: Structure ID
        xtal: Structure

    Returns:
        ``"name": <ase json>,`` followed by a newline
    """
    return f'"{name}": {encode(xtal)},\n'


def decode_struct(line: str) -> tuple[str, Atoms]:
    """
    Parse a line written by ``encode_struct``.

    Args:
        line: One structure line, with or without the trailing comma

    Returns:
        Structure ID and structure
    """
    name, xtal = line.split(":", 1)
    return name.strip().strip('"'), decode(xtal.strip().rstrip(","))


def write_parallel(file_path: str, struct_dict: dict) -> None:
    """
    Convert structures to JSON strings, gather and store to file.

    Args:
        file_path: Path to output file
        struct_dict: Dictionary of structures to write
    """
    str_list = [encode_struct(k, v) for k, v in struct_dict.items()]
    str_list = gp.comm.gather(str_list, root=0)
    if not gp.is_master:
        return

    str_list = [s for sublist in str_list for s in sublist]
    if not str_list:
        logger.warning("No structures to write!")
        return
    logger.info(f"Writing {len(str_list)} structures to file")
    str_list[-1] = str_list[-1][:-2]
    with open(file_path, "w") as wfile:
        wfile.write("{\n")
        wfile.writelines(str_list)
        wfile.write("\n}")


def read_parallel(file_path: str) -> dict:
    """
    Reads JSON database of structures and scatters it to all processes.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary mapping IDs to Atoms objects
    """
    logger.info(f"Reading structures from {file_path}")

    str_list = None
    if gp.is_master:
        with open(file_path) as rfile:
            # Drop the lines with the opening and closing braces
            str_list = _make_scatterable_form(rfile.readlines()[1:-1])

    str_list = gp.comm.scatter(str_list, root=0)
    return dict(decode_struct(str_struct) for str_struct in str_list)


def _make_scatterable_form(str_list: list) -> list:
    """
    Construct a list of length comm.size with padding for even distribution.

    Args:
        str_list: List of strings to distribute

    Returns:
        List of sublists for each process
    """
    ave, res = divmod(len(str_list), gp.size)
    # The first ``res`` ranks get one extra item
    bounds = [p * ave + min(p, res) for p in range(gp.size + 1)]
    return [str_list[bounds[p] : bounds[p + 1]] for p in range(gp.size)]
