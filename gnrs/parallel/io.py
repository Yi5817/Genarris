"""
This module provides functionality for reading and writing parallel data.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations
__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import logging
import random

from ase import Atoms
from ase.io.jsonio import encode, decode

import gnrs.parallel as gp

logger = logging.getLogger("parallel_io")


def read_geometry_out(file_path: str) -> dict:
    """
    Master process reads geometry file and scatters data to other processes.
    
    Args:
        file_path: Path to the geometry output file
        
    Returns:
        Dictionary mapping random IDs to Atoms objects
    """
    if gp.is_master:
        with open(file_path, "r") as gfile:
            str_data = gfile.read()
        str_data = str_data.split("#######  END  STRUCTURE #######")
        str_data = str_data[:-1]
        str_data = _make_scatterable_form(str_data)
    else:
        str_data = None

    str_data = gp.comm.scatter(str_data, root=0)
    struct_list = [
        str2atoms(str_geo.split("\n"))
        for str_geo in str_data
        if str_geo is not None
    ]
    # random IDs
    struct_dict = {f"{random.getrandbits(60):x}": s for s in struct_list}
    
    return struct_dict


def str2atoms(geometry_str: list) -> Atoms:
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

    xtal = Atoms(symbols="".join(species), positions=pos, cell=cell, pbc=True)
    if spg is not None:
        xtal.info["spg"] = spg

    return xtal


def write_parallel(file_path: str, struct_dict: dict, 
                  gather: bool = True, mode: str = "w") -> None:
    """
    Convert structures to JSON strings, gather and store to file.
    
    Args:
        file_path: Path to output file
        struct_dict: Dictionary of structures to write
        gather: Whether to gather data from all processes
        mode: File opening mode
    """
    
    if not struct_dict:
        logger.info("No structures to write!")
        return

    # Convert to list of JSON strings
    str_list = [f'"{k}": {encode(v)},\n' for k, v in struct_dict.items()]
    
    if gather:
        str_list = gp.comm.gather(str_list, root=0)
        if gp.is_master and str_list:
            num_structs = sum(len(e) for e in str_list)
            logger.info(f"Writing {num_structs} structures to file")

    if gp.is_master:
        # Flatten
        if gather and str_list:
            str_list = [s for sublist in str_list for s in sublist]
            
        if str_list:
            str_list[-1] = str_list[-1][:-2]
            
            with open(file_path, mode) as wfile:
                wfile.write("{\n")
                wfile.writelines(str_list)
                wfile.write("\n}")


def read_parallel(file_path: str, scatter: bool = True) -> dict:
    """
    Reads JSON database of structures.
    
    Args:
        file_path: Path to JSON file
        scatter: Whether to scatter data to all processes
        
    Returns:
        Dictionary mapping IDs to Atoms objects
    """
    logger.info(f"Reading structures from {file_path}")
    
    if gp.is_master:
        with open(file_path, "r") as rfile:
            str_list = rfile.readlines()
        # Remove {} and add comma to the last element
        str_list = str_list[1:-1]
        if str_list:
            str_list[-1] = str_list[-1] + ","
            str_list = _make_scatterable_form(str_list)
    else:
        str_list = None

    str_list = gp.comm.scatter(str_list, root=0)
    struct_list = []
    # Construct struct_list
    for str_struct in str_list:
        if str_struct is None:
            continue
        s_id, s = str_struct.split(":", 1)
        s_id = s_id.strip('"')
        s = s[:-2]  # Remove comma and newline
        struct_list.append([s_id, decode(s)])

    if not scatter:
        struct_list = gp.comm.gather(struct_list, root=0)
        if struct_list:
            struct_list = [item for sublist in struct_list for item in sublist]

    struct_dict = {s[0]: s[1] for s in struct_list}
    return struct_dict


def _make_scatterable_form(str_list: list) -> list:
    """
    Construct a list of length comm.size with padding for even distribution.
    
    Args:
        str_list: List of strings to distribute
        
    Returns:
        List of sublists for each process
    """
    ave, res = divmod(len(str_list), gp.size)
    counts = [ave + 1 if p < res else ave for p in range(gp.size)]
    # Determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(gp.size)]
    ends = [sum(counts[: p + 1]) for p in range(gp.size)]
    new_list = [str_list[starts[p]: ends[p]] for p in range(gp.size)]
    return new_list
