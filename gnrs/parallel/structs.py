"""
This module contains the DistributedStructs class, which is used to handle distributed structure dictionaries.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import json
import logging
import os
import numpy as np
from pathlib import Path

from ase.atoms import Atoms
from ase.io.jsonio import decode, encode

import gnrs.parallel as gp

logger = logging.getLogger("DistributedStructs")


class DistributedStructs:
    """
    Contains functions for handling distributed structure dictionaries.
    """

    def __init__(self, structs: dict):
        """
        Initialize with a dictionary of structures.
        
        Args:
            structs: Dictionary mapping structure names to ASE Atoms objects
        
        Raises:
            ValueError: If structs is not a dictionary
        """
        self.structs = structs
        self.logger = logger

    def get_num_structs(self) -> int:
        """
        Get the total number of structures in a distributed
        structures dictionary.
        
        Returns:
            Total number of structures across all ranks
        """
        # Handle structs = None case
        if self.structs is not None:
            num_each_rank = len(self.structs)
        else:
            num_each_rank = 0

        total_num_list = gp.comm.gather(num_each_rank, root=0)
        if gp.is_master:
            total_num = sum(total_num_list)
            logger.debug(f"xtal distribution across cores: {total_num_list}")
        else:
            total_num = None

        total_num = gp.comm.bcast(total_num, root=0)
        return total_num

    def find_matches(self, target: Atoms, settings: dict | None = None) -> list:
        """
        Runs pymatgen duplicate checks on a distributed struct dictionary.
        
        Args:
            target: Target structure to be matched
            settings: Settings for pymatgen StructureMatcher
            
        Returns:
            List of matching structure IDs
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.analysis.structure_matcher import StructureMatcher

        pmg_target = AseAtomsAdaptor.get_structure(target)

        if settings is None:
            settings = {"stol": 0.5, "ltol": 0.5, "angle_tol": 10, "attempt_supercell": True}
        matcher = StructureMatcher(**settings)

        match_list = []
        for name, xtal in self.structs.items():
            pmg_xtal = AseAtomsAdaptor.get_structure(xtal)
            if matcher.fit(pmg_target, pmg_xtal):
                match_list.append(name)

        # Combine match list and flatten
        match_list = gp.comm.allgather(match_list)
        match_list = [item for sublist in match_list for item in sublist]

        logger.info(f"Matched with {len(match_list)} structures")
        logger.debug(f"Matched structures = {match_list}")

        return match_list

    def collect_property(self, prpty: str, ptype: str = "info") -> list:
        """
        Collects the property of all the structures into a list.
        
        Args:
            prpty: Property name to collect
            ptype: Property type, either 'info' or 'method'
            
        Returns:
            List of property values on master rank, None on other ranks
        """
        # Construct property list for each rank
        prpty_list = []
        for xtal in self.structs.values():
            if ptype == "method":
                prop = getattr(xtal, prpty)()
            elif ptype == "info":
                prop = xtal.info.get(prpty)
            prpty_list.append(prop)

        # Combine
        prpty_list = gp.comm.gather(prpty_list)
        if gp.is_master:
            prpty_list = [item for sublist in prpty_list for item in sublist]
        return prpty_list

    def get_statistics(self, prpty: str, ptype: str = "info") -> dict:
        """
        Gets the statistics on a property of interest.
        
        Args:
            prpty: Property name to analyze
            ptype: Get property from either 'info' or 'method'
            
        Returns:
            Dictionary with statistics on master rank, None on other ranks
        """
        prpty_list = self.collect_property(prpty, ptype)
        
        if not gp.is_master:
            return None
            
        prpty_array = np.array(prpty_list)
        stats = {
            "Minimum": np.min(prpty_array),
            "Maximum": np.max(prpty_array),
            "Average": np.average(prpty_array),
            "Std": np.std(prpty_array),
        }
        return stats

    def find_spg(self, tol: float = 0.001) -> None:
        """
        Finds the space group of all structures.
        Space group number is stored in info["spg"]
        
        Args:
            tol: Tolerance for symmetry finding
        """
        from ase.spacegroup.spacegroup import get_spacegroup

        for struct in self.structs.values():
            struct.info["spg"] = get_spacegroup(struct, symprec=tol).no

    def checkpoint_save(self, path: str) -> None:
        """
        Checkpoints partially done calculation for restart.
        This routine doesn't communicate to others so that
        ranks can execute independently.
        
        Args:
            path: Directory path to save checkpoint
        """
        structs_str = {name: encode(xtal) for name, xtal in self.structs.items()}
        save_path = os.path.join(path, f"{gp.rank}.save")
        with open(save_path, "w") as chk:
            json.dump(structs_str, chk)

    def checkpoint_load(self, path: str) -> None:
        """
        Loads checkpoint. Unlike save, load is a collective
        and blocking operation.
        
        Args:
            path: Directory path to load checkpoint from
        """
        # Get all checkpoint files
        checkpoints = None
        if gp.is_master:
            checkpoints = list(Path(path).rglob("*.save"))

        # Bcast and partition among ranks
        checkpoints = gp.comm.bcast(checkpoints, root=0)
        checkpoints = checkpoints[gp.rank::gp.size]

        saved_structs = {}
        for checkpoint in checkpoints:
            with open(checkpoint, "r") as chk:
                saved_str = json.load(chk)
                saved_structs.update({name: decode(xtal) for name, xtal in saved_str.items()})

        self.structs = saved_structs
        self.redistribute()

        self.logger.debug(f"Read {self.get_num_structs()} from checkpoint")

    def redistribute(self) -> None:
        """
        Redistribute structures such that all ranks have almost
        equal number of structures. Helpful for balancing load
        across cores
        """
        allstructs = gp.comm.gather(self.structs, root=0)
        scatter_list = None

        # Assemble the list to be scattered
        if gp.is_master:
            # Combine all dictionaries
            combined_structs = {}
            for struct_dict in allstructs:
                combined_structs.update(struct_dict)
                
            # Split dict into list of dicts
            items = list(combined_structs.items())
            num_per_rank = len(combined_structs) // gp.size
            remainder = len(combined_structs) % gp.size
            
            scatter_list = []
            start_idx = 0
            
            for rank in range(gp.size):
                slice_size = num_per_rank + (1 if rank < remainder else 0)
                end_idx = start_idx + slice_size
                scatter_list.append(dict(items[start_idx:end_idx]))
                start_idx = end_idx
                
            scatter_list.reverse()

        self.structs = gp.comm.scatter(scatter_list, root=0)
