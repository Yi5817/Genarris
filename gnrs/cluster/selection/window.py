"""
This module provides the WindowSelection class for selecting structures within a energy window.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import re
import logging

import gnrs.output as gout
from gnrs.core.selection import SelectionABC
from gnrs.gnrsutil.core import eV2kJ

logger = logging.getLogger("WindowSelection")


class WINDOWSelection(SelectionABC):
    """
    Selection class that selects structures within a energy window from clusters.
    """
    def initialize(self) -> None:
        """
        Initialize the window selection process.
        """
        self.logger = logger
        super().initialize()
        # Number of structures to select from each cluster
        self.n_structs = self.settings.get("n_structs_per_cluster", None)
        # Energy range above minimum energy to include structures 
        self.e_window = self.settings.get("energy_window", None)
        
        if self.settings["final_num_cluster"] is not None:
            self.n_clusters = self.settings["final_num_cluster"]
        else:
            self.n_clusters = self.settings["n_clusters"]
        logger.debug(f"Final cluster {self.n_clusters}")
        self.clst_name = self.settings["cluster_name"]
        self.filter = self.settings["filter"]
        self.Z = self.settings["z"]
        gout.emit(f"Selecting structures within {self.e_window} energy window from clusters...")
        logger.info(f"Selecting structures within {self.e_window} energy window from clusters")

    def finalize(self) -> None:
        """
        Finalize the selection process.
        """
        logger.info(f"Completed selecting structures within {self.e_window} energy window from clusters")
        gout.emit(f"Completed selecting structures within {self.e_window} energy window from clusters.\n")
        gout.emit("")

    def select(self, struct_dict: dict) -> None:
        """
        Select structures within energy window from each cluster.
        
        Args:
            struct_dict: Crystals dictionary
        """
        self.struct_dict = struct_dict
        sel_xtals = []
        for idx in range(self.n_clusters):
            clst_structs = self.get_energy(idx)
            clst_structs = self.get_range_allranks(clst_structs)
            sel_xtals.extend(clst_structs)

        logger.info(f"Total selected structures: {len(sel_xtals)}")
        self.comm.Barrier()
        self._filter_xtals(sel_xtals)
        self.comm.Barrier()

    def get_energy(self, idx: int) -> list:
        """
        Get energy for structures in the specified cluster.
        
        Args:
            idx: Cluster index
            
        Returns:
            List of [name, energy] pairs for structures in cluster
        """
        structs = []
        for _id, xtal in self.struct_dict.items():
            xtal_cluster = int(re.search(r'\d+', xtal.info[self.clst_name]).group())
            if idx == xtal_cluster:
                e = float(xtal.info[self.filter])
                structs.append([_id, e])
        return structs

    def get_window_across_ranks(self, structs: list) -> list:
        """
        Get structures within energy window across all ranks.
        
        Args:
            structs: List of [xtal_id, energy] pairs for structures in a cluster
            
        Returns:
            List of xtal ids within energy window
        """
        structs = self.comm.gather(structs, root=0)
        
        if self.is_master:
            # Flatten list of lists from all ranks
            all_structs = [s for rank_structs in structs if rank_structs 
                         for s in rank_structs]
            
            if not all_structs:
                logger.debug("No structures found in cluster")
                selected = []
            else:
                # Find minimum energy and calculate relative lattice energies
                min_e = min(s[1] for s in all_structs)
                rel_energies = sorted([(_id, eV2kJ((e - min_e) / self.Z)) 
                              for _id, e in all_structs], key=lambda x: x[1])
                
                # Select structures within energy window
                selected = [_id for _id, rel_e in rel_energies 
                          if 0 <= rel_e <= self.e_window]
                
                # Limit number of structures if specified
                if self.n_structs is not None:
                    selected = selected[:min(len(selected), self.n_structs)]
                    
                logger.info(f"Selected {len(selected)} structures within lattice energy "
                           f"window of {self.e_window:.3f} kJ/mol")
        else:
            selected = None
            
        selected = self.comm.bcast(selected, root=0)
        return selected

    def _filter_xtals(self, keep_ids: list) -> None:
        """
        Remove structures not in selected list.
        
        Args:
            keep_ids: List of structure names to keep
        """
        for _id in list(self.struct_dict.keys()):
            if _id not in keep_ids:
                del self.struct_dict[_id]