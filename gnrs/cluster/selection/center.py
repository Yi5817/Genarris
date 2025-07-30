"""
This module provides the CenterSelection class for selecting the center of a cluster.

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

logger = logging.getLogger("CenterSelection")


class CENTERSelection(SelectionABC):
    """
    Selection class that identifies and keeps only the center crystals from clusters.
    
    This class can select centers based on either:
    1. Crystals marked as centers during clustering
    2. Crystals with minimum value of a specified property within each cluster
    """
    
    def initialize(self) -> None:
        """
        Initialize the center selection process.
        """
        self.logger = logger
        super().initialize()
        self.settings["property_name"] = None
        if self.settings["final_num_cluster"] is not None:
            self.num_clusters = self.settings["final_num_cluster"]
        else:
            self.num_clusters = self.settings["n_clusters"]
        logger.debug(f"Final cluster {self.num_clusters}")
        self.cluster_name = self.settings["cluster_name"]
        self.filter = self.settings.get("filter", 'center')
        
        gout.emit(f"Selecting cluster centers using {self.filter}...")
        logger.info(f"Selecting cluster centers using {self.filter}...")

    def finalize(self) -> None:
        """
        Finalize the selection process.
        """
        logger.info(f"Completed selecting cluster centers using {self.filter}.")
        gout.emit(f"Completed selecting cluster centers using {self.filter}.\n")
        gout.emit("")

    def select(self, struct_dict: dict) -> None:
        """
        Select crystals based on configured criteria.
        
        Args:
            struct_dict: Crystals dictionary
        """
        self.struct_dict = struct_dict
        
        if self.filter != 'center':
            # Select based on minimum property value
            self._select_by_property()
        else:
            # Select based on center flag from clustering
            self._select_by_center_flag()

    def _select_by_property(self) -> None:
        """
        Select crystals with minimum property value in each cluster.
        """
        min_xtals = []
        
        # Find minimum property xtal for each cluster
        for idx in range(self.num_clusters):
            min_proper = self._get_min_property_xtal(idx)
            min_proper_xtal = self._get_min_across_ranks(min_proper)
            if min_proper_xtal is not None:
                min_xtals.append(min_proper_xtal)
                
        self.comm.Barrier()
        self._filter_xtals(min_xtals)
        self.comm.Barrier()

    def _select_by_center_flag(self) -> None:
        """
        Select crystals marked as centers during clustering.
        """
        # Gather centers from all ranks
        center_xtals = self._gather_center_xtals()
        
        self.comm.Barrier()
        self._filter_xtals(center_xtals)
        self.comm.Barrier()

    def _get_min_property_xtal(self, idx: int) -> list:
        """
        Find crystal with minimum property value in specified cluster on this rank.
        
        Args:
            idx: Cluster index
            
        Returns:
            List containing [xtal_id, property_value] or empty list
        """
        min_xtal = []
        min_value = float('inf')
        
        for _id, xtal in self.struct_dict.items():
            xtal_cluster = int(re.search(r'\d+', xtal.info[self.cluster_name]).group())
            if xtal_cluster != idx:
                continue
                    
            property_value = float(xtal.info[self.filter])
            if property_value < min_value:
                min_value = property_value
                min_xtal = [_id, property_value]

        return min_xtal

    def _gather_center_xtals(self) -> list:
        """
        Gather all crystals marked as centers across all ranks.
        
        Returns:
            List of crystal names that are centers
        """
        center_list = []
        for _id, xtal in self.struct_dict.items():
            if "center" in xtal.info.get(self.cluster_name, ""):
                center_list.append(_id)

        centers = self.comm.gather(center_list, root=0)
        
        if self.is_master:
            center_xtals = [item for sublist in centers if sublist for item in sublist]
        else:
            center_xtals = None
            
        return self.comm.bcast(center_xtals, root=0)

    def _get_min_across_ranks(self, min_list: list) -> str | None:
        """
        Find crystal with minimum property value across all ranks.
        
        Args:
            min_list: [xtal_id, property_value]
            
        Returns:
            ID of crystal with minimum property value
        """
        all_min_lists = self.comm.gather(min_list, root=0)
        
        if self.is_master:
            min_entries = [e for e in all_min_lists if e]
            
            if not min_entries:
                min_all_ranks = []
            else:
                min_val = min(e[1] for e in min_entries)
                min_entries = [e for e in min_entries if e[1] == min_val]
                min_all_ranks = min(min_entries, key=lambda x: x[0]) if min_entries else []
        else:
            min_all_ranks = None

        min_all_ranks = self.comm.bcast(min_all_ranks, root=0)
        
        if min_all_ranks:
            return min_all_ranks[0]
        else:
            return None

    def _filter_xtals(self, keep_ids: list) -> None:
        """
        Remove all crystals except those in the keep list.
        
        Args:
            keep_ids: List of crystal IDs to keep
        """
        for _id in list(self.struct_dict.keys()):
            if _id not in keep_ids:
                del self.struct_dict[_id]