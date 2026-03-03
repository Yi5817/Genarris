"""
This module provides the DuplicateRemovalTask class for removing duplicate crystal structures from the pool.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import logging

from mpi4py import MPI
from pymatgen.analysis.structure_matcher import StructureMatcher

import gnrs.output as gout
import gnrs.parallel as gp
from gnrs.core.task import TaskABC
from gnrs.parallel.structs import DistributedStructs
from gnrs.deduplication.dedup import group_by_spg, dedup_group

logger = logging.getLogger("DuplicateRemovalTask")


class DuplicateRemovalTask(TaskABC):
    """
    Task for removing duplicate crystal structures from the pool.
    """

    TASK_NAME = "dedup"

    def __init__(
        self,
        comm: MPI.Comm,
        config: dict,
        gnrs_info: dict,
        instance_id: str | None = None,
    ) -> None:
        """
        Initialize the duplicate removal task.

        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
            instance_id: Unique ID for this task instance
        """
        super().__init__(comm, config, gnrs_info, instance_id=instance_id)

    def initialize(self) -> None:
        """
        Initialize the duplicate removal task.
        """
        iid = self._instance_id or self.TASK_NAME
        title = f"Duplicate Removal: {iid}"
        super().initialize(self.TASK_NAME, title)

    def pack_settings(self) -> dict:
        """
        Pack settings needed for duplicate removal.

        Returns:
            Task settings dictionary
        """
        cfg = self._merge_config(self.TASK_NAME, self._active_instance_id)
        if not cfg:
            cfg = self.config.get("dedup", {})
        task_set = {
            "stol": cfg.get("stol", 0.5),
            "ltol": cfg.get("ltol", 0.5),
            "angle_tol": cfg.get("angle_tol", 10),
            "energy_key": cfg.get("energy_key", None),
            "group_by_spg": cfg.get("group_by_spg", True),
        }
        return task_set

    def print_settings(self, task_set: dict) -> None:
        """
        Print task settings in a formatted table.

        Args:
            task_set: Task settings dictionary
        """
        gout.emit("Duplicate Removal Settings:")
        super().print_settings(task_set)

    def create_folders(self) -> None:
        """
        Create output folders."""
        super().create_folders()

    def perform_task(self, task_set: dict) -> None:
        """
        Execute the duplicate removal task.

        This method:
        1. Groups structures by space group
        2. Removes duplicates from each space group in parallel
        3. Scatters the deduplicated pool back across ranks

        Args:
            task_set: Task settings dictionary
        """
        os.chdir(self.calc_dir)

        energy_key = task_set.pop("energy_key")
        use_spg_groups = task_set.pop("group_by_spg")
        matcher = StructureMatcher(**task_set)

        all_structs = gp.comm.gather(self.structs, root=0)

        combined = {}
        if gp.is_master:
            for d in all_structs:
                combined.update(d)
            del all_structs

        if use_spg_groups:
            spg_groups = {}
            spg_keys = []
            if gp.is_master:
                spg_groups = group_by_spg(combined)
                spg_keys = sorted(spg_groups.keys())
                gout.emit(f"Deduplicating {len(combined)} structures across {len(spg_keys)} space groups")
                del combined

            spg_keys = gp.comm.bcast(spg_keys, root=0)

            unique = {}
            for spg in spg_keys:
                pool = spg_groups.pop(spg, {}) if gp.is_master else {}
                kept = dedup_group(pool, matcher, spg, energy_key)
                unique.update(kept)
        else:
            if gp.is_master:
                gout.emit(
                    f"Deduplicating all {len(combined)} structures"
                )
            unique = dedup_group(
                combined if gp.is_master else {},
                matcher, None, energy_key,
            )

        # Scatter deduplicated pool back across ranks
        ds = DistributedStructs(unique)
        ds.redistribute()
        self.structs = ds.structs

    def collect_results(self) -> None:
        """
        Write surviving structures to disk.
        """
        super().collect_results()

    def analyze(self) -> None:
        """
        Analyze the results of the task.
        """
        logger.debug("Performing analysis")
        sdict = DistributedStructs(self.structs)
        num_structs = sdict.get_num_structs()
        vol_stat = sdict.get_statistics("get_volume", ptype="method")
        gout.print_sub_section("Pool Analysis")
        gout.emit(f"Total number of structures after deduplication = {num_structs}")
        gout.emit("")
        gout.emit(f"Unit Cell Volume Statistics after deduplication:")
        gout.print_dict_table(vol_stat, header=["Stat", "Volume (A^3)"])

    def finalize(self) -> None:
        """
        Finalize the task and update runtime settings.
        """
        logger.info("Completed duplicate removal")
        super().finalize(self.TASK_NAME)
