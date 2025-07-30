"""
This module provides the ClusterSelectionTask class for performing cluster selection tasks.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import logging
import importlib
from mpi4py import MPI

import gnrs.output as gout
from gnrs.core.task import TaskABC
from gnrs.parallel.structs import DistributedStructs

AVAILABLE_CLUSTERS = ["ap", "kmeans"]
AVAILABLE_SELECTIONS = ["center", "window"]

logger = logging.getLogger("ClusterSelectionTask")


class ClusterSelectionTask(TaskABC):
    """
    Task for performing cluster selection.
    """
    def __init__(self, comm: MPI.Comm, config: dict, gnrs_info: dict, cluster: str, selection: str) -> None:
        """
        Initialize the cluster selection task.
        
        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
            cluster: Clustering class name
            selection: Selection class name
        """
        super().__init__(comm, config, gnrs_info)
        self.clstr_name = cluster.lower()
        self.clstr_file = f"gnrs.cluster.{self.clstr_name}"
        self.clstr_class = f"{cluster.upper()}Cluster"
        
        self.slct_name = selection.lower()
        self.slct_file = f"gnrs.cluster.selection.{self.slct_name}"
        self.slct_class = f"{selection.upper()}Selection"
        self.task_name = f"{self.clstr_name}-{self.slct_name}"

    def initialize(self) -> None:
        """
        Initialize the cluster selection task.
        """
        title = f"Cluster-Selection: {self.task_name}"
        super().initialize(self.task_name, title)
        # Import required modules
        try:
            clstr_module = importlib.import_module(self.clstr_file)
            self.clstr = getattr(clstr_module, self.clstr_class)
        except (ImportError, AttributeError):
            logger.warning(f"Unable to find cluster method {self.clstr_name}")
            logger.warning(f"Available cluster methods: {AVAILABLE_CLUSTERS}")
            raise
        try:
            select_module = importlib.import_module(self.slct_file)
            self.slct = getattr(select_module, self.slct_class)
        except (ImportError, AttributeError):
            logger.warning(f"Cannot find selection method {self.slct_name}")
            logger.warning(f"Available selection methods: {AVAILABLE_SELECTIONS}")
            raise

        logger.info(f"Starting Cluster-Selection task: {self.task_name}")

    def pack_settings(self) -> dict:
        """
        Pack settings needed for cluster selection.
        
        Returns:
            Task settings dictionary
        """
        task_set = {}
        task_set[self.clstr_name] = self.config[self.clstr_name].copy()
        task_set[self.slct_name] = self.config[self.slct_name].copy()
        return task_set

    def print_settings(self, task_set: dict) -> None:
        """
        Print task settings in a formatted table.
        
        Args:
            task_set: Task settings dictionary
        """
        gout.emit("Clustering Settings:")
        super().print_settings(task_set[self.clstr_name])
        gout.emit("Selection Settings:")
        super().print_settings(task_set[self.slct_name])

    def create_folders(self) -> None:
        """
        Create folders needed for the task.
        """
        super().create_folders()

    def perform_task(self, task_set: dict) -> None:
        """
        Execute the cluster selection task.
        
        Args:
            task_set: Task settings dictionary
        """
        os.chdir(self.calc_dir)

        n_structs = DistributedStructs(self.structs).get_num_structs()
        if type(task_set[self.clstr_name]["n_clusters"]) is float:
            task_set[self.clstr_name]["n_clusters"] = int(n_structs * task_set[self.clstr_name]["n_clusters"])
        # If total number of structs too low, dont cluster
        if n_structs < task_set[self.clstr_name]["n_clusters"]:
            gout.emit(
                f"Total number of structures = {n_structs},"
                f" less than number of clusters. Skipping task!"
            )
            return
        
        if type(task_set[self.clstr_name]["clusters_tol"]) is float:
            task_set[self.clstr_name]["clusters_tol"] = int(task_set[self.clstr_name]["clusters_tol"] * task_set[self.clstr_name]["n_clusters"])
        
        self._run_cluster(task_set[self.clstr_name])
        self._run_selection(task_set[self.slct_name])
        return

    def collect_results(self) -> None:
        """
        Collect and save the results of the task.
        """
        super().collect_results()

    def analyze(self) -> None:
        """
        Analyze the results of the task.
        """
        dsdict = DistributedStructs(self.structs)
        n_structs = dsdict.get_num_structs()
        vol_stat = dsdict.get_statistics("get_volume", ptype="method")
        gout.emit(f"Number of structures after selection: {n_structs}")
        gout.emit("Unit Cell Volume Statistics")
        gout.print_dict_table(vol_stat, header=["Stat", "Volume (A^3)"])

    def finalize(self) -> None:
        """
        Finalize the task and update runtime settings.
        """
        logger.info("Completed clustering-selection")
        super().finalize(self.task_name)

    def _run_cluster(self, cluster_settings: dict) -> None:
        """
        Run the clustering algorithm.
        
        Args:
            cluster_settings: Clustering task settings
        """
        calc = self.clstr(self.comm, cluster_settings)
        self.final_num_cluster = calc.run(self.structs)
        return

    def _run_selection(self, selection_settings: dict) -> None:
        """
        Run the selection algorithm.
        
        Args:
            selection_settings: Selection task settings
        """
        selection_settings["final_num_cluster"] = self.final_num_cluster
        calc = self.slct(self.comm, selection_settings)
        calc.run(self.structs)
        return
