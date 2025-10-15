"""
This module computes the energy.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import importlib
import json
import logging
import os

from mpi4py import MPI

from gnrs.core.task import TaskABC
from gnrs.parallel.structs import DistributedStructs
import gnrs.output as gout

AVAILABLE_CALCULATORS = ["DFTBP", "AIMS", "MACEOFF", "UMA", "VASP"]
logger = logging.getLogger("EnergyCalcTask")

class EnergyCalculationTask(TaskABC):
    """
    Task for computing energy using DFT or semi-empirical method.
    Uses ASE calculators for energy evaluation.
    """

    def __init__(
        self, 
        comm: MPI.Comm, 
        config: dict, 
        gnrs_info: dict, 
        energy_method: str
    ) -> None:
        """Initialize the energy calculation task.
        
        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
            energy_method: Energy calculation method
        """
        super().__init__(comm, config, gnrs_info)
        self.energy_name = energy_method.lower()
        self.energy_class = self.energy_name.upper() + "Energy"
        self.energy_file = f"gnrs.energy.{self.energy_name}"

        try:
            energy_module = importlib.import_module(self.energy_file)
            self.energy_calc = getattr(energy_module, self.energy_class)
        except (ImportError, AttributeError):
            logger.warn("Unable to find requested energy calculation method.")
            logger.warn(f"Available calculators: {AVAILABLE_CALCULATORS}")
            raise

    def initialize(self) -> None:
        """
        Initialize the energy calculation task.
        """
        title = "Energy Calculation: " + self.energy_name
        super().initialize(self.energy_name, title)
        logger.info(f"Starting energy calculation task: {self.energy_name}")

    def pack_settings(self) -> dict:
        """
        Pack the settings for the energy calculation task.
        """
        task_set = {**self.config[self.energy_name]}
        return task_set

    def print_settings(self, task_settings: dict) -> None:
        """
        Print the settings for the energy calculation task.
        """
        super().print_settings(task_settings)

    def create_folders(self) -> None:
        """
        Create the folders for the energy calculation task.
        """
        super().create_folders()

    def perform_task(self, task_settings: dict) -> None:
        """
        Perform the energy calculation task.
        """
        if "energy_settings_path" in task_settings:
            set_file = task_settings["energy_settings_path"]
            with open(set_file, "r") as jfile:
                task_settings["energy_settings"] = json.load(jfile)

        # Create and change dirs
        os.chdir(self.calc_dir)
        dir_name = "rank_" + str(self.rank)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        os.chdir(dir_name)

        self.rank_calc_dir = os.path.join(self.calc_dir, dir_name)
        self._load_save_files()

        # Calculate energy
        calc = self.energy_calc(self.comm, task_settings, self.energy_name)
        for xtal in self.structs.values():
            calc.run(xtal)
            if task_settings["save_flag"]:
                self.dsdict.checkpoint_save(self.rank_calc_dir)

    def collect_results(self) -> None:
        """
        Collect the results from the energy calculation task.
        """
        super().collect_results()

    def analyze(self) -> None:
        """
        Analyze the results from the energy calculation task.
        """
        dsdict = DistributedStructs(self.structs)
        stat_dict = dsdict.get_statistics(self.energy_name)
        gout.print_sub_section("Energy Statistics")
        gout.print_dict_table(stat_dict, header=["Stat", "eV"])

    def finalize(self) -> None:
        """
        Finalize the energy calculation task.
        """
        logger.info("Completed energy calculation")
        super().finalize(self.energy_name)

    def _load_save_files(self) -> None:
        """
        Load the save files for the energy calculation task.
        """
        ds = DistributedStructs({})
        ds.checkpoint_load(self.calc_dir)
        n_struct = ds.get_num_structs()
        if n_struct > 0:
            self.structs = ds.structs
            gout.emit("Save files of previous calculation found.")
            gout.emit(f"Loaded {n_struct} structure(s) from save files.")

        self.dsdict = DistributedStructs(self.structs)
        n_completed = None
        completed = self.dsdict.collect_property(self.energy_name, "info")
        if self.is_master:
            n_completed = sum(x is not None for x in completed)
        if n_struct > 0:
            gout.emit(f"{n_completed} calculation(s) were completed previously.")
            gout.emit("")
