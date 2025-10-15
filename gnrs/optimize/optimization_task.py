"""
This module provides the GeometryOptimizationTask class for performing geometry optimization tasks.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import json
import importlib
import logging

from mpi4py import MPI
import gnrs.output as gout
from gnrs.core.task import TaskABC
from gnrs.parallel.io import read_parallel
from gnrs.parallel.structs import DistributedStructs
from gnrs.gnrsutil.molecule_bonding import get_vdw_distance_cutoff_matrix

AVAILABLE_METHODS = ["LBFGS", "BFGS", "RIGID_PRESS", "SYMM_RIGID_PRESS"]
AVAILABLE_ENERGY_METHODS = ["DFTBP", "AIMS", "MACEOFF", "UMA", "VASP"]
logger = logging.getLogger("GeoOptTask")


class GeometryOptimizationTask(TaskABC):
    """
    Task for relaxing geometry of crystal structures.
    """

    def __init__(
        self, 
        comm: MPI.Comm, 
        config: dict, 
        gnrs_info: dict, 
        optimizer: str, 
        energy_method: str | None = None
    ) -> None:
        """
        Initialize the geometry optimization task.
        
        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
            optimizer: Optimizer
            energy_method: Energy calculator(optional)
        """
        super().__init__(comm, config, gnrs_info)
        self.opt_name = optimizer.lower()
        self.opt_class = f"{self.opt_name.upper()}Optimizer"
        # Set task name and energy method
        if energy_method is not None:
            self.energy_method = energy_method.lower()
            self.energy_class = f"{self.energy_method.upper()}Energy"
            self.task_name = f"{self.opt_name}-{self.energy_method}"
        else:
            self.energy_method = None
            self.task_name = self.opt_name
            
        self.energy_set = {}
        self.structs = None
        self.dsdict = None
        self.rank_calc_dir = None
        self.opt_calc = None
        self.energy_calc = None

    def initialize(self) -> None:
        """
        Initialize the optimization task.
        """
        title = f"Geometry Optimization: {self.task_name}"
        super().initialize(self.task_name, title)
        logger.info(f"Starting geometry optimization task: {self.task_name}")

        # If struct_path specified, use it instead of default
        spath = self.config[self.opt_name].get("struct_path")
        if spath is not None:
            logger.info(f"Reading from user given file {spath}")
            self.structs = read_parallel(spath)
            self.config[self.opt_name].pop("struct_path")

        # Log the optimizer being used
        if self.energy_method is not None:
            gout.emit(f"Using ASE {self.opt_name} optimizer with {self.energy_method} energy method.")
        else:
            gout.emit("Using builtin optimizer.")

        self._load_modules()

    def _load_modules(self) -> None:
        """
        Load the required optimizer and energy calculator modules.
        
        Raises:
            ImportError: If the requested optimization method or energy calculator is not found.
        """
        # Check if optimization method is implemented
        self.opt_file = f"gnrs.optimize.{self.opt_name}"
        try:
            opt_module = importlib.import_module(self.opt_file)
            self.opt_calc = getattr(opt_module, self.opt_class)
        except (ImportError, AttributeError):
            logger.error("Unable to find requested optimization method.")
            logger.error(f"Available methods: {AVAILABLE_METHODS}")
            raise

        # If the builtin optimizer is used instead of ASE
        if self.energy_method is None:
            return

        # Check if energy calculator is implemented within gnrs
        self.energy_file = f"gnrs.energy.{self.energy_method}"
        try:
            energy_module = importlib.import_module(self.energy_file)
            self.energy_calc = getattr(energy_module, self.energy_class)
        except (ImportError, AttributeError):
            logger.error(f"Unable to find requested energy calculation method.")
            logger.error(f"Available methods: {AVAILABLE_ENERGY_METHODS}")
            raise

    def pack_settings(self) -> dict:
        """
        Pack settings for the optimization task.
        
        Returns:
            dict: Task settings dictionary
        """
        task_set = {}
        if self.opt_name in self.config:
            task_set.update(self.config[self.opt_name])

        if self.opt_name in ["rigid_press", "symm_rigid_press"]:
            task_set["z"] = self.config["master"]["z"]
            task_set["mol_path"] = self.gnrs_info["molecule_path"][0]
            sr = task_set.pop("sr")
            cutoff_mult = task_set.pop("natural_cutoff_mult")
            cutoff_matrix, hbond = get_vdw_distance_cutoff_matrix(
                self.gnrs_info["molecule_path"], task_set["z"], sr, cutoff_mult
            )
            task_set["cutoff_matrix"] = cutoff_matrix

        # Pack settings for energy method separately in self.energy_set
        if self.energy_method is not None:
            energy_method = task_set.pop("energy_method")
            self.energy_set = self.config[energy_method]
            
        return task_set

    def print_settings(self, task_set: dict) -> None:
        """
        Print the task settings.
        
        Args:
            task_set: Task settings dictionary
        """
        gout.emit("Optimization Settings:")
        gout.print_dict_table(task_set, ["Option", "Value"], skip=("cutoff_matrix"))
        if self.energy_method is not None:
            gout.emit("Energy Settings:")
            super().print_settings(self.energy_set)


    def create_folders(self) -> None:
        """
        Create the necessary folders for the optimization task.
        """
        super().create_folders()

    def perform_task(self, task_set: dict) -> None:
        """
        Perform the optimization task.
        
        Args:
            task_set: Task settings dictionary
        """
        # Load settings if neccessary and Create energy calculator
        if self.energy_method is not None:
            set_file = self.energy_set.get("energy_settings_path")
            if set_file is not None:
                with open(set_file, "r") as jfile:
                    self.energy_set["energy_settings"] = json.load(jfile)
            # Get energy calculator
            ec_obj = self.energy_calc(self.comm, self.energy_set, self.energy_method)
            e_calc = ec_obj.get_calculator()
        else:
            e_calc = None

        # Create and change into dir for each rank.
        os.chdir(self.calc_dir)
        dir_name = f"rank_{self.rank}"
        os.makedirs(dir_name, exist_ok=True)
        self.rank_calc_dir = os.path.join(self.calc_dir, dir_name)
        self._load_save_files()

        # Run optimization
        gout.emit("Optimizing structures...")
        opt = self.opt_calc(
            self.comm, task_set, self.opt_name, self.energy_method, e_calc
        )
        
        for _id, xtal in self.structs.items():
            opt.run(xtal)
            self.dsdict.checkpoint_save(self.rank_calc_dir)

        self.comm.barrier()
        gout.emit("Completed optimizations.")

    def collect_results(self):
        super().collect_results()

    def analyze(self) -> None:
        """
        Analyze the results of the optimization task.
        """
        dsdict = DistributedStructs(self.structs)
        vol_stat = dsdict.get_statistics("get_volume", ptype="method")
        
        gout.print_sub_section("Unit Cell Volume Statistics")
        gout.print_dict_table(vol_stat, header=["Stat", "Volume (A^3)"])
        
        if self.energy_method is not None:
            energy_stat = dsdict.get_statistics(f"{self.opt_name}_{self.energy_method}")
            gout.print_sub_section("Energy Statistics")
            gout.print_dict_table(energy_stat, header=["Stat", "Energy (eV)"])

    def finalize(self) -> None:
        """
        Finalize the optimization task.
        """
        logger.info("Completed optimization task")
        if self.energy_method is not None:
            self.gnrs_info["energy_list"].append(self.energy_method)
        super().finalize(self.task_name)

    def _load_save_files(self) -> None:
        """
        Load checkpoint files from previous calculations if they exist.
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
        completed = self.dsdict.collect_property(self.opt_name, "info")
        
        if self.is_master:
            n_completed = sum(x is not None for x in completed)
            
        if n_struct > 0:
            gout.emit(f"{n_completed} calculation(s) were completed previously.")
            gout.emit("")
