"""
This module provides the workflow orchestration for Genarris.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import time
import logging

from mpi4py import MPI

from gnrs.core import folders
from gnrs.core.logging import GenarrisLogger
import gnrs.output as gout
from gnrs.parallel import init_parallel
from gnrs.parser import UserSettingsParser, UserSettingsSanityChecker
from gnrs.parallel.test import test_bcast
from gnrs.restart import restart_init, is_task_completed, load_restart, write_restart
from gnrs.gnrsutil.core import check_if_exp_found

import argparse


class Genarris:
    """Defines the flow of control in Genarris for crystal structure generation and optimization."""

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize Genarris.
        """

        self.config = {}
        self.gnrs_info = {}
        self.restart = args.restart
        self.executors = {}

        self._mpi_init()
        self._log_init()
        self._output_init()
        self._parallel_init()
        self._gnrs_info_init()
        self._config_init(args)
        restart_init(self.comm, self.config, self.gnrs_info)
        if not self.restart:
            self._folders_init()
        else:
            self.attempt_restart()
        
        self._register_tasks()
        self.comm.barrier()
        self.logger.info("Genarris initialized successfully")

    def run(self) -> None:
        """
        Execute Genarris with the configured tasks.
        """
        self.logger.info("Starting Genarris Tasks")
        tasks = self.config.get("workflow", {}).get("tasks", [])
        self._run_tasks(tasks)

    def _log_init(self) -> None:
        """
        Initialize logger with MPI communicator.
        """
        self.Genlogger = GenarrisLogger(self.comm)
        self.logger = logging.getLogger("genarris")

    def _mpi_init(self) -> None:
        """
        Initialize MPI.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_master = self.rank == 0

    def _parallel_init(self) -> None:
        """
        Initialize parallel processing environment.
        """
        init_parallel(self.comm)

    def _output_init(self) -> None:
        """
        Initialize output system and display welcome message.
        """
        gout.init_output(self.comm)
        gout.welcome_message()

    def _config_init(self, args: argparse.Namespace) -> None:
        """
        Parse configuration settings from input file.
        """

        self.config_path = os.path.abspath(args.config)
        self.gnrs_info["config_path"] = self.config_path

        # Parse Config file
        gout.print_title("Parsing User Config File")
        gout.emit(f"Reading {self.config_path}.")
        # parse config file
        if self.is_master:
            parser = UserSettingsParser(self.config_path)
            config = parser.load_config()
            
            # Update log level
            new_level = config["master"]["log_level"]
            self.Genlogger.reset_loglevel(new_level)
            UserSettingsSanityChecker(config)
            self.config.update(config)

        # Broadcast settings to all processes
        self.config = self.comm.bcast(self.config, root=0)
        gout.print_configs(self.config)

    def _gnrs_info_init(self) -> None:
        """
        Initialize Genarris information with paths and execution metadata.
        """
        self.logger.info("Setting runtime values")
        
        # Set working directories
        self.work_dir = os.getcwd()
        self.gnrs_info["work_dir"] = self.work_dir
        self.gnrs_info["struct_dir"] = os.path.join(self.work_dir, "structures")
        self.gnrs_info["tmp_dir"] = os.path.join(self.work_dir, "tmp")
        
        # Initialize data containers
        self.gnrs_info["energy_list"] = []
        self.gnrs_info["genarris_start_time"] = time.time()
        self.gnrs_info["size"] = self.size

    def attempt_restart(self) -> None:
        """
        Load restart data if restart flag is set
        """
            
        load_restart()
        gout.print_title("Restarting Genarris")
        gout.print_configs(self.config)
        gout.double_separator()

    def _folders_init(self) -> None:
        """
        Initialize folder structure for execution.
        
        Creates tmp and structures directories and copies molecule data
        when not in restart mode.
        """
        folders.init_folders(self.is_master)
        
        if not self.restart:
            self.logger.info("Setting up folders: structures and tmp")
            folders.setup_main_folders(self.gnrs_info)
            folders.copy_molecule(self.config, self.gnrs_info)

    def _register_tasks(self) -> None:
        """
        Register all available job types that can be executed
        """
        from gnrs.generation import StructureGenerationTask
        from gnrs.optimize import GeometryOptimizationTask
        from gnrs.descriptor import DescriptorEvaluationTask
        from gnrs.cluster import ClusterSelectionTask
        from gnrs.energy import EnergyCalculationTask

        # Register generation task
        self.executors["generation"] = lambda: StructureGenerationTask(
            self.comm, self.config, self.gnrs_info
        ).run()

        # Register optimization tasks
        self.executors["symm_rigid_press"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "symm_rigid_press"
        ).run()

        self.executors["rigid_press"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "rigid_press"
        ).run()

        # Register descriptor tasks
        self.executors["acsf"] = lambda: DescriptorEvaluationTask(
            self.comm, self.config, self.gnrs_info, "acsf"
        ).run()

        # Register cluster tasks
        self.executors["ap_center"] = lambda: ClusterSelectionTask(
            self.comm, self.config, self.gnrs_info, "ap", "center"
        ).run()

        self.executors["ap_window"] = lambda: ClusterSelectionTask(
            self.comm, self.config, self.gnrs_info, "ap", "window"
        ).run()

        # Register enegy tasks
        self.executors["maceoff"] = lambda: EnergyCalculationTask(
            self.comm, self.config, self.gnrs_info, "maceoff"
        ).run()

        self.executors["uma"] = lambda: EnergyCalculationTask(
            self.comm, self.config, self.gnrs_info, "uma"
        ).run()

        self.executors["aimnet"] = lambda: EnergyCalculationTask(
            self.comm, self.config, self.gnrs_info, "aimnet"
        ).run()

        self.executors["aims"] = lambda: EnergyCalculationTask(
            self.comm, self.config, self.gnrs_info, "aims"
        ).run()

        self.executors["vasp"] = lambda: EnergyCalculationTask(
            self.comm, self.config, self.gnrs_info, "vasp"
        ).run()

        self.executors["dftb"] = lambda: EnergyCalculationTask(
            self.comm, self.config, self.gnrs_info, "dftb"
        ).run()

        # Register bfgs tasks
        self.executors["bfgs_mace"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "bfgs", "maceoff"
        ).run()

        self.executors["bfgs_uma"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "bfgs", "uma"
        ).run()

        self.executors["bfgs_aimnet"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "bfgs", "aimnet"
        ).run()

        self.executors["bfgs_aims"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "bfgs", "aims"
        ).run()

        self.executors["bfgs_vasp"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "bfgs", "vasp"
        ).run()

        # Register lbfgs tasks
        self.executors["lbfgs_mace"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "lbfgs", "maceoff"
        ).run()

        self.executors["lbfgs_uma"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "lbfgs", "uma"
        ).run()

        self.executors["lbfgs_aimnet"] = lambda: GeometryOptimizationTask(
            self.comm, self.config, self.gnrs_info, "lbfgs", "aimnet"
        ).run()

    def _run_tasks(self, tasks: list) -> None:
        """
        Run specific tasks in config file
        
        Args:
            tasks: List of task names to execute
        """
        self.logger.info(f"Running configured tasks: {tasks}")
        gout.emit(f"Executing {len(tasks)} configured tasks")
        
        for task in tasks:
            if task not in self.executors:
                self.logger.error(f"Unknown task: {task}")
                gout.emit(f"Error: Task '{task}' is not registered. Skipping.")
                continue
                
            if not is_task_completed(task):
                gout.emit(f"Running task: {task}")
                self.executors[task]()
                write_restart()
                test_bcast()
                check_if_exp_found(self.config, self.gnrs_info)
            else:
                self.logger.info(f"{task} task was completed before restart")
                gout.skip_task(task)
