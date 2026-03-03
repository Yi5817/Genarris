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
from gnrs.core.registry import resolve_tasks
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
        self.seed = args.seed
        self.restart = args.restart

        self._mpi_init()
        self._log_init()
        self._output_init()
        self._parallel_init(seed=self.seed)
        self._gnrs_info_init()
        self._config_init(args)
        restart_init(self.comm, self.config, self.gnrs_info)
        if not self.restart:
            self._folders_init()
        else:
            self.attempt_restart()

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

    def _parallel_init(self, seed: int = 42) -> None:
        """
        Initialize parallel processing environment.
        """
        init_parallel(self.comm, seed=seed)

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

    def _run_tasks(self, tasks: list) -> None:
        """
        Run specific tasks in config file
        
        Args:
            tasks: List of task names to execute
        """
        try:
            task_specs = resolve_tasks(tasks)
        except ValueError as exc:
            self.logger.error(str(exc))
            gout.emit(f"Error: {exc}")
            return

        self.logger.info(f"Running configured tasks: {[s.instance_id for s in task_specs]}")
        gout.emit(f"Executing {len(task_specs)} configured tasks")
        
        for spec in task_specs:
            if not is_task_completed(spec.instance_id):
                gout.emit(f"Running task: {spec.instance_id}")
                spec.cls(
                    self.comm, self.config, self.gnrs_info,
                    *spec.extra_args,
                    instance_id=spec.instance_id,
                ).run()
                write_restart()
                test_bcast()
                check_if_exp_found(self.config, self.gnrs_info)
            else:
                self.logger.info(f"{spec.instance_id} task was completed before restart")
                gout.skip_task(spec.instance_id)
