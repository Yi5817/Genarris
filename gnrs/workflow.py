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
from gnrs.parallel.structs import DistributedStructs
from gnrs.parser import UserSettingsParser, UserSettingsSanityChecker
from gnrs.parallel.test import test_bcast
from gnrs.core.restart import (
    RestartError,
    restart_init,
    is_task_completed,
    load_restart,
    write_restart,
)
from gnrs.gnrsutil.core import check_if_exp_found

import argparse


class Genarris:
    """
    Defines the flow of control in Genarris for crystal structure
    generation and optimization.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize Genarris.
        """

        self.config = {}
        self.gnrs_info = {}
        self.seed = args.seed
        self.restart = args.restart
        self.overwrite = args.overwrite

        self._mpi_init()
        self._log_init()
        self._output_init()
        self._parallel_init(seed=self.seed)
        self._gnrs_info_init()
        self._config_init(args)
        restart_init(self.comm, self.config, self.gnrs_info)
        if self.restart:
            self.attempt_restart()
        else:
            self._check_previous_run()
        self._folders_init()

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
        self.gnrs_info["restart"] = self.restart

    def attempt_restart(self) -> None:
        """
        Load restart data if restart flag is set.

        Raises:
            RestartError: If no restart file exists in the current directory.
        """
        gout.print_title("Restarting Genarris")
        if not load_restart():
            raise RestartError(
                "--restart was requested, but no restart file was found at "
                f"{os.path.join(self.gnrs_info['work_dir'], 'restart.json')}. "
                "Run from the directory of a previous Genarris run, or start "
                "a new run without --restart. Runs made with older Genarris "
                "versions keep this file at tmp/restart.json; move it to the "
                "run directory to resume them."
            )
        self._print_restart_summary()
        gout.double_separator()

    def _check_previous_run(self) -> None:
        """
        Refuse to start over on top of a previous run unless --overwrite
        was given; with it, discard the previous run's progress record and
        the checkpoints of every task.

        Raises:
            RestartError: If a previous run exists and --overwrite is not set.
        """
        # Older releases kept the restart file under tmp/
        candidates = [
            os.path.join(self.gnrs_info["work_dir"], "restart.json"),
            os.path.join(self.gnrs_info["tmp_dir"], "restart.json"),
        ]
        found = None
        if self.is_master:
            found = [path for path in candidates if os.path.isfile(path)]
        found = self.comm.bcast(found, root=0)
        if not found:
            return

        if not self.overwrite:
            raise RestartError(
                "This directory already contains a Genarris run "
                f"({found[0]}). Rerun with --restart to resume it, or "
                "with --overwrite to discard its progress and start over."
            )
        self.logger.warning("Discarding previous run record (--overwrite)")
        gout.emit(
            "NOTE: --overwrite given. Discarding the previous run's progress "
            "record; results in structures/ will be overwritten as tasks "
            "complete."
        )
        gout.emit("")
        if self.is_master:
            for restart_file in found:
                os.remove(restart_file)
            # Tasks this run never reaches would otherwise keep their old
            # checkpoints, which a later --restart would merge into the pool
            tmp_dir = self.gnrs_info["tmp_dir"]
            if os.path.isdir(tmp_dir):
                for entry in os.scandir(tmp_dir):
                    if entry.is_dir():
                        DistributedStructs.checkpoint_clear(entry.path)

    def _print_restart_summary(self) -> None:
        """
        Report which tasks are already completed and where the run resumes.
        """
        tasks = self.config.get("workflow", {}).get("tasks", [])
        try:
            specs = resolve_tasks(tasks)
        except ValueError:
            return  # _run_tasks reports the invalid task list

        completed, pending = [], []
        for spec in specs:
            if is_task_completed(spec.instance_id):
                completed.append(spec.instance_id)
            else:
                pending.append(spec.instance_id)
        gout.emit(
            f"Restart file loaded: {len(completed)} of {len(specs)} tasks "
            "already completed."
        )
        if completed:
            gout.emit(f"Completed tasks (will be skipped): {', '.join(completed)}")
        if pending:
            gout.emit(f"Resuming from task: {pending[0]}")
        else:
            gout.emit("All tasks were already completed. Nothing to do.")

    def _folders_init(self) -> None:
        """
        Initialize folder structure for execution.

        Creates tmp and structures directories and copies molecule data.
        Runs in restart mode too, so a cleaned tmp/ dir is recreated.
        """
        folders.init_folders(self.is_master)
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
