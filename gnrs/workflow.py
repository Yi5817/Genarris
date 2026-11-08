"""
This module provides the workflow orchestration for Genarris.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import argparse
import logging
import os
import time

from mpi4py import MPI

import gnrs.output as gout
import gnrs.parallel as gp
from gnrs.core import folders
from gnrs.core.logging import GenarrisLogger
from gnrs.core.registry import resolve_tasks
from gnrs.core.restart import Restart, RestartError
from gnrs.gnrsutil.core import check_if_exp_found
from gnrs.parser import UserSettingsParser, UserSettingsSanityChecker


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
        self._gnrs_info_init()
        self._config_init(args)
        tasks = self.config.get("workflow", {}).get("tasks", [])
        self.task_specs = resolve_tasks(tasks)
        self.restart_manager = Restart(self.comm, self.config, self.gnrs_info)
        if self.restart:
            self.attempt_restart()
        else:
            self._check_previous_run()

        gp.base_seed = self.gnrs_info["seed"]
        gout.emit(f"Random seed: {gp.base_seed}")
        self._folders_init()
        self.restart_manager.write()

        self.comm.barrier()
        self.Genlogger.sync()
        self.logger.info("Genarris initialized successfully")

    def run(self) -> None:
        """
        Execute Genarris with the configured tasks.
        """
        self.logger.info("Starting Genarris Tasks")
        self._run_tasks(self.task_specs)

    def _log_init(self) -> None:
        """
        Initialize logger with MPI communicator.
        """
        self.Genlogger = GenarrisLogger(self.comm)
        self.logger = logging.getLogger(__name__)

    def _mpi_init(self) -> None:
        """
        Initialize MPI and the package communicator.
        """
        gp.init_parallel(MPI.COMM_WORLD)
        self.comm = gp.comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_master = self.rank == 0

    def _output_init(self) -> None:
        """
        Initialize output system and display welcome message.
        """
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
        self.logger.debug("Setting runtime values")

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
        self.gnrs_info["seed"] = self.seed

    def attempt_restart(self) -> None:
        """
        Load restart data if restart flag is set.

        Raises:
            RestartError: If no restart file exists in the current directory.
        """
        gout.print_title("Restarting Genarris")
        discarded = self.restart_manager.load()
        if discarded is None:
            raise RestartError(
                "--restart was requested, but no restart file was found at "
                f"{self.restart_manager.restart_file}. "
                "Run from the directory of a previous Genarris run, or start "
                "a new run without --restart."
            )
        self._print_restart_summary()
        for task in discarded:
            gout.emit(
                f"NOTE: The checkpoints of task '{task}' from the previous "
                "run are discarded, because the task or one before it "
                "changed; it starts from scratch."
            )
        gout.double_separator()

    def _check_previous_run(self) -> None:
        """
        Refuse to start over on top of a previous run unless --overwrite
        was given; with it, discard the previous run's progress record. A
        fresh run never keeps checkpoints of an earlier one.

        Raises:
            RestartError: If a previous run exists and --overwrite is not set.
        """
        found = self.restart_manager.find_records()
        if found and not self.overwrite:
            if found[0] == self.restart_manager.restart_file:
                hint = "Rerun with --restart to resume it, or --overwrite to start over"
            else:
                hint = "It is from an older release; start over with --overwrite"
            raise RestartError(
                f"This directory already contains a Genarris run ({found[0]}). {hint}."
            )
        if found:
            gout.emit(
                "NOTE: --overwrite given. The previous run's progress record is "
                "discarded; every task starts from scratch and overwrites its "
                "results in structures/."
            )
            gout.emit("")
            if self.is_master:
                self.logger.warning("Discarding previous run record (--overwrite)")
                for restart_file in found:
                    os.remove(restart_file)
        self.restart_manager.discard_checkpoints()

    def _print_restart_summary(self) -> None:
        """
        Report which tasks are already completed and where the run resumes.
        """
        ids = [spec.instance_id for spec in self.task_specs]
        completed = [i for i in ids if self.restart_manager.is_task_completed(i)]
        pending = [i for i in ids if i not in completed]
        gout.emit(
            f"Restart file loaded: {len(completed)} of {len(ids)} tasks "
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
        self.logger.debug("Setting up folders: structures and tmp")
        folders.setup_main_folders(self.gnrs_info)
        folders.copy_molecule(self.config, self.gnrs_info)

    def _run_tasks(self, task_specs: list) -> None:
        """
        Run specific tasks in config file

        Args:
            task_specs: Resolved specs of the tasks to execute
        """
        self.logger.info(
            f"Running configured tasks: {[s.instance_id for s in task_specs]}"
        )
        gout.emit(f"Executing {len(task_specs)} configured tasks")

        for spec in task_specs:
            if not self.restart_manager.is_task_completed(spec.instance_id):
                spec.cls(
                    self.comm,
                    self.config,
                    self.gnrs_info,
                    *spec.extra_args,
                    instance_id=spec.instance_id,
                ).run()
                self.restart_manager.write()
                check_if_exp_found(self.config, self.gnrs_info)
            else:
                self.logger.info(
                    f"{spec.instance_id} task was completed before restart"
                )
                gout.skip_task(spec.instance_id)
            self.Genlogger.sync()
