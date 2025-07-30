"""
This module provides functionality for saving and loading program state for
restart functionality.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import json
import logging
from mpi4py import MPI

logger = logging.getLogger("restart")

class Restart:
    """
    Manages saving and loading program state for restart functionality.
    """

    def __init__(self) -> None:
        self.comm: MPI.Comm | None = None
        self.config: dict = {}
        self.gnrs_info: dict = {}
        self.restart_file: str | None = None
        self.is_master: bool = False

    def initialize(
        self,
        comm: MPI.Comm,
        config: dict,
        gnrs_info: dict
    ) -> None:
        """
        Initialize the restart manager.

        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
        """
        self.comm = comm
        self.config = config
        self.gnrs_info = gnrs_info
        self.is_master = comm.Get_rank() == 0
        self.restart_file = os.path.join(self.gnrs_info["tmp_dir"], "restart.json")

    def write_restart(self) -> None:
        """
        Write current program state to restart file.
        """
        if not self.is_master:
            return
        logger.info("Writing restart file")
        restart = {"config": self.config, "gnrs_info": self.gnrs_info}
        with open(self.restart_file, "w") as rfile:
            json.dump(restart, rfile, indent=4, sort_keys=True)

    def load_restart(self) -> None:
        """
        Load program state from restart file.
        """
        self.comm.barrier()
        restart_data = None
        if self.is_master:
            if self.check_restart(bcast=False):
                logger.info("Reading restart file")
                with open(self.restart_file, "r") as rfile:
                    restart_data = json.load(rfile)
            else:
                logger.info("No restart file found")

        restart_data = self.comm.bcast(restart_data, root=0)
        if restart_data:
            self.config.update(restart_data["config"])
            self.gnrs_info.update(restart_data["gnrs_info"])

    def check_restart(self, bcast: bool = True) -> bool:
        """
        Check if restart file exists.
        """
        exists = False
        if self.is_master:
            exists = os.path.isfile(self.restart_file)

        if bcast:
            exists = self.comm.bcast(exists, root=0)

        return exists

    def check_task_completion(self, task_name: str) -> bool:
        """
        Check if a task has been completed.
        """
        task = self.gnrs_info.get(task_name, {})
        return task.get("status") == "completed"


# Global singleton instance
_restart = Restart()

def restart_init(comm, config, gnrs_info):
    """
    Initialize the global restart manager.
    """
    _restart.initialize(comm, config, gnrs_info)

def write_restart():
    """
    Write current program state to restart file.
    """
    _restart.write_restart()

def load_restart():
    """
    Load program state from restart file.
    """
    return _restart.load_restart()

def check_restart():
    """
    Check if restart file exists.
    """
    return _restart.check_restart()

def is_task_completed(task_name):
    """
    Check if a task has been completed.
    """
    return _restart.check_task_completion(task_name)

__all__ = [
    "restart_init", "write_restart", "load_restart", 
    "check_restart", "is_task_completed"
]
