"""
Core task class for all tasks.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import abc
import time

from mpi4py import MPI

import gnrs.output as gout
from gnrs.core import folders
from gnrs.parallel.io import read_parallel, write_parallel
from gnrs.parallel.structs import DistributedStructs


class TaskABC(abc.ABC):
    """
    Abstract base class for all tasks.
    
    This class defines the common interface and workflow for tasks like:
    - Structure generation
    - Descriptor evaluation
    - Energy evaluation
    - Geometry optimization
    - Clustering and selection
    """

    def __init__(self, comm: MPI.Comm, config: dict, gnrs_info: dict) -> None:
        """
        Initialize the task with MPI communicator and settings.
        
        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_master = self.rank == 0
        self.config = config
        self.gnrs_info = gnrs_info

    def run(self) -> None:
        """
        Execute the complete task workflow.
        
        The workflow consists of:
        1. Initialize task
        2. Pack settings
        3. Print settings
        4. Create folders
        5. Perform task
        6. Collect results
        7. Analyze results
        8. Finalize task
        """
        self.initialize()
        task_set = self.pack_settings()
        self.print_settings(task_set)
        self.create_folders()
        self.perform_task(task_set)
        self.collect_results()
        self.analyze()
        self.finalize()

    @abc.abstractmethod
    def initialize(self, task_name: str, title: str) -> None:
        """
        Initialize the task with required setup.
        
        Args:
            task_name: Name of the task
            title: Title to display for the task
        """
        self.comm.barrier()
        gout.print_title(title)

        self.start_time = time.time()
        self.debug_mode = self.config["master"].get("debug_mode", False)
        self.struct_dir = os.path.join(self.gnrs_info["struct_dir"], task_name)
        self.struct_path = os.path.join(self.struct_dir, "structures.json")
        self.calc_dir = os.path.join(self.gnrs_info["tmp_dir"], task_name)

        self.gnrs_info[task_name] = {
            "start_time": self.start_time,
            "status": "running",
            "struct_dir": self.struct_dir,
            "struct_path": self.struct_path,
            "calc_dir": self.calc_dir
        }

        # Get structs path from last run task
        last_struct_path = self.gnrs_info.get("last_struct_path")
        if last_struct_path is not None:  # if not generation task
            self.structs = read_parallel(last_struct_path)
            ds = DistributedStructs(self.structs)
            n_structs = ds.get_num_structs()
            gout.emit(f"Starting {task_name} task with {n_structs} Structures.")

    @abc.abstractmethod
    def pack_settings(self) -> dict:
        """
        Collect and pack settings needed for the task.
        
        Returns:
            Task settings dictionary 
        """
        pass

    @abc.abstractmethod
    def print_settings(self, task_set: dict) -> None:
        """
        Print task settings in a formatted table.
        
        Args:
            task_set: Task settings dictionary
        """
        gout.print_dict_table(task_set, ["Option", "Value"])

    @abc.abstractmethod
    def create_folders(self) -> None:
        """
        Create the folder structure required for the task.
        """
        folders.mkdir(self.struct_dir)
        folders.mkdir(self.calc_dir)
        self.comm.barrier()  # Wait for folder creation

    @abc.abstractmethod
    def perform_task(self, task_set: dict) -> None:
        """
        Execute the main task.
        
        Args:
            task_set: Task settings dictionary
        """
        pass

    @abc.abstractmethod
    def collect_results(self) -> None:
        """
        Collect and save the results of the task.
        """
        write_parallel(self.struct_path, self.structs)

    @abc.abstractmethod
    def analyze(self) -> None:
        """
        Analyze the results of the task.
        """
        pass

    @abc.abstractmethod
    def finalize(self, task_name: str) -> None:
        """
        Finalize the task and update runtime settings.
        
        Args:
            task_name: Name of the task
        """
        self.gnrs_info["last_struct_path"] = self.struct_path
        self.gnrs_info[task_name]["results"] = self.struct_path
        self.gnrs_info[task_name]["status"] = "completed"
        end_time = self.gnrs_info[task_name]["end_time"] = time.time()
        elapsed = end_time - self.start_time
        gout.emit(f"Completed {task_name} task in {elapsed:.2f} seconds.")
        gout.section_complete()
        os.chdir(self.gnrs_info["work_dir"])
        self.comm.Barrier()
