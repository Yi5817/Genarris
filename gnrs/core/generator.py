"""
Abstract base class for structure generators.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = "Yi Yang"
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import abc

from mpi4py import MPI

import gnrs.output as gout


class GeneratorABC(abc.ABC):
    """
    Abstract base class for structure generators.

    A generator writes its structures to ``geometry.out`` in the folder of
    the generation task. All generator implementations should inherit from
    this class and implement the abstract methods.
    """

    title = "Generation"

    def __init__(self, comm: MPI.Comm, config: dict, gnrs_info: dict) -> None:
        """
        Initialize the generator.

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

    @abc.abstractmethod
    def pack_settings(self, settings: dict) -> dict:
        """
        Pack settings needed for the generation.

        Args:
            settings: Options of the task's config section

        Returns:
            Task settings dictionary
        """
        pass

    def print_settings(self, task_set: dict) -> None:
        """
        Print task settings in a formatted table.

        Args:
            task_set: Task settings dictionary
        """
        gout.print_dict_table(task_set, ["Option", "Value"])

    def write_inputs(self, calc_dir: str) -> None:
        """
        Write the input files the generator needs.

        Args:
            calc_dir: Folder of the generation task
        """
        pass

    @abc.abstractmethod
    def generate(self, task_set: dict, calc_dir: str) -> None:
        """
        Generate the structures and write them to ``geometry.out``.

        Args:
            task_set: Task settings dictionary
            calc_dir: Folder of the generation task
        """
        pass

    @abc.abstractmethod
    def analyze(self, structs: dict) -> None:
        """
        Report statistics of the generated structures.

        Args:
            structs: Generated structures on this rank
        """
        pass
