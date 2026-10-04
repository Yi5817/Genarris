"""
This module provides the StructureGenerationTask class for performing structure generation tasks.

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
from gnrs.core.task import TaskABC
from gnrs.parallel.io import read_geometry_out
from gnrs.parallel.structs import DistributedStructs

GENERATORS = {
    "crystal_generation": "crystal",
    "asu_generation": "asu",
    "generation": "crystal",
}
logger = logging.getLogger("GenerationTask")


class StructureGenerationTask(TaskABC):
    """
    Task for generating structures: crystals or asymmetric units.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        config: dict,
        gnrs_info: dict,
        task_name: str = "generation",
        instance_id: str | None = None,
    ) -> None:
        """
        Initialize the structure generation task.

        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
            task_name: Generation task name; selects the generator
            instance_id: Unique ID for this task instance
        """
        super().__init__(comm, config, gnrs_info, instance_id=instance_id)
        self.task_name = task_name
        self.gen_name = GENERATORS[task_name]
        self.gen_file = f"gnrs.generation.{self.gen_name}"
        self.gen_class = f"{self.gen_name.upper()}Generator"
        self.structs = None

        gen_module = importlib.import_module(self.gen_file)
        self.generator = getattr(gen_module, self.gen_class)(comm, config, gnrs_info)

    def initialize(self) -> None:
        """
        Initialize the structure generation task.
        """
        logger.info(f"Starting random {self.gen_name} generation")
        super().initialize(self.task_name, self.generator.title)

    def pack_settings(self) -> dict:
        """
        Pack settings needed for structure generation.

        Returns:
            Task settings dictionary
        """
        settings = self._merge_config(self.task_name, self._active_instance_id)
        return self.generator.pack_settings(settings)

    def print_settings(self, task_set: dict) -> None:
        """
        Print settings for the generation task.

        Args:
            task_set: Task settings dictionary
        """
        self.generator.print_settings(task_set)

    def create_folders(self) -> None:
        """
        Create necessary folders and prepare input files.
        """
        super().create_folders()
        self.generator.write_inputs(self.calc_dir)

    def perform_task(self, task_set: dict) -> None:
        """
        Perform the structure generation task.

        Args:
            task_set: Task settings dictionary
        """
        # A crystal generation after another task starts from its pool of ASUs
        if self.gen_name == "crystal" and self.structs is not None:
            self.generator.generate_from_asus(task_set, self.calc_dir, self.structs)
        else:
            self.generator.generate(task_set, self.calc_dir)

    def collect_results(self) -> None:
        """
        Collect and save the results of the task.
        """
        logger.info(f"Collecting generated {self.gen_name} structures")
        geometry_out = os.path.join(self.calc_dir, "geometry.out")
        self.structs = read_geometry_out(geometry_out)
        if DistributedStructs(self.structs).get_num_structs() == 0:
            raise RuntimeError(
                f"{self.task_name} generated no structures, so the task is not "
                "completed. Relax the generation settings and run again."
            )
        super().collect_results()

    def analyze(self) -> None:
        """
        Analyze the results of the task.
        """
        self.generator.analyze(self.structs)

    def finalize(self) -> None:
        """
        Finalize the task and update runtime settings.
        """
        logger.info(f"Finalizing {self.task_name}")
        super().finalize(self.task_name)
