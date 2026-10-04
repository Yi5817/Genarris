"""
This module provides the asymmetric unit (ASU) generator.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import logging

import numpy as np

import gnrs.output as gout
import gnrs.parallel as gp
from gnrs.core.generator import GeneratorABC
from gnrs.core.molecule import Molecule
from gnrs.parallel.structs import DistributedStructs
from gnrs.cgenarris import pygenarris_mpi as pg_mpi

logger = logging.getLogger("asu_generation")


class ASUGenerator(GeneratorABC):
    """
    Generates random asymmetric units (ASUs) of several molecules.
    """

    title = "ASU Generation"

    def pack_settings(self, settings: dict) -> dict:
        """
        Pack settings needed for ASU generation.

        Args:
            settings: Options of the task's config section

        Returns:
            Task settings dictionary

        Raises:
            ValueError: If the input is not two molecules with stoichiometry 1:1
        """
        molecule_path = self.config["master"]["molecule_path"]
        if len(molecule_path) != 2 or settings["stoichiometry"] != [1, 1]:
            raise ValueError(
                "ASU generation currently supports exactly two components "
                "with stoichiometry 1:1."
            )
        return {"molecule_path": molecule_path, "seed": gp.base_seed, **settings}

    def generate(self, task_set: dict, calc_dir: str) -> None:
        """
        Generate the asymmetric units with cgenarris.

        Args:
            task_set: Task settings dictionary
            calc_dir: Folder of the generation task
        """
        molecules = [
            Molecule.read(path, parallel=False)
            for path in self.gnrs_info["molecule_path"]
        ]
        positions = np.concatenate([mol.get_positions() for mol in molecules])

        species = "".join(
            symbol.ljust(2) for mol in molecules for symbol in mol.get_chemical_symbols()
        )
        n_atoms_per_mol = np.array([len(mol) for mol in molecules], dtype=np.int32)
        stoichiometry = np.array(task_set["stoichiometry"], dtype=np.int32)

        num_asus = task_set["num_asus"]
        sr_min = task_set["sr_min"]
        sr_max = task_set["sr_max"]
        max_attempts = task_set["max_attempts_per_asu"]
        seed = task_set["seed"]
        output_file = os.path.join(calc_dir, "geometry.out")

        num_generated = pg_mpi.generate_asymmetric_units(
            positions,
            species,
            n_atoms_per_mol,
            stoichiometry,
            num_asus,
            sr_min,
            sr_max,
            max_attempts,
            seed,
            output_file,
            self.comm,
        )

        if num_generated < 0:
            raise RuntimeError(
                "cgenarris could not generate asymmetric units; "
                "see the ***ERROR message above."
            )
        if num_generated < num_asus:
            gout.emit(
                f"WARNING: Only {num_generated} of {num_asus} asymmetric units "
                "were generated before max_attempts_per_asu was reached."
            )
        logger.info("Completed ASU generation")

    def analyze(self, structs: dict) -> None:
        """
        Report the number of generated asymmetric units.

        Args:
            structs: Generated structures on this rank
        """
        num_asus = DistributedStructs(structs).get_num_structs()
        gout.print_sub_section("Pool Analysis")
        gout.emit(f"Total number of generated asymmetric units = {num_asus}")
