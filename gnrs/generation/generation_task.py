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
import time
import logging

import numpy as np
from mpi4py import MPI
import gnrs.output as gout
from gnrs.core.task import TaskABC
from gnrs.core.molecule import Molecule
from gnrs.gnrsutil.volume_estimation import predict_cell_volume
from gnrs.gnrsutil.molecule_bonding import get_vdw_distance_cutoff_matrix
from gnrs.parallel.io import read_geometry_out
from gnrs.parallel.structs import DistributedStructs
from gnrs.cgenarris import pygenarris_mpi as pg_mpi

logger = logging.getLogger("generation")


class StructureGenerationTask(TaskABC):
    """
    Task for generating crystal structures.
    """
    
    def __init__(self, comm: MPI.Comm, config: dict, gnrs_info: dict) -> None:
        """
        Initialize the structure generation task.
        
        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
        """
        super().__init__(comm, config, gnrs_info)
        self.task_name = "generation"
        self.generation_type = None
        self.spg_distribution = None
        self.stoic = None
        self.ucv_mean = None
        self.ucv_std = None
        self.ucv_mult = None
        self.sr = None
        self.cutoff_matrix = None
        self.hbond = None
        self.structs = None

    def initialize(self) -> None:
        """
        Initialize the structure generation task.
        """
        logger.info("Starting random structure (raw pool) generation")
        title = "Generation"
        super().initialize(self.task_name, title)

    def pack_settings(self) -> dict:
        """
        Pack settings needed for structure generation.
        
        Returns:
            Task settings dictionary
        """
        seed = self.config["master"].get("seed", np.random.randint(0, 2**31))
        task_set = {
            "seed": seed,
            "z": self.config["master"]["z"],
            "molecule_path": self.config["master"]["molecule_path"],
            **self.config["generation"]
        }
        
        self.generation_type = task_set["generation_type"]
        self.spg_distribution = task_set["spg_distribution_type"]
        self.stoic = task_set["stoichiometry"]
        
        if isinstance(self.spg_distribution, list):
            task_set["spg_distribution_type"] = "custom"

        self.ucv_mean = task_set.pop("ucv_mean", task_set.pop("unit_cell_volume_mean", None))
        self.ucv_std = task_set.pop("ucv_std", task_set.pop("unit_cell_volume", None))
        self.ucv_mult = task_set.pop("ucv_mult", task_set.pop("volume_mult", 1.5))
        self.sr = task_set.pop("sr", task_set.pop("specific_radius_proportion", 0.95))
        self._predict_cell_volume(task_set["z"])
        task_set["ucv_mean"] = self.ucv_mean
        task_set["ucv_std"] = self.ucv_std
        task_set["sr"] = self.sr

        # Get van der waal cutoff matrix for structure checks
        gout.emit("Constructing Van der Waal cutoff matrix...")
        logger.info("Getting van der waal distance cutoff matrix")
        self.cutoff_matrix, self.hbond = get_vdw_distance_cutoff_matrix(
            mol_path = task_set["molecule_path"],
            z = task_set["z"],
            sr = task_set["sr"],
            natural_cutoff_mult = task_set["natural_cutoff_mult"]
            )
        task_set["cutoff_matrix"] = self.cutoff_matrix
        gout.emit("Van der Waal cutoff matrix constructed")
        return task_set

    def print_settings(self, task_set: dict) -> None:
        """
        Print settings for the generation task.
        
        Args:
            task_set: Task settings dictionary
        """
        logger.debug("Printing settings for generation")
        ucv_mean = task_set["ucv_mean"]
        ucv_std = task_set["ucv_std"]
        gout.emit(f"Predicted unit cell volume = {ucv_mean:.1f} A^3.")
        gout.emit(
            f"Standard deviation of unit cell"
            f" volume distribution = {ucv_std:.1f} A^3."
        )
        gout.emit("")

        if self.hbond:
            gout.emit(f"Hydrogen bond corrections applied to: {self.hbond}")
        else:
            gout.emit("No Hydrogen bond corrections applied")
        gout.emit("")

        gout.print_dict_table(
            task_set, ["Option", "Value"], skip=("cutoff_matrix")
        )

        gout.emit(
            "Passing control to cgenarris, fast"
            " and scalable structure generator...\n"
        )
        gout.single_separator()
        gout.emit("")

    def create_folders(self) -> None:
        """
        Create necessary folders and prepare input files.
        """
        super().create_folders()

        # Copy molecule to tmp/generation folder
        # Save number of atoms in the molecule in rtm_set
        self.gnrs_info["n_atoms_in_mol"] = []
        for i, mpth in enumerate(self.gnrs_info["molecule_path"]):
            mol = Molecule.read(mpth)
            gen_mol_path = os.path.join(self.calc_dir, "geometry.in")
            self.gnrs_info["n_atoms_in_mol"].append(len(mol))
            mol.write(gen_mol_path, format="aims")

        # Check if only selected spacegroups are requested
        # And create spg file for the given spg
        if isinstance(self.spg_distribution, list) and self.is_master:
            spg_file = os.path.join(self.calc_dir, "spg")
            with open(spg_file, "w") as sfile:
                for spg in self.spg_distribution:
                    print(spg, file=sfile)

        # Write cutoff matrix to file
        if self.is_master:
            np.savetxt(
                os.path.join(self.calc_dir, "cutoff_matrix.txt"), self.cutoff_matrix
            )

    def perform_task(self, task_set: dict) -> None:
        """
        Perform the structure generation task.
        
        Args:
            task_set: Task settings dictionary
        """
        # change working dir to generation
        os.chdir(self.calc_dir)

        # Unpack and call cgenarris
        seed = task_set["seed"]
        z = task_set["z"]
        tol = task_set["tol"]
        ucv_mean = task_set["ucv_mean"]
        ucv_std = task_set["ucv_std"]
        cutoff_matrix = np.array(task_set["cutoff_matrix"], dtype="float32")
        num_structs = task_set["num_structures_per_spg"]
        max_attempts = task_set["max_attempts_per_spg"]
        vol_attempts = task_set["max_attempts_per_volume"]
        spg_type = task_set["spg_distribution_type"]
        norm_dev = task_set["lattice_norm_dev"]
        angle_std = task_set["lattice_angle_std"]
        stoic = task_set["stoichiometry"]
        
        if self.generation_type == "crystal":
            pg_mpi.mpi_generate_molecular_crystals_with_vdw_cutoff_matrix(
                cutoff_matrix,
                num_structs,
                z,
                ucv_mean,
                ucv_std,
                tol,
                max_attempts,
                spg_type,
                vol_attempts,
                seed,
                norm_dev,
                angle_std,
                self.comm,
            )
        else:
            raise ValueError(f"Generation type {self.generation_type} not supported")

        os.chdir(self.gnrs_info["work_dir"])
        gout.single_separator()
        gout.emit("")
        logger.info("Completed generation")

    def collect_results(self) -> None:
        """
        Collect and save the results of the task.
        """
        # Move to structures dir
        logger.info("Collecting generated crystals")
        geometry_out = os.path.join(self.calc_dir, "geometry.out")
        self.structs = read_geometry_out(geometry_out)
        super().collect_results()

    def analyze(self) -> None:
        """
        Analyze the results of the task.
        """
        logger.debug("Performing analysis")
        sdict = DistributedStructs(self.structs)
        num_structs = sdict.get_num_structs()
        vol_stat = sdict.get_statistics("get_volume", ptype="method")
        gout.print_sub_section("Pool Analysis")
        gout.emit(f"Total number of generated structures = {num_structs}")
        gout.emit("")
        gout.emit(f"Unit Cell Volume Statistics:")
        gout.print_dict_table(vol_stat, header=["Stat", "Volume (A^3)"])

    def finalize(self) -> None:
        """
        Finalize the task and update runtime settings.
        """
        logger.info("Finalizing generation")
        super().finalize(self.task_name)

    def _predict_cell_volume(self, Z: int) -> None:
        """
        Predict the unit cell volume.
        
        Args:
            Z: Number of molecules in the unit cell
        """
        # Estimate unit cell volume and send it to other cores
        if self.ucv_mean == "predict" and self.is_master:
            logger.info("Predicting unit cell volume using builtin PyMoVE model...")
            gout.emit("Predicting unit cell volume using builtin PyMoVE model.")
            start_time = time.time()
            
            pred_volume = 0.0
            for molecule_path, st in zip(self.gnrs_info["molecule_path"], self.stoic):
                pred_volume += predict_cell_volume(molecule_path) * st
            
            self.ucv_mean = pred_volume * Z * self.ucv_mult
            elapsed_time = time.time() - start_time
            
            logger.debug(f"Predicted molecular volume: {pred_volume:.2f} A^3")
            logger.debug(f"Final unit cell volume: {self.ucv_mean:.2f} A^3")
            gout.emit(f"Unit cell volume prediction completed in {elapsed_time:.1f} seconds.")

        self.ucv_mean = self.comm.bcast(self.ucv_mean, root=0)
        self.ucv_std = self.ucv_mean * self.ucv_std  # Get std in A^3
