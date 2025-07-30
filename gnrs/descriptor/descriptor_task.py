"""
This module provides the DescriptorEvaluationTask class for evaluating descriptors.

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

import numpy as np
from mpi4py import MPI
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import gnrs.output as gout
from gnrs.core.task import TaskABC
from gnrs.parallel.io import write_parallel

AVAILABLE_DESCRIPTORS = ["ACSF"]
logger = logging.getLogger("DescriptorTask")


class DescriptorEvaluationTask(TaskABC):
    """
    Task for evaluating crystal structure descriptors.
    """

    def __init__(
        self, 
        comm: MPI.Comm, 
        config: dict, 
        gnrs_info: dict, 
        descriptor: str
    ) -> None:
        """Initialize the descriptor evaluation task.
        
        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
            descriptor: Descriptor class name
        """
        super().__init__(comm, config, gnrs_info)
        self.desc_name = descriptor.lower()
        self.task_name = self.desc_name
        self.desc_file = f"gnrs.descriptor.{self.desc_name}"
        self.desc_class = f"{descriptor.upper()}Descriptor"
        self.explain_variance: float | None = None
        self.desc = None
        
        # Check if the descriptor is implemented
        try:
            desc_module = importlib.import_module(self.desc_file)
            self.desc = getattr(desc_module, self.desc_class)
        except (ImportError, AttributeError) as e:
            logger.error(f"Unable to find requested descriptor: {str(e)}")
            logger.error(f"Available descriptors: {AVAILABLE_DESCRIPTORS}")
            raise

    def initialize(self) -> None:
        """
        Initialize the descriptor evaluation task.
        """
        title = f"Descriptor Evaluation: {self.desc_name}"
        super().initialize(self.task_name, title)
        logger.info(f"Starting descriptor evaluation task: {self.desc_name}")

    def pack_settings(self) -> dict:
        """
        Pack settings needed for descriptor evaluation.
        
        Returns:
            Task settings dictionary
        """
        task_set = {"molecule_path": self.gnrs_info["molecule_path"]}
        task_set.update(self.config[self.task_name])
        return task_set

    def print_settings(self, task_set: dict) -> None:
        """
        Print task settings in a formatted table.
        
        Args:
            task_set: Task settings dictionary
        """
        super().print_settings(task_set)

    def create_folders(self) -> None:
        """
        Create folders needed for the task.
        """
        super().create_folders()

    def perform_task(self, task_set: dict) -> None:
        """
        Execute the descriptor evaluation task.
        
        This method:
        1. Computes descriptors for all structures
        2. Optionally performs PCA compression
        
        Args:
            task_set: Task settings dictionary
        """
        os.chdir(self.calc_dir)

        # Check if PCA compression is requested
        self.pca = task_set.pop("pca", False)
        self.n_components = task_set.pop("n_components", None)
        
        if self.debug_mode:
            dir_name = f"rank_{self.rank}"
            os.makedirs(dir_name, exist_ok=True)
            os.chdir(dir_name)

        desc = self.desc(self.comm, task_set)

        for _id, xtal in self.structs.items():
            try:
                desc.run(xtal)
            except MemoryError:
                if self.debug_mode:
                    xtal.write(f"{_id}.in", parallel=False)
                    logger.debug(f"{_id} got MemoryError on Rank {self.rank}")
                self.structs.pop(_id)
            
        # Perform PCA compression if requested
        if self.pca:
            self._standardize()
            self._pca_compression()
            self._clean_full_descriptor()
            gout.emit("Completed PCA compression.")
            gout.emit("")

    def collect_results(self) -> None:
        """
        Collect and save the results of the task.
        """
        write_parallel(self.struct_path, self.structs)

    def analyze(self) -> None:
        """
        Analyze the results of the task.
        """
        pass

    def finalize(self) -> None:
        """
        Finalize the task and update runtime settings.
        """
        logger.info(f"Completed {self.desc_name} descriptor evaluation")
        super().finalize(self.task_name)

    def _pca_compression(self) -> None:
        """
        Perform PCA compression on the descriptor.
        """
        logger.info("Performing PCA compression")
        local_features = np.array(
            [xtal.info[self.task_name][0, :] for xtal in self.structs.values()]
        )
        
        all_features = self.comm.gather(local_features, root=0)
        
        if self.is_master:
            features = np.vstack([
                feat for sublist in all_features if sublist is not None 
                for feat in sublist
            ])
            
            n_samples, n_features = features.shape
            logger.info(f"PCA input: {n_samples} samples with {n_features} features")
            pca = PCA(n_components=self.n_components, whiten=True)
            pca.fit(features)
            explained_var = np.sum(pca.explained_variance_ratio_)
            logger.info(f"PCA compression: {pca.n_components_} components explain {explained_var:.4f} of variance")
        else:
            pca = None
            
        pca = self.comm.bcast(pca, root=0)

        for xtal in self.structs.values():
            feature = xtal.info[self.task_name]
            compressed = pca.transform(feature.reshape(1, -1))
            xtal.info[f"{self.task_name}_pca"] = compressed

    def _clean_full_descriptor(self) -> None:
        """
        Remove the uncompressed descriptor to save memory.
        """
        for xtal in self.structs.values():
            del xtal.info[self.task_name]

    def _standardize(self) -> None:
        """
        Standardize the descriptor using StandardScaler.
        
        Args:
            name: Name of the descriptor to standardize
        """
        local_features = np.array([x.info[self.task_name][0, :] for x in self.structs.values()])
        fp = np.memmap(
            f"features_{self.rank}.dat", dtype="float32", mode="w+", shape=local_features.shape
        )
        fp[:] = local_features[:]
        fp.flush()

        all_features = self.comm.gather(fp, root=0)
        
        if self.is_master:
            scaler = StandardScaler()
            features = np.vstack([
                feat for sublist in all_features if sublist is not None 
                for feat in sublist
            ])
            scaler.fit(features)
        else:
            scaler = None
            
        scaler = self.comm.bcast(scaler, root=0)

        for xtal in self.structs.values():
            feature = xtal.info[self.task_name]
            scaled = scaler.transform(feature.reshape(1, -1))
            xtal.info[self.task_name] = scaled
