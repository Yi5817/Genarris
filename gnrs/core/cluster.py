"""
Abstract base class for clustering methods.

This module provides the base class for implementing clustering methods.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import abc
from mpi4py import MPI


class ClusterABC(abc.ABC):
    """
    Abstract base class for clustering methods.
    
    This class defines the interface for implementing crystal structure clustering algorithms.
    All clustering implementations should inherit from this class and implement
    the abstract methods.
    """

    def __init__(self, comm: MPI.Comm, task_settings: dict) -> None:
        """
        Initialize the clustering method.
        
        Args:
            comm: MPI communicator
            task_settings: Task settings
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_master = self.rank == 0
        self.tsk_set = task_settings

    def run(self, struct_dict: dict) -> int:
        """
        Run the clustering workflow.
        
        1. Initialize
        2. Fit the clustering model
        3. Predict clusters
        4. Finalize
        
        Args:
            struct_dict: Crystal structures
            
        Returns:
            Number of clusters found
        """
        self.structs = struct_dict
        self.initialize()
        self.fit()
        self.final_n_clusters = self.predict()
        self.finalize()
        return self.final_n_clusters

    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initialize for clustering.
        """
        pass

    @abc.abstractmethod
    def fit(self) -> None:
        """
        Fit the clustering model to the data.
        """
        pass

    @abc.abstractmethod
    def predict(self) -> int:
        """
        Predict clusters for the data.
        
        Returns:
            Number of clusters
        """
        pass

    @abc.abstractmethod
    def finalize(self) -> None:
        """
        Finalize the clustering process.
        """
        pass
