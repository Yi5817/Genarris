"""
This module provides k-means clustering.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import logging
from mpi4py import MPI

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import gnrs.output as gout
from gnrs.core.cluster import ClusterABC

logger = logging.getLogger("kmeans")


class KMEANSCluster(ClusterABC):
    """
    K-means clustering.
    """

    def __init__(self, comm: MPI.Comm, task_settings: dict) -> None:
        """
        Initialize the k-means clustering.

        Args:
            comm: MPI communicator
            task_settings: Task settings
        """
        super().__init__(comm, task_settings)
        self.cluster_name = "kmeans"
        self.feature_name = task_settings.pop("feature_name")
        self.save_info = task_settings.pop("save_info", False)

    def initialize(self) -> None:
        """
        Initialize the k-means clustering.
        """
        self.features = np.array(
            [x.info[self.feature_name][0, :] for x in self.structs.values()]
        )
        self.kmeans = MiniBatchKMeans(**self.tsk_set, batch_size=len(self.features))
        logger.info("Started kmeans clustering")
        gout.emit("Running kmeans clustering...")

    def fit(self) -> None:
        """
        Fit the k-means clustering.
        """
        # Gather and fit
        all_features = self.comm.gather(self.features, root=0)
        if self.is_master:
            X = np.array([i for sublist in all_features for i in sublist])
            self.kmeans.fit(X)
            gout.emit(f"Computed minimum inertia = {self.kmeans.inertia_}.")
        self.kmeans = self.comm.bcast(self.kmeans, root=0)
        logger.info("Completed kmeans fitting")
        return

    def predict(self) -> None:
        """
        Predict the clusters.
        """
        for xtal, sf in zip(self.structs.values(), self.features):
            label = self.kmeans.predict(sf.reshape(1, -1))
            xtal.info[self.cluster_name] = label
            if self.save_info:
                distance = self.kmeans.transform(sf.reshape(1, -1)).min()
                distance = float(distance)
                xtal.info[self.cluster_name + "_dist"] = distance
        logger.info("Completed predicting clusters")
        return

    def finalize(self) -> None:
        """
        Finalize the k-means clustering.
        """
        logger.info("Completed kmeans clustering")
        gout.emit("Completed kmeans clustering.\n")
        gout.emit("")
