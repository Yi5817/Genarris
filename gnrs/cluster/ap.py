"""
This module provides the AP clustering algorithm.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import numpy as np
import logging

from mpi4py import MPI
from bisect import bisect_left
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import euclidean_distances

import gnrs.output as gout
from gnrs.core.cluster import ClusterABC

logger = logging.getLogger("AP")


class APCluster(ClusterABC):
    """
    Affinity Propagation clustering implementation.

    This class implements the Affinity Propagation clustering algorithm with
    support for distributed computation using MPI.
    """

    def __init__(self, comm: MPI.Comm, task_settings: dict) -> None:
        """
        Initialize the APCluster object.

        Args:
            comm: MPI communicator
            task_settings: Task settings for AP clustering
        """
        super().__init__(comm, task_settings)

        self.result = None
        self.cluster_method = "ap"
        self.feature_name = self.tsk_set["feature_name"]
        # AP parameters
        self.damping = self.tsk_set.get("damping", 0.5)
        self.max_iter = self.tsk_set.get("max_iter", 200)
        self.convergence_iter = self.tsk_set.get("convergence_iter", 15)
        self.preference_range = self.tsk_set.get("preference_range", "quantile")
        self.max_sampling_attempts = self.tsk_set.get("max_sampling_attempts", 10)
        # Clustering parameters
        self.n_clusters = self.tsk_set.get("n_clusters")
        self.clusters_tol = self.tsk_set.get("clusters_tol", 0.5)
        self.debug_mode = self.tsk_set.get("debug_mode", False)
        # Files
        self.ids_file = self.tsk_set.get("ids_file", "ids.dat")
        self.simmat_file = self.tsk_set.get("simmat_file", "similarity_matrix.dat")
        self.feature_file = self.tsk_set.get("feature_file", "acsf.dat")
        # Internal variables
        self.preference = None
        self.success_rank = None
        self.final_n_clusters = None
        # AP model
        self.ap_model = None
        self.max_ap_attempts = self.tsk_set.get("max_ap_attempts", 10)

    def initialize(self) -> None:
        """
        Initialize the clustering by gathering crystal structures and computing similarity matrix.

        This method:
        1. Gathers structures from all processes
        2. Computes or loads feature vectors
        3. Computes or loads similarity matrix
        4. Sets up preference range for AP clustering
        5. Initializes the AP model
        """
        local_features = np.array(
            [xtal.info[self.feature_name][0, :] for xtal in self.structs.values()]
        )
        local_ids = list(self.structs.keys())
        
        all_features = self.comm.gather(local_features, root=0)
        all_ids = self.comm.gather(local_ids, root=0)
        if self.is_master:
            all_features = np.vstack([feat for sublist in all_features if sublist is not None for feat in sublist], dtype=np.float32)
            n_samples, n_features = all_features.shape
            if not os.path.exists(self.feature_file):
                gout.emit(f"Creating feature file: {self.feature_file}")
                features = np.memmap(
                    self.feature_file,
                    dtype="float32",
                    mode="w+",
                    shape=(n_samples, n_features),
                )
                features[:] = all_features[:]
                features.flush()
            else:
                gout.emit(f"Loading feature file: {self.feature_file}")
                features = np.memmap(
                    self.feature_file,
                    dtype="float32",
                    mode="r",
                    shape=(n_samples, n_features),
                )
            del all_features

            if not os.path.exists(self.simmat_file):
                gout.emit(f"Creating similarity matrix file: {self.simmat_file}")
                sim_mat = -euclidean_distances(features, squared=True)
                sim_mat_file = np.memmap(
                    self.simmat_file,
                    dtype="float32",
                    mode="w+",
                    shape=(n_samples, n_samples),
                )
                sim_mat_file[:] = sim_mat[:]
                sim_mat_file.flush()
                self.sim_mat = sim_mat_file
                del sim_mat
            else:
                gout.emit(f"Loading similarity matrix file: {self.simmat_file}")
                self.sim_mat = np.memmap(
                    self.simmat_file,
                    dtype="float32",
                    mode="r",
                    shape=(n_samples, n_samples),
                )
            del features

            ids_array = np.vstack([
                _id for sublist in all_ids if sublist is not None for _id in sublist
            ])
            if not os.path.exists(self.ids_file):
                gout.emit(f"Creating IDs file: {self.ids_file}")
                self.ids = np.memmap(
                    self.ids_file, dtype="<U15", mode="w+", shape=ids_array.shape
                )
                self.ids[:] = ids_array[:]
                self.ids.flush()
            else:
                gout.emit(f"Loading IDs file: {self.ids_file}")
                self.ids = np.memmap(
                    self.ids_file,
                    dtype="<U15",
                    mode="r",
                    shape=(n_samples,)
                )
            del ids_array
            del all_ids
        else:
            self.ids = None
            self.sim_mat = None
            n_samples = None
            
        n_samples = self.comm.bcast(n_samples, root=0)
        if not self.is_master:
            self.sim_mat = np.memmap(
                self.simmat_file,
                dtype="float32",
                mode="r",
                shape=(n_samples, n_samples),
            )
            self.ids = np.memmap(
                self.ids_file, dtype="<U15", mode="r", shape=(n_samples,)
            )

        if self.preference_range == "quantile":
            self.preference_range = [
                np.quantile(self.sim_mat, 0.05),
                np.quantile(self.sim_mat, 0.95),
            ]
            gout.emit(f"Using quantile preference range: {self.preference_range}")
        elif self.preference_range == "mean-median":
            self.preference_range = sorted(
                [np.mean(self.sim_mat), np.median(self.sim_mat)]
            )
            gout.emit(f"Using mean-median preference range: {self.preference_range}")

        self.ap_model = AffinityPropagation(
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            copy=True,
            affinity="precomputed",
        )

        logger.info(f"Started AP clustering")
        gout.emit(f"Running AP clustering targeting {self.n_clusters} clusters")

    def fit(self) -> None:
        """
        Fit the clustering model to the data.
        """
        n_attempts = 0
        pref_range = self.preference_range

        while n_attempts < self.max_sampling_attempts:
            gout.emit(
                f"Beginning attempt {n_attempts} with preference range [{pref_range[0]:.4f}, {pref_range[-1]:.4f}] on {self.size} processes"
            )

            pref = round(
                pref_range[-1]
                + float(pref_range[0] - pref_range[-1])
                * float(self.rank + 1)
                / float(self.size + 1),
                4
            )

            try:
                result = self._affinity_propagation(pref, pref_range)
            except StopIteration:
                continue

            n_clusters_pref = self.comm.gather(
                [result["n_clusters"], result["preference"]], root=0
            )
            if self.is_master:
                converged_result = self.check_convergence(
                    n_clusters_pref, pref_range, n_attempts
                )
            else:
                converged_result = None
            del n_clusters_pref
            converged_result = self.comm.bcast(converged_result, root=0)

            if converged_result["converged"]:
                if converged_result["rank"] == self.rank:
                    self.result = result
                break
            else:
                pref_range = converged_result["new_pref_range"]
            gout.emit(
                f"Attempt {n_attempts}: Preference range updated: [{pref_range[0]:.4f}, {pref_range[-1]:.4f}]"
            )
            n_attempts += 1

        if converged_result["converged"]:
            gout.emit(f"Affinity Propagation with fixed number of clusters succeeded!")
        else:
            gout.emit(
                f"Failed to cluster to {self.n_clusters} clusters with tolerance {self.clusters_tol}"
            )

        self.result = self.comm.bcast(result, root=converged_result["rank"])
        n_clusters_done = self.result["n_clusters"]
        gout.emit(f"Affinity Propagation completed with {n_clusters_done} clusters")
        logger.info(f"Completed AP fitting with {n_clusters_done} clusters")

    def predict(self) -> int:
        """
        Assign cluster labels to structures.

        Returns:
            int: Final number of clusters
        """
        # Assign cluster labels to structures
        for ids, xtal in self.structs.items():
            label = self.result["assigned_cluster"][ids]
            if ids in self.result["exemplar_ids"]:
                xtal.info[self.cluster_method] = str(label) + "_center"
            else:
                xtal.info[self.cluster_method] = str(label)

        logger.info("Completed predicting clusters")
        return self.result["n_clusters"]

    def finalize(self) -> None:
        """
        Finalize the clustering process.
        """
        logger.info("Completed AP clustering")
        gout.emit("Completed AP clustering.\n")
        gout.emit("")

    def _affinity_propagation(self, pref: float, pref_range: list) -> dict:
        """
        Run Affinity Propagation clustering with current preference value.

        Returns:
            dict: Clustering results
        """
        for iteration in range(self.max_ap_attempts):
            self.ap_model.preference = pref
            clustering = self.ap_model.fit(self.sim_mat)

            # Check if clustering converged
            if clustering.labels_[0] != -1:
                break

            # If not converged, try a random preference value
            factor = (iteration + 1) / self.max_ap_attempts
            mid_point = (pref_range[0] + pref_range[1]) / 2

            if iteration % 2 == 0:
                new_pref = mid_point + (
                    (pref_range[1] - pref_range[0]) * 0.5 * factor
                )
            else:
                new_pref = mid_point - (
                    (pref_range[1] - pref_range[0]) * 0.5 * factor
                )

            pref = round(new_pref, 4)

            if self.debug_mode:
                print(
                    f"preference: {pref} failed to converge. Trying a random preference: {new_pref}",
                    flush=True
                )

            if iteration == self.max_ap_attempts - 1:
                raise StopIteration()

        cluster_centers = clustering.cluster_centers_indices_
        n_clusters = len(cluster_centers)
        labels = clustering.labels_

        exemplar_indices = np.array(cluster_centers, dtype="int")
        exemplar_ids = [self.ids[idx] for idx in exemplar_indices]

        assigned_cluster = {str(_id): label for _id, label in zip(self.ids, labels)}

        return {
            "n_clusters": n_clusters,
            "assigned_cluster": assigned_cluster,
            "exemplar_ids": exemplar_ids,
            "preference": pref,
        }

    def check_convergence(
        self,
        n_clusters_pref: list,
        pref_range: list,
        n_attempts: int,
    ) -> dict:
        """
        Check if the clustering has converged.

        Args:
            n_clusters_pref: List of [num_clusters, preference] pairs from all processes
            pref_range: List of preference values from all processes
            n_attempts: Current attempt number

        Returns:
            Dictionary containing:
                - converged: Whether a suitable clustering was found
                - rank: Rank with suitable clustering
                - pref: preference used for clustering
                - n_cluster: Number of clusters found
                - new_pref_range: New preference range to try
        """
        n_clusters, pref_list = map(list, zip(*n_clusters_pref))
        sorted_clusters_with_indices = sorted(
            (clu, idx) for idx, clu in enumerate(n_clusters)
        )
        sorted_n_clusters = [clu for clu, _ in sorted_clusters_with_indices]

        # Find the insertion point
        idx = bisect_left(sorted_n_clusters, self.n_clusters)
        candidates = []
        if idx > 0:
            candidates.append(sorted_clusters_with_indices[idx - 1])
        if idx < len(n_clusters):
            candidates.append(sorted_clusters_with_indices[idx])

        # Find the closest cluster
        if not candidates:
            closest_clu, closest_idx = None, None
        else:
            closest_clu, closest_idx = min(
                candidates, key=lambda x: abs(x[0] - self.n_clusters)
            )

        is_within_tolerance = (
            self.clusters_tol is not None
            and closest_clu is not None
            and abs(closest_clu - self.n_clusters) <= self.clusters_tol
        )

        # Return success if we found a good solution or reached max iterations
        if is_within_tolerance or n_attempts == self.max_sampling_attempts - 1:
            return {
                "converged": True,
                "rank": closest_idx,
                "pref": pref_list[closest_idx],
                "n_cluster": closest_clu,
                "new_pref_range": None,
            }

        if idx == 0:
            # Target clusters is lower than any value obtained - decrease preference range
            new_pref_range = [
                pref_range[0] - abs(pref_range[0]) * 0.2,
                pref_list[sorted_clusters_with_indices[0][1]]
            ]
        elif idx == len(pref_list):
            # Target clusters is higher than any value obtained - increase preference range
            new_pref_range = [
                pref_list[sorted_clusters_with_indices[-1][1]],
                pref_range[1] + abs(pref_range[1]) * 0.2
            ]
        else:
            # Target clusters is between values obtained
            # Get the preference values on either side of where self.n_clusters would be inserted
            # Find the indices in the original list that correspond to the sorted positions
            new_pref_range = sorted([
                pref_list[sorted_clusters_with_indices[idx - 1][1]],
                pref_list[sorted_clusters_with_indices[idx][1]]
            ])

        return {
            "converged": False,
            "rank": None,
            "pref": None,
            "n_cluster": None,
            "new_pref_range": new_pref_range,
        }
