"""
GPU device management for MPI-parallel workloads.

Implements a worker/feeder pattern where only a subset of MPI ranks
(GPU workers) load models onto GPUs, while the remaining ranks (feeders)
send structures to workers via MPI for computation. This avoids GPU OOM
when running with many MPI ranks and few GPUs.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import logging
from typing import Optional

import torch
from mpi4py import MPI

logger = logging.getLogger("gpu")

# MPI tags for worker/feeder communication
TAG_WORK_REQUEST = 100
TAG_WORK_DATA = 101
TAG_WORK_RESULT = 102
TAG_SHUTDOWN = 103


class GPUDeviceManager:
    """
    Manages GPU device allocation across MPI ranks.

    Partitions ranks into GPU workers and CPU feeders. Workers are assigned
    to GPUs on their own node. Feeders offload computation to workers via
    MPI. If no rank has a GPU, every rank is a worker computing on CPU.

    Typical usage in HPC:
        - 1 GPU node with 1-4 GPUs, 32-128 CPU cores
        - Workers: 1 per GPU (or configurable)
        - Feeders: all remaining ranks
    """

    def __init__(
        self,
        comm: MPI.Comm,
        max_workers_per_gpu: int = 1,
    ) -> None:
        """
        Initialize GPU device manager.

        Args:
            comm: MPI communicator.
            max_workers_per_gpu: Maximum number of worker ranks per GPU.
        """
        if max_workers_per_gpu < 1:
            raise ValueError(
                f"max_workers_per_gpu must be >= 1, got {max_workers_per_gpu}"
            )

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.max_workers_per_gpu = max_workers_per_gpu

        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.local_rank = node_comm.Get_rank()
        self.local_size = node_comm.Get_size()
        node_comm.Free()

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Assign GPUs by node-local rank: the first
        # num_gpus * max_workers_per_gpu ranks on each node become workers.
        self.gpu_id: Optional[int] = None
        if self.num_gpus > 0:
            if self.local_rank < self.num_gpus * self.max_workers_per_gpu:
                self.gpu_id = self.local_rank % self.num_gpus

        # Share worker flags so all ranks agree on the global partition,
        # even when nodes have different GPU counts.
        worker_flags = comm.allgather(self.gpu_id is not None)
        if not any(worker_flags):
            worker_flags = [True] * self.size

        self._worker_ranks = [r for r, flag in enumerate(worker_flags) if flag]
        self._feeder_ranks = [r for r, flag in enumerate(worker_flags) if not flag]
        self._is_worker = worker_flags[self.rank]

        if self.gpu_id is not None:
            # Pin this rank to its GPU and expose the plain "cuda" device
            # string: some calculators reject "cuda:N" and accept only
            # "cuda" or "cpu". With the device pinned, "cuda" resolves to
            # the assigned GPU.
            torch.cuda.set_device(self.gpu_id)
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info(
            f"GPU Device Manager: rank={self.rank} local_rank={self.local_rank} "
            f"gpus={self.num_gpus} gpu_id={self.gpu_id} "
            f"workers={self.num_workers} feeders={self.num_feeders}"
        )

    @property
    def device(self) -> str:
        """
        The torch device string for this rank ("cuda" or "cpu").
        """
        return self._device

    @property
    def is_worker(self) -> bool:
        """
        Whether this rank is a GPU worker.
        """
        return self._is_worker

    @property
    def is_feeder(self) -> bool:
        """
        Whether this rank is a CPU feeder.
        """
        return not self._is_worker

    @property
    def num_workers(self) -> int:
        """
        Total number of GPU worker ranks.
        """
        return len(self._worker_ranks)

    @property
    def num_feeders(self) -> int:
        """
        Total number of CPU feeder ranks.
        """
        return len(self._feeder_ranks)

    @property
    def worker_ranks(self) -> list[int]:
        """
        List of all worker rank IDs.
        """
        return list(self._worker_ranks)

    @property
    def feeder_ranks(self) -> list[int]:
        """
        List of all feeder rank IDs.
        """
        return list(self._feeder_ranks)

    def assigned_worker(self) -> int:
        """
        Return the worker rank this feeder is assigned to (round-robin).

        Returns:
            Worker rank ID
        """
        if self._is_worker:
            return self.rank
        feeder_index = self._feeder_ranks.index(self.rank)
        return self._worker_ranks[feeder_index % len(self._worker_ranks)]

    def assigned_feeders(self) -> list[int]:
        """
        Return the feeder ranks assigned to this worker (round-robin).

        Returns:
            Feeder rank IDs; empty if this rank is a feeder.
        """
        if not self._is_worker:
            return []
        worker_index = self._worker_ranks.index(self.rank)
        return [
            feeder_rank
            for i, feeder_rank in enumerate(self._feeder_ranks)
            if i % len(self._worker_ranks) == worker_index
        ]
