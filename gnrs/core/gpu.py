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

import gnrs.output as gout

logger = logging.getLogger("GPUManager")

# MPI tags for worker/feeder communication
TAG_WORK_REQUEST = 100
TAG_WORK_DATA = 101
TAG_WORK_RESULT = 102
TAG_SHUTDOWN = 103


class GPUDeviceManager:
    """
    Manages GPU device allocation across MPI ranks.

    Partitions ranks into GPU workers and CPU feeders. Workers are assigned
    to GPUs. Feeders offload computation to workers via MPI.

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
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.max_workers_per_gpu = max_workers_per_gpu

        self._num_workers = min(
            self.num_gpus * self.max_workers_per_gpu,
            self.size,
        )
        if self.num_gpus == 0:
            self._num_workers = self.size

        self._is_worker = self.rank < self._num_workers
        self._device: Optional[str] = None
        self._assign_device()

        logger.debug(
            f"Rank {self.rank}: role={self._is_worker} device={self._device} (gpus={self.num_gpus} workers={self._num_workers} feeders={self.size - self._num_workers})"
        )

    def _assign_device(self) -> None:
        """
        Assign a CUDA device to worker ranks, CPU to feeders.
        """
        if not self._is_worker:
            self._device = "cpu"
            return

        if self.num_gpus == 0:
            self._device = "cpu"
            return

        gpu_id = self.rank % self.num_gpus
        self._device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

    @property
    def device(self) -> str:
        """
        The torch device string for this rank.
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
        return self._num_workers

    @property
    def num_feeders(self) -> int:
        """
        Total number of CPU feeder ranks.
        """
        return self.size - self._num_workers

    @property
    def worker_ranks(self) -> list[int]:
        """
        List of all worker rank IDs.
        """
        return list(range(self._num_workers))

    @property
    def feeder_ranks(self) -> list[int]:
        """
        List of all feeder rank IDs.
        """
        return list(range(self._num_workers, self.size))

    def assigned_worker(self) -> int:
        """
        Return the worker rank this feeder is assigned to (round-robin).
        
        Returns:
            Worker rank ID
        """
        if self._is_worker:
            return self.rank
        feeder_index = self.rank - self._num_workers
        return feeder_index % self._num_workers
