"""
Abstract base class for energy calculators.

This module provides the base class for implementing energy calculators.

It supports a GPU worker/feeder pattern: when running with more MPI ranks
than GPUs, only a subset of ranks (workers) load models onto GPUs.
The remaining ranks (feeders) send structures to workers via MPI.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import abc
import logging
from collections import deque
from typing import Any, Callable, Optional

from mpi4py import MPI
from ase import Atoms

from gnrs.core.gpu import GPUDeviceManager, TAG_WORK_DATA, TAG_WORK_RESULT, TAG_SHUTDOWN

logger = logging.getLogger("EnergyCalculator")


class EnergyCalculatorABC(abc.ABC):
    """
    Abstract base class for energy calculators.

    Supports two execution modes:
    1. Direct mode (ranks <= GPUs or CPU-only calculators):
       Every rank loads the model and computes locally.

    2. Worker/feeder mode (ranks > GPUs, GPU-based calculators):
       Worker ranks (one per GPU) load models. Feeder ranks send
       structures to their assigned worker and receive results back.
    """

    requires_gpu: bool = False

    def __init__(self, comm: MPI.Comm, task_settings: dict, energy_name: str) -> None:
        """
        Initialize the energy calculations.

        Args:
            comm: MPI communicator
            task_settings: Task settings
            energy_name: Energy name
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_master = True if self.rank == 0 else False
        self.tsk_set = task_settings
        self.energy_name = energy_name
        self.calc = None

        self._gpu_mgr = None
        self._use_worker_feeder = False

        if self.requires_gpu:
            max_workers = task_settings.get("max_workers_per_gpu", 1)
            self._gpu_mgr = GPUDeviceManager(
                comm, max_workers_per_gpu=max_workers,
            )
            self._use_worker_feeder = self._gpu_mgr.num_feeders > 0

    @property
    def device(self) -> str:
        """
        Torch device string for this rank.
        """
        if self._gpu_mgr is not None:
            return self._gpu_mgr.device
        return "cpu"

    def run(self, xtal: Atoms) -> None:
        """
        Run the energy calculation on a single structure (direct mode only).

        Args:
            xtal: Crystal structure
        """
        if self.energy_name in xtal.info:
            return
        self.initialize()
        self.compute(xtal)
        self.finalize()

    def run_batch(
        self,
        structs: dict[str, Atoms],
        on_structure_done: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Run energy calculations on a batch of structures.

        Args:
            structs: structure dictionary
            on_structure_done: used for checkpoint saves
        """
        if not self._use_worker_feeder:
            for xtal in structs.values():
                self.run(xtal)
                if on_structure_done is not None:
                    on_structure_done()
        elif self._gpu_mgr.is_worker:
            self._worker_loop(structs, on_structure_done)
        else:
            self._feeder_loop(structs, on_structure_done)

        self.comm.Barrier()

    def _worker_loop(
        self,
        local_structs: dict[str, Atoms],
        on_structure_done: Optional[Callable[[], None]],
    ) -> None:
        """
        GPU worker: interleave local computation with feeder requests
        """
        my_feeders = set()
        for feeder_rank in self._gpu_mgr.feeder_ranks:
            feeder_index = feeder_rank - self._gpu_mgr.num_workers
            if feeder_index % self._gpu_mgr.num_workers == self.rank:
                my_feeders.add(feeder_rank)

        local_queue: deque[Atoms] = deque(
            xtal for xtal in local_structs.values()
            if self.energy_name not in xtal.info
        )

        while local_queue or my_feeders:
            served = self._drain_feeder_requests(my_feeders)

            if local_queue:
                xtal = local_queue.popleft()
                self.run(xtal)
                if on_structure_done is not None:
                    on_structure_done()
                continue

            if my_feeders and not served:
                status = MPI.Status()
                data = self.comm.recv(
                    source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status,
                )
                self._handle_worker_msg(
                    data, status.Get_source(), status.Get_tag(), my_feeders,
                )

    def _drain_feeder_requests(self, active_feeders: set[int]) -> bool:
        """
        Non-blocking: serve all pending feeder messages

        Returns:
            True if at least one message was processed
        """
        served_any = False
        while True:
            status = MPI.Status()
            has_msg = self.comm.iprobe(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status,
            )
            if not has_msg:
                break
            data = self.comm.recv(source=status.Get_source(), tag=status.Get_tag())
            self._handle_worker_msg(
                data, status.Get_source(), status.Get_tag(), active_feeders,
            )
            served_any = True
        return served_any

    def _handle_worker_msg(
        self, data: Any, source: int, tag: int, active_feeders: set[int],
    ) -> None:
        """
        Process a single message received by a worker
        """
        if tag == TAG_SHUTDOWN:
            active_feeders.discard(source)
            return

        if tag == TAG_WORK_DATA:
            name, xtal = data
            self.initialize()
            self.compute(xtal)
            self.finalize()
            energy = xtal.info.get(self.energy_name, 0)
            self.comm.send((name, energy), dest=source, tag=TAG_WORK_RESULT)

    def _feeder_loop(
        self,
        local_structs: dict[str, Atoms],
        on_structure_done: Optional[Callable[[], None]],
    ) -> None:
        """
        CPU feeder: delegate GPU computation to assigned worker
        """
        worker = self._gpu_mgr.assigned_worker()

        for name, xtal in local_structs.items():
            if self.energy_name in xtal.info:
                continue
            self.comm.send((name, xtal), dest=worker, tag=TAG_WORK_DATA)
            _, energy = self.comm.recv(source=worker, tag=TAG_WORK_RESULT)
            xtal.info[self.energy_name] = energy
            if on_structure_done is not None:
                on_structure_done()

        self.comm.send(None, dest=worker, tag=TAG_SHUTDOWN)

    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initialize the energy calculations.
        """
        pass

    def get_calculator(self) -> Any:
        """
        Returns the calculator.
        """
        return self.calc

    @abc.abstractmethod
    def compute(self, xtal: Atoms) -> None:
        """
        Compute the energy.

        Args:
            xtal: Crystal structure
        """
        pass

    @abc.abstractmethod
    def finalize(self) -> None:
        """
        Finalize the energy calculations.
        """
        pass
