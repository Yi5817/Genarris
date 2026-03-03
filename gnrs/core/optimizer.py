"""
Abstract base class for geometry optimization.

This module provides the base class for implementing geometry optimization.

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
from typing import Callable, Optional

import numpy as np
from mpi4py import MPI
from ase.atoms import Atoms

from gnrs.core.gpu import GPUDeviceManager

logger = logging.getLogger("optimizer")

# MPI tags specific to optimization worker/feeder (offset from energy tags)
TAG_OPT_DATA = 200
TAG_OPT_RESULT = 201
TAG_OPT_SHUTDOWN = 202


class GeometryOptimizerABC(abc.ABC):
    """
    Abstract class for Geometry optimization methods.

    Supports three execution modes:
    1. Direct mode (ranks <= GPUs or CPU-only calculators):
    Every rank runs the optimizer locally.

    2. Worker/feeder mode (ranks > GPUs, GPU-based calculators):
    Worker ranks run optimizations on GPU.  Feeder ranks send
    structures to their assigned worker and receive optimized
    results back.

    3. Serial DFT mode:
    Only rank 0 runs optimizations.  Results are broadcast to all
    ranks.  Use when the DFT binary needs the full allocation.

    All optimizers should inherit this class.
    """

    def __init__(
        self, 
        comm: MPI.Comm, 
        task_set: dict,
        opt_name: str = "relax",
        energy_method: str | None = None,
        energy_calc: any | None = None,
        gpu_mgr: GPUDeviceManager | None = None,
        dft_serial_mode: bool = False,
    ) -> None:
        """
        Initialize the geometry optimizer.
        
        Args:
            comm: MPI communicator for parallel computation
            task_set: Optimization settings
            opt_name: Optimizer
            energy_method: Energy calculation method
            energy_calc: ASE calculator
            gpu_mgr: GPU device manager
            dft_serial_mode: If True, only rank 0 runs optimizations and
            results are broadcast to all ranks.
        """
        self.opt_name = opt_name
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_master = self.rank == 0
        self.tsk_set = task_set
        self.energy_method = energy_method
        self.energy_calc = energy_calc
        self.converged = False
        self._gpu_mgr = gpu_mgr
        self._use_worker_feeder = (
            gpu_mgr is not None and gpu_mgr.num_feeders > 0
        )
        self._dft_serial_mode = dft_serial_mode

    def run(self, xtal: Atoms) -> None:
        """
        Run the optimization workflow.
        
        1. Initialize
        2. Perform optimization
        3. Update structure information
        4. Finalize
        
        Args:
            xtal: ASE Atoms object representing the crystal structure
        """

        self.initialize()
        self.optimize(xtal)
        self.update(xtal)
        self.finalize(xtal)

    def run_batch(
        self,
        structs: dict[str, Atoms],
        on_structure_done: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Run optimization on a batch of structures.

        Args:
            structs: structure dictionary
            on_structure_done: used for checkpoint saves
        """
        if self._dft_serial_mode:
            self._serial_dft_batch(structs, on_structure_done)
        elif not self._use_worker_feeder:
            for xtal in structs.values():
                self.run(xtal)
                if on_structure_done is not None:
                    on_structure_done()
        elif self._gpu_mgr.is_worker:
            self._worker_loop(structs, on_structure_done)
        else:
            self._feeder_loop(structs, on_structure_done)

        self.comm.Barrier()

    def _serial_dft_batch(
        self,
        structs: dict[str, Atoms],
        on_structure_done: Optional[Callable[[], None]],
    ) -> None:
        """
        Serial DFT mode: only rank 0 runs optimizations.
        """
        local_items = list(structs.items())
        all_items = self.comm.gather(local_items, root=0)

        results: dict[str, tuple[dict, np.ndarray, np.ndarray]] | None = None
        if self.is_master:
            flat = [(n, x) for rank_items in all_items for n, x in rank_items]
            logger.info(
                "dft_mode=serial: rank 0 optimizing %d structures",
                len(flat),
            )
            results = {}
            for name, xtal in flat:
                self.run(xtal)
                results[name] = (
                    xtal.info.copy(),
                    np.array(xtal.positions),
                    np.array(xtal.cell),
                )
                if on_structure_done is not None:
                    on_structure_done()

        results = self.comm.bcast(results, root=0)

        for name, xtal in structs.items():
            if name in results:
                info, positions, cell = results[name]
                xtal.info.update(info)
                xtal.positions = positions
                xtal.cell = cell

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
            if self.opt_name not in xtal.info
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
            data = self.comm.recv(
                source=status.Get_source(), tag=status.Get_tag(),
            )
            self._handle_worker_msg(
                data, status.Get_source(), status.Get_tag(), active_feeders,
            )
            served_any = True
        return served_any

    def _handle_worker_msg(
        self,
        data: object,
        source: int,
        tag: int,
        active_feeders: set[int],
    ) -> None:
        """
        Process a single message received by a worker from a feeder.
        """
        if tag == TAG_OPT_SHUTDOWN:
            active_feeders.discard(source)
            return

        if tag == TAG_OPT_DATA:
            name, xtal = data
            self.run(xtal)
            result = (
                name,
                xtal.info.copy(),
                np.array(xtal.positions),
                np.array(xtal.cell),
            )
            self.comm.send(result, dest=source, tag=TAG_OPT_RESULT)

    def _feeder_loop(
        self,
        local_structs: dict[str, Atoms],
        on_structure_done: Optional[Callable[[], None]],
    ) -> None:
        """
        CPU feeder: delegate optimization to assigned GPU worker.
        """
        worker = self._gpu_mgr.assigned_worker()

        for name, xtal in local_structs.items():
            if self.opt_name in xtal.info:
                continue
            self.comm.send((name, xtal), dest=worker, tag=TAG_OPT_DATA)
            _, info, positions, cell = self.comm.recv(
                source=worker, tag=TAG_OPT_RESULT,
            )
            xtal.info.update(info)
            xtal.positions = positions
            xtal.cell = cell
            if on_structure_done is not None:
                on_structure_done()

        self.comm.send(None, dest=worker, tag=TAG_OPT_SHUTDOWN)

    def initialize(self) -> None:
        """
        Initialize for optimization.
        """
        pass

    @abc.abstractmethod
    def optimize(self, xtal: Atoms) -> None:
        """
        Perform optimization.
        
        Args:
            xtal: ASE Atoms object
        """
        pass

    @abc.abstractmethod
    def update(self, xtal: Atoms) -> None:
        """
        Update the geometry and add energy information.
        
        Args:
            xtal: ASE Atoms object
        """
        pass

    def finalize(self, xtal: Atoms) -> None:
        """
        Finalize the optimization and clean up.
        
        Args:
            xtal: ASE Atoms object
        """
        xtal.calc = None
