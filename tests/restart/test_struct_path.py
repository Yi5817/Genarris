"""
``[<optimizer>] struct_path`` feeds the first task in the workflow that uses
the optimizer; later ones continue from the previous task's output. Decided
from the task list, so it also holds on a restart that skips the first task.
Run in a single process (no mpirun).
"""
from __future__ import annotations

import pytest
from mpi4py import MPI

from gnrs.optimize.optimization_task import GeometryOptimizationTask

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() != 1, reason="single-process unit test"
)


def _task(tasks: list[str], instance_id: str) -> GeometryOptimizationTask:
    config = {"workflow": {"tasks": tasks}, "bfgs": {"struct_path": "in.json"}}
    task = GeometryOptimizationTask(
        MPI.COMM_WORLD, config, {}, "bfgs", "maceoff", instance_id=instance_id
    )
    task._active_instance_id = instance_id
    return task


def test_first_task_using_the_optimizer_reads_struct_path() -> None:
    tasks = ["bfgs_maceoff", "maceoff", "bfgs_maceoff"]
    assert _task(tasks, "bfgs_maceoff_1")._reads_struct_path()
    assert not _task(tasks, "bfgs_maceoff_2")._reads_struct_path()
    assert not _task(["bfgs_maceoff", "bfgs_uma"], "bfgs_uma")._reads_struct_path()
    assert _task(["generation", "bfgs_maceoff"], "bfgs_maceoff")._reads_struct_path()


def test_struct_path_is_not_consumed() -> None:
    task = _task(["bfgs_maceoff"], "bfgs_maceoff")
    assert task._reads_struct_path()
    assert task.config["bfgs"] == {"struct_path": "in.json"}
