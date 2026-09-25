"""
Unit tests for the restart manager. Run in a single process (no mpirun).
"""
from __future__ import annotations

import pytest

from gnrs.core.restart import Restart, RestartError, _diff_configs, _remap_paths


def _manager(current_tasks: list[str], completed: list[str]) -> Restart:
    manager = Restart()
    manager.config = {"workflow": {"tasks": current_tasks}}
    manager.gnrs_info = {task: {"status": "completed"} for task in completed}
    return manager


def test_task_list_unchanged_is_fine() -> None:
    saved = {"workflow": {"tasks": ["generation", "symm_rigid_press"]}}
    _manager(["generation", "symm_rigid_press"], ["generation"])._check_task_list(saved)


def test_appending_tasks_is_fine() -> None:
    saved = {"workflow": {"tasks": ["generation", "symm_rigid_press"]}}
    manager = _manager(
        ["generation", "symm_rigid_press", "dedup"],
        ["generation", "symm_rigid_press"],
    )
    manager._check_task_list(saved)


def test_appending_duplicate_task_type_is_refused() -> None:
    # Duplicates are renumbered, so the completed symm_rigid_press would be
    # looked up as symm_rigid_press_1 and rerun
    saved = {"workflow": {"tasks": ["generation", "symm_rigid_press"]}}
    manager = _manager(
        ["generation", "symm_rigid_press", "symm_rigid_press"],
        ["generation", "symm_rigid_press"],
    )
    with pytest.raises(RestartError, match="task list changed"):
        manager._check_task_list(saved)


def test_changing_pending_tasks_is_fine() -> None:
    saved = {"workflow": {"tasks": ["generation", "rigid_press"]}}
    _manager(["generation", "symm_rigid_press"], ["generation"])._check_task_list(saved)


def test_inserting_before_completed_task_is_refused() -> None:
    saved = {"workflow": {"tasks": ["generation", "symm_rigid_press"]}}
    manager = _manager(
        ["generation", "dedup", "symm_rigid_press"],
        ["generation", "symm_rigid_press"],
    )
    with pytest.raises(RestartError, match="task list changed"):
        manager._check_task_list(saved)


def test_removing_completed_task_is_refused() -> None:
    saved = {"workflow": {"tasks": ["generation", "symm_rigid_press"]}}
    manager = _manager(["symm_rigid_press"], ["generation", "symm_rigid_press"])
    with pytest.raises(RestartError, match="task list changed"):
        manager._check_task_list(saved)


def test_remap_paths_requires_path_boundary() -> None:
    data = {"a": "/data/run/x", "b": ["/data/run_backup/y", "/data/run"], "c": 1}
    assert _remap_paths(data, "/data/run", "/new") == {
        "a": "/new/x",
        "b": ["/data/run_backup/y", "/new"],
        "c": 1,
    }


def test_diff_configs_reports_nested_changes() -> None:
    saved = {"master": {"z": 2, "name": "a"}, "gen": {"n": 1}}
    current = {"master": {"z": 4, "name": "a"}, "gen": {"n": 1}, "new": {"k": 0}}
    assert _diff_configs(saved, current) == [
        "master.z: 2 -> 4",
        "new: '<not set>' -> {'k': 0}",
    ]
