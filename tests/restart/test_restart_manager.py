"""
Unit tests for the restart manager. Run in a single process (no mpirun).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from gnrs.core.restart import (
    RESTART_VERSION,
    Restart,
    RestartError,
    _diff_configs,
    _json_default,
    _remap_paths,
)

WORKFLOW = ["generation", "symm_rigid_press"]


def _manager(
    tmp_path: Path,
    current_tasks: list[str] = WORKFLOW,
    completed: list[str] = (),
    config: dict | None = None,
) -> Restart:
    config = {"workflow": {"tasks": current_tasks}, **(config or {})}
    gnrs_info = {"work_dir": str(tmp_path)}
    gnrs_info.update({task: {"status": "completed"} for task in completed})
    return Restart(MPI.COMM_WORLD, config, gnrs_info)


def _saved(tasks: list[str] = WORKFLOW, **sections: dict) -> dict:
    return {"workflow": {"tasks": tasks}, **sections}


# --- task list -------------------------------------------------------------


def test_task_list_unchanged_is_fine(tmp_path: Path) -> None:
    manager = _manager(tmp_path, completed=["generation"])
    completed = manager._check_task_list(_saved())
    assert [s.instance_id for s in completed] == ["generation"]


def test_appending_tasks_is_fine(tmp_path: Path) -> None:
    manager = _manager(tmp_path, WORKFLOW + ["dedup"], completed=WORKFLOW)
    manager._check_task_list(_saved())


def test_appending_duplicate_task_type_is_refused(tmp_path: Path) -> None:
    # Duplicates are renumbered, so the completed symm_rigid_press would be
    # looked up as symm_rigid_press_1 and rerun
    manager = _manager(tmp_path, WORKFLOW + ["symm_rigid_press"], completed=WORKFLOW)
    with pytest.raises(RestartError, match="'symm_rigid_press' \\(position 2\\)"):
        manager._check_task_list(_saved())


def test_changing_pending_tasks_is_fine(tmp_path: Path) -> None:
    manager = _manager(tmp_path, completed=["generation"])
    manager._check_task_list(_saved(["generation", "rigid_press"]))


def test_inserting_before_completed_task_is_refused(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path, ["generation", "dedup", "symm_rigid_press"], completed=WORKFLOW
    )
    with pytest.raises(RestartError, match="task list changed"):
        manager._check_task_list(_saved())


def test_removing_completed_task_is_refused(tmp_path: Path) -> None:
    manager = _manager(tmp_path, ["symm_rigid_press"], completed=WORKFLOW)
    with pytest.raises(RestartError, match="task list changed"):
        manager._check_task_list(_saved())


# --- config ----------------------------------------------------------------


def test_changing_settings_of_pending_task_is_allowed(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path, completed=["generation"],
        config={"generation": {"sr": 0.95}, "symm_rigid_press": {"sr": 0.9}},
    )
    saved = _saved(generation={"sr": 0.95}, symm_rigid_press={"sr": 0.85})
    completed = manager._check_task_list(saved)
    manager._check_config(saved, completed)


def test_changing_settings_of_completed_task_is_refused(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path, completed=["generation"], config={"generation": {"sr": 0.9}}
    )
    saved = _saved(generation={"sr": 0.95})
    completed = manager._check_task_list(saved)
    with pytest.raises(RestartError, match="generation.sr: 0.95 -> 0.9"):
        manager._check_config(saved, completed)


def test_removing_section_of_completed_task_is_refused(tmp_path: Path) -> None:
    manager = _manager(tmp_path, completed=["generation"])
    saved = _saved(generation={"sr": 0.95})
    completed = manager._check_task_list(saved)
    with pytest.raises(RestartError, match="already completed tasks changed"):
        manager._check_config(saved, completed)


def test_instance_overrides_of_completed_task_are_frozen(tmp_path: Path) -> None:
    tasks = ["generation", "dedup", "dedup"]
    manager = _manager(
        tmp_path, tasks, completed=["generation", "dedup_1"],
        config={"dedup": {"tol": 0.1}, "dedup_1": {"tol": 0.3}},
    )
    saved = _saved(tasks, dedup={"tol": 0.1}, dedup_1={"tol": 0.2})
    completed = manager._check_task_list(saved)
    with pytest.raises(RestartError, match="dedup_1.tol: 0.2 -> 0.3"):
        manager._check_config(saved, completed)


# --- file round trip -------------------------------------------------------


def test_write_then_load_round_trip(tmp_path: Path) -> None:
    manager = _manager(tmp_path, completed=["generation"])
    manager.gnrs_info["energy_list"] = [np.float64(1.5)]
    manager.write()

    fresh = _manager(tmp_path)
    assert fresh.load() is True
    assert fresh.is_task_completed("generation")
    assert fresh.gnrs_info["energy_list"] == [1.5]
    assert not (tmp_path / "restart.json.tmp").exists()


def test_load_without_file_returns_false(tmp_path: Path) -> None:
    assert _manager(tmp_path).load() is False


def test_load_rejects_newer_format(tmp_path: Path) -> None:
    (tmp_path / "restart.json").write_text(
        json.dumps({"version": RESTART_VERSION + 1, "config": {}, "gnrs_info": {}})
    )
    with pytest.raises(RestartError, match="newer Genarris"):
        _manager(tmp_path).load()


def test_load_accepts_file_without_version(tmp_path: Path) -> None:
    (tmp_path / "restart.json").write_text(
        json.dumps({"config": _saved(), "gnrs_info": {}})
    )
    assert _manager(tmp_path).load() is True


def test_load_rejects_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "restart.json").write_text("{not json")
    with pytest.raises(RestartError, match="corrupted"):
        _manager(tmp_path).load()


def test_unserializable_value_is_refused_not_stringified() -> None:
    assert _json_default(np.int64(3)) == 3
    assert _json_default(np.array([1, 2])) == [1, 2]
    with pytest.raises(TypeError, match="cannot be stored"):
        _json_default(Path("/x"))


# --- helpers ---------------------------------------------------------------


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
