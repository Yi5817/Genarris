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

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() != 1, reason="single-process unit test"
)

WORKFLOW = ["generation", "symm_rigid_press"]


def _manager(
    tmp_path: Path,
    current_tasks: list[str] = WORKFLOW,
    completed: list[str] = (),
    config: dict | None = None,
) -> Restart:
    config = {"workflow": {"tasks": current_tasks}, **(config or {})}
    gnrs_info = {"work_dir": str(tmp_path), "tmp_dir": str(tmp_path / "tmp")}
    gnrs_info.update({task: {"status": "completed"} for task in completed})
    return Restart(MPI.COMM_WORLD, config, gnrs_info)


def _saved(tasks: list[str] = WORKFLOW, **sections: dict) -> dict:
    return {"workflow": {"tasks": tasks}, **sections}


def _write(
    tmp_path: Path,
    saved: dict,
    gnrs_info: dict | None = None,
    version: int | None = RESTART_VERSION,
) -> None:
    data = {"config": saved, "gnrs_info": gnrs_info or {}}
    if version is not None:
        data["version"] = version
    (tmp_path / "restart.json").write_text(json.dumps(data))


def _apply(manager: Restart, saved: dict, gnrs_info: dict | None = None) -> None:
    """
    Load a restart file holding the saved config into the manager.
    """
    _write(Path(manager.gnrs_info["work_dir"]), saved, gnrs_info)
    assert manager.load()


def _checkpoint(tmp_path: Path, task: str) -> Path:
    ckpt = tmp_path / "tmp" / task / "rank_0" / "0.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("")
    return ckpt


# --- task list -------------------------------------------------------------


def test_task_list_unchanged_is_fine(tmp_path: Path) -> None:
    manager = _manager(tmp_path, completed=["generation"])
    _apply(manager, _saved())
    assert manager.is_task_completed("generation")
    assert not manager.is_task_completed("symm_rigid_press")


def test_appending_tasks_is_fine(tmp_path: Path) -> None:
    manager = _manager(tmp_path, WORKFLOW + ["dedup"], completed=WORKFLOW)
    _apply(manager, _saved())
    assert not manager.is_task_completed("dedup")


def test_appending_duplicate_task_type_renumbers_completed_task(
    tmp_path: Path,
) -> None:
    # Duplicates are renumbered, so the completed symm_rigid_press is now
    # symm_rigid_press_1 and must keep its record under that id
    ckpt = _checkpoint(tmp_path, "symm_rigid_press")
    manager = _manager(tmp_path, WORKFLOW + ["symm_rigid_press"], completed=WORKFLOW)
    _apply(manager, _saved())
    assert manager.is_task_completed("symm_rigid_press_1")
    assert not manager.is_task_completed("symm_rigid_press_2")
    assert "symm_rigid_press" not in manager.gnrs_info
    # Its scratch directory follows it
    assert not ckpt.exists()
    assert (tmp_path / "tmp" / "symm_rigid_press_1" / "rank_0" / "0.ckpt").is_file()


def test_appending_duplicate_of_pending_task_keeps_its_checkpoints(
    tmp_path: Path,
) -> None:
    # The interrupted maceoff becomes maceoff_1 and resumes from its checkpoints
    ckpt = _checkpoint(tmp_path, "maceoff")
    tasks = ["generation", "maceoff", "bfgs_maceoff", "maceoff"]
    manager = _manager(tmp_path, tasks, completed=["generation"])
    _apply(manager, _saved(["generation", "maceoff"]))
    assert not ckpt.exists()
    assert (tmp_path / "tmp" / "maceoff_1" / "rank_0" / "0.ckpt").is_file()


def test_removing_pending_duplicate_task_is_fine(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path, ["generation", "dedup"], completed=["generation", "dedup_1"]
    )
    _apply(manager, _saved(["generation", "dedup", "dedup"]))
    assert manager.is_task_completed("dedup")
    assert "dedup_1" not in manager.gnrs_info


def test_changing_pending_tasks_is_fine(tmp_path: Path) -> None:
    manager = _manager(tmp_path, completed=["generation"])
    _apply(manager, _saved(["generation", "rigid_press"]))


def test_inserting_before_completed_task_is_refused(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path, ["generation", "dedup", "symm_rigid_press"], completed=WORKFLOW
    )
    with pytest.raises(RestartError, match="task list changed"):
        _apply(manager, _saved())


def test_removing_completed_task_is_refused(tmp_path: Path) -> None:
    manager = _manager(tmp_path, ["symm_rigid_press"], completed=WORKFLOW)
    with pytest.raises(RestartError, match="task list changed"):
        _apply(manager, _saved())


def test_invalid_task_list_is_refused_before_touching_anything(
    tmp_path: Path,
) -> None:
    ckpt = _checkpoint(tmp_path, "symm_rigid_press")
    manager = _manager(tmp_path, ["generation", "not_a_task"], completed=["generation"])
    with pytest.raises(RestartError, match="current task list is invalid"):
        _apply(manager, _saved())
    assert ckpt.is_file()


# --- config ----------------------------------------------------------------


def test_changing_settings_of_pending_task_is_allowed(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path, completed=["generation"],
        config={"generation": {"sr": 0.95}, "symm_rigid_press": {"sr": 0.9}},
    )
    _apply(manager, _saved(generation={"sr": 0.95}, symm_rigid_press={"sr": 0.85}))


def test_changing_settings_of_completed_task_is_refused(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path, completed=["generation"], config={"generation": {"sr": 0.9}}
    )
    with pytest.raises(RestartError, match="generation.sr: 0.95 -> 0.9"):
        _apply(manager, _saved(generation={"sr": 0.95}))


def test_removing_section_of_completed_task_is_reported_only(tmp_path: Path) -> None:
    # Nothing the completed task read changed its value, so this is a
    # reported difference, not a refused one
    manager = _manager(tmp_path, completed=["generation"])
    _apply(manager, _saved(generation={"sr": 0.95}))


def test_instance_overrides_of_completed_task_are_frozen(tmp_path: Path) -> None:
    tasks = ["generation", "dedup", "dedup"]
    manager = _manager(
        tmp_path, tasks, completed=["generation", "dedup_1"],
        config={"dedup": {"tol": 0.1}, "dedup_1": {"tol": 0.3}},
    )
    with pytest.raises(RestartError, match="dedup_1.tol: 0.2 -> 0.3"):
        _apply(manager, _saved(tasks, dedup={"tol": 0.1}, dedup_1={"tol": 0.2}))


def test_effective_settings_follow_renumbered_task(tmp_path: Path) -> None:
    # The completed dedup_1 becomes dedup; its override moved into [dedup]
    manager = _manager(
        tmp_path, ["generation", "dedup"], completed=["generation", "dedup_1"],
        config={"dedup": {"tol": 0.2}},
    )
    tasks = ["generation", "dedup", "dedup"]
    _apply(manager, _saved(tasks, dedup={"tol": 0.1}, dedup_1={"tol": 0.2}))


def test_method_sections_of_completed_task_are_frozen(tmp_path: Path) -> None:
    tasks = ["generation", "bfgs_maceoff"]
    manager = _manager(
        tmp_path, tasks, completed=tasks,
        config={"bfgs": {"maxiter": 100}, "maceoff": {"model_size": "small"}},
    )
    saved = _saved(tasks, bfgs={"maxiter": 100}, maceoff={"model_size": "large"})
    with pytest.raises(RestartError, match="maceoff.model_size: 'large' -> 'small'"):
        _apply(manager, saved)


def test_base_setting_masked_by_override_may_change(tmp_path: Path) -> None:
    # The completed bfgs_maceoff_1 never used [bfgs].maxiter, so it may
    # change for the pending bfgs_maceoff_2
    tasks = ["generation", "bfgs_maceoff", "bfgs_maceoff"]
    manager = _manager(
        tmp_path, tasks, completed=["generation", "bfgs_maceoff_1"],
        config={"bfgs": {"maxiter": 300}, "bfgs_maceoff_1": {"maxiter": 100}},
    )
    saved = _saved(tasks, bfgs={"maxiter": 200}, bfgs_maceoff_1={"maxiter": 100})
    _apply(manager, saved)


def test_changed_override_of_method_section_is_reported_once(tmp_path: Path) -> None:
    tasks = ["generation", "bfgs_maceoff", "bfgs_maceoff"]
    manager = _manager(
        tmp_path, tasks, completed=["generation", "bfgs_maceoff_1"],
        config={"bfgs": {"maxiter": 200}, "bfgs_maceoff_1": {"maxiter": 150}},
    )
    saved = _saved(tasks, bfgs={"maxiter": 200}, bfgs_maceoff_1={"maxiter": 100})
    with pytest.raises(RestartError, match="maxiter: 100 -> 150") as info:
        _apply(manager, saved)
    assert str(info.value).count("maxiter") == 1
    assert "bfgs_maceoff_1.maxiter" in str(info.value)


def test_removed_override_falls_back_to_base_setting(tmp_path: Path) -> None:
    tasks = ["generation", "dedup", "dedup"]
    manager = _manager(
        tmp_path, tasks, completed=["generation", "dedup_1"],
        config={"dedup": {"tol": 0.2}},
    )
    _apply(manager, _saved(tasks, dedup={"tol": 0.1}, dedup_1={"tol": 0.2}))
    with pytest.raises(RestartError, match="dedup_1.tol: 0.1 -> 0.2"):
        _apply(manager, _saved(tasks, dedup={"tol": 0.2}, dedup_1={"tol": 0.1}))


def test_run_settings_are_frozen_even_without_completed_tasks(tmp_path: Path) -> None:
    manager = _manager(tmp_path, config={"master": {"z": 4, "log_level": "debug"}})
    with pytest.raises(RestartError, match="master.z: 2 -> 4") as info:
        _apply(manager, _saved(master={"z": 2, "log_level": "info"}))
    assert "log_level" not in str(info.value)


def test_settings_only_one_run_has_do_not_freeze_completed_task(
    tmp_path: Path,
) -> None:
    # e.g. a default added or removed by a newer release, or a setting the
    # user added after the task completed; the results cannot depend on it
    manager = _manager(
        tmp_path, completed=["generation"],
        config={"generation": {"sr": 0.95, "new_opt": 1}},
    )
    _apply(manager, _saved(generation={"sr": 0.95, "old_opt": 2}))


def test_setting_added_to_pending_task_discards_its_checkpoints(
    tmp_path: Path,
) -> None:
    ckpt = _checkpoint(tmp_path, "symm_rigid_press")
    manager = _manager(
        tmp_path, completed=["generation"],
        config={"symm_rigid_press": {"sr": 0.85, "maxiter": 10}},
    )
    _apply(manager, _saved(symm_rigid_press={"sr": 0.85}))
    assert not ckpt.exists()


def test_settings_changed_by_running_tasks_are_not_saved(tmp_path: Path) -> None:
    config = {"workflow": {"tasks": WORKFLOW}, "generation": {"sr": 0.95}}
    gnrs_info = {"work_dir": str(tmp_path), "generation": {"status": "completed"}}
    manager = Restart(MPI.COMM_WORLD, config, gnrs_info)
    # e.g. optimization tasks pop struct_path and add energy_settings
    config["generation"]["energy_settings"] = {"xc": "pbe"}
    manager.write()

    fresh = _manager(tmp_path, config={"generation": {"sr": 0.95}})
    assert fresh.load() is True


# --- checkpoints of pending tasks -----------------------------------------


def test_checkpoints_of_unchanged_pending_task_are_kept(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path, "symm_rigid_press")
    manager = _manager(
        tmp_path, completed=["generation"], config={"symm_rigid_press": {"sr": 0.85}}
    )
    _apply(manager, _saved(symm_rigid_press={"sr": 0.85}))
    assert ckpt.is_file()


def test_checkpoints_of_completed_tasks_are_left_alone(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path, "generation")
    manager = _manager(tmp_path, completed=["generation"])
    _apply(manager, _saved())
    assert ckpt.is_file()


def test_checkpoints_of_changed_pending_task_are_discarded(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path, "symm_rigid_press")
    manager = _manager(
        tmp_path, completed=["generation"], config={"symm_rigid_press": {"sr": 0.8}}
    )
    _apply(manager, _saved(symm_rigid_press={"sr": 0.85}))
    assert not ckpt.exists()


def test_checkpoints_after_an_inserted_task_are_discarded(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path, "symm_rigid_press")
    manager = _manager(
        tmp_path, ["generation", "dedup", "symm_rigid_press"], completed=["generation"]
    )
    _apply(manager, _saved())
    assert not ckpt.exists()


def test_checkpoints_of_removed_tasks_are_discarded(tmp_path: Path) -> None:
    # A later run adding a second dedup would otherwise pick them up again
    ckpt = _checkpoint(tmp_path, "dedup_2")
    manager = _manager(
        tmp_path, ["generation", "dedup"], completed=["generation", "dedup_1"]
    )
    _apply(manager, _saved(["generation", "dedup", "dedup"]))
    assert not ckpt.exists()


def test_discard_checkpoints_reports_what_it_removed(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "generation")
    _checkpoint(tmp_path, "symm_rigid_press")
    (tmp_path / "tmp" / "molecule").mkdir()
    manager = _manager(tmp_path)
    assert manager.discard_checkpoints(keep=["generation"]) == ["symm_rigid_press"]
    assert manager.discard_checkpoints() == ["generation"]
    assert manager.discard_checkpoints() == []


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


def test_moved_run_directory_remaps_saved_config_paths(tmp_path: Path) -> None:
    old_dir = "/old/run"
    manager = _manager(
        tmp_path, completed=["generation"],
        config={"generation": {"path": str(tmp_path / "in.json")}},
    )
    saved = _saved(generation={"path": f"{old_dir}/in.json"})
    _apply(manager, saved, gnrs_info={"work_dir": old_dir})


def test_load_rejects_newer_format(tmp_path: Path) -> None:
    (tmp_path / "restart.json").write_text(
        json.dumps({"version": RESTART_VERSION + 1, "config": {}, "gnrs_info": {}})
    )
    with pytest.raises(RestartError, match="newer Genarris"):
        _manager(tmp_path).load()


def test_load_rejects_file_of_older_release(tmp_path: Path) -> None:
    _write(tmp_path, _saved(), version=None)
    with pytest.raises(RestartError, match="older Genarris release.*--overwrite"):
        _manager(tmp_path).load()


def test_load_rejects_old_layout_run(tmp_path: Path) -> None:
    # Older releases kept the restart file under tmp/
    (tmp_path / "tmp").mkdir()
    _write(tmp_path / "tmp", _saved(), version=None)
    with pytest.raises(RestartError, match="tmp/restart.json was written by an older"):
        _manager(tmp_path).load()


def test_load_rejects_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "restart.json").write_text("{not json")
    with pytest.raises(RestartError, match="corrupted.*--overwrite"):
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
