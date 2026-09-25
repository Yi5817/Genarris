"""
This module provides functionality for saving and loading program state for
restart functionality.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import copy
import json
import logging
import os
from contextlib import suppress
from typing import Callable, Collection, TypeVar

import numpy as np
from mpi4py import MPI

import gnrs.output as gout
from gnrs.core.registry import TaskSpec, resolve_tasks
from gnrs.parallel.structs import DistributedStructs

logger = logging.getLogger("restart")

RESTART_FILE = "restart.json"

# Format version stamped into restart files. Files without one were written
# by releases before the stamp existed and use the same layout.
RESTART_VERSION = 1

# [master] settings that define the run itself. Every task's results depend
# on them, so a restart may never change them.
_RUN_SETTINGS = ("z", "molecule_path")

# Settings of older releases that no longer exist. Their value has no bearing
# on the results, so restart files that still carry them are not refused.
_REMOVED_SETTINGS = {"maceoff": ("save_flag",)}

T = TypeVar("T")


class RestartError(Exception):
    """
    Raised when a requested restart cannot be performed.
    """


def restart_path(directory: str) -> str:
    """
    Path of the restart file inside a run directory.

    Args:
        directory: Run directory.

    Returns:
        Absolute path of the restart file.
    """
    return os.path.join(directory, RESTART_FILE)


def _json_default(obj: object) -> object:
    """
    Convert NumPy scalars and arrays for json; refuse anything else.

    Args:
        obj: Object that json could not serialize.

    Returns:
        A JSON-serializable representation of the object.

    Raises:
        TypeError: If the object has no faithful JSON representation. It
            would otherwise silently come back as a string on restart.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        f"{type(obj).__name__} value {obj!r} cannot be stored in the restart file"
    )


def _remap_paths(obj: object, old_root: str, new_root: str) -> object:
    """
    Recursively rewrite stored paths when the run directory has moved.

    Args:
        obj: Loaded restart data (nested dicts/lists/strings).
        old_root: Work directory recorded in the restart file.
        new_root: Current work directory.

    Returns:
        Data with old_root path prefixes replaced by new_root.
    """
    if isinstance(obj, str):
        # Require a path boundary so e.g. /data/run cannot match a
        # sibling directory such as /data/run_backup
        if obj == old_root or obj.startswith(old_root + os.sep):
            return new_root + obj[len(old_root):]
        return obj
    if isinstance(obj, dict):
        return {k: _remap_paths(v, old_root, new_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_remap_paths(v, old_root, new_root) for v in obj]
    return obj


def _diff_configs(saved: dict, current: dict, prefix: str = "") -> list[str]:
    """
    List settings that differ between the saved and current config.

    Args:
        saved: Config stored in the restart file.
        current: Config parsed from the user's config file.
        prefix: Key prefix used for nested sections.

    Returns:
        Every differing setting as ``"section.key: old -> new"``.
    """
    diffs = []
    for key in sorted(set(saved) | set(current), key=str):
        old_val = saved.get(key, "<not set>")
        new_val = current.get(key, "<not set>")
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            diffs.extend(_diff_configs(old_val, new_val, f"{prefix}{key}."))
        elif old_val != new_val:
            diffs.append(f"{prefix}{key}: {old_val!r} -> {new_val!r}")
    return diffs


def _drop_removed_settings(config: dict) -> dict:
    """
    Copy a config without the settings listed in ``_REMOVED_SETTINGS``.

    Args:
        config: Saved or current config.

    Returns:
        A copy of the config with those settings removed.
    """
    config = copy.deepcopy(config)
    for section, keys in _REMOVED_SETTINGS.items():
        for key in keys:
            config.get(section, {}).pop(key, None)
    return config


def _resolve(which: str, config: dict) -> list[TaskSpec]:
    """
    Resolve the task list of a config.

    Args:
        which: ``"previous"`` or ``"current"``, for the error message.
        config: Saved or current config.

    Returns:
        The resolved task specs.

    Raises:
        RestartError: If the list holds an unknown task. Checked here, before
            anything is changed on disk, rather than when the tasks run.
    """
    try:
        return resolve_tasks(config.get("workflow", {}).get("tasks", []))
    except ValueError as exc:
        raise RestartError(f"The {which} task list is invalid: {exc}") from exc


def _task_settings(config: dict, task_type: str, instance_id: str) -> dict:
    """
    Effective settings of one task: its type section updated with its
    per-instance overrides, as ``TaskABC._merge_config`` does.

    Args:
        config: Saved or current config.
        task_type: Task type, e.g. ``dedup``.
        instance_id: Task instance id, e.g. ``dedup_2``.

    Returns:
        The merged settings.
    """
    settings = dict(config.get(task_type, {}))
    if instance_id != task_type:
        settings.update(config.get(instance_id, {}))
    return settings


def _task_diffs(
    saved_config: dict, current_config: dict, spec: TaskSpec, saved_id: str
) -> list[str]:
    """
    List the settings of one task that differ between two configs.

    Args:
        saved_config: Config stored in the restart file.
        current_config: Config parsed from the user's config file.
        spec: The task, resolved from the current task list.
        saved_id: The task's instance id in the saved task list. Differs from
            ``spec.instance_id`` when repeated tasks were renumbered.

    Returns:
        Every differing setting as ``"section.key: old -> new"``.
    """
    diffs = _diff_configs(
        _task_settings(saved_config, spec.task_type, saved_id),
        _task_settings(current_config, spec.task_type, spec.instance_id),
        f"{spec.instance_id}.",
    )
    for section in spec.sections:
        if section in (spec.task_type, spec.instance_id):
            continue
        diffs.extend(_diff_configs(
            saved_config.get(section, {}),
            current_config.get(section, {}),
            f"{section}.",
        ))
    return diffs


class Restart:
    """
    Keeps the progress record of a run: which tasks completed, where their
    results are, and the config they ran with.

    The record is a JSON file in the run directory, written on the master
    rank after every completed task. Loading it is collective: every rank
    ends up with the same state, or every rank raises the same error.
    """

    def __init__(self, comm: MPI.Comm, config: dict, gnrs_info: dict) -> None:
        """
        Args:
            comm: MPI communicator.
            config: Parsed config dictionary. A copy is kept, so settings that
                tasks add to the live config while running are neither
                written to the restart file nor reported as changes.
            gnrs_info: Live Genarris info dictionary; updated in place on load.
        """
        self.comm = comm
        self.config = copy.deepcopy(config)
        self.gnrs_info = gnrs_info
        self.is_master = comm.Get_rank() == 0
        # The restart file lives in the work directory, next to the run's
        # results, so cleaning the tmp/ scratch dir cannot destroy it.
        self.restart_file = restart_path(gnrs_info["work_dir"])

    def write(self) -> None:
        """
        Write the current program state to the restart file.

        The file is written atomically (temporary file, then rename) so a
        crash mid-write can never corrupt an existing restart file. A
        failure to write is reported but does not stop the run.
        """
        if not self.is_master:
            return
        logger.info("Writing restart file")
        restart = {
            "version": RESTART_VERSION,
            "config": self.config,
            "gnrs_info": self.gnrs_info,
        }
        tmp_file = self.restart_file + ".tmp"
        try:
            with open(tmp_file, "w") as rfile:
                json.dump(
                    restart, rfile, indent=4, sort_keys=True, default=_json_default
                )
                rfile.flush()
                os.fsync(rfile.fileno())
            os.replace(tmp_file, self.restart_file)
        except (OSError, TypeError, ValueError) as exc:
            logger.error(f"Failed to write restart file: {exc}")
            gout.emit(
                f"WARNING: Could not write restart file {self.restart_file}: "
                f"{exc}. The run continues, but restarting from this point "
                "will not be possible."
            )
            with suppress(OSError):
                os.remove(tmp_file)

    def load(self) -> bool:
        """
        Load program state from the restart file. Collective.

        Settings of completed tasks and the [master] settings that define
        the run must match the saved ones; for every other setting the
        current config file takes precedence and the change is reported.
        Checkpoints of a pending task are discarded if the task or one
        before it changed.

        Returns:
            True if a restart file was found and loaded, False otherwise.

        Raises:
            RestartError: If the restart file exists but cannot be used.
        """
        payload = self._collective(self._read_restart_file)
        if payload is None:
            logger.info("No restart file found")
            return False

        self._apply_restart(payload)
        self._collective(self._check_last_struct)
        return True

    def _collective(self, master_func: Callable[[], T]) -> T:
        """
        Run a step on the master rank and share its result or its error.

        Args:
            master_func: Callable run on the master rank only. It may raise
                RestartError.

        Returns:
            The callable's return value, on every rank.

        Raises:
            RestartError: On every rank, if the master rank raised one.
                Non-master ranks would otherwise block in the broadcast.
        """
        result, error = None, None
        if self.is_master:
            try:
                result = master_func()
            except RestartError as exc:
                error = str(exc)
        result, error = self.comm.bcast((result, error), root=0)
        if error is not None:
            raise RestartError(error)
        return result

    def _read_restart_file(self) -> dict | None:
        """
        Read and validate the restart file. Master rank only.

        Returns:
            Parsed restart data, or None if no restart file exists.

        Raises:
            RestartError: If the file is unreadable, malformed or written by
                a newer Genarris.
        """
        path = self.restart_file
        if not os.path.isfile(path):
            return None

        logger.info(f"Reading restart file {path}")
        try:
            with open(path, "r") as rfile:
                data = json.load(rfile)
        except (OSError, ValueError) as exc:
            raise RestartError(
                f"Could not read restart file {path}: {exc}. "
                "The file may be corrupted. Start over with --overwrite."
            ) from exc

        if not (
            isinstance(data, dict)
            and isinstance(data.get("config"), dict)
            and isinstance(data.get("gnrs_info"), dict)
        ):
            raise RestartError(
                f"Restart file {path} has an unexpected format. "
                "Start over with --overwrite."
            )

        version = data.get("version", 1)
        if not isinstance(version, int) or version > RESTART_VERSION:
            raise RestartError(
                f"Restart file {path} was written by a newer Genarris "
                f"(format version {version}, this release reads up to "
                f"{RESTART_VERSION}). Use that Genarris version to resume, "
                "or start over with --overwrite."
            )
        return data

    def _check_last_struct(self) -> None:
        """
        Fail early if the structure file needed to resume is gone.

        Raises:
            RestartError: If the last completed task's result file is missing.
        """
        last_struct = self.gnrs_info.get("last_struct_path")
        if last_struct and not os.path.isfile(last_struct):
            raise RestartError(
                f"The structure file needed to resume ({last_struct}) no "
                "longer exists. Start over with --overwrite."
            )

    def _apply_restart(self, payload: dict) -> None:
        """
        Apply loaded restart data to the live config and gnrs_info.

        Args:
            payload: Validated restart data. Identical on all ranks.
        """
        saved_config = payload["config"]
        saved_info = payload["gnrs_info"]

        # Rewrite stored paths if the run directory was moved or renamed
        old_work_dir = saved_info.get("work_dir")
        new_work_dir = self.gnrs_info.get("work_dir")
        if old_work_dir and new_work_dir and old_work_dir != new_work_dir:
            logger.info(
                f"Run directory moved from {old_work_dir} to {new_work_dir}; "
                "remapping saved paths"
            )
            gout.emit(
                f"Previous run directory was {old_work_dir}; saved paths "
                f"remapped to {new_work_dir}."
            )
            saved_info = _remap_paths(saved_info, old_work_dir, new_work_dir)

        # Keep values that describe the current run, not the previous one
        # (the molecule files under tmp/ are recreated if they are missing)
        for key in (
            "size", "genarris_start_time", "config_path", "restart",
            "molecule_path",
        ):
            saved_info.pop(key, None)
        self.gnrs_info.update(saved_info)

        saved_config = _drop_removed_settings(saved_config)
        # Normalize the live config the way the saved one was stored
        current_config = _drop_removed_settings(
            json.loads(json.dumps(self.config, default=_json_default))
        )
        saved_specs = _resolve("previous", saved_config)
        current_specs = _resolve("current", current_config)

        completed = self._check_task_list(saved_specs, current_specs)
        self._check_config(saved_config, current_config, completed)
        self._discard_stale_checkpoints(
            saved_config, current_config, saved_specs, current_specs
        )

    def _check_task_list(
        self, saved_specs: list[TaskSpec], current_specs: list[TaskSpec]
    ) -> list[tuple[str, TaskSpec]]:
        """
        Refuse to resume if the completed tasks no longer line up with the
        current task list. Completed tasks are matched by position and type,
        so inserting, removing or reordering tasks before them would skip
        the wrong task; tasks after the last completed one may change
        freely. Repeated task types are renumbered (``dedup`` becomes
        ``dedup_1``, ``dedup_2``) when one of them is added or removed, so a
        completed task's record is moved to its new id.

        Args:
            saved_specs: Task list of the previous run.
            current_specs: Task list of the current config.

        Returns:
            The completed tasks in workflow order, each as its id in the
            previous task list and its spec in the current one.

        Raises:
            RestartError: If a completed task moved or changed.
        """
        completed, renamed = [], {}
        for pos, saved in enumerate(saved_specs):
            if not self.is_task_completed(saved.instance_id):
                continue
            current = current_specs[pos] if pos < len(current_specs) else None
            if current is None or current.task_type != saved.task_type:
                raise RestartError(
                    f"The task list changed since the previous run, so the "
                    f"completed task '{saved.instance_id}' (position "
                    f"{pos + 1}) no longer lines up with it:\n"
                    f"    previous: {[s.instance_id for s in saved_specs]}\n"
                    f"    current:  {[s.instance_id for s in current_specs]}\n"
                    "Completed tasks are matched by their position in "
                    "[workflow] tasks, so only tasks after the last completed "
                    "one may be changed, removed or added. Restore the "
                    "previous list to resume, or start over with --overwrite."
                )
            if current.instance_id != saved.instance_id:
                renamed[saved.instance_id] = current.instance_id
            completed.append((saved.instance_id, current))
        for saved_id, current_id in renamed.items():
            logger.info(f"Completed task {saved_id} is now {current_id}")
            self.gnrs_info[current_id] = self.gnrs_info.pop(saved_id)
        return completed

    def _check_config(
        self,
        saved_config: dict,
        current_config: dict,
        completed: list[tuple[str, TaskSpec]],
    ) -> None:
        """
        Compare the saved config with the current one.

        The settings of completed tasks (their own sections and the method
        sections they read, e.g. ``[bfgs]`` and ``[maceoff]`` for
        ``bfgs_maceoff``) and the ``[master]`` settings that define the run
        are frozen: the results were produced with the saved values, so a
        change is refused rather than silently ignored. Any other change is
        reported and the current config wins.

        Args:
            saved_config: Config stored in the restart file.
            current_config: Config parsed from the user's config file.
            completed: Tasks already completed, from ``_check_task_list``.

        Raises:
            RestartError: If a frozen setting changed.
        """
        saved_master = saved_config.get("master", {})
        current_master = current_config.get("master", {})
        blocked = _diff_configs(
            {key: saved_master.get(key) for key in _RUN_SETTINGS},
            {key: current_master.get(key) for key in _RUN_SETTINGS},
            "master.",
        )
        for saved_id, spec in completed:
            blocked.extend(_task_diffs(saved_config, current_config, spec, saved_id))
        if blocked:
            raise RestartError(
                "Settings the previous run depends on changed:\n    "
                + "\n    ".join(blocked)
                + "\nThe [master] molecule and z define the run, and the "
                "results of completed tasks were produced with the previous "
                "settings. Restore them to resume, or start over with "
                "--overwrite."
            )
        diffs = _diff_configs(saved_config, current_config)
        if diffs:
            logger.warning(
                "Config differs from the previous run: " + "; ".join(diffs)
            )
            gout.emit(
                "WARNING: Settings changed since the previous run "
                "(the current config file takes precedence):"
            )
            for diff in diffs:
                gout.emit(f"    {diff}")

    def _discard_stale_checkpoints(
        self,
        saved_config: dict,
        current_config: dict,
        saved_specs: list[TaskSpec],
        current_specs: list[TaskSpec],
    ) -> None:
        """
        Remove the checkpoints of pending tasks that cannot reuse them.

        A pending task's checkpoints hold results of the previous run, which
        are only valid if the task has the same id at the same position and
        neither its settings nor any task before it changed. Structures are
        matched by name and names are reproducible across runs, so stale
        checkpoints would otherwise be merged silently into the new pool.

        Args:
            saved_config: Config stored in the restart file.
            current_config: Config parsed from the user's config file.
            saved_specs: Task list of the previous run.
            current_specs: Task list of the current config.
        """
        keep, intact = [], True
        for pos, spec in enumerate(current_specs):
            if self.is_task_completed(spec.instance_id):
                keep.append(spec.instance_id)
                continue
            intact = (
                intact
                and pos < len(saved_specs)
                and saved_specs[pos].instance_id == spec.instance_id
                and not _task_diffs(
                    saved_config, current_config, spec, spec.instance_id
                )
            )
            if intact:
                keep.append(spec.instance_id)

        for task in self.discard_checkpoints(keep):
            gout.emit(
                f"NOTE: The checkpoints of task '{task}' from the previous "
                "run are discarded, because the task or one before it "
                "changed; it starts from scratch."
            )

    def discard_checkpoints(self, keep: Collection[str] = ()) -> list[str]:
        """
        Remove the checkpoints of every task under tmp/. Collective.

        Args:
            keep: Instance ids of the tasks whose checkpoints are kept.

        Returns:
            Instance ids of the tasks whose checkpoints were removed.
        """
        discarded = []
        tmp_dir = self.gnrs_info.get("tmp_dir")
        if self.is_master and tmp_dir and os.path.isdir(tmp_dir):
            for entry in os.scandir(tmp_dir):
                if entry.is_dir() and entry.name not in keep:
                    if DistributedStructs.checkpoint_clear(entry.path):
                        discarded.append(entry.name)
        discarded = self.comm.bcast(discarded, root=0)
        for task in discarded:
            logger.warning(f"Discarded checkpoints of task {task}")
        return discarded

    def is_task_completed(self, task_name: str) -> bool:
        """
        Check if a task has been completed.

        Args:
            task_name: Task instance id, e.g. ``dedup_2``.

        Returns:
            True if the task finished in a previous run.
        """
        task = self.gnrs_info.get(task_name)
        return isinstance(task, dict) and task.get("status") == "completed"


__all__ = ["RESTART_FILE", "Restart", "RestartError", "restart_path"]
