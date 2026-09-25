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

import json
import logging
import os
from contextlib import suppress

import numpy as np
from mpi4py import MPI

import gnrs.output as gout
from gnrs.core.registry import TaskSpec, resolve_tasks

logger = logging.getLogger("restart")

RESTART_FILE = "restart.json"

# Format version stamped into restart files. Files without one were written
# by releases before the stamp existed and use the same layout.
RESTART_VERSION = 1


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
            config: Live config dictionary; updated in place on load.
            gnrs_info: Live Genarris info dictionary; updated in place on load.
        """
        self.comm = comm
        self.config = config
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

        Settings of completed tasks must match the saved ones; for every
        other setting the current config file takes precedence and the
        change is reported.

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

    def _collective(self, master_func):
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
                "The file may be corrupted. Delete it and rerun without "
                "--restart to start over."
            ) from exc

        if not (
            isinstance(data, dict)
            and isinstance(data.get("config"), dict)
            and isinstance(data.get("gnrs_info"), dict)
        ):
            raise RestartError(
                f"Restart file {path} has an unexpected format. "
                "Delete it and rerun without --restart to start over."
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
                "longer exists. Start a fresh run without --restart."
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
        # (molecule files are re-copied from the current config every run)
        for key in (
            "size", "genarris_start_time", "config_path", "restart",
            "molecule_path",
        ):
            saved_info.pop(key, None)
        self.gnrs_info.update(saved_info)

        completed = self._check_task_list(saved_config)
        self._check_config(saved_config, completed)

    def _check_task_list(self, saved_config: dict) -> list[TaskSpec]:
        """
        Refuse to resume if the completed tasks no longer line up with the
        current task list. Completed tasks are matched by position, so
        inserting, removing or reordering tasks before them would skip the
        wrong task. Appending tasks at the end is fine, unless the appended
        task's type already appears in the list: duplicates are renumbered
        (``dedup`` becomes ``dedup_1``, ``dedup_2``), so the completed task
        would no longer be found under its saved id.

        Args:
            saved_config: Config stored in the restart file.

        Returns:
            The completed tasks, in workflow order.

        Raises:
            RestartError: If a completed task moved or changed.
        """
        saved_tasks = saved_config.get("workflow", {}).get("tasks", [])
        current_tasks = self.config.get("workflow", {}).get("tasks", [])
        try:
            saved_specs = resolve_tasks(saved_tasks)
            current_ids = [s.instance_id for s in resolve_tasks(current_tasks)]
        except ValueError:
            return []  # an invalid task list is reported when the tasks run

        completed = []
        for pos, spec in enumerate(saved_specs):
            if not self.is_task_completed(spec.instance_id):
                continue
            if pos < len(current_ids) and current_ids[pos] == spec.instance_id:
                completed.append(spec)
                continue
            raise RestartError(
                f"The task list changed since the previous run, so the "
                f"completed task '{spec.instance_id}' (position {pos + 1}) "
                "no longer lines up with it:\n"
                f"    previous: {saved_tasks}\n"
                f"    current:  {current_tasks}\n"
                "Completed tasks are matched by their position in [workflow] "
                "tasks. Restore the previous list to resume (adding tasks of "
                "a new type at the end is fine; adding a second task of a "
                "type already in the list is not), or start over with "
                "--overwrite."
            )
        return completed

    def _check_config(self, saved_config: dict, completed: list[TaskSpec]) -> None:
        """
        Compare the saved config with the current one.

        Sections of completed tasks are frozen: their results were produced
        with the saved settings, so a change is refused rather than silently
        ignored. Any other change is reported and the current config wins.

        Args:
            saved_config: Config stored in the restart file.
            completed: Tasks already completed, from ``_check_task_list``.

        Raises:
            RestartError: If a setting of a completed task changed.
        """
        # Normalize the live config the way the saved one was stored
        current_config = json.loads(
            json.dumps(self.config, default=_json_default)
        )
        frozen = {s.task_type for s in completed} | {s.instance_id for s in completed}
        blocked = []
        for section in sorted(frozen):
            blocked.extend(_diff_configs(
                saved_config.get(section, {}),
                current_config.get(section, {}),
                f"{section}.",
            ))
        if blocked:
            raise RestartError(
                "Settings of already completed tasks changed since the "
                "previous run:\n    "
                + "\n    ".join(blocked)
                + "\nTheir results were produced with the previous settings. "
                "Restore them to resume, or start over with --overwrite."
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
