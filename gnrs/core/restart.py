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

import os
import json
import logging
from contextlib import suppress

import numpy as np
from mpi4py import MPI

import gnrs.output as gout
from gnrs.core.registry import resolve_tasks

logger = logging.getLogger("restart")

# Format version stamped into restart files for future checks
RESTART_VERSION = 1


class RestartError(Exception):
    """
    Raised when a requested restart cannot be performed.
    """


def _json_default(obj: object) -> object:
    """
    Convert objects that json cannot serialize natively.

    Args:
        obj: Object that json could not serialize.

    Returns:
        A JSON-serializable representation of the object.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


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
        Every differing setting.
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
    Manages saving and loading program state for restart functionality.
    """

    def __init__(self) -> None:
        self.comm: MPI.Comm | None = None
        self.config: dict = {}
        self.gnrs_info: dict = {}
        self.restart_file: str | None = None
        self.is_master: bool = False

    def initialize(
        self,
        comm: MPI.Comm,
        config: dict,
        gnrs_info: dict
    ) -> None:
        """
        Initialize the restart manager.

        Args:
            comm: MPI communicator
            config: Config dictionary
            gnrs_info: Genarris info dictionary
        """
        self.comm = comm
        self.config = config
        self.gnrs_info = gnrs_info
        self.is_master = comm.Get_rank() == 0
        # The restart file lives in the work directory, next to the run's
        # results, so cleaning the tmp/ scratch dir cannot destroy it.
        self.restart_file = os.path.join(
            self.gnrs_info["work_dir"], "restart.json"
        )

    def write_restart(self) -> None:
        """
        Write current program state to restart file.

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

    def load_restart(self) -> bool:
        """
        Load program state from restart file.

        The current config file takes precedence over the saved config;
        any settings that changed since the original run are reported.

        Returns:
            True if a restart file was found and loaded, False otherwise.

        Raises:
            RestartError: If the restart file exists but cannot be used.
        """
        payload = None
        error = None
        if self.is_master:
            try:
                payload = self._read_restart_file()
            except RestartError as exc:
                error = str(exc)

        # Fail together on all ranks with a clear message instead of
        # leaving non-master ranks blocked in the broadcast.
        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise RestartError(error)

        payload = self.comm.bcast(payload, root=0)
        if payload is None:
            logger.info("No restart file found")
            return False

        self._apply_restart(payload)

        # Fail early if the structure file needed to resume is gone
        error = None
        last_struct = self.gnrs_info.get("last_struct_path")
        if self.is_master and last_struct and not os.path.isfile(last_struct):
            error = (
                f"The structure file needed to resume ({last_struct}) no "
                "longer exists. Start a fresh run without --restart."
            )
        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise RestartError(error)

        return True

    def _read_restart_file(self) -> dict | None:
        """
        Read and validate the restart file. Master rank only.

        Returns:
            Parsed restart data, or None if no restart file exists.

        Raises:
            RestartError: If the file is unreadable or malformed.
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

        return data

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
        self._check_task_list(saved_config)

        # The current config file wins; tell the user what changed.
        current_config = json.loads(
            json.dumps(self.config, default=_json_default)
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

    def _check_task_list(self, saved_config: dict) -> None:
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

        Raises:
            RestartError: If a completed task moved or changed.
        """
        saved_tasks = saved_config.get("workflow", {}).get("tasks", [])
        current_tasks = self.config.get("workflow", {}).get("tasks", [])
        if saved_tasks == current_tasks:
            return
        try:
            saved_ids = [s.instance_id for s in resolve_tasks(saved_tasks)]
            current_ids = [s.instance_id for s in resolve_tasks(current_tasks)]
        except ValueError:
            return  # an invalid task list is reported when the tasks run

        for pos, task_id in enumerate(saved_ids):
            if not self.check_task_completion(task_id):
                continue
            if pos < len(current_ids) and current_ids[pos] == task_id:
                continue
            raise RestartError(
                "The task list changed since the previous run, so its "
                "completed tasks no longer line up with it:\n"
                f"    previous: {saved_tasks}\n"
                f"    current:  {current_tasks}\n"
                "Completed tasks are matched by their position in [workflow] "
                "tasks. Restore the previous list to resume (adding tasks of "
                "a new type at the end is fine; adding a second task of a "
                "type already in the list is not), or start over with "
                "--overwrite."
            )

    def check_task_completion(self, task_name: str) -> bool:
        """
        Check if a task has been completed.
        """
        task = self.gnrs_info.get(task_name)
        return isinstance(task, dict) and task.get("status") == "completed"


# Global singleton instance
_restart = Restart()

def restart_init(comm: MPI.Comm, config: dict, gnrs_info: dict) -> None:
    """
    Initialize the global restart manager.
    """
    _restart.initialize(comm, config, gnrs_info)

def write_restart() -> None:
    """
    Write current program state to restart file.
    """
    _restart.write_restart()

def load_restart() -> bool:
    """
    Load program state from restart file.
    """
    return _restart.load_restart()

def is_task_completed(task_name: str) -> bool:
    """
    Check if a task has been completed.
    """
    return _restart.check_task_completion(task_name)

__all__ = [
    "RestartError", "restart_init", "write_restart",
    "load_restart", "is_task_completed"
]
