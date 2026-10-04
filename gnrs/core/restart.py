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
from collections.abc import Callable, Collection
from contextlib import suppress
from typing import TypeVar

import numpy as np
from mpi4py import MPI

import gnrs.output as gout
from gnrs.core.registry import TaskSpec, resolve_tasks
from gnrs.parallel.structs import DistributedStructs

logger = logging.getLogger(__name__)

RESTART_FILE = "restart.json"
RESTART_VERSION = 1

# [master] settings that define the run itself. Every task's results depend
# on them, so a restart may never change them.
_RUN_SETTINGS = ("z", "molecule_path")

_OLDER_RELEASE = (
    "was written by an older Genarris release and cannot be resumed with "
    "this one. Rerun the workflow with this release, starting over with "
    "--overwrite."
)

T = TypeVar("T")


class RestartError(Exception):
    """
    Raised when a requested restart cannot be performed.
    """


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


def _diff_configs(
    saved: dict, current: dict, prefix: str = "", shared_only: bool = False
) -> list[str]:
    """
    List settings that differ between the saved and current config.

    Args:
        saved: Config stored in the restart file.
        current: Config parsed from the user's config file.
        prefix: Key prefix used for nested sections.
        shared_only: Compare only settings both configs have, so a setting
            that one of them lacks (e.g. a default added by a newer release)
            is not a difference.

    Returns:
        Every differing setting as ``"section.key: old -> new"``.
    """
    keys = set(saved) & set(current) if shared_only else set(saved) | set(current)
    diffs = []
    for key in sorted(keys):
        old_val = saved.get(key, "<not set>")
        new_val = current.get(key, "<not set>")
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            diffs.extend(
                _diff_configs(old_val, new_val, f"{prefix}{key}.", shared_only)
            )
        elif old_val != new_val:
            diffs.append(f"{prefix}{key}: {old_val!r} -> {new_val!r}")
    return diffs


def _task_diffs(
    saved_config: dict,
    current_config: dict,
    spec: TaskSpec,
    shared_only: bool = False,
) -> list[str]:
    """
    List the settings in the sections one task reads that differ between
    two configs.

    Args:
        saved_config: Config stored in the restart file.
        current_config: Config parsed from the user's config file.
        spec: The task, at the same position in both task lists.
        shared_only: See ``_diff_configs``. Never applies to the override
            section of a repeated task (``[dedup_2]``), where a removed key
            changes the value the task reads.

    Returns:
        Every differing setting as ``"section.key: old -> new"``.
    """
    diffs = []
    for section in spec.sections:
        is_override = section == spec.instance_id != spec.task_type
        diffs += _diff_configs(
            saved_config.get(section, {}),
            current_config.get(section, {}),
            f"{section}.",
            shared_only and not is_override,
        )
    return diffs


class Restart:
    """
    Keeps the progress record of a run: which tasks completed, where their
    results are, and the config they ran with.

    The record is a JSON file in the run directory, written on the master
    rank when the run starts and after every completed task, so it always
    holds the settings the pending tasks' checkpoints are made with. Loading
    it is collective: every rank ends up with the same state, or every rank
    raises the same error.
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
        self.restart_file = os.path.join(gnrs_info["work_dir"], RESTART_FILE)

    def write(self) -> None:
        """
        Write the current program state to the restart file.

        The file is written atomically (temporary file, then rename) so a
        crash mid-write can never corrupt an existing restart file. A
        failure to write is reported but does not stop the run.
        """
        if not self.is_master:
            return
        logger.debug("Writing restart file")
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
            gout.warning(
                f"Could not write restart file {self.restart_file}: "
                f"{exc}. The run continues, but restarting from this point "
                "will not be possible."
            )
            with suppress(OSError):
                os.remove(tmp_file)

    def load(self) -> list[str] | None:
        """
        Load program state from the restart file. Collective.

        Settings of completed tasks and the [master] settings that define
        the run must match the saved ones; for every other setting the
        current config file takes precedence and the change is reported.
        Everything is validated before anything on disk is touched; only
        then are the checkpoints of pending tasks that cannot reuse them
        discarded.

        Returns:
            Instance ids of the tasks whose checkpoints were discarded, or
            None if there is no restart file.

        Raises:
            RestartError: If the restart file exists but cannot be used.
        """
        found = self.find_records()
        payload = self._collective(lambda: self._read_record(found))
        if payload is None:
            logger.info("No restart file found")
            return None
        self._collective(lambda: self._check_paths(payload["gnrs_info"]))
        kept = self._apply_restart(payload)
        return self.discard_checkpoints([spec.instance_id for spec in kept])

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

    def find_records(self) -> list[str]:
        """
        Restart files present in the run directory. Collective.

        Returns:
            The current file first, then the ``tmp/restart.json`` of older
            releases; only the ones that exist.
        """
        legacy = os.path.join(self.gnrs_info["tmp_dir"], RESTART_FILE)
        return self._collective(
            lambda: [p for p in (self.restart_file, legacy) if os.path.isfile(p)]
        )

    def _read_record(self, found: list[str]) -> dict | None:
        """
        Read and validate the restart file. Master rank only.

        Args:
            found: Restart files present, from ``find_records``.

        Returns:
            Parsed restart data, or None if no restart file exists.

        Raises:
            RestartError: If the file is unreadable or written by another
                Genarris release.
        """
        if not found:
            return None
        path = found[0]
        if path != self.restart_file:
            raise RestartError(f"Restart file {path} {_OLDER_RELEASE}")

        logger.info(f"Reading restart file {path}")
        try:
            with open(path) as rfile:
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
        if data.get("version") != RESTART_VERSION:
            raise RestartError(f"Restart file {path} {_OLDER_RELEASE}")
        return data

    def _check_paths(self, saved_info: dict) -> None:
        """
        Fail early if the run is not where the restart file says or the
        structure file needed to resume is gone. Master rank only, so every
        rank gets the same verdict from ``_collective``.

        Args:
            saved_info: The ``gnrs_info`` stored in the restart file.

        Raises:
            RestartError: If the run directory changed (the saved paths would
                point outside it) or the last result file is missing.
        """
        old_work_dir = saved_info.get("work_dir")
        if old_work_dir != self.gnrs_info["work_dir"]:
            raise RestartError(
                f"The previous run was made in {old_work_dir}, but this "
                f"directory is {self.gnrs_info['work_dir']}. Move it back "
                "to resume, or start over with --overwrite."
            )
        last_struct = saved_info.get("last_struct_path")
        if last_struct and not os.path.isfile(last_struct):
            raise RestartError(
                f"The structure file needed to resume ({last_struct}) no "
                "longer exists. Start over with --overwrite."
            )

    def _apply_restart(self, payload: dict) -> list[TaskSpec]:
        """
        Validate loaded restart data against the current config and apply
        it to the live gnrs_info. Touches nothing on disk.

        Args:
            payload: Validated restart data. Identical on all ranks.

        Returns:
            The tasks whose checkpoints are kept: completed tasks and the
            pending tasks that can resume from them.
        """
        saved_config = payload["config"]
        saved_info = payload["gnrs_info"]

        # Keep values that describe the current run, not the previous one
        # (the molecule files under tmp/ are recreated if they are missing)
        for key in (
            "size",
            "genarris_start_time",
            "config_path",
            "restart",
            "molecule_path",
        ):
            saved_info.pop(key, None)
        self.gnrs_info.update(saved_info)

        # Normalize the live config the way the saved one was stored
        current_config = json.loads(json.dumps(self.config, default=_json_default))
        saved_specs = resolve_tasks(saved_config.get("workflow", {}).get("tasks", []))
        current_specs = resolve_tasks(current_config["workflow"]["tasks"])

        completed = self._check_task_list(saved_specs, current_specs)
        self._check_config(saved_config, current_config, completed)
        return completed + self._resumable(
            saved_config, current_config, saved_specs, current_specs
        )

    def _check_task_list(
        self, saved_specs: list[TaskSpec], current_specs: list[TaskSpec]
    ) -> list[TaskSpec]:
        """
        Refuse to resume if the completed tasks no longer line up with the
        current task list. Completed tasks are matched by position and
        instance id, so inserting, removing or reordering tasks before them
        would skip the wrong task; tasks after the last completed one may
        change freely. Adding or removing a repeated task type renumbers the
        others (``dedup`` becomes ``dedup_1``, ``dedup_2``) and counts as a
        change too.

        Args:
            saved_specs: Task list of the previous run.
            current_specs: Task list of the current config.

        Returns:
            The completed tasks in workflow order.

        Raises:
            RestartError: If a completed task moved or changed.
        """
        completed = []
        for pos, saved in enumerate(saved_specs):
            if not self.is_task_completed(saved.instance_id):
                continue
            current = current_specs[pos] if pos < len(current_specs) else None
            if current is None or current.instance_id != saved.instance_id:
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
            completed.append(current)
        return completed

    def _check_config(
        self, saved_config: dict, current_config: dict, completed: list[TaskSpec]
    ) -> None:
        """
        Compare the saved config with the current one.

        The sections completed tasks read (their own, the method sections,
        e.g. ``[bfgs]`` and ``[maceoff]`` for ``bfgs_maceoff``, and the
        override section of a repeated task) and the ``[master]`` settings
        that define the run are frozen: the results were produced with the
        saved values, so a changed value is refused rather than silently
        ignored. A setting that only one of the two runs has (e.g. a default
        added or removed by a newer release) cannot have changed the results
        and does not block. Every other difference is reported and the
        current config wins.

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
        for spec in completed:
            blocked.extend(
                _task_diffs(saved_config, current_config, spec, shared_only=True)
            )
        blocked = list(dict.fromkeys(blocked))  # sections shared by several tasks
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
            gout.warning(
                "Settings changed since the previous run "
                "(the current config file takes precedence):\n    "
                + "\n    ".join(diffs)
            )

    def _resumable(
        self,
        saved_config: dict,
        current_config: dict,
        saved_specs: list[TaskSpec],
        current_specs: list[TaskSpec],
    ) -> list[TaskSpec]:
        """
        Find the pending tasks that can resume from their checkpoints.

        Checkpoints hold results of the previous run, which are only valid
        if the task has the same id at the same position and neither its
        settings nor any task before it changed. Structures are matched by
        name and names are reproducible across runs, so stale checkpoints
        would otherwise be merged silently into the new pool.

        Args:
            saved_config: Config stored in the restart file.
            current_config: Config parsed from the user's config file.
            saved_specs: Task list of the previous run.
            current_specs: Task list of the current config.

        Returns:
            The resumable tasks.
        """
        resumable = []
        for spec, saved in zip(current_specs, saved_specs):
            if self.is_task_completed(spec.instance_id):
                continue
            if saved.instance_id != spec.instance_id or _task_diffs(
                saved_config, current_config, spec
            ):
                break
            resumable.append(spec)
        return resumable

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
                        logger.warning(f"Discarded checkpoints of task {entry.name}")
        discarded = self.comm.bcast(discarded, root=0)
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
