"""
Task registry for Genarris.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib
from collections import Counter
from typing import NamedTuple

_TASK_TYPES = {
    "generation": ("gnrs.generation", "StructureGenerationTask"),
    "energy": ("gnrs.energy", "EnergyCalculationTask"),
    "optimize": ("gnrs.optimize", "GeometryOptimizationTask"),
    "descriptor": ("gnrs.descriptor", "DescriptorEvaluationTask"),
    "cluster": ("gnrs.cluster", "ClusterSelectionTask"),
    "dedup": ("gnrs.deduplication", "DuplicateRemovalTask"),
}

_ENERGY_METHODS = {"maceoff", "uma", "aimnet", "aims", "vasp", "dftb"}

_OPTIMIZERS = {"bfgs", "lbfgs"}

_RIGID_PRESS_OPTIMIZERS = {"rigid_press", "symm_rigid_press"}

_CLUSTERERS = {"ap", "kmeans"}

_SELECTORS = {"center", "window"}

_DESCRIPTORS = {"acsf"}


class TaskSpec(NamedTuple):
    """
    Specification for a single task.

    Args:
        task_type: Task type name (e.g. "dedup", "acsf").
        instance_id: Unique ID for folders and restart tracking.
        cls: Task class.
        extra_args: Extra args for the task constructor.
    """
    task_type: str
    instance_id: str
    cls: type
    extra_args: tuple


def resolve_tasks(task_list: list[str]) -> list[TaskSpec]:
    """
    Resolve a full workflow task list into `TaskSpec` objects.

    Duplicate task names are auto-indexed (e.g. "dedup_1", "dedup_2").

    Args:
        task_list: Raw task names from `config["workflow"]["tasks"]`.

    Returns:
        List of TaskSpec objects.
    """
    counts = Counter(task_list)
    seen = {}
    specs = []

    for raw_name in task_list:
        cls, extra_args = resolve_task(raw_name)
        task_type = raw_name.strip().lower()

        if counts[raw_name] > 1:
            idx = seen.get(task_type, 0) + 1
            seen[task_type] = idx
            instance_id = f"{task_type}_{idx}"
        else:
            instance_id = task_type

        specs.append(TaskSpec(task_type, instance_id, cls, extra_args))

    return specs


def resolve_task(task_name: str):
    """
    Resolve a config task name into (task_class, extra_args).

    Args:
        task_name: Task name from the config file.
    """
    name = task_name.strip().lower()

    # 1) generation
    if name == "generation":
        cls = _import_class(*_TASK_TYPES["generation"])
        return cls, ()

    # 2) rigid press optimizers: rigid_press, symm_rigid_press
    if name in _RIGID_PRESS_OPTIMIZERS:
        cls = _import_class(*_TASK_TYPES["optimize"])
        return cls, (name,)
    
    # duplicate removal
    if name == "dedup":
        cls = _import_class(*_TASK_TYPES["dedup"])
        return cls, ()

    # 3) optimizer + energy: bfgs_maceoff, lbfgs_uma, ...
    for opt in _OPTIMIZERS:
        prefix = opt + "_"
        if name.startswith(prefix):
            energy_method = name[len(prefix) :]
            if energy_method in _ENERGY_METHODS:
                cls = _import_class(*_TASK_TYPES["optimize"])
                return cls, (opt, energy_method)

    # 4) SPE: maceoff, uma, vasp, ...
    if name in _ENERGY_METHODS:
        cls = _import_class(*_TASK_TYPES["energy"])
        return cls, (name,)

    # 5) descriptor: acsf, ...
    if name in _DESCRIPTORS:
        cls = _import_class(*_TASK_TYPES["descriptor"])
        return cls, (name,)

    # 6) cluster + selection: ap_center, kmeans_window, ...
    for cm in _CLUSTERERS:
        prefix = cm + "_"
        if name.startswith(prefix):
            selection = name[len(prefix) :]
            if selection in _SELECTORS:
                cls = _import_class(*_TASK_TYPES["cluster"])
                return cls, (cm, selection)

    raise ValueError(
        f"Unknown task: {task_name}. "
        f"Could not resolve to any registered task type."
    )


def _import_class(module_path: str, class_name: str):
    """
    Import and return a class from the given module path.
    """
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)
