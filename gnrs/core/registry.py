"""
Task registry for Genarris.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib

_TASK_TYPES = {
    "generation": ("gnrs.generation", "StructureGenerationTask"),
    "energy": ("gnrs.energy", "EnergyCalculationTask"),
    "optimize": ("gnrs.optimize", "GeometryOptimizationTask"),
    "descriptor": ("gnrs.descriptor", "DescriptorEvaluationTask"),
    "cluster": ("gnrs.cluster", "ClusterSelectionTask"),
}

_ENERGY_METHODS = {"maceoff", "uma", "aimnet", "aims", "vasp", "dftb"}

_OPTIMIZERS = {"bfgs", "lbfgs"}

_RIGID_PRESS_OPTIMIZERS = {"rigid_press", "symm_rigid_press"}

_CLUSTERERS = {"ap", "kmeans"}

_SELECTORS = {"center", "window"}

_DESCRIPTORS = {"acsf"}


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
