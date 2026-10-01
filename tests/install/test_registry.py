"""
Task names from a config file resolve to task classes without needing any
optional calculator packages.
"""
from __future__ import annotations

import sys

import pytest

from gnrs.core.registry import resolve_tasks


def test_core_task_names_resolve() -> None:
    names = [
        "generation", "rigid_press", "symm_rigid_press", "dedup", "acsf",
        "ap_center", "kmeans_window",
    ]
    specs = resolve_tasks(names)
    assert [spec.instance_id for spec in specs] == names
    assert all(isinstance(spec.cls, type) for spec in specs)


def test_energy_tasks_resolve_without_optional_calculators() -> None:
    """
    Resolving an energy task must load only the task class. The calculator
    module (and its optional dependency such as fairchem or mace) is imported
    when the task runs, so a core install can still parse such workflows.
    """
    names = ["maceoff", "uma", "aimnet", "aims", "vasp", "dftb", "bfgs_uma"]
    specs = resolve_tasks(names)
    assert [spec.instance_id for spec in specs] == names
    calculators = [
        name for name in sys.modules
        if name.startswith("gnrs.energy.") and name != "gnrs.energy.energy_task"
    ]
    assert not calculators, f"calculator modules imported too early: {calculators}"


def test_repeated_tasks_get_unique_ids() -> None:
    specs = resolve_tasks(["generation", "dedup", "symm_rigid_press", "dedup"])
    assert [spec.instance_id for spec in specs] == [
        "generation", "dedup_1", "symm_rigid_press", "dedup_2",
    ]


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        resolve_tasks(["generation", "not_a_task"])
