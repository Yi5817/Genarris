"""
The batch runners of energy calculators and optimizers skip structures the
task already completed (restored from checkpoints) and report every newly
completed structure by name. Run in a single process (no mpirun).
"""
from __future__ import annotations

import pytest
from ase import Atoms
from mpi4py import MPI

from gnrs.core.energy import EnergyCalculatorABC
from gnrs.core.optimizer import GeometryOptimizerABC

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() != 1, reason="single-process unit test"
)


class _Energy(EnergyCalculatorABC):
    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(MPI.COMM_WORLD, settings or {}, "maceoff")
        self.computed: list[Atoms] = []

    def initialize(self) -> None:
        pass

    def compute(self, xtal: Atoms) -> None:
        xtal.info[self.energy_name] = -1.0
        self.computed.append(xtal)

    def finalize(self) -> None:
        pass


class _Optimizer(GeometryOptimizerABC):
    def __init__(self, dft_serial_mode: bool = False) -> None:
        super().__init__(
            MPI.COMM_WORLD, {}, "bfgs", dft_serial_mode=dft_serial_mode
        )
        self.optimized: list[Atoms] = []

    def optimize(self, xtal: Atoms) -> None:
        self.optimized.append(xtal)

    def update(self, xtal: Atoms) -> None:
        xtal.info[self.opt_name] = "converged"


def _pool(n: int) -> dict[str, Atoms]:
    return {
        f"s{i}": Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[3.0] * 3, pbc=True)
        for i in range(n)
    }


@pytest.mark.parametrize("serial", [False, True])
def test_optimizer_skips_done_and_reports_the_rest(serial: bool) -> None:
    pool = _pool(3)
    # A result of an earlier task of the same type (e.g. bfgs_maceoff before
    # bfgs_uma) must not count as done for this task
    pool["s1"].info["bfgs"] = "stale"
    opt = _Optimizer(dft_serial_mode=serial)
    reported = []

    opt.run_batch(pool, lambda name, xtal: reported.append(name), done={"s0"})

    # Serial mode optimizes gathered copies, so compare results, not objects
    assert len(opt.optimized) == 2
    assert reported == ["s1", "s2"]
    assert pool["s1"].info["bfgs"] == pool["s2"].info["bfgs"] == "converged"
    assert "bfgs" not in pool["s0"].info


@pytest.mark.parametrize("serial", [False, True])
def test_energy_skips_done_and_reports_the_rest(serial: bool) -> None:
    pool = _pool(3)
    pool["s1"].info["maceoff"] = -9.0  # stale energy from before a relaxation
    calc = _Energy({"dft_mode": "serial"} if serial else None)
    reported = []

    calc.run_batch(pool, lambda name, xtal: reported.append(name), done={"s0"})

    assert len(calc.computed) == 2
    assert reported == ["s1", "s2"]
    assert pool["s1"].info["maceoff"] == pool["s2"].info["maceoff"] == -1.0
    assert "maceoff" not in pool["s0"].info


def test_failed_optimizations_are_dropped_from_the_pool() -> None:
    class _Failing(_Optimizer):
        def optimize(self, xtal: Atoms) -> None:
            raise RuntimeError("no convergence")

    pool = _pool(2)
    _Failing().run_batch(pool, done={"s0"})
    assert list(pool) == ["s0"]
