"""
Unit tests for structure checkpoints. Run in a single process (no mpirun).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io.jsonio import encode
from mpi4py import MPI

import gnrs.parallel as gp
from gnrs.parallel.structs import DistributedStructs

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() != 1, reason="single-process unit test"
)

KEY = "opt"


@pytest.fixture(autouse=True)
def _parallel() -> None:
    gp.init_parallel(MPI.COMM_WORLD)


def _atoms(result: float | None = None) -> Atoms:
    xtal = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 3.0], pbc=True)
    if result is not None:
        xtal.info[KEY] = result
    return xtal


def _pool(n: int) -> dict[str, Atoms]:
    return {f"s{i}": _atoms() for i in range(n)}


def _logged_names(ckpt: Path) -> list[str]:
    return [line.split(":", 1)[0].strip('"') for line in ckpt.read_text().splitlines()]


def test_save_appends_each_completed_structure_once(tmp_path: Path) -> None:
    rank_dir = tmp_path / "rank_0"
    rank_dir.mkdir()
    ds = DistributedStructs(_pool(3))

    ds.checkpoint_save(str(rank_dir), KEY)
    assert not list(rank_dir.iterdir()), "nothing completed, nothing written"

    ds.structs["s0"].info[KEY] = -1.0
    ds.checkpoint_save(str(rank_dir), KEY)
    ds.checkpoint_save(str(rank_dir), KEY)
    ds.structs["s1"].info[KEY] = -2.0
    ds.checkpoint_save(str(rank_dir), KEY)

    assert _logged_names(rank_dir / "0.ckpt") == ["s0", "s1"]


def test_load_merges_with_pool_and_skips_damaged_line(tmp_path: Path) -> None:
    rank_dir = tmp_path / "rank_0"
    rank_dir.mkdir()
    ds = DistributedStructs({"s0": _atoms(-1.0), "s1": _atoms(-2.0)})
    ds.checkpoint_save(str(rank_dir), KEY)
    with open(rank_dir / "0.ckpt", "a") as chk:
        chk.write('"s2": {"numbers": [1], "posi')  # cut short by a job kill

    ds = DistributedStructs(_pool(4))
    n_restored = ds.checkpoint_load(str(tmp_path), KEY)

    assert n_restored == 2
    assert sorted(ds.structs) == ["s0", "s1", "s2", "s3"]
    assert ds.structs["s0"].info[KEY] == -1.0
    assert KEY not in ds.structs["s2"].info, "damaged entry is recomputed"

    # Restored results are already on disk and must not be logged again
    ds.checkpoint_save(str(rank_dir), KEY)
    assert _logged_names(rank_dir / "0.ckpt") == ["s0", "s1", "s2"]

    # New results go on a fresh line after the damaged one
    ds.structs["s3"].info[KEY] = -4.0
    ds.checkpoint_save(str(rank_dir), KEY)
    assert _logged_names(rank_dir / "0.ckpt") == ["s0", "s1", "s2", "s3"]
    ds = DistributedStructs(_pool(4))
    assert ds.checkpoint_load(str(tmp_path), KEY) == 3
    assert ds.structs["s3"].info[KEY] == -4.0


def test_load_ignores_structures_outside_the_pool(tmp_path: Path) -> None:
    rank_dir = tmp_path / "rank_0"
    rank_dir.mkdir()
    # Log left behind by an unrelated earlier run in the same directory
    DistributedStructs({"old": _atoms(-9.0)}).checkpoint_save(str(rank_dir), KEY)
    DistributedStructs({"s0": _atoms(-1.0)}).checkpoint_save(str(rank_dir), KEY)

    ds = DistributedStructs(_pool(2))
    assert ds.checkpoint_load(str(tmp_path), KEY) == 1
    assert sorted(ds.structs) == ["s0", "s1"]
    assert ds.structs["s0"].info[KEY] == -1.0


def test_load_reads_legacy_snapshot(tmp_path: Path) -> None:
    rank_dir = tmp_path / "rank_0"
    rank_dir.mkdir()
    (rank_dir / "0.save").write_text(json.dumps({"s0": encode(_atoms(-1.0))}))

    ds = DistributedStructs(_pool(2))
    assert ds.checkpoint_load(str(tmp_path), KEY) == 1
    assert ds.structs["s0"].info[KEY] == -1.0
    assert KEY not in ds.structs["s1"].info


def test_load_without_checkpoints_keeps_pool(tmp_path: Path) -> None:
    ds = DistributedStructs(_pool(2))
    assert ds.checkpoint_load(str(tmp_path), KEY) == 0
    assert sorted(ds.structs) == ["s0", "s1"]


def test_clear_removes_all_checkpoint_files(tmp_path: Path) -> None:
    for rank in range(2):
        rank_dir = tmp_path / f"rank_{rank}"
        rank_dir.mkdir()
        (rank_dir / f"{rank}.ckpt").write_text("")
        (rank_dir / f"{rank}.save").write_text("{}")
    DistributedStructs.checkpoint_clear(str(tmp_path))
    assert not list(tmp_path.glob("rank_*/*"))
